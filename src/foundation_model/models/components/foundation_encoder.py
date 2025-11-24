# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Foundation encoder components for multi-task learning models.

This module provides the core encoder components that transform input features
into latent representations for multi-task learning models. Two encoder
variants are supported:

* A feed-forward multi-layer perceptron (MLP) implemented with ``LinearBlock``.
* A lightweight Transformer encoder that treats each scalar feature as a
  sequence token and aggregates the contextual representation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from foundation_model.models.model_config import EncoderConfig, MLPEncoderConfig, TransformerEncoderConfig

from .fc_layers import LinearBlock


class _TransformerBackbone(nn.Module):
    """Simple Transformer backbone for tabular features.

    The module interprets each scalar feature as a token, projects it to the
    model dimension, applies a stack of ``nn.TransformerEncoderLayer`` blocks
    and aggregates the contextualized representation either via a learnable
    ``[CLS]`` token or mean pooling.

    When ``use_cls_token`` is enabled the downstream task heads only see
    the hidden state of the classifier token. The remaining feature tokens still
    participate in training because self-attention allows gradients to flow from
    the ``[CLS]`` query back through the full sequence: every token contributes
    keys and values to the attention updates that the classifier consumes.
    Disabling the ``[CLS]`` token switches to mean pooling, which exposes the
    aggregated hidden states of all tokens directly to the task heads and
    distributes gradients evenly across the sequence.

    Both modes therefore provide supervised training signals to every token
    representation without relying on masked language modeling style
    pre-training.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        *,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        apply_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"Transformer d_model must be divisible by nhead. Received d_model={d_model} and nhead={nhead}."
            )

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.input_dim = input_dim
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # Project scalar features (treated as tokens) to the transformer space.
        self.input_projection = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.register_buffer("cls_token", None, persistent=False)

        self.position_encoding: torch.Tensor
        position_encoding = self._build_positional_encoding(
            input_dim + (1 if use_cls_token else 0),
            d_model,
        )
        self.register_buffer("position_encoding", position_encoding, persistent=False)
        self.output_norm = nn.LayerNorm(d_model) if apply_layer_norm else nn.Identity()

    @staticmethod
    def _build_positional_encoding(seq_len: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""

        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                "Transformer encoder expects 2D input tensors with shape (batch, features). "
                f"Received tensor with shape {tuple(x.shape)}."
            )

        batch_size, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                "Input feature dimension mismatch for transformer encoder. "
                f"Configured for {self.input_dim} features but received {feature_dim}."
            )

        # Treat each scalar feature as a token by projecting it individually.
        tokens = self.input_projection(x.unsqueeze(-1))

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = tokens + self.position_encoding[:, : tokens.size(1), :]
        hidden = self.transformer(tokens)

        if self.use_cls_token:
            # Gradients from the downstream task heads flow into the `[CLS]`
            # token and, via self-attention, influence all feature token
            # representations even though only the classifier embedding is
            # returned here.
            latent = hidden[:, 0, :]
        else:
            # Mean pooling exposes every contextualised feature token to the
            # task heads while still producing a fixed-width latent vector.
            latent = hidden.mean(dim=1)

        return self.output_norm(latent)


class FoundationEncoder(nn.Module):
    """
    Foundation model encoder providing shared latent representations for multi-task learning.

    This module encapsulates the core encoding layers that transform input features
    into a latent representation. Task-specific activation (Tanh) is applied at the
    FlexibleMultiTaskModel level.

    Parameters
    ----------
    encoder_config : BaseEncoderConfig
        Encoder configuration defining the backbone implementation, latent
        dimensionality, and input dimension. ``MLPEncoderConfig`` yields the
        fully connected stack, while ``TransformerEncoderConfig`` enables the
        transformer backbone.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
    ):
        super().__init__()
        self.shared: LinearBlock | _TransformerBackbone
        self.encoder_config = encoder_config
        # Both MLPEncoderConfig and TransformerEncoderConfig define input_dim
        self.input_dim = encoder_config.input_dim

        if isinstance(encoder_config, MLPEncoderConfig):
            hidden_dims = list(encoder_config.hidden_dims)
            # hidden_dims includes input_dim as the first element
            if len(hidden_dims) < 2:
                raise ValueError("MLP encoder requires input_dim + at least one hidden/latent dimension")
            self.shared = LinearBlock(
                hidden_dims,
                normalization=encoder_config.norm,
                residual=encoder_config.residual,
            )
            latent_dim = hidden_dims[-1]
        elif isinstance(encoder_config, TransformerEncoderConfig):
            self.shared = _TransformerBackbone(
                input_dim=encoder_config.input_dim,
                d_model=encoder_config.d_model,
                num_layers=encoder_config.num_layers,
                nhead=encoder_config.nhead,
                dim_feedforward=encoder_config.dim_feedforward,
                dropout=encoder_config.dropout,
                use_cls_token=encoder_config.use_cls_token,
                apply_layer_norm=encoder_config.apply_layer_norm,
            )
            latent_dim = encoder_config.d_model
        else:  # pragma: no cover - defensive branch
            raise TypeError("encoder_config must be an instance of MLPEncoderConfig or TransformerEncoderConfig")

        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the foundation encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (B, input_dim).

        Returns
        -------
        latent : torch.Tensor
            Latent representation, shape (B, latent_dim).
            Task-specific activation (Tanh) is applied at the FlexibleMultiTaskModel level.
        """
        return self.shared(x)
