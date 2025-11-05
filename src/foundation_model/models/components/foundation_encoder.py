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

from ..model_config import BaseEncoderConfig, MLPEncoderConfig, TransformerEncoderConfig
from .fc_layers import LinearBlock


class _TransformerBackbone(nn.Module):
    """Simple Transformer backbone for tabular features.

    The module interprets each scalar feature as a token, projects it to the
    model dimension, applies a stack of ``nn.TransformerEncoderLayer`` blocks
    and aggregates the contextualized representation either via a learnable
    ``[CLS]`` token or mean pooling.

    When ``use_cls_token`` is enabled the downstream ``deposit`` layer only sees
    the hidden state of the classifier token. The remaining feature tokens still
    participate in training because self-attention allows gradients to flow from
    the ``[CLS]`` query back through the full sequence: every token contributes
    keys and values to the attention updates that the classifier consumes.
    Disabling the ``[CLS]`` token switches to mean pooling, which exposes the
    aggregated hidden states of all tokens directly to the deposit layer and
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
            # Gradients from the downstream deposit layer flow into the `[CLS]`
            # token and, via self-attention, influence all feature token
            # representations even though only the classifier embedding is
            # returned here.
            latent = hidden[:, 0, :]
        else:
            # Mean pooling exposes every contextualised feature token to the
            # deposit layer while still producing a fixed-width latent vector.
            latent = hidden.mean(dim=1)

        return self.output_norm(latent)

    def encode_masked(self, x_masked: torch.Tensor) -> torch.Tensor:
        return self.forward(x_masked)


class FoundationEncoder(nn.Module):
    """
    Foundation model encoder providing shared representations for multi-task learning.

    This module encapsulates the core encoding layers that transform input features
    into a latent representation, followed by a deposit layer that serves as a buffer
    between the shared encoder and task-specific heads.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    deposit_dim : int | None
        Output dimension of the deposit layer.
    encoder_config : BaseEncoderConfig
        Encoder configuration defining the backbone implementation and latent
        dimensionality. ``MLPEncoderConfig`` yields the legacy fully connected
        stack, while ``TransformerEncoderConfig`` enables the transformer
        backbone.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_config: BaseEncoderConfig,
        deposit_dim: int | None = None,
    ):
        super().__init__()

        if deposit_dim is not None and deposit_dim <= 0:
            raise ValueError("deposit_dim must be positive when provided")

        self.encoder_config = encoder_config

        if isinstance(encoder_config, MLPEncoderConfig):
            hidden_dims = list(encoder_config.hidden_dims)
            if not hidden_dims:
                raise ValueError("MLP encoder requires at least one hidden dimension")
            self.shared = LinearBlock(
                [input_dim] + hidden_dims,
                normalization=encoder_config.norm,
                residual=encoder_config.residual,
            )
            latent_dim = hidden_dims[-1]
        elif isinstance(encoder_config, TransformerEncoderConfig):
            self.shared = _TransformerBackbone(
                input_dim=input_dim,
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
            raise TypeError(
                "encoder_config must be an instance of MLPEncoderConfig or TransformerEncoderConfig"
            )

        if deposit_dim is None:
            deposit_dim = latent_dim

        self.latent_dim = latent_dim
        self.deposit_dim = deposit_dim

        # Deposit layer serves as a buffer between shared encoder and task heads
        self.deposit = nn.Sequential(
            nn.Linear(latent_dim, deposit_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the foundation encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (B, input_dim).

        Returns
        -------
        latent : torch.Tensor
            Latent representation. The trailing dimension equals
            ``encoder_config.latent_dim``.
        task_representation : torch.Tensor
            Task representation after deposit layer, shape (B, deposit_dim).
        """
        latent = self.shared(x)
        task_representation = self.deposit(latent)

        return latent, task_representation

    def encode_masked(self, x_masked: torch.Tensor, is_structure: bool = False) -> torch.Tensor:
        """
        Encode masked features for self-supervised learning.

        Parameters
        ----------
        x_masked : torch.Tensor
            Masked input features.
        is_structure : bool
            Retained for backward compatibility. Ignored in this encoder.

        Returns
        -------
        torch.Tensor
            Encoded representation of masked features.
        """
        if hasattr(self.shared, "encode_masked"):
            return self.shared.encode_masked(x_masked)
        return self.shared(x_masked)
