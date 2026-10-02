# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Foundation encoder components for multi-task learning models.

This module provides the core encoder components that transform input features
into latent representations for multi-task learning models. Two encoder
variants are supported:

* A feed-forward multi-layer perceptron (MLP) implemented with ``LinearBlock``.
* A Transformer encoder using shared scalar or feature-specific scalar/group
  tokens and CLS, mean or concatenation pooling. A token-wise MLP is available
  as a no-attention control.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from foundation_model.models.model_config import (
    EncoderActivation,
    EncoderConfig,
    EncoderPooling,
    FeatureTokenization,
    MLPEncoderConfig,
    TransformerEncoderConfig,
)

from .fc_layers import LinearBlock


class _TransformerBackbone(nn.Module):
    """Simple Transformer backbone for tabular features.

    The module interprets each scalar feature as a token, projects it to the
    model dimension, applies a stack of ``nn.TransformerEncoderLayer`` blocks
    and aggregates the contextualized representation via a learnable
    ``[CLS]`` token, mean pooling or concatenation followed by projection.

    When ``use_cls_token`` is enabled the downstream task heads only see
    the hidden state of the classifier token. The remaining feature tokens still
    participate in training because self-attention allows gradients to flow from
    the ``[CLS]`` query back through the full sequence: every token contributes
    keys and values to the attention updates that the classifier consumes.
    Disabling the ``[CLS]`` token switches to mean pooling, which exposes the
    aggregated hidden states of all tokens directly to the task heads and
    distributes gradients evenly across the sequence.

    All modes therefore provide supervised training signals to every token
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
        pooling: EncoderPooling = EncoderPooling.CLS,
        output_dim: int | None = None,
        norm_first: bool = False,
        activation: str = "relu",
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
        self.pooling = pooling

        # Project scalar features (treated as tokens) to the transformer space.
        self.input_projection = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=activation,
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
        pooled_dim = input_dim * d_model if pooling is EncoderPooling.CONCAT else d_model
        self.output_norm = nn.LayerNorm(pooled_dim) if apply_layer_norm else nn.Identity()
        self.output_projection = (
            nn.Linear(pooled_dim, output_dim or d_model) if pooled_dim != (output_dim or d_model) else nn.Identity()
        )

    @staticmethod
    def _build_positional_encoding(seq_len: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""

        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: dim // 2])
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
        elif self.pooling is EncoderPooling.MEAN:
            # Mean pooling exposes every contextualised feature token to the
            # task heads while still producing a fixed-width latent vector.
            latent = hidden.mean(dim=1)
        else:
            latent = hidden.flatten(1)

        return self.output_projection(self.output_norm(latent))


class _FeatureBackbone(nn.Module):
    """Feature-specific scalar/group tokens, with attention or a token-wise MLP control."""

    def __init__(self, config: TransformerEncoderConfig) -> None:
        super().__init__()
        self.input_dim = config.input_dim
        self.group_size = config.group_size
        self.n_tokens = config.input_dim // config.group_size
        self.pooling = EncoderPooling(config.pooling or EncoderPooling.CLS)
        self.feature_weight = nn.Parameter(torch.empty(self.n_tokens, self.group_size, config.d_model))
        self.feature_bias = nn.Parameter(torch.empty(self.n_tokens, config.d_model))
        nn.init.uniform_(self.feature_weight, -1 / math.sqrt(self.group_size), 1 / math.sqrt(self.group_size))
        nn.init.uniform_(self.feature_bias, -1 / math.sqrt(self.group_size), 1 / math.sqrt(self.group_size))
        if self.pooling is EncoderPooling.CLS:
            self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.register_buffer("cls_token", None, persistent=False)
        ff_dim = config.dim_feedforward or config.d_model * 4
        self.transformer: nn.TransformerEncoder | nn.Sequential
        if config.use_attention:
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=ff_dim,
                dropout=config.dropout,
                batch_first=True,
                norm_first=bool(config.norm_first),
                activation=EncoderActivation(config.activation or EncoderActivation.GELU).value,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers, enable_nested_tensor=False)
            for block in self.transformer.layers:
                # Packed QKV parameters are not covered by the model's nn.Linear initializer.
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
        else:
            self.transformer = nn.Sequential(*[_TokenMLPBlock(config, ff_dim) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.d_model) if config.apply_layer_norm else nn.Identity()
        pooled_dim = self.n_tokens * config.d_model if self.pooling is EncoderPooling.CONCAT else config.d_model
        self.output_projection = (
            nn.Linear(pooled_dim, config.latent_dim) if pooled_dim != config.latent_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Feature encoder expects shape (batch, {self.input_dim}); received {tuple(x.shape)}")
        groups = x.reshape(x.shape[0], self.n_tokens, self.group_size)
        tokens = torch.einsum("bng,ngd->bnd", groups, self.feature_weight) + self.feature_bias
        if self.cls_token is not None:
            tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), tokens], dim=1)
        hidden = self.output_norm(self.transformer(tokens))
        if self.pooling is EncoderPooling.CLS:
            pooled = hidden[:, 0]
        elif self.pooling is EncoderPooling.MEAN:
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden.flatten(1)
        return self.output_projection(pooled)


class _TokenMLPBlock(nn.Module):
    """Residual token-wise feed-forward control without token-to-token attention."""

    def __init__(self, config: TransformerEncoderConfig, hidden: int) -> None:
        super().__init__()
        self.norm_first = config.norm_first
        self.norm = nn.LayerNorm(config.d_model)
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU() if config.activation is EncoderActivation.GELU else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            return x + self.net(self.norm(x))
        return self.norm(x + self.net(x))


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
        self.shared: LinearBlock | _TransformerBackbone | _FeatureBackbone
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
            if encoder_config.tokenization is not FeatureTokenization.SHARED:
                self.shared = _FeatureBackbone(encoder_config)
            else:
                self.shared = _TransformerBackbone(
                    input_dim=encoder_config.input_dim,
                    d_model=encoder_config.d_model,
                    num_layers=encoder_config.num_layers,
                    nhead=encoder_config.nhead,
                    dim_feedforward=encoder_config.dim_feedforward,
                    dropout=encoder_config.dropout,
                    use_cls_token=encoder_config.use_cls_token,
                    apply_layer_norm=encoder_config.apply_layer_norm,
                    pooling=EncoderPooling(
                        encoder_config.pooling
                        or (EncoderPooling.CLS if encoder_config.use_cls_token else EncoderPooling.MEAN)
                    ),
                    output_dim=encoder_config.output_dim,
                    norm_first=bool(encoder_config.norm_first),
                    activation=EncoderActivation(encoder_config.activation or EncoderActivation.RELU).value,
                )
            latent_dim = encoder_config.latent_dim
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
