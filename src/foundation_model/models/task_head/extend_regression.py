# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Extended regression task head for handling both x and t inputs with interaction terms.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray

from foundation_model.models.fc_layers import LinearBlock
from foundation_model.models.model_config import ExtendRegressionTaskConfig

from .base import BaseTaskHead


class FourierFeatures(nn.Module):
    """
    Encode scalar t into Fourier features
    """

    def __init__(self, input_dim: int, mapping_size: int, scale: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn((input_dim, mapping_size)) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # Ensure t is float type
        t = t.float()
        # Core operation: (batch_size, 1) @ (1, mapping_size) -> (batch_size, mapping_size)
        x_proj = 2 * math.pi * t @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ExtendRegressionHead(BaseTaskHead):
    """
    Extended regression task head for predicting continuous values with x and t inputs.

    This head handles both x (feature) and t (temporal/scalar) inputs, computing:
    y = f_x(x) + f_t(t) + interaction(g_x(x), g_t(t))

    Parameters
    ----------
    config : ExtendRegressionTaskConfig
        Configuration object containing parameters like x_dim, t_dim, interaction_dim,
        t_encoding_method, normalization, and residual connections.
    """

    def __init__(self, config: ExtendRegressionTaskConfig):
        super().__init__(config)

        # Calculate t_embedding_dim and t encoding configuration
        t_embedding_dim = config.t_dim[0]

        if config.t_encoding_method == "fourier":
            mapping_size = math.ceil(t_embedding_dim / 2)
            self.t_encoder = FourierFeatures(input_dim=1, mapping_size=mapping_size)
            t_input_dim = mapping_size * 2
        elif config.t_encoding_method == "fc":
            t_input_dim = t_embedding_dim
            self.t_encoder = LinearBlock(
                [1] + config.t_dim[:-1],
                normalization=config.norm,
                residual=config.residual,
                dim_output_layer=t_embedding_dim,
            )
        else:
            raise ValueError(f"Unknown t_encoding_method: {config.t_encoding_method}")

        # Build network components using LinearBlock
        # f_x: x -> ... -> 1 (direct effect of x)
        self.f_x = LinearBlock(
            config.x_dim,
            normalization=config.norm,
            residual=config.residual,
            dim_output_layer=1,
        )

        # f_t: t_encoded -> ... -> 1 (direct effect of t)
        self.f_t = LinearBlock(
            [t_input_dim] + config.t_dim,
            normalization=config.norm,
            residual=config.residual,
            dim_output_layer=1,
        )

        # g_x: x -> ... -> interaction_dim (for interaction with t)
        self.g_x = LinearBlock(
            config.x_dim,
            normalization=config.norm,
            residual=config.residual,
            dim_output_layer=config.interaction_dim,
        )

        # g_t: t_encoded -> ... -> interaction_dim (for interaction with x)
        self.g_t = LinearBlock(
            [t_input_dim] + config.t_dim,
            normalization=config.norm,
            residual=config.residual,
            dim_output_layer=config.interaction_dim,
        )

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass of the extended regression head.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor, shape (B, x_dim).
        t : torch.Tensor, optional
            Temporal/scalar input tensor, shape (B,) or (B, 1).
        **kwargs : dict
            Additional arguments (not used).

        Returns
        -------
        torch.Tensor
            Predicted regression values, shape (B, 1).
        """
        if t is None:
            raise ValueError("t parameter is required for ExtendRegressionHead")

        # Ensure t has correct dimension for encoding
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # Encode t according to the selected method
        t_encoded = self.t_encoder(t)

        # Compute model output components
        fx_out = self.f_x(x)  # Direct effect of x
        ft_out = self.f_t(t_encoded)  # Direct effect of t
        gx_out = self.g_x(x)  # x component for interaction
        gt_out = self.g_t(t_encoded)  # t component for interaction

        # Compute interaction term
        interaction = (gx_out * gt_out).sum(dim=1, keepdim=True)

        # Final output: additive decomposition
        y_hat = fx_out + ft_out + interaction
        return y_hat

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute masked MSE loss for extended regression.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values, shape (B, output_dim).
        target : torch.Tensor
            Target values, shape (B, output_dim).
        mask : torch.Tensor, optional
            Binary mask indicating valid targets, shape (B, output_dim).
            If None, all targets are considered valid.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (total_loss, per_dim_loss) where total_loss is a scalar tensor
            and per_dim_loss contains loss per output dimension.
        """
        if mask is None:
            mask = torch.ones_like(target)

        # Apply mask to both predictions and targets
        losses = F.mse_loss(pred, target, reduction="none") * mask

        # Compute per-dimension losses (average over batch)
        per_dim_loss = torch.nan_to_num(losses.sum(0) / mask.sum(0), nan=0.0, posinf=0.0, neginf=0.0)

        # Compute total loss (sum over all dimensions, average over valid points)
        total_loss = losses.sum() / mask.sum().clamp_min(1.0)

        return total_loss, per_dim_loss

    def _predict_impl(self, x: torch.Tensor, additional: bool = False) -> dict[str, ndarray]:
        """
        Core prediction logic for extended regression.

        Parameters
        ----------
        x : torch.Tensor
            Raw output from the forward pass (regression values).
        additional : bool, optional
            If True, return additional prediction information. Currently unused
            for regression, but kept for interface consistency. Defaults to False.

        Returns
        -------
        dict[str, ndarray]
            A dictionary containing the prediction: {"value": x}.
        """
        return {"value": x.detach().cpu().numpy()}
