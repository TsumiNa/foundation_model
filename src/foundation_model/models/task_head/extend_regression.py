# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

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
    Encode scalar t into Fourier features using random Fourier features.
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
    Extended regression head for handling variable-length sequences of (t, target) pairs.

    Uses a decomposition approach: y = f_x(x) + f_t(t) + g_x(x) * g_t(t)
    where f_x, f_t, g_x, g_t are implemented using LinearBlock networks.

    Parameters
    ----------
    config : ExtendRegressionTaskConfig
        Configuration object containing network dimensions, normalization settings,
        and other parameters.
    """

    def __init__(self, config: ExtendRegressionTaskConfig):
        super().__init__(config)

        # Store configuration parameters
        self.t_encoding_method = config.t_encoding_method

        # Calculate t_embedding_dim from t_dim[0]
        t_embedding_dim = config.t_dim[0]

        # Initialize t encoder based on encoding method
        self.t_encoder: nn.Module  # Accept both FourierFeatures and Sequential

        if self.t_encoding_method == "fourier":
            # Calculate t_input_dim for Fourier encoding
            # If t_embedding_dim is odd, round up to ensure sufficient features
            if t_embedding_dim % 2 == 1:
                t_input_dim = math.ceil(t_embedding_dim / 2)
            else:
                t_input_dim = t_embedding_dim // 2

            self.t_encoder = FourierFeatures(input_dim=1, mapping_size=t_input_dim)
            encoded_t_dim = t_input_dim * 2  # Fourier features: sin + cos

        elif self.t_encoding_method == "fc":
            # For FC encoding, t_input_dim equals t_embedding_dim
            self.t_encoder = nn.Sequential(nn.Linear(1, t_embedding_dim), nn.LeakyReLU(0.1))
            encoded_t_dim = t_embedding_dim
        else:
            raise ValueError(f"Unsupported t_encoding_method: {self.t_encoding_method}. Must be 'fourier' or 'fc'.")

        # Initialize networks using LinearBlock
        # f_x: processes material features x
        self.f_x = LinearBlock(
            config.x_dim,
            normalization=config.norm,
            residual=config.residual,
            dim_output_layer=1,  # Output scalar value
        )

        # f_t: processes encoded t features
        self.f_t = LinearBlock(
            [encoded_t_dim] + config.t_dim[1:],
            normalization=config.norm,
            residual=config.residual,
            dim_output_layer=1,  # Output scalar value
        )

        # g_x: processes material features for interaction
        self.g_x = LinearBlock(config.x_dim, normalization=config.norm, residual=config.residual)

        # g_t: processes encoded t features for interaction
        self.g_t = LinearBlock(
            [encoded_t_dim] + config.t_dim[1:],
            normalization=config.norm,
            residual=config.residual,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_) -> torch.Tensor:
        """
        Forward pass of the extended regression head.

        Parameters
        ----------
        x : torch.Tensor
            Material features, shape (N, x_dim[å0])
        t : torch.Tensor
            Parameter values (e.g., energy, temperature), shape (N,) or (N, 1)

        Returns
        -------
        torch.Tensor
            Predicted values, shape (N, 1)
        """
        # Ensure t has correct dimension
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # Encode t using the configured encoding method
        t_encoded = self.t_encoder(t)

        # Compute model output using decomposition
        fx_out = self.f_x(x)  # (N, 1)
        ft_out = self.f_t(t_encoded)  # (N, 1)
        gx_out = self.g_x(x)  # (N, interaction_dim)
        gt_out = self.g_t(t_encoded)  # (N, interaction_dim)

        # Compute interaction term
        interaction = (gx_out * gt_out).sum(dim=1, keepdim=True)  # (N, 1)

        # Final output: additive decomposition + interaction
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
            Predicted values, shape (N, 1) or (N,)
        target : torch.Tensor
            Target values, shape (N, 1) or (N,)
        mask : torch.Tensor, optional
            Binary mask indicating valid targets, shape (N, 1) or (N,).
            If None, all targets are considered valid.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (total_loss, per_dim_loss) where total_loss is a scalar tensor
            and per_dim_loss contains loss per output dimension.
        """
        # Ensure consistent shapes
        if pred.dim() == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 2 and target.shape[1] == 1:
            target = target.squeeze(1)

        if mask is None:
            mask = torch.ones_like(target)
        elif mask.dim() == 2 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        # Apply mask to both predictions and targets
        losses = F.mse_loss(pred, target, reduction="none") * mask

        # Compute per-dimension losses (average over batch)
        # For extended regression, there's only one dimension
        per_dim_loss = torch.nan_to_num(losses.sum() / mask.sum().clamp_min(1.0), nan=0.0, posinf=0.0, neginf=0.0)

        # Compute total loss
        total_loss = per_dim_loss

        return total_loss, per_dim_loss.unsqueeze(0)

    def _predict_impl(self, x: torch.Tensor, additional: bool = False) -> dict[str, ndarray]:
        """
        Core prediction logic for extended regression.

        Parameters
        ----------
        x : torch.Tensor
            Raw output from the forward pass (regression values).
        additional : bool, optional
            If True, return additional prediction information. Currently unused
            for extended regression, but kept for interface consistency. Defaults to False.

        Returns
        -------
        dict[str, ndarray]
            A dictionary containing the prediction: {"value": x}.
        """
        return {"value": x.detach().cpu().numpy()}
