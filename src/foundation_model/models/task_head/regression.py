"""
Regression task head for the FlexibleMultiTaskModel.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray

from foundation_model.models.fc_layers import LinearBlock
from foundation_model.models.model_config import RegressionTaskConfig

from .base import BaseTaskHead


class RegressionHead(BaseTaskHead):
    """
    Regression task head for predicting continuous values.

    Parameters
    ----------
    config : RegressionTaskConfig
        Configuration object containing parameters like input dimension (`d_in`),
        task name (`name`), hidden dimensions (`dims`), normalization (`norm`),
        and residual connections (`residual`).
    """

    def __init__(self, config: RegressionTaskConfig):  # Changed signature
        super().__init__(config)

        if not hasattr(config, "dims") or not config.dims:
            raise ValueError("RegressionHead config must have 'dims' attribute, and it cannot be empty.")
        d_in = config.dims[0]  # d_in sourced from config
        # head_internal_dims are dimensions after d_in, including the final output dimension
        head_internal_dims = config.dims[1:]
        if not head_internal_dims:
            raise ValueError(
                "RegressionHead config 'dims' must include at least an output dimension after the input dimension."
            )

        norm = config.norm
        residual = config.residual

        self.net = LinearBlock(
            [d_in] + head_internal_dims[:-1],  # Input to LinearBlock is d_in, hidden are head_internal_dims[:-1]
            normalization=norm,
            residual=residual,
            dim_output_layer=head_internal_dims[-1],  # Output layer size is the last element of head_internal_dims
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the regression head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, d_in).
        **kwargs : dict
            Additional arguments (not used in regression head).

        Returns
        -------
        torch.Tensor
            Predicted regression values, shape (B, output_dim).
        """
        return self.net(x)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute masked MSE loss for regression.

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
        torch.Tensor | None
            Total loss as a scalar tensor, or None if no valid samples in the batch.
        """
        if mask is None:
            mask = torch.ones_like(target)

        # Check if there are any valid samples
        valid_count = mask.sum()
        if valid_count == 0:
            # No valid samples in this batch for this task
            return None

        # Apply mask to both predictions and targets
        losses = F.mse_loss(pred, target, reduction="none") * mask

        # Compute total loss - simple division without defensive clamp
        total_loss = losses.sum() / valid_count

        return total_loss

    def _predict_impl(self, x: torch.Tensor) -> dict[str, ndarray]:
        """
        Core prediction logic for regression.

        Parameters
        ----------
        x : torch.Tensor
            Raw output from the forward pass (regression values).
        additional : bool, optional
            If True, return additional prediction information. Currently unused
            for regression, but kept for interface consistency. Defaults to False.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the prediction: {"value": x}.
        """
        return {"value": x.detach().cpu().numpy()}
