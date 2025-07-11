"""
Regression task head for the FlexibleMultiTaskModel.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray

from foundation_model.models.components.lora_adapter import LoRAAdapter
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
        residual connections (`residual`), LoRA rank (`lora_rank`), and
        LoRA alpha (`lora_alpha`).
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

        # LoRA specific attributes from config
        lora_enabled = getattr(config, "lora_enabled", False)
        lora_rank = getattr(config, "lora_rank", 0)
        lora_alpha = getattr(config, "lora_alpha", 1.0)
        lora_freeze_base = getattr(config, "lora_freeze_base", True)

        self.net = LinearBlock(
            [d_in] + head_internal_dims[:-1],  # Input to LinearBlock is d_in, hidden are head_internal_dims[:-1]
            normalization=norm,
            residual=residual,
            dim_output_layer=head_internal_dims[-1],  # Output layer size is the last element of head_internal_dims
        )

        # Apply LoRA to the final layer if requested
        if lora_enabled and lora_rank > 0:  # Check enabled flag and rank
            # Access the output layer of LinearBlock using its attribute (assumed to be 'output_layer')
            if hasattr(self.net, "output_layer") and isinstance(self.net.output_layer, nn.Linear):
                self.net.output_layer = LoRAAdapter(
                    self.net.output_layer, r=lora_rank, alpha=lora_alpha, freeze_base=lora_freeze_base
                )
            elif (
                hasattr(self.net, "output_layer")
                and isinstance(self.net.output_layer, nn.Module)
                and hasattr(self.net.output_layer, "layer")
                and isinstance(self.net.output_layer.layer, nn.Linear)
            ):
                self.net.output_layer.layer = LoRAAdapter(
                    self.net.output_layer.layer, r=lora_rank, alpha=lora_alpha, freeze_base=lora_freeze_base
                )
            # else:
            # Consider logging a warning if LoRA couldn't be applied as expected
            # print(f"Warning: Could not apply LoRA to RegressionHead {config.name} as expected.")

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
