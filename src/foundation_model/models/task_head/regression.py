"""
Regression task head for the FlexibleMultiTaskModel.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.lora_adapter import LoRAAdapter
from ..fc_layers import LinearBlock
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

    def __init__(self, config: object):  # TODO: Use specific RegressionTaskConfig type hint
        super().__init__(config)

        # Extract parameters from config, providing defaults if necessary
        d_in = config.d_in
        dims = getattr(config, "dims", [1])  # Default to single output if dims missing
        norm = getattr(config, "norm", True)
        residual = getattr(config, "residual", False)
        lora_rank = getattr(config, "lora_rank", None)
        lora_alpha = getattr(config, "lora_alpha", 1.0)

        # Construct the network using LinearBlock for internal layers
        # Ensure dims has at least one element for output layer size
        if not dims:
            raise ValueError("RegressionHead config 'dims' cannot be empty.")

        self.net = LinearBlock(
            [d_in] + dims[:-1],
            normalization=norm,
            residual=residual,
            dim_output_layer=dims[-1],  # Adds final layer automatically
        )

        # Apply LoRA to the final layer if requested
        if lora_rank is not None and lora_rank > 0:
            # The last layer in LinearBlock is the output layer
            last_layer = self.net[-1].layer if hasattr(self.net[-1], "layer") else self.net[-1]

            if isinstance(last_layer, nn.Linear):
                # Replace with LoRA adapter
                self.net[-1] = LoRAAdapter(last_layer, r=lora_rank, alpha=lora_alpha, freeze_base=True)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _predict_impl(self, x: torch.Tensor, additional: bool = False) -> dict[str, torch.Tensor]:
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
            A dictionary containing the prediction: {"prediction": x}.
        """
        return {"prediction": x}
