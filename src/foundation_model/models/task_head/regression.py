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
    d_in : int
        Input dimension to the task head.
    name : str
        Name of the regression task.
    dims : list[int]
        Hidden dimensions of the regression network. The last dimension should be 1
        for scalar regression or > 1 for multi-target regression.
    norm : bool
        Whether to apply layer normalization.
    residual : bool
        Whether to use residual connections.
    lora_rank : Optional[int]
        If not None, apply LoRA adaptation with the specified rank.
    lora_alpha : float
        Scaling factor for LoRA adaptation.
    """

    def __init__(
        self,
        d_in: int,
        name: str,
        dims: list[int],
        norm: bool = True,
        residual: bool = False,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
    ):
        super().__init__(d_in, name)

        # Construct the network using LinearBlock for internal layers
        self.net = LinearBlock(
            [d_in] + dims[:-1],
            normalization=norm,
            residual=residual,
            dim_output_layer=dims[-1],  # Adds final layer automatically
        )

        # Apply LoRA to the final layer if requested
        if lora_rank is not None and lora_rank > 0:
            # The last layer in LinearBlock is the output layer
            last_layer = (
                self.net[-1].layer if hasattr(self.net[-1], "layer") else self.net[-1]
            )

            if isinstance(last_layer, nn.Linear):
                # Replace with LoRA adapter
                self.net[-1] = LoRAAdapter(
                    last_layer, r=lora_rank, alpha=lora_alpha, freeze_base=True
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
        per_dim_loss = torch.nan_to_num(
            losses.sum(0) / mask.sum(0), nan=0.0, posinf=0.0, neginf=0.0
        )

        # Compute total loss (sum over all dimensions, average over valid points)
        total_loss = losses.sum() / mask.sum().clamp_min(1.0)

        return total_loss, per_dim_loss
