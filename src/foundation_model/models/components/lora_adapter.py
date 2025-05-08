"""
LoRA (Low-Rank Adaptation) implementation for the FlexibleMultiTaskModel.
"""

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """
    LoRA: Low-Rank Adaptation of Large Language Models.

    Implements ΔW = α / r · U @ V
    Wraps an existing nn.Linear layer; base weights can be frozen.

    Parameters
    ----------
    base_linear : nn.Linear
        The base linear layer to adapt.
    r : int
        Rank of the low-rank adaptation matrices. Lower values give fewer
        parameters but potentially less expressive power.
    alpha : float
        Scaling factor for the adaptation. Higher values give stronger
        adaptation effect.
    freeze_base : bool
        If True, the base layer's parameters are frozen during fine-tuning.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = base_linear

        # Freeze base parameters if requested
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

        # Get dimensions from base linear layer
        in_dim, out_dim = base_linear.in_features, base_linear.out_features

        # Initialize low-rank matrices with small random values
        self.U = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.V = nn.Parameter(torch.randn(r, out_dim) * 0.01)

        # Scaling factor
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LoRA adapter.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with LoRA adaptation applied.
        """
        # Original output + scaled low-rank adaptation
        return self.base(x) + self.scale * (x @ self.U @ self.V)
