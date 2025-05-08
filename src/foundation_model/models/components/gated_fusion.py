"""
Gated fusion module for combining formula and structure representations.
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining formula and structure representations.

    Implements h = h_formula + sigmoid(W [h_f, h_s]) * h_structure

    Parameters
    ----------
    dim : int
        Dimension of the input feature vectors.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Gate network takes concatenated features as input and outputs a scalar
        self.gate = nn.Linear(dim * 2, 1)

    def forward(
        self, h_f: torch.Tensor, h_s: torch.Tensor, has_s: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the gated fusion module.

        Parameters
        ----------
        h_f : torch.Tensor
            Formula representation, shape (B, dim).
        h_s : torch.Tensor
            Structure representation, shape (B, dim).
        has_s : torch.Tensor
            Binary indicator if structure information is available, shape (B, 1).
            Used to mask the gating mechanism when no structure is available.

        Returns
        -------
        torch.Tensor
            Fused representation, shape (B, dim).
        """
        # Compute gating factor and apply mask for samples without structure
        g = torch.sigmoid(self.gate(torch.cat([h_f, h_s], dim=-1))) * has_s

        # Fuse the representations using the gate
        return h_f + g * h_s
