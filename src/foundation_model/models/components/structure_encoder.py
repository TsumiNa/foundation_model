"""
Structure encoder module for the FlexibleMultiTaskModel.
"""

import torch
import torch.nn as nn

from ..fc_layers import LinearBlock


class StructureEncoder(nn.Module):
    """
    Simple MLP encoder for structure descriptors (vector form).

    Parameters
    ----------
    d_in : int
        Input dimension of structure descriptor vector.
    hidden_dims : list[int]
        Widths of hidden layers (last element should equal formula latent dim).
    norm, residual : bool
        Same switches as LinearBlock.
    """

    def __init__(
        self,
        d_in: int,
        hidden_dims: list[int],
        norm: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        self.net = LinearBlock(
            [d_in] + hidden_dims,
            normalization=norm,
            residual=residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the structure encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input structure descriptor, shape (B, d_in).

        Returns
        -------
        torch.Tensor
            Encoded structure representation, shape (B, hidden_dims[-1]).
        """
        return self.net(x)
