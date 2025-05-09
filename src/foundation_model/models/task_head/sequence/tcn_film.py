"""
Dilated Temporal Convolutional Network with FiLM modulation for sequence prediction.
"""

import torch
import torch.nn as nn

from ..sequence.base import SequenceBaseHead


class _DilatedTCN(nn.Module):
    """Simple 1-D Dilated TCN block stack with residual connections."""

    def __init__(self, channels: int, n_layers: int = 4, kernel_size: int = 3):
        super().__init__()
        dilations = [2**i for i in range(n_layers)]
        pads = [((kernel_size - 1) * d) // 2 for d in dilations]
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=p,
                )
                for d, p in zip(dilations, pads)
            ]
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, C, L).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (B, C, L).
        """
        # x: (B,C,L)
        out = x
        for conv in self.layers:
            res = out
            out = self.act(conv(out))
            out = out + res
        return out


class SequenceHeadTCNFiLM(SequenceBaseHead):
    """
    Dilated-TCN + FiLM modulation sequence head.

    This head processes sequence points using a dilated temporal convolutional network
    and modulates the features using FiLM conditioning from the deposit layer representation.

    The processing flow is:
    * temps → linear → (B,hidden,L) → Dilated-TCN
    * FiLM: γ,β from latent h (B,1,hidden)
    * output linear → (B,L)

    Parameters
    ----------
    d_in : int
        Dimension of the latent input.
    name : str
        Name of the task.
    hidden : int
        Hidden dimension for the TCN.
    n_layers : int
        Number of dilated convolutional layers.
    """

    def __init__(self, d_in: int, name: str, hidden: int = 128, n_layers: int = 4):
        super().__init__(d_in, name)
        self.temp_proj = nn.Linear(1, hidden)
        self.tcn = _DilatedTCN(hidden, n_layers=n_layers)
        self.film = nn.Linear(d_in, 2 * hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Task-specific representation tensor from the deposit layer, shape (B, D).
        temps : torch.Tensor
            Temperature points, shape (B, L, 1).

        Returns
        -------
        torch.Tensor
            Predicted sequence, shape (B, L).
        """
        # temps:(B,L,1)
        B, L, _ = temps.shape
        x = self.temp_proj(temps).permute(0, 2, 1)  # (B,hidden,L)
        x = self.tcn(x).permute(0, 2, 1)  # (B,L,hidden)
        gamma, beta = self.film(h).unsqueeze(1).chunk(2, dim=-1)  # (B,1,H)
        fused = gamma * x + beta
        y = self.out(fused).squeeze(-1)  # (B,L)
        return y
