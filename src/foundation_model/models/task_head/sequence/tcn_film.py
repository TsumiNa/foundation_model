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
    config : SequenceTaskConfig (or similar)
        Configuration object containing parameters like input dimension (`d_in`),
        task name (`name`), hidden dimension (`hidden`), and number of TCN layers (`n_layers`).
    """

    def __init__(self, config: object):  # TODO: Use specific SequenceTaskConfig type hint
        super().__init__(config)

        # Extract parameters from config
        d_in = config.d_in
        hidden = getattr(config, "hidden", 128)  # Default hidden size
        n_layers = getattr(config, "n_layers", 4)  # Default number of layers
        # kernel_size could also be configurable
        kernel_size = getattr(config, "kernel_size", 3)

        self.temp_proj = nn.Linear(1, hidden)  # Project input sequence points
        self.tcn = _DilatedTCN(hidden, n_layers=n_layers, kernel_size=kernel_size)
        self.film = nn.Linear(d_in, 2 * hidden)  # FiLM generator
        self.out = nn.Linear(hidden, 1)  # Output layer

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
        # temps: (B, L, 1) -> x: (B, hidden, L)
        # 1. Project temperature points to hidden dimension
        x = self.temp_proj(temps).permute(0, 2, 1)  # Shape: (B, hidden, L)

        # 2. Process projected points through TCN
        x = self.tcn(x)  # Shape remains (B, hidden, L)

        # 3. Transpose back for FiLM modulation: (B, L, hidden)
        x = x.permute(0, 2, 1)

        # 4. Generate FiLM parameters from task representation h
        gamma_beta = self.film(h).unsqueeze(1).chunk(2, dim=-1)  # Shapes: (B, 1, hidden) each
        gamma, beta = gamma_beta

        # 5. Apply FiLM modulation
        fused = gamma * x + beta  # Broadcasting applies gamma/beta to each time step

        # 6. Project to output dimension
        y = self.out(fused).squeeze(-1)  # Shape: (B, L)
        return y

    def _predict_impl(self, x: torch.Tensor, additional: bool = False) -> dict[str, torch.Tensor]:
        """
        Core prediction logic for TCN-FiLM sequence head.

        Parameters
        ----------
        x : torch.Tensor
            Raw output from the forward pass (predicted sequence).
        additional : bool, optional
            If True, return additional prediction information. Currently unused,
            but kept for interface consistency. Defaults to False.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the prediction: {"prediction": x}.
        """
        # For TCN-FiLM sequence head, the raw output is the prediction
        return {"prediction": x}
