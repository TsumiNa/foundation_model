# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
RNN-based sequence head for the FlexibleMultiTaskModel.
"""

import torch
import torch.nn as nn

from ..sequence.base import SequenceBaseHead


class SequenceHeadRNN(SequenceBaseHead):
    """
    GRU/LSTM sequence head with FiLM conditioning.

    Parameters
    ----------
    d_in : int
        Dimension of the latent vector from the encoder.
    name : str
        Name of the sequence task.
    hidden : int, optional
        Hidden size of the recurrent layer (default: 128).
    cell : {"gru","lstm"}, optional
        Select GRU or LSTM cell (default: "gru").
    """

    def __init__(self, d_in: int, name: str, hidden: int = 128, cell: str = "gru"):
        super().__init__(d_in, name)

        rnn_cls = nn.GRU if cell.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True)
        self.film = nn.Linear(d_in, 2 * hidden)  # γ & β
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
        # h: (B,D) / temps: (B,L,1)  -> y:(B,L)
        gamma_beta = self.film(h).unsqueeze(1)  # (B,1,2H)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        rnn_out, _ = self.rnn(temps)  # (B,L,H)
        fused = gamma * rnn_out + beta
        return self.out(fused).squeeze(-1)
