# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
RNN-based sequence head for the FlexibleMultiTaskModel.
"""

import torch
import torch.nn as nn
from numpy import ndarray

from foundation_model.configs.model_config import SequenceTaskConfig
from foundation_model.models.task_head.base import SequenceBaseHead


class SequenceHeadRNN(SequenceBaseHead):
    """
    GRU/LSTM sequence head with FiLM conditioning.

    Parameters
    ----------
    config : SequenceTaskConfig (or similar)
        Configuration object containing parameters like input dimension (`d_in`),
        task name (`name`), hidden size (`hidden`), and cell type (`cell`).
    """

    def __init__(self, config: SequenceTaskConfig):  # TODO: Use specific SequenceTaskConfig type hint
        super().__init__(config)

        # Extract parameters from config
        d_in = config.d_in
        hidden = config.hidden  # Default hidden size
        cell = config.cell.lower()  # Default cell type

        if cell not in ["gru", "lstm"]:
            raise ValueError(f"Unsupported RNN cell type: {cell}. Choose 'gru' or 'lstm'.")

        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        # TODO: Consider making num_layers configurable via config
        num_layers = config.num_layers if hasattr(config, "num_layers") else 2
        self.rnn = rnn_cls(input_size=1, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.film = nn.Linear(d_in, 2 * hidden)  # γ & β for FiLM conditioning
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
        # h: (B, D_in) / temps: (B, L, 1) -> y: (B, L)
        # 1. Generate FiLM parameters (gamma, beta) from task representation h
        gamma_beta = self.film(h).unsqueeze(1)  # Shape: (B, 1, 2 * hidden)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Shapes: (B, 1, hidden) each

        # 2. Process sequence points through RNN
        rnn_out, _ = self.rnn(temps)  # Shape: (B, L, hidden)

        # 3. Apply FiLM modulation
        fused = gamma * rnn_out + beta  # Broadcasting applies gamma/beta to each time step

        # 4. Project to output dimension
        output_sequence = self.out(fused).squeeze(-1)  # Shape: (B, L)
        return output_sequence

    def _predict_impl(self, x: torch.Tensor, additional: bool = False) -> dict[str, ndarray]:
        """
        Core prediction logic for RNN sequence head.

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
        # For RNN sequence head, the raw output is the prediction
        return {"prediction": x.cpu().numpy()}
