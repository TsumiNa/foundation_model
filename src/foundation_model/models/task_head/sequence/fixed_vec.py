# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Fixed vector sequence head for the FlexibleMultiTaskModel.
"""

from typing import Optional

import torch
import torch.nn as nn
from numpy import ndarray

from foundation_model.configs.model_config import SequenceTaskConfig
from foundation_model.models.task_head.base import SequenceBaseHead


class SequenceHeadFixedVec(SequenceBaseHead):
    """
    Simple multi-output MLP that predicts a *fixed-length* vector.

    Parameters
    ----------
    config : SequenceTaskConfig (or similar)
        Configuration object containing parameters like input dimension (`d_in`),
        task name (`name`), and sequence length (`seq_len`).
    """

    def __init__(self, config: SequenceTaskConfig):  # TODO: Use specific SequenceTaskConfig type hint
        super().__init__(config)

        # Extract parameters from config
        d_in = config.d_in
        seq_len = getattr(config, "seq_len", None)
        if seq_len is None or seq_len <= 0:
            raise ValueError("SequenceHeadFixedVec config must specify a positive 'seq_len'.")

        self.seq_len = seq_len
        # Define the network structure based on config if needed, or keep it simple
        # Example: Use hidden dims from config if provided
        hidden_dims = getattr(config, "hidden_dims", [d_in])  # Default to one layer if not specified

        layers = []
        current_dim = d_in
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, seq_len))  # Final output layer

        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, temps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Task-specific representation tensor from the deposit layer, shape (B, D).
        temps : torch.Tensor | None
            Ignored; kept for interface compatibility.

        Returns
        -------
        torch.Tensor
            Fixed-length output (B, seq_len).
        """
        return self.net(h)  # (B, seq_len)

    def _predict_impl(self, x: torch.Tensor, additional: bool = False) -> dict[str, ndarray]:
        """
        Core prediction logic for fixed vector sequence head.

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
            A dictionary containing the prediction: {"values": x}.
        """
        # For fixed vector, the raw output is the prediction
        return {"values": x.cpu().numpy()}
