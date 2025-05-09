# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Fixed vector sequence head for the FlexibleMultiTaskModel.
"""

from typing import Optional

import torch
import torch.nn as nn

from ..sequence.base import SequenceBaseHead


class SequenceHeadFixedVec(SequenceBaseHead):
    """
    Simple multi-output MLP that predicts a *fixed-length* vector.

    Parameters
    ----------
    d_in : int
        Latent dimension.
    name : str
        Name of the sequence task.
    seq_len : int
        Length of the output sequence (each element is a scalar).
    """

    def __init__(self, d_in: int, name: str, seq_len: int):
        super().__init__(d_in, name)

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, seq_len),
        )

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
