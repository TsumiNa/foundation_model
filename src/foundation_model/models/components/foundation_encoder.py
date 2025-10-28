# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Foundation encoder components for multi-task learning models.

This module provides the core encoder components that transform input features
into latent representations for multi-task learning models.
"""

import torch
import torch.nn as nn

from ..fc_layers import LinearBlock


class FoundationEncoder(nn.Module):
    """
    Foundation model encoder providing shared representations for multi-task learning.

    This module encapsulates the core encoding layers that transform input features
    into a latent representation, followed by a deposit layer that serves as a buffer
    between the shared encoder and task-specific heads.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list[int]
        Hidden dimensions for the encoder network. The last element defines the
        latent representation dimension.
    deposit_dim : int
        Output dimension of the deposit layer.
    norm : bool
        Whether to apply layer normalization.
    residual : bool
        Whether to use residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        deposit_dim: int,
        norm: bool = True,
        residual: bool = False,
    ):
        super().__init__()

        # Construct shared encoder
        self.shared = LinearBlock(
            [input_dim] + hidden_dims,
            normalization=norm,
            residual=residual,
        )

        # Deposit layer serves as a buffer between shared encoder and task heads
        self.deposit = nn.Sequential(
            nn.Linear(hidden_dims[-1], deposit_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the foundation encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (B, input_dim).

        Returns
        -------
        latent : torch.Tensor
            Latent representation, shape (B, hidden_dims[-1]).
        task_representation : torch.Tensor
            Task representation after deposit layer, shape (B, deposit_dim).
        """
        latent = self.shared(x)
        task_representation = self.deposit(latent)

        return latent, task_representation

    def encode_masked(self, x_masked: torch.Tensor, is_structure: bool = False) -> torch.Tensor:
        """
        Encode masked features for self-supervised learning.

        Parameters
        ----------
        x_masked : torch.Tensor
            Masked input features.
        is_structure : bool
            Retained for backward compatibility. Ignored in this encoder.

        Returns
        -------
        torch.Tensor
            Encoded representation of masked features.
        """
        return self.shared(x_masked)


