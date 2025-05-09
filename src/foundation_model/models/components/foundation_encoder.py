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
from .gated_fusion import GatedFusion
from .structure_encoder import StructureEncoder


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
            Ignored in this class, included for consistency with MultiModalFoundationEncoder.

        Returns
        -------
        torch.Tensor
            Encoded representation of masked features.
        """
        return self.shared(x_masked)


class MultiModalFoundationEncoder(nn.Module):
    """
    Multi-modal foundation encoder with fusion capabilities.

    This module extends the foundation encoder to handle multiple input modalities
    (formula and structure), with gated fusion to combine their representations.

    Parameters
    ----------
    formula_input_dim : int
        Input dimension for formula features.
    formula_hidden_dims : list[int]
        Hidden dimensions for formula encoder.
    structure_input_dim : int
        Input dimension for structure features.
    structure_hidden_dims : list[int]
        Hidden dimensions for structure encoder.
    deposit_dim : int
        Output dimension of the deposit layer.
    norm : bool
        Whether to apply layer normalization.
    residual : bool
        Whether to use residual connections.
    """

    def __init__(
        self,
        formula_input_dim: int,
        formula_hidden_dims: list[int],
        structure_input_dim: int,
        structure_hidden_dims: list[int],
        deposit_dim: int,
        norm: bool = True,
        residual: bool = False,
    ):
        super().__init__()

        # Validate that both encoders output the same latent dimension
        if formula_hidden_dims[-1] != structure_hidden_dims[-1]:
            raise ValueError("Formula and structure latent dimensions must match for fusion")

        self.latent_dim = formula_hidden_dims[-1]

        # Formula encoder
        self.formula_encoder = LinearBlock(
            [formula_input_dim] + formula_hidden_dims,
            normalization=norm,
            residual=residual,
        )

        # Structure encoder
        self.structure_encoder = StructureEncoder(
            structure_input_dim,
            structure_hidden_dims,
            norm=norm,
            residual=residual,
        )

        # Gated fusion
        self.fusion = GatedFusion(self.latent_dim)

        # Deposit layer
        self.deposit = nn.Sequential(
            nn.Linear(self.latent_dim, deposit_dim),
            nn.Tanh(),
        )

    def forward(
        self, x_formula: torch.Tensor, x_structure: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the multi-modal foundation encoder.

        Parameters
        ----------
        x_formula : torch.Tensor
            Formula input features, shape (B, formula_input_dim).
        x_structure : torch.Tensor, optional
            Structure input features, shape (B, structure_input_dim).
            If None, only formula features will be used.

        Returns
        -------
        h_formula : torch.Tensor
            Formula latent representation, shape (B, latent_dim).
        h_structure : torch.Tensor | None
            Structure latent representation, shape (B, latent_dim).
            None if x_structure is None.
        h_fused : torch.Tensor
            Fused latent representation, shape (B, latent_dim).
        task_representation : torch.Tensor
            Task representation after deposit layer, shape (B, deposit_dim).
        """
        # Encode formula
        h_formula = self.formula_encoder(x_formula)

        # Encode structure if provided
        if x_structure is not None:
            h_structure = self.structure_encoder(x_structure)
            has_structure = torch.ones(h_formula.size(0), 1, device=h_formula.device)
        else:
            h_structure = torch.zeros_like(h_formula)
            has_structure = torch.zeros(h_formula.size(0), 1, device=h_formula.device)

        # Apply fusion
        h_fused = self.fusion(h_formula, h_structure, has_structure)

        # Apply deposit layer
        task_representation = self.deposit(h_fused)

        return h_formula, h_structure if x_structure is not None else None, h_fused, task_representation

    def encode_formula(self, x_formula: torch.Tensor) -> torch.Tensor:
        """
        Encode formula features.

        Parameters
        ----------
        x_formula : torch.Tensor
            Formula features.

        Returns
        -------
        torch.Tensor
            Encoded formula representation.
        """
        return self.formula_encoder(x_formula)

    def encode_structure(self, x_structure: torch.Tensor) -> torch.Tensor:
        """
        Encode structure features.

        Parameters
        ----------
        x_structure : torch.Tensor
            Structure features.

        Returns
        -------
        torch.Tensor
            Encoded structure representation.
        """
        return self.structure_encoder(x_structure)

    def encode_masked(self, x_masked: torch.Tensor, is_structure: bool = False) -> torch.Tensor:
        """
        Encode masked features for self-supervised learning.

        Parameters
        ----------
        x_masked : torch.Tensor
            Masked input features.
        is_structure : bool
            Whether the masked features are structure features.

        Returns
        -------
        torch.Tensor
            Encoded representation of masked features.
        """
        if is_structure:
            return self.structure_encoder(x_masked)
        else:
            return self.formula_encoder(x_masked)
