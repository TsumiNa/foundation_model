# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Self-supervised learning components for foundation models.

This module provides self-supervised training objectives that can be used
to improve representation learning without requiring labeled data.
"""

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfSupervisedModule(nn.Module):
    """
    Self-supervised learning module supporting various auxiliary training objectives.

    This module provides several self-supervised learning objectives that can be
    used to improve representation learning without requiring labeled data:

    1. Masked Feature Modeling (MFM): Predict randomly masked input features
    2. Contrastive Learning: Align representations from different modalities
    3. Cross-Reconstruction: Reconstruct one modality from another's representation

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent representation.
    formula_dim : int
        Input dimension of the formula features.
    structure_dim : int | None
        Input dimension of the structure features.
    mask_ratio : float
        Ratio of features to be randomly masked in MFM.
    temperature : float
        Temperature parameter for contrastive learning.
    """

    def __init__(
        self,
        latent_dim: int,
        formula_dim: int,
        structure_dim: int | None = None,
        mask_ratio: float = 0.15,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.temperature = temperature

        # Decoders for reconstruction tasks
        self.formula_decoder = nn.Linear(latent_dim, formula_dim, bias=False)

        # Structure decoder (only if structure modality is used)
        if structure_dim is not None:
            self.structure_decoder = nn.Linear(latent_dim, structure_dim, bias=False)

    def _mask_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a masked version of the input features for masked feature modeling.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        x_masked : torch.Tensor
            Masked version of the input (masked positions set to 0).
        mask : torch.Tensor
            Binary mask indicating which positions were masked (1 = masked).
        """
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def compute_masked_feature_loss(
        self,
        encoder_fn: Callable,
        x_formula: torch.Tensor,
        x_structure: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute masked feature modeling loss.

        Parameters
        ----------
        encoder_fn : callable
            Function that encodes masked inputs to latent representations.
            Should accept (x, is_structure) arguments.
        x_formula : torch.Tensor
            Original formula features.
        x_structure : torch.Tensor | None
            Original structure features.

        Returns
        -------
        total_loss : torch.Tensor
            Combined masked feature modeling loss.
        component_losses : dict
            Dictionary with individual loss components for logging.
        """
        # Formula MFM
        x_formula_masked, formula_mask = self._mask_features(x_formula)
        h_formula_masked = encoder_fn(x_formula_masked, False)
        formula_recon = self.formula_decoder(h_formula_masked)
        formula_loss = F.mse_loss(formula_recon[formula_mask], x_formula[formula_mask])

        total_loss = formula_loss
        component_losses = {"mfm_formula": formula_loss.detach()}

        # Structure MFM (if available)
        if x_structure is not None and hasattr(self, "structure_decoder"):
            x_structure_masked, structure_mask = self._mask_features(x_structure)
            h_structure_masked = encoder_fn(x_structure_masked, True)
            structure_recon = self.structure_decoder(h_structure_masked)
            structure_loss = F.mse_loss(structure_recon[structure_mask], x_structure[structure_mask])

            total_loss = 0.5 * (formula_loss + structure_loss)
            component_losses["mfm_structure"] = structure_loss.detach()

        return total_loss, component_losses

    def compute_contrastive_loss(self, h_formula: torch.Tensor, h_structure: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between formula and structure representations.

        Parameters
        ----------
        h_formula : torch.Tensor
            Formula latent representations.
        h_structure : torch.Tensor
            Structure latent representations.

        Returns
        -------
        loss : torch.Tensor
            Contrastive loss value.
        """
        # L2 normalize embeddings
        h_formula = F.normalize(h_formula, dim=-1)
        h_structure = F.normalize(h_structure, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(h_formula, h_structure.T) / self.temperature

        # Contrastive loss (InfoNCE)
        labels = torch.arange(h_formula.size(0), device=h_formula.device)
        loss = 0.5 * (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels))

        return loss

    def compute_cross_reconstruction_loss(
        self, h_formula: torch.Tensor, h_structure: torch.Tensor, x_formula: torch.Tensor, x_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-reconstruction loss between modalities.

        The model must reconstruct one modality's input from the other modality's
        latent representation.

        Parameters
        ----------
        h_formula : torch.Tensor
            Formula representations.
        h_structure : torch.Tensor
            Structure representations.
        x_formula : torch.Tensor
            Original formula features.
        x_structure : torch.Tensor
            Original structure features.

        Returns
        -------
        loss : torch.Tensor
            Cross-reconstruction loss value.
        """
        # Reconstruct structure from formula representation
        structure_from_formula = self.structure_decoder(h_formula)

        # Reconstruct formula from structure representation
        formula_from_structure = self.formula_decoder(h_structure)

        # Compute reconstruction losses
        loss = 0.5 * (F.mse_loss(structure_from_formula, x_structure) + F.mse_loss(formula_from_structure, x_formula))

        return loss
