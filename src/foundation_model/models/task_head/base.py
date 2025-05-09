# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Base task head interface for the FlexibleMultiTaskModel.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTaskHead(nn.Module, ABC):
    """
    Abstract base class for all task heads.

    Parameters
    ----------
    d_in : int
        Input dimension to the task head.
    name : str
        Name of the task.
    """

    def __init__(self, d_in: int, name: str):
        super().__init__()
        self.d_in = d_in
        self.name = name

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the task head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        **kwargs : dict
            Additional task-specific arguments.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute task-specific loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values.
        target : torch.Tensor
            Target values.
        mask : torch.Tensor, optional
            Binary mask for valid values.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (total_loss, per_dim_loss) where total_loss is a scalar tensor
            and per_dim_loss contains loss per dimension or class.
        """
        pass


class SequenceBaseHead(BaseTaskHead, ABC):
    """
    Abstract base class for sequence task heads.

    Parameters
    ----------
    d_in : int
        Input dimension to the task head.
    name : str
        Name of the task.
    """

    def __init__(self, d_in: int, name: str):
        super().__init__(d_in, name)

    @abstractmethod
    def forward(self, h: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sequence head.

        Parameters
        ----------
        h : torch.Tensor
            Task-specific representation tensor from the deposit layer.
        temps : torch.Tensor
            Temperature or sequence points tensor.

        Returns
        -------
        torch.Tensor
            Predicted sequence values.
        """
        pass

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sequence-specific loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted sequence values, shape (B, L).
        target : torch.Tensor
            Target sequence values, shape (B, L).
        mask : torch.Tensor, optional
            Binary mask for valid values, shape (B, L).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (total_loss, per_step_loss) where total_loss is a scalar tensor
            and per_step_loss contains loss per sequence step.
        """
        if mask is None:
            mask = torch.ones_like(target)

        # Apply mask to both predictions and targets
        losses = torch.nn.functional.mse_loss(pred, target, reduction="none") * mask

        # Compute per-step losses (average over batch)
        per_step_loss = torch.nan_to_num(losses.sum(0) / mask.sum(0), nan=0.0, posinf=0.0, neginf=0.0)

        # Compute total loss (sum over all steps, average over valid points)
        total_loss = losses.sum() / mask.sum().clamp_min(1.0)

        return total_loss, per_step_loss
