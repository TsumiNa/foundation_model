# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Base task head interface for the FlexibleMultiTaskModel.
"""

import re  # For snake_case conversion
from abc import ABC, abstractmethod
from typing import Dict  # For type hinting

import torch
import torch.nn as nn


# Helper function to convert camelCase or PascalCase to snake_case
def _to_snake_case(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class BaseTaskHead(nn.Module, ABC):
    """
    Abstract base class for all task heads.

    Each task head is responsible for taking a shared representation
    and producing task-specific outputs. It also defines how to compute
    the loss for that task and how to format predictions.

    Parameters
    ----------
    config : object # Should be a specific TaskConfig dataclass instance
        Configuration object for the task head, containing at least `d_in` and `name`.
    """

    def __init__(self, config: object):  # TODO: Replace object with a more specific TaskConfig base type if available
        super().__init__()
        self.config = config

        # --- DEBUGGING ---
        print(f"DEBUG: In BaseTaskHead.__init__ for config.name = {getattr(config, 'name', 'NAME_NOT_FOUND')}")
        print(f"DEBUG: type(config) = {type(config)}")
        print(f"DEBUG: hasattr(config, 'd_in') = {hasattr(config, 'd_in')}")
        if hasattr(config, "d_in"):
            print(f"DEBUG: config.d_in = {config.d_in}")
        print(f"DEBUG: hasattr(config, 'name') = {hasattr(config, 'name')}")
        if hasattr(config, "name"):
            print(f"DEBUG: config.name = {config.name}")
        # --- END DEBUGGING ---

        # Ensure d_in and name are accessible, assuming they are attributes of config
        if not hasattr(config, "d_in") or not hasattr(config, "name"):
            raise ValueError("Task head config must have 'd_in' and 'name' attributes.")

        # Additional check: if d_in exists but is None, this could also be an issue for heads expecting an int
        if config.d_in is None and not isinstance(
            self, SequenceBaseHead
        ):  # Sequence heads might handle d_in differently if it comes from latent_dim
            # For Regression/Classification, d_in should be an integer (deposit_dim)
            print(
                f"WARNING: config.d_in is None for non-sequence task head {getattr(config, 'name', 'Unknown')}. This might cause issues."
            )
            # We might still proceed and let the specific head fail if it can't handle None d_in.
            # Or raise an error here:
            # raise ValueError(f"config.d_in cannot be None for task head {getattr(config, 'name', 'Unknown')}")

        self.d_in = config.d_in
        self.name = config.name

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # Raw output from the network
        """
        Forward pass through the task head, returning raw model outputs (e.g., logits).

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
            and per_dim_loss contains loss per dimension or class (unweighted).
        """
        pass

    @abstractmethod
    def _predict_impl(self, x: torch.Tensor, additional: bool = False) -> Dict[str, torch.Tensor]:
        """
        Core prediction logic implemented by subclasses.

        This method should perform any necessary post-processing on the raw
        model output (e.g., applying softmax for classification).

        Parameters
        ----------
        x : torch.Tensor
            Raw output from the forward pass of this task head.
        additional : bool, optional
            If True, return additional prediction information (e.g., probabilities
            for classification tasks). Defaults to False.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing prediction results. Keys should be generic
            (e.g., "prediction", "labels", "probabilities") without task name prefixes.
        """
        pass

    def predict(self, x: torch.Tensor, additional: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generates predictions with task-specific post-processing and formatted keys.

        This method calls the subclass-specific `_predict_impl` and then
        prefixes the keys in the returned dictionary with the task's name
        in snake_case.

        Parameters
        ----------
        x : torch.Tensor
            Raw output from the forward pass of this task head.
        additional : bool, optional
            If True, request additional prediction information from `_predict_impl`.
            Defaults to False.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary where keys are prefixed with the snake_case task name
            (e.g., "task_name_labels", "task_name_probabilities").
        """
        core_results = self._predict_impl(x, additional)
        snake_name = _to_snake_case(self.config.name)
        prefixed_results = {f"{snake_name}_{key}": value for key, value in core_results.items()}
        return prefixed_results


class SequenceBaseHead(BaseTaskHead, ABC):
    """
    Abstract base class for sequence task heads.

    Parameters
    ----------
    config : object # Should be a specific SequenceTaskConfig dataclass instance
        Configuration object for the sequence task head.
    """

    def __init__(self, config: object):  # TODO: Replace object with a more specific TaskConfig base type
        super().__init__(config)

    @abstractmethod
    def forward(self, h: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:  # Raw output
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
