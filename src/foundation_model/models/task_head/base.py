# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Base task head interface for the FlexibleMultiTaskModel.
"""

import re  # For snake_case conversion
from abc import ABC, abstractmethod
from typing import Dict, Optional  # For type hinting

import torch
import torch.nn as nn
from numpy import ndarray

from foundation_model.models.model_config import BaseTaskConfig


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

    def __init__(self, config: BaseTaskConfig):
        super().__init__()
        self.config = config

        # Ensure d_in and name are accessible, assuming they are attributes of config
        if not hasattr(config, "name"):
            raise ValueError("Task head config must have 'name' attribute.")

        self.name = config.name

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # Raw output from the network
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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss for the task.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values from the model.
        target : torch.Tensor
            Ground truth target values.
        mask : torch.Tensor, optional
            Binary mask indicating valid samples. If None, all samples are considered valid.

        Returns
        -------
        torch.Tensor
            Total loss as a scalar tensor.
        """
        pass

    @abstractmethod
    def _predict_impl(self, x: torch.Tensor) -> Dict[str, ndarray]:
        """
        Core prediction logic implemented by subclasses.

        This method should perform any necessary post-processing on the raw
        model output (e.g., applying softmax for classification).

        Parameters
        ----------
        x : torch.Tensor
            Raw output from the forward pass of this task head.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing prediction results. Keys should be generic
            (e.g., "prediction", "labels", "probabilities") without task name prefixes.
        """
        pass

    def predict(self, x: torch.Tensor, additional: bool = False) -> Dict[str, ndarray]:
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
