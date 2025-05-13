# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Foundation model implementation and related modules.
"""

from foundation_model.configs.model_config import (
    BaseTaskConfig,
    ClassificationTaskConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    SequenceTaskConfig,
    TaskType,
)

from .flexible_multi_task_model import FlexibleMultiTaskModel

__all__ = [
    "FlexibleMultiTaskModel",
    "BaseTaskConfig",
    "TaskType",
    "RegressionTaskConfig",
    "ClassificationTaskConfig",
    "SequenceTaskConfig",
    "OptimizerConfig",
]
