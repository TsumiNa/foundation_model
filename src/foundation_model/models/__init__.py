# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Foundation model implementation and related modules.
"""

from .flexible_multi_task_model import FlexibleMultiTaskModel
from .model_config import (
    BaseTaskConfig,
    ClassificationTaskConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    SequenceTaskConfig,
    TaskType,
)

__all__ = [
    "FlexibleMultiTaskModel",
    "BaseTaskConfig",
    "TaskType",
    "RegressionTaskConfig",
    "ClassificationTaskConfig",
    "SequenceTaskConfig",
    "OptimizerConfig",
]
