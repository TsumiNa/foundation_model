"""
Foundation model implementation and related modules.
"""

from .flexible_multi_task_model import FlexibleMultiTaskModel
from .task_config import (
    BaseTaskConfig,
    ClassificationTaskConfig,
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
]
