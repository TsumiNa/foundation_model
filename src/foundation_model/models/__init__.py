# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import torch

from foundation_model.models.model_config import (
    BaseTaskConfig,
    ClassificationTaskConfig,
    ExtendRegressionTaskConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    TaskType,
)

from .flexible_multi_task_model import FlexibleMultiTaskModel

__all__ = [
    "FlexibleMultiTaskModel",
    "BaseTaskConfig",
    "TaskType",
    "RegressionTaskConfig",
    "ClassificationTaskConfig",
    "ExtendRegressionTaskConfig",
    "OptimizerConfig",
]

torch.set_float32_matmul_precision("medium")  # 推荐选项
