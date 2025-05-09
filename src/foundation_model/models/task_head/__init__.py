# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Task heads for the FlexibleMultiTaskModel.
"""

import torch.nn as nn

from ..model_config import (
    ClassificationTaskConfig,
    RegressionTaskConfig,
    SequenceTaskConfig,
    TaskType,
)
from .classification import ClassificationHead
from .regression import RegressionHead
from .sequence import create_sequence_head


def create_task_heads(
    task_configs: list[RegressionTaskConfig | ClassificationTaskConfig | SequenceTaskConfig],
    deposit_dim: int,
    latent_dim: int,
) -> nn.ModuleDict:
    """
    Create task heads based on configurations.

    Parameters
    ----------
    task_configs : list[RegressionTaskConfig | ClassificationTaskConfig | SequenceTaskConfig]
        List of task configurations.
    deposit_dim : int
        Dimension of the deposit layer output (input to regression/classification heads).
    latent_dim : int
        Dimension of the latent representation (input to sequence heads).

    Returns
    -------
    nn.ModuleDict
        Dictionary of task heads, keyed by task name.
    """
    task_heads = nn.ModuleDict()

    for config in task_configs:
        if not config.enabled:
            continue

        # Determine LoRA parameters if applicable
        lora_params = {}
        if hasattr(config, "lora_enabled") and config.lora_enabled:
            lora_params = {
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "freeze_base": config.lora_freeze_base,
            }

        if config.type == TaskType.REGRESSION:
            assert isinstance(config, RegressionTaskConfig)
            task_heads[config.name] = RegressionHead(
                d_in=deposit_dim,
                name=config.name,
                dims=config.dims,
                norm=config.norm,
                residual=config.residual,
                **lora_params,
            )
        elif config.type == TaskType.CLASSIFICATION:
            assert isinstance(config, ClassificationTaskConfig)
            task_heads[config.name] = ClassificationHead(
                d_in=deposit_dim,
                name=config.name,
                dims=config.dims,
                num_classes=config.num_classes,
                norm=config.norm,
                residual=config.residual,
                **lora_params,
            )
        elif config.type == TaskType.SEQUENCE:
            assert isinstance(config, SequenceTaskConfig)
            task_heads[config.name] = create_sequence_head(
                d_in=latent_dim,
                name=config.name,
                config=config,
            )

    return task_heads
