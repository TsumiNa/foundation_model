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
    lora_rank: int | None = None,
    lora_alpha: float = 1.0,
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
    lora_rank : int | None
        If not None, apply LoRA adaptation with the specified rank.
    lora_alpha : float
        Scaling factor for LoRA adaptation.

    Returns
    -------
    nn.ModuleDict
        Dictionary of task heads, keyed by task name.
    """
    task_heads = nn.ModuleDict()

    for config in task_configs:
        if not config.enabled:
            continue

        if config.type == TaskType.REGRESSION:
            assert isinstance(config, RegressionTaskConfig)
            task_heads[config.name] = RegressionHead(
                d_in=deposit_dim,
                name=config.name,
                dims=config.dims,
                norm=config.norm,
                residual=config.residual,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
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
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        elif config.type == TaskType.SEQUENCE:
            assert isinstance(config, SequenceTaskConfig)
            task_heads[config.name] = create_sequence_head(
                d_in=latent_dim,
                name=config.name,
                config=config,
            )

    return task_heads
