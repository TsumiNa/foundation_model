# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration classes for the foundation model.
"""

from dataclasses import dataclass, field  # Import field
from enum import Enum
from typing import List, Literal, Optional, Tuple


class TaskType(str, Enum):  # Inherit from str
    """Types of tasks supported by the model."""

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
    SEQUENCE = "SEQUENCE"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer and learning rate scheduler."""

    # Optimizer settings
    optimizer_type: Literal["AdamW", "Adam", "SGD"] = "AdamW"  # Type of optimizer
    lr: float = 5e-3  # Learning rate
    weight_decay: float = 1e-3  # Weight decay (L2 penalty)
    eps: float = 1e-6  # Term added to denominator for numerical stability
    betas: Tuple[float, float] = (0.9, 0.999)  # Coefficients for computing running averages of gradient
    freeze_parameters: bool = (
        False  # If True, parameters associated with this optimizer will be frozen (requires_grad=False)
    )

    # Scheduler settings
    scheduler_type: Literal["ReduceLROnPlateau", "StepLR", "None"] = (
        "ReduceLROnPlateau"  # Type of learning rate scheduler
    )
    mode: Literal["min", "max"] = "min"  # Mode for ReduceLROnPlateau
    factor: float = 0.5  # Factor by which the learning rate will be reduced
    patience: int = 5  # Number of epochs with no improvement after which learning rate will be reduced
    min_lr: float = 1e-4  # A lower bound on the learning rate
    monitor: str = "train_total_loss"  # Quantity to monitor
    interval: str = "epoch"  # Interval for monitoring
    frequency: int = 1  # Frequency of monitoring


@dataclass
class BaseTaskConfig:
    """Base configuration for all task types."""

    name: str  # Name of the task
    type: TaskType  # Type of the task (will be overridden by subclasses with a default)
    data_column: str = ""  # Column name in attributes_df for primary task data

    # Default fields below are now keyword-only
    enabled: bool = field(default=True, kw_only=True)  # Whether the task is enabled
    weight: float = field(default=1.0, kw_only=True)  # Weight of the task in the loss function

    # LoRA configuration
    lora_enabled: bool = field(default=False, kw_only=True)  # Whether to enable LoRA adaptation
    lora_rank: int = field(default=0, kw_only=True)  # Rank for LoRA adaptation, 0 means disabled
    lora_alpha: float = field(default=1.0, kw_only=True)  # Scaling factor for LoRA adaptation
    lora_freeze_base: bool = field(default=True, kw_only=True)  # Whether to freeze the base weights when using LoRA

    # Optimizer configuration
    optimizer: Optional[OptimizerConfig] = field(default=None, kw_only=True)  # Optimizer configuration for this task


@dataclass
class RegressionTaskConfig(BaseTaskConfig):
    """Configuration for regression tasks."""

    dims: List[int] = field(default_factory=lambda: [256, 128, 64])  # positional argument
    type: TaskType = TaskType.REGRESSION  # Overrides Base.type, provides default, remains positional
    norm: bool = True  # New positional argument with default
    residual: bool = False  # New positional argument with default


@dataclass
class ClassificationTaskConfig(BaseTaskConfig):
    """Configuration for classification tasks."""

    dims: List[int] = field(default_factory=lambda: [256, 128, 64])  # positional argument
    num_classes: int = field(default=2, kw_only=True)  # positional argument
    type: TaskType = TaskType.CLASSIFICATION  # Overrides Base.type, provides default, remains positional
    norm: bool = True  # New positional argument with default
    residual: bool = False  # New positional argument with default


@dataclass
class SequenceTaskConfig(BaseTaskConfig):
    """Configuration for sequence prediction tasks."""

    subtype: str = field(default="rnn", kw_only=True)  # positional argument
    d_in: int = field(default=64, kw_only=True)  # positional argument
    type: TaskType = TaskType.SEQUENCE  # Overrides Base.type, provides default, remains positional

    # Common parameters - these become positional with defaults
    hidden: int = 128  # Hidden dimension size
    steps_column: str = ""  # Column name in attributes_df for sequence steps/x-axis

    # RNN-specific parameters
    cell: str = "gru"  # Cell type for RNN (gru or lstm)

    # Fixed vector-specific parameters
    seq_len: Optional[int] = None  # Sequence length for fixed vector output

    # TCN-specific parameters
    n_tcn_layers: int = 4  # Number of TCN layers
