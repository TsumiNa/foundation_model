"""
Configuration classes for the foundation model.
"""

from enum import Enum, auto
from typing import Literal

from pydantic import BaseModel, Field


class TaskType(Enum):
    """Types of tasks supported by the model."""

    REGRESSION = auto()
    CLASSIFICATION = auto()
    SEQUENCE = auto()


class OptimizerConfig(BaseModel):
    """Configuration for optimizer and learning rate scheduler."""

    # Optimizer settings
    optimizer_type: Literal["AdamW", "Adam", "SGD"] = Field("AdamW", description="Type of optimizer")
    lr: float = Field(5e-3, description="Learning rate")
    weight_decay: float = Field(1e-3, description="Weight decay (L2 penalty)")
    eps: float = Field(1e-6, description="Term added to denominator for numerical stability")
    betas: tuple[float, float] = Field(
        (0.9, 0.999), description="Coefficients for computing running averages of gradient"
    )

    # Scheduler settings
    scheduler_type: Literal["ReduceLROnPlateau", "StepLR", "None"] = Field(
        "ReduceLROnPlateau", description="Type of learning rate scheduler"
    )
    mode: Literal["min", "max"] = Field("min", description="Mode for ReduceLROnPlateau")
    factor: float = Field(0.5, description="Factor by which the learning rate will be reduced")
    patience: int = Field(
        5, description="Number of epochs with no improvement after which learning rate will be reduced"
    )
    min_lr: float = Field(1e-4, description="A lower bound on the learning rate")
    monitor: str = Field("train_loss", description="Quantity to monitor")
    interval: str = Field("epoch", description="Interval for monitoring")
    frequency: int = Field(1, description="Frequency of monitoring")


class BaseTaskConfig(BaseModel):
    """Base configuration for all task types."""

    name: str = Field(..., description="Name of the task")
    type: TaskType = Field(..., description="Type of the task")
    enabled: bool = Field(True, description="Whether the task is enabled")
    weight: float = Field(1.0, description="Weight of the task in the loss function")

    # Optimizer configuration
    optimizer: OptimizerConfig | None = Field(None, description="Optimizer configuration for this task")


class RegressionTaskConfig(BaseTaskConfig):
    """Configuration for regression tasks."""

    type: Literal[TaskType.REGRESSION] = TaskType.REGRESSION
    dims: list[int] = Field(..., description="Dimensions of the regression head")
    norm: bool = Field(True, description="Whether to use normalization layers")
    residual: bool = Field(False, description="Whether to use residual connections")


class ClassificationTaskConfig(BaseTaskConfig):
    """Configuration for classification tasks."""

    type: Literal[TaskType.CLASSIFICATION] = TaskType.CLASSIFICATION
    dims: list[int] = Field(..., description="Dimensions of the classification head")
    num_classes: int = Field(..., description="Number of classes")
    norm: bool = Field(True, description="Whether to use normalization layers")
    residual: bool = Field(False, description="Whether to use residual connections")


class SequenceTaskConfig(BaseTaskConfig):
    """Configuration for sequence prediction tasks."""

    type: Literal[TaskType.SEQUENCE] = TaskType.SEQUENCE
    subtype: str = Field(..., description="Subtype of sequence head (rnn, vec, tcn)")

    # Common parameters
    hidden: int = Field(128, description="Hidden dimension size")

    # RNN-specific parameters
    cell: str = Field("gru", description="Cell type for RNN (gru or lstm)")

    # Fixed vector-specific parameters
    seq_len: int | None = Field(None, description="Sequence length for fixed vector output")

    # TCN-specific parameters
    n_tcn_layers: int = Field(4, description="Number of TCN layers")
