"""
Configuration classes for tasks in FlexibleMultiTaskModel.
"""

from enum import Enum, auto
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TaskType(Enum):
    """Types of tasks supported by the model."""

    REGRESSION = auto()
    CLASSIFICATION = auto()
    SEQUENCE = auto()


class BaseTaskConfig(BaseModel):
    """Base configuration for all task types."""

    name: str = Field(..., description="Name of the task")
    type: TaskType = Field(..., description="Type of the task")
    enabled: bool = Field(True, description="Whether the task is enabled")
    weight: float = Field(1.0, description="Weight of the task in the loss function")


class RegressionTaskConfig(BaseTaskConfig):
    """Configuration for regression tasks."""

    type: Literal[TaskType.REGRESSION] = TaskType.REGRESSION
    dims: List[int] = Field(..., description="Dimensions of the regression head")
    norm: bool = Field(True, description="Whether to use normalization layers")
    residual: bool = Field(False, description="Whether to use residual connections")


class ClassificationTaskConfig(BaseTaskConfig):
    """Configuration for classification tasks."""

    type: Literal[TaskType.CLASSIFICATION] = TaskType.CLASSIFICATION
    dims: List[int] = Field(..., description="Dimensions of the classification head")
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
    seq_len: Optional[int] = Field(
        None, description="Sequence length for fixed vector output"
    )

    # TCN-specific parameters
    n_tcn_layers: int = Field(4, description="Number of TCN layers")
