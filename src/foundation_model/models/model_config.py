# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration classes for the foundation model.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple


class TaskType(str, Enum):
    """Types of tasks supported by the model."""

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
    KERNEL_REGRESSION = "KernelRegression"
    AUTOENCODER = "AUTOENCODER"


class EncoderType(str, Enum):
    """Available foundation encoder backbones."""

    MLP = "mlp"
    TRANSFORMER = "transformer"


@dataclass(kw_only=True)
class BaseEncoderConfig:
    """Base class for encoder configuration objects."""

    type: EncoderType
    use_deposit_layer: bool = True

    def __post_init__(self) -> None:  # pragma: no cover - simple normalization
        if not isinstance(self.type, EncoderType):
            self.type = EncoderType(str(self.type).lower())
        if not isinstance(self.use_deposit_layer, bool):
            raise TypeError("BaseEncoderConfig.use_deposit_layer must be a boolean value")

    @property
    def latent_dim(self) -> int:
        """Return the latent dimension produced by the encoder."""

        raise NotImplementedError("Subclasses must define a latent_dim property")


@dataclass(kw_only=True)
class MLPEncoderConfig(BaseEncoderConfig):
    """Configuration for the MLP foundation encoder."""

    hidden_dims: Sequence[int]
    norm: bool = True
    residual: bool = False
    type: EncoderType = EncoderType.MLP

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.hidden_dims or len(self.hidden_dims) < 2:
            raise ValueError(
                "MLPEncoderConfig.hidden_dims must include input_dim as the first element and at least one hidden/latent dimension"
            )
        self.hidden_dims = tuple(int(dim) for dim in self.hidden_dims)
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("MLPEncoderConfig.hidden_dims entries must be positive integers")

    @property
    def latent_dim(self) -> int:
        return int(self.hidden_dims[-1])

    @property
    def input_dim(self) -> int:
        return int(self.hidden_dims[0])


@dataclass(kw_only=True)
class TransformerEncoderConfig(BaseEncoderConfig):
    """Configuration for the transformer foundation encoder.

    ``use_cls_token`` determines how the encoder aggregates feature tokens
    before passing them into the deposit layer: enabling it selects the
    contextualised ``[CLS]`` embedding, while disabling it applies mean pooling
    over all tokens. In both cases gradients still reach every feature token via
    the self-attention blocks.
    """

    d_model: int
    num_layers: int = 2
    nhead: int = 4
    dim_feedforward: int | None = None
    dropout: float = 0.1
    use_cls_token: bool = True
    apply_layer_norm: bool = True
    type: EncoderType = EncoderType.TRANSFORMER
    input_dim: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            input_dim_int = int(self.input_dim)
        except (TypeError, ValueError) as exc:
            raise TypeError("TransformerEncoderConfig.input_dim must be convertible to int") from exc
        if isinstance(self.input_dim, bool) or input_dim_int <= 0:
            raise ValueError("TransformerEncoderConfig.input_dim must be a positive integer")
        self.input_dim = input_dim_int
        if self.d_model <= 0:
            raise ValueError("TransformerEncoderConfig.d_model must be positive")
        if self.num_layers <= 0:
            raise ValueError("TransformerEncoderConfig.num_layers must be positive")
        if self.nhead <= 0:
            raise ValueError("TransformerEncoderConfig.nhead must be positive")
        if self.dim_feedforward is not None and self.dim_feedforward <= 0:
            raise ValueError("TransformerEncoderConfig.dim_feedforward must be positive when provided")

    @property
    def latent_dim(self) -> int:
        return int(self.d_model)


EncoderConfig = MLPEncoderConfig | TransformerEncoderConfig


def build_encoder_config(config: BaseEncoderConfig | Mapping[str, Any]) -> EncoderConfig:
    """Normalize encoder configuration inputs to dataclass instances.

    For MLP, ``hidden_dims`` must include the input dimension as the first element.
    For Transformer, ``input_dim`` must be provided explicitly.
    """

    if isinstance(config, BaseEncoderConfig):
        return config

    if not isinstance(config, Mapping):
        raise TypeError(
            f"encoder_config must be a BaseEncoderConfig, mapping, or None; received {type(config).__name__}"
        )

    config_dict = dict(config)
    config_type = config_dict.pop("type", EncoderType.MLP)
    if not isinstance(config_type, EncoderType):
        config_type = EncoderType(str(config_type).lower())

    if config_type is EncoderType.MLP:
        if "hidden_dims" not in config_dict:
            raise ValueError(
                "MLP encoder_config mapping must include 'hidden_dims' with input_dim as the first element"
            )
        hidden_dims = config_dict.pop("hidden_dims")
        input_dim = config_dict.pop("input_dim", None)
        if input_dim is not None:
            try:
                input_dim_int = int(input_dim)
            except (TypeError, ValueError) as exc:
                raise TypeError("encoder_config.input_dim must be convertible to int") from exc
            if input_dim_int <= 0:
                raise ValueError("encoder_config.input_dim must be positive")
            if int(hidden_dims[0]) != input_dim_int:
                raise ValueError(
                    f"MLP encoder_config input_dim ({input_dim_int}) must match hidden_dims[0] ({hidden_dims[0]})"
                )
        return MLPEncoderConfig(hidden_dims=hidden_dims, **config_dict)

    if config_type is EncoderType.TRANSFORMER:
        if "input_dim" not in config_dict:
            raise ValueError("Transformer encoder_config mapping must include 'input_dim'")
        if "d_model" not in config_dict:
            raise ValueError("Transformer encoder_config mapping must include 'd_model'")
        d_model = config_dict.pop("d_model")
        return TransformerEncoderConfig(d_model=d_model, **config_dict)

    raise ValueError(f"Unsupported encoder type: {config_type}")


@dataclass
class OptimizerConfig:
    """Configuration for optimizer and learning rate scheduler."""

    # Optimizer settings
    optimizer_type: Literal["AdamW", "Adam", "SGD"] = "AdamW"  # Type of optimizer
    lr: float = 5e-3  # Learning rate
    weight_decay: float = 1e-3  # Weight decay (L2 penalty)
    eps: float = 1e-6  # Term added to denominator for numerical stability
    betas: Tuple[float, float] = (0.9, 0.999)  # Coefficients for computing running averages of gradient

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
    loss_weight: float = field(default=1.0, kw_only=True)  # Static weight applied to this task's loss
    freeze_parameters: bool = field(default=False, kw_only=True)  # Whether to freeze this task's parameters

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
class KernelRegressionTaskConfig(BaseTaskConfig):
    """
    Configuration for extended regression (kernel regression) tasks that handle variable-length sequences.

    This configuration supports tasks like DOS prediction, temperature-dependent properties,
    time series analysis, etc., where each sample has a sequence of (t, target) pairs.

    Attributes:
        x_dim: Base layer dimensions for networks operating on shared X features.
        t_dim: Base layer dimensions for networks operating on encoded t features.
        t_encoding_method: Method for encoding t-parameters ("fourier" or "fc").
        t_column: Column name in attributes DataFrame containing t-parameter sequences.
        norm: Whether to use normalization in LinearBlocks.
        residual: Whether to use residual connections in LinearBlocks.
        kernel_num_centers: Number of Gaussian kernel centres.
        kernel_centers_init: Optional initial kernel centres.
        kernel_sigmas_init: Optional initial kernel bandwidths.
        kernel_init_sigma: Default bandwidth initialisation when none is provided.
        kernel_init_range: Range used to initialise equally spaced kernel centres.
        kernel_learnable_centers: Whether kernel centres are learnable.
        kernel_learnable_sigmas: Whether kernel bandwidths are learnable.
        kernel_min_sigma: Lower bound for kernel bandwidths.
        enable_mu3: Whether to enable the joint X–t baseline branch.
        mu*_hidden_dims: Optional overrides for hidden dimensions in the respective branches.
    """

    x_dim: List[int] = field(default_factory=lambda: [256, 128, 64])
    t_dim: List[int] = field(default_factory=lambda: [256, 128, 64])
    t_encoding_method: Literal["fourier", "fc"] = "fourier"  # Encoding method for t-parameters
    t_column: str = ""  # Column name containing t-parameter sequences (e.g., energy, temperature, time)
    type: TaskType = TaskType.KERNEL_REGRESSION
    norm: bool = True
    residual: bool = False
    # Kernel regression specific parameters
    kernel_num_centers: int = field(default=32, kw_only=True)
    kernel_centers_init: Optional[List[float]] = field(default=None, kw_only=True)
    kernel_sigmas_init: Optional[List[float]] = field(default=None, kw_only=True)
    kernel_init_sigma: float = field(default=0.15, kw_only=True)
    kernel_init_range: Tuple[float, float] = field(default=(0.0, 1.0), kw_only=True)
    kernel_learnable_centers: bool = field(default=False, kw_only=True)
    kernel_learnable_sigmas: bool = field(default=True, kw_only=True)
    kernel_min_sigma: float = field(default=1e-3, kw_only=True)
    enable_mu3: bool = field(default=True, kw_only=True)
    mu3_hidden_dims: Optional[List[int]] = field(default=None, kw_only=True)
    beta_hidden_dims: Optional[List[int]] = field(default=None, kw_only=True)
    mu1_hidden_dims: Optional[List[int]] = field(default=None, kw_only=True)
    mu2_hidden_dims: Optional[List[int]] = field(default=None, kw_only=True)


@dataclass
class AutoEncoderTaskConfig(BaseTaskConfig):
    """Configuration for autoencoder tasks (reconstruction)."""

    dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    type: TaskType = TaskType.AUTOENCODER
    norm: bool = True
    residual: bool = False


TaskConfigType = RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig | AutoEncoderTaskConfig
