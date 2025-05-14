# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration classes for PyTorch Lightning callbacks.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class EarlyStoppingConfig:
    """Configuration for the EarlyStopping callback."""

    enabled: bool = True  # Whether to enable early stopping.
    monitor: str = "val_loss"  # Quantity to be monitored.
    min_delta: float = 0.0  # Minimum change in the monitored quantity to qualify as an improvement.
    patience: int = 10  # Number of epochs with no improvement after which training will be stopped.
    verbose: bool = False  # Verbosity mode.
    mode: Literal["min", "max"] = (
        "min"  # One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing.
    )
    strict: bool = True  # Whether to crash the training if 'monitor' is not found in the validation metrics.
    check_finite: bool = True  # When set True, stops training when the monitored quantity becomes NaN or infinite.
    stopping_threshold: Optional[float] = (
        None  # Stop training immediately once the monitored quantity reaches this threshold.
    )
    divergence_threshold: Optional[float] = (
        None  # Stop training as soon as the monitored quantity becomes worse than this threshold.
    )
    check_on_train_epoch_end: Optional[bool] = (
        None  # Whether to run early stopping check on trainer.train_epoch_end. # Added based on PyTorch Lightning docs, default is None (False)
    )
    log_rank_zero_only: bool = False  # Set to True to log only on rank zero.


@dataclass
class ModelCheckpointConfig:
    """Configuration for the ModelCheckpoint callback."""

    enabled: bool = True  # Whether to enable model checkpointing.
    dirpath: Optional[str] = (
        None  # Directory to save the checkpoints. If None, uses 'default_root_dir/checkpoints/log_name'.
    )
    filename: Optional[str] = (
        None  # Checkpoint filename. If None, a default name is generated (e.g., '{epoch}-{step}-{val_loss:.2f}').
    )
    monitor: Optional[str] = (
        "val_loss"  # Quantity to monitor. If None, it will operate in 'every_n_epochs' mode and save continuously.
    )
    verbose: bool = False  # Verbosity mode.
    save_last: Optional[bool] = (
        True  # When True, always saves the model at the end of the training epoch or when training completes. Default: None (True if monitor is set, False otherwise)
    )
    save_top_k: int = 1  # The best k models according to the quantity monitored will be saved. If -1, all models are saved. If 0, no models are saved.
    save_weights_only: bool = False  # If True, then only the model’s weights will be saved.
    mode: Literal["min", "max"] = (
        "min"  # One of {'min', 'max'}. If `save_top_k != 0`, the decision to overwrite the checkpoint is based on either the maximization or the minimization of the monitored quantity.
    )
    every_n_epochs: Optional[int] = None  # Number of epochs when checkpoints are saved.
    every_n_train_steps: Optional[int] = (
        None  # Number of training steps when checkpoints are saved. If both `every_n_train_steps` and `every_n_epochs` are specified, `every_n_train_steps` takes precedence.
    )
    train_time_interval: Optional[str] = (
        None  # Time interval at which to save checkpoints. E.g., '00:01:00' for 1 minute. Requires `datetime.timedelta`. # Pydantic doesn't directly support timedelta, using str for now
    )
    save_on_train_epoch_end: Optional[bool] = (
        None  # Whether to run checkpointing at the end of the training epoch. If this is False, then the check runs at the end of the validation.
    )
    auto_insert_metric_name: bool = True  # When True, the checkpoints filenames will contain the metric name.
    log_rank_zero_only: bool = False  # Set to True to log only on rank zero.
