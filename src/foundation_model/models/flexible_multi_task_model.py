# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


"""
Module: flexible_multi_task_model
---------------------------------

A flexible multi-task model with foundation model capabilities.

Tensor shape legend (used across all docstrings):
* **B** - batch size
* **L** - sequence length (e.g. number of temperature points)
* **D** - latent / embedding feature dimension
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, cast

import lightning as L
import numpy as np
import pandas as pd  # Added
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from loguru import logger  # Replaced logging with loguru
from torch.optim.lr_scheduler import LRScheduler  # Changed from _LRScheduler
from torchmetrics.regression import R2Score

try:  # pragma: no cover - optional distributed import
    import torch.distributed as dist
except Exception:  # noqa: BLE001 - keep fallback for CPU-only environments
    dist = None  # type: ignore[assignment]

from .components.foundation_encoder import FoundationEncoder
from .model_config import (
    BaseEncoderConfig,
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    MLPEncoderConfig,
    OptimizerConfig,
    TransformerEncoderConfig,
    RegressionTaskConfig,
    TaskConfigType,
    TaskType,
    _AEConfig,
    build_encoder_config,
)
from .task_head.autoencoder import AutoEncoderHead
from .task_head.base import BaseTaskHead
from .task_head.classification import ClassificationHead
from .task_head.kernel_regression import (
    KernelRegressionHead,
    expand_for_kernel_regression,
    reshape_kernel_regression_predictions,
)
from .task_head.regression import RegressionHead
from .inverse_design import (
    CompositionOptimizationResult,
    InverseDesignMixin,
    OptimizationResult,
    OptimizationTarget,
)

# Re-exported: inverse design moved to .inverse_design, but workflows/inverse.py and the tests
# import these from here. Keeping the names bound preserves those call sites, so the move stays a
# pure relocation.
__all__ = [
    "FlexibleMultiTaskModel",
    "TaskPredictions",
    "OptimizationTarget",
    "OptimizationResult",
    "CompositionOptimizationResult",
]

#: One task's predictions as the heads actually produce them: NumPy, one array per prediction
#: channel. Kernel-regression heads reshape theirs to one array per sample, because their sequences
#: have different lengths — the same asymmetry the collate function has on the input side.
#:
#: ``predict_step`` was annotated ``dict[str, torch.Tensor]``, which described neither. The
#: mismatch was invisible while a ``# type: ignore`` on ``head.predict`` erased the type at source.
TaskPredictions = dict[str, "np.ndarray | list[np.ndarray]"]


class FlexibleMultiTaskModel(InverseDesignMixin, L.LightningModule):
    """
    Foundation model with flexible task heads.

    This model implements a flexible multi-task learning framework with foundation model capabilities.
    The core architecture includes:

    1. Shared Encoder Layer (Foundation Encoder):
       Extracts general representations from input features, serving as a shared foundation for all tasks.

    2. Deposit Layer:
       Acts as a buffer between the shared encoder and task heads, providing an extensible design for continual learning.

    3. Multi-task Heads:
       Supports various types of prediction tasks:
       - Regression tasks: Predict continuous value attributes
       - Classification tasks: Predict discrete categories
       - KernelRegression tasks: Predict variable-length sequences (e.g., DOS, temperature-dependent properties)

    Training Process:
    - Each batch's loss includes task-specific losses
    - Different components (shared encoder, task heads, etc.) can use different optimizer configurations

    Usage Scenarios:
    1. Multi-task Learning: Predict multiple related tasks simultaneously
    2. Transfer Learning: Pre-train shared encoder, then fine-tune specific tasks
    3. Multi-modal Fusion: Combine data from different sources
    4. Continual Learning: Support model updates via modular architecture

    Parameters
    ----------
    task_configs : list[RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig]
        List of task configurations, each defining a prediction task. Each configuration must specify
        task type, name, dimensions, etc. Regression and classification task heads receive Tanh-activated
        latent representations, while KernelRegression task heads receive both latent representations and sequence points.
        A task-specific `loss_weight` (defaults to 1.0) can be set in each configuration to scale its loss.
    encoder_config : BaseEncoderConfig | Mapping[str, Any]
        Configuration controlling the foundation encoder backbone.
        For MLP, hidden_dims must include input_dim as the first element.
    shared_block_optimizer : OptimizerConfig | None
        Optimizer configuration for the shared foundation encoder.
    enable_learnable_loss_balancer : bool
        Whether to use learnable log_sigma_t parameters for each supervised task to weight their losses.
    """

    def __init__(
        self,
        task_configs: Sequence[RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig],
        *,
        encoder_config: BaseEncoderConfig | Mapping[str, Any],
        # Freezing parameters
        freeze_shared_encoder: bool = False,
        # Optimization parameters
        shared_block_optimizer: OptimizerConfig | None = None,
        enable_learnable_loss_balancer: bool = False,
        # Loss calculation behavior
        allow_all_missing_in_batch: bool = True,
        # AutoEncoder head
        enable_autoencoder: bool = False,
        autoencoder_nonnegative: bool = False,
    ):
        super().__init__()
        # logger=False: saves all hparams to checkpoint (pickle, not OmegaConf) but skips
        # logger.log_hyperparams(), which is where OmegaConf chokes on Union[str, Sequence[str]].
        self.save_hyperparameters(logger=False)

        # Store the new parameters
        self.enable_learnable_loss_balancer = enable_learnable_loss_balancer
        self.allow_all_missing_in_batch = allow_all_missing_in_batch

        # Validate inputs
        if not task_configs and not enable_autoencoder:
            raise ValueError("At least one task configuration must be provided")

        if encoder_config is None:
            raise ValueError("encoder_config must be provided")
        # build_encoder_config already returns a passed-in config unchanged, so branching here
        # only duplicated it — and did so against the abstract BaseEncoderConfig rather than the
        # concrete EncoderConfig union, so a third subclass would pass this check and then fail
        # inside FoundationEncoder with a worse message. One call, one place that decides.
        self.encoder_config = build_encoder_config(encoder_config)
        # Dimension of latent representation (input to task heads after Tanh activation)
        self.latent_dim = self.encoder_config.latent_dim
        self.task_configs: list = list(task_configs)
        self.task_configs_map: dict = {cfg.name: cfg for cfg in self.task_configs}
        for cfg in self.task_configs:
            cfg.loss_weight = self._normalize_loss_weight(getattr(cfg, "loss_weight", 1.0), cfg.name)

        # Auto-create reconstruction head if requested
        if enable_autoencoder:
            _AE_NAME = "__reconstruction__"
            if _AE_NAME in self.task_configs_map:
                raise ValueError(
                    f"Task name '{_AE_NAME}' is reserved for the built-in autoencoder head; "
                    "rename the conflicting task."
                )
            ae_cfg = _AEConfig(
                dims=self._derive_ae_dims(self.encoder_config),
                nonnegative=autoencoder_nonnegative,
            )
            self.task_configs.append(ae_cfg)
            self.task_configs_map[ae_cfg.name] = ae_cfg

        # Freezing parameters
        self.freeze_shared_encoder = freeze_shared_encoder

        # Optimizer configurations
        self.shared_block_optimizer = shared_block_optimizer or OptimizerConfig(weight_decay=1e-2)

        # Initialize learnable uncertainty parameters (log(sigma_t))
        self.task_log_sigmas = nn.ParameterDict()
        self._disabled_task_log_sigma_buffers: dict[str, torch.Tensor] = {}
        if self.enable_learnable_loss_balancer:
            logger.info("Learnable task uncertainty (task_log_sigmas) is ENABLED.")
        else:
            logger.info("Learnable task uncertainty (task_log_sigmas) is DISABLED.")

        # Initialize model components
        self._init_foundation_encoder()
        self._init_task_heads()
        # Track task types
        self._track_task_types()

        # Initialize weights
        self._init_weights()

        # Set to manual optimization as we handle multiple optimizers
        self.automatic_optimization = False

        # Distributed metric tracking
        self.val_r2_metrics = nn.ModuleDict()
        self.test_r2_metrics = nn.ModuleDict()
        self._metrics_updated: dict[str, set[str]] = {"val": set(), "test": set()}
        self._stage_index_trackers: dict[str, dict[str, Any] | None] = {"val": None, "test": None}
        # Rebuilt by configure_optimizers; defined here so on_train_epoch_end never depends on
        # that having run (e.g. a hook called directly).
        self._scheduler_monitors: list[str] = []
        self._init_stage_metrics()

        logger.info("Initializing FlexibleMultiTaskModel...")
        logger.info("Registered Task Heads:")
        task_info_df = self.registered_tasks_info
        if not task_info_df.empty:
            # Log as a formatted table if pandas is available and dataframe is not empty
            # For a cleaner log, convert DataFrame to string
            task_info_str = task_info_df.to_string(index=False)
            for line in task_info_str.split("\n"):
                logger.info(f"  {line}")
        else:
            logger.info("  No task heads configured.")

        logger.info("FlexibleMultiTaskModel Structure:")
        # Convert the model's string representation into multiple log lines for readability
        model_structure_str = str(self)
        for line in model_structure_str.split("\n"):
            logger.info(f"  {line}")
        logger.info("FlexibleMultiTaskModel initialization complete.")

    def _init_foundation_encoder(self):
        """Initialize the foundation encoder."""
        self.encoder = FoundationEncoder(
            encoder_config=self.encoder_config,
        )

    def _infer_parameter_device(self) -> torch.device:
        """Infer the device for newly created modules/parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _get_task_static_weight(self, task_name: str) -> float:
        """Return the configured static loss weight for a task."""
        cfg = self.task_configs_map.get(task_name)
        if cfg is None:
            return 1.0
        return self._normalize_loss_weight(getattr(cfg, "loss_weight", 1.0), task_name)

    # Logging helpers to allow mixins to respect module-level logger patches in tests
    def _log_debug(self, message: str) -> None:
        logger.debug(message)

    def _log_info(self, message: str) -> None:
        logger.info(message)

    def _log_warning(self, message: str) -> None:
        logger.warning(message)

    def _init_stage_metrics(self) -> None:
        """Initialize per-task R² metrics for validation and test stages."""
        for cfg in self.task_configs:
            if not getattr(cfg, "enabled", True) or cfg.type != TaskType.REGRESSION:
                continue
            # R2Score in torchmetrics>=1.4.0 auto-detects output dimensions from first update
            # No need to specify num_outputs parameter
            self.val_r2_metrics[cfg.name] = R2Score()
            self.test_r2_metrics[cfg.name] = R2Score()

    def _reset_stage_metrics(self, stage: str) -> None:
        metrics = self.val_r2_metrics if stage == "val" else self.test_r2_metrics
        for metric in metrics.values():
            cast(R2Score, metric).reset()
        self._metrics_updated[stage] = set()

    def _init_stage_index_tracker(self, stage: str) -> None:
        dataset_len = None
        datamodule = getattr(self.trainer, "datamodule", None) if self.trainer is not None else None
        if datamodule is not None:
            dataset = getattr(datamodule, f"{stage}_dataset", None)
            if dataset is not None:
                dataset_len = len(dataset)
        self._stage_index_trackers[stage] = self._build_index_tracker(dataset_len)

    def _build_index_tracker(self, dataset_len: int | None) -> dict[str, Any] | None:
        if dataset_len is None:
            return None
        is_distributed = dist is not None and dist.is_available() and dist.is_initialized()
        if not is_distributed:
            return None
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        num_samples = math.ceil(dataset_len / world_size)
        total_size = num_samples * world_size
        base_indices = list(range(dataset_len))
        if len(base_indices) < total_size:
            base_indices.extend(base_indices[: total_size - len(base_indices)])
        indices_for_rank = base_indices[rank:total_size:world_size]
        return {"indices": indices_for_rank, "cursor": 0, "seen": set()}

    def _get_batch_valid_mask(
        self,
        *,
        stage: str,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[bool]] | None:
        tracker = self._stage_index_trackers.get(stage)
        if not tracker:
            return None
        start = tracker["cursor"]
        end = start + batch_size
        indices = tracker["indices"]
        batch_indices = indices[start:end]
        tracker["cursor"] = min(end, len(indices))
        if len(batch_indices) < batch_size and indices:
            batch_indices.extend(indices[-1:] * (batch_size - len(batch_indices)))
        seen: set[int] = tracker["seen"]
        valid_flags: list[bool] = []
        for idx in batch_indices:
            if idx in seen:
                valid_flags.append(False)
            else:
                seen.add(idx)
                valid_flags.append(True)
        if not valid_flags:
            return None
        mask_tensor = torch.tensor(valid_flags, dtype=torch.bool, device=device)
        return mask_tensor, valid_flags

    def _apply_stage_valid_mask(
        self,
        *,
        sample_mask: torch.Tensor | list[torch.Tensor] | None,
        target: torch.Tensor | list[torch.Tensor],
        batch_valid_mask: torch.Tensor | None,
        batch_valid_list: list[bool] | None,
        is_sequence: bool,
    ) -> torch.Tensor | list[torch.Tensor] | None:
        """Apply distributed duplicate filtering to per-task masks."""
        if batch_valid_mask is None and batch_valid_list is None:
            return sample_mask

        if is_sequence:
            if not isinstance(target, list) or batch_valid_list is None:
                return sample_mask
            if sample_mask is None:
                sample_mask = [torch.ones_like(seq, dtype=torch.bool) for seq in target]
            assert isinstance(sample_mask, list)
            adjusted_masks: list[torch.Tensor] = []
            for valid, mask in zip(batch_valid_list, sample_mask):
                if valid:
                    adjusted_masks.append(mask)
                else:
                    adjusted_masks.append(torch.zeros_like(mask, dtype=torch.bool))
            return adjusted_masks

        if not isinstance(target, torch.Tensor):
            raise TypeError("Expected tensor target for non-sequence task.")
        if batch_valid_mask is None:
            return sample_mask
        if sample_mask is None:
            sample_mask = torch.ones_like(target, dtype=torch.bool)
        if not isinstance(sample_mask, torch.Tensor):
            raise TypeError("Expected tensor mask for non-sequence task.")
        valid_tensor = batch_valid_mask
        while valid_tensor.ndim < sample_mask.ndim:
            valid_tensor = valid_tensor.unsqueeze(-1)
        return sample_mask & valid_tensor

    def _update_r2_metric(
        self,
        *,
        stage: str,
        task_name: str,
        preds: torch.Tensor,
        targets: torch.Tensor,
        sample_mask: torch.Tensor | None,
    ) -> None:
        metrics = self.val_r2_metrics if stage == "val" else self.test_r2_metrics
        metric = cast("R2Score | None", metrics._modules.get(task_name))
        if metric is None:
            return
        if sample_mask is None:
            mask_bool = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask_bool = sample_mask.to(dtype=torch.bool)
        preds_flat = preds.reshape(preds.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)
        mask_flat = mask_bool.reshape(mask_bool.shape[0], -1)
        if mask_flat.shape[1] > 1:
            row_mask = mask_flat.all(dim=1)
        else:
            row_mask = mask_flat.squeeze(-1)
        if not torch.any(row_mask):
            return
        valid_preds = preds_flat[row_mask]
        valid_targets = targets_flat[row_mask]
        if valid_preds.numel() == 0:
            return
        metric.update(valid_preds.detach().to(torch.float32), valid_targets.detach().to(torch.float32))
        self._metrics_updated[stage].add(task_name)

    def _log_stage_r2_metrics(self, stage: str) -> None:
        metrics = self.val_r2_metrics if stage == "val" else self.test_r2_metrics
        for name in self._metrics_updated[stage]:
            metric = cast("R2Score | None", metrics._modules.get(name))
            if metric is None:
                continue
            self.log(
                f"{stage}_{name}_r2",
                metric,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    @staticmethod
    def _normalize_loss_weight(weight_value: float | None, task_name: str) -> float:
        """Validate and normalize a configured loss weight."""
        if weight_value is None:
            return 1.0
        try:
            numeric_value = float(weight_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Task '{task_name}' has non-numeric loss_weight: {weight_value!r}") from exc
        if numeric_value < 0:
            raise ValueError(f"Task '{task_name}' has negative loss_weight; expected non-negative value.")
        return numeric_value

    def _validate_task_config(
        self,
        config_item: RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig,
    ):
        """Validate that the task configuration is compatible with the shared encoder."""
        if not config_item.name:
            raise ValueError("Task config must have a non-empty name.")
        if config_item.name in self.task_configs_map:
            raise ValueError(f"Task '{config_item.name}' already exists in the model.")

        expected_input_dim = self.latent_dim

        if config_item.type == TaskType.KERNEL_REGRESSION:
            assert isinstance(config_item, KernelRegressionTaskConfig)
            if not config_item.x_dim:
                raise ValueError(f"KernelRegression task '{config_item.name}' requires at least one x_dim entry.")
            if config_item.x_dim[0] != expected_input_dim:
                raise ValueError(
                    f"KernelRegression task '{config_item.name}' expects x_dim[0]=={expected_input_dim}, "
                    f"but received {config_item.x_dim[0]}."
                )
        else:
            assert isinstance(config_item, (RegressionTaskConfig, ClassificationTaskConfig))
            if not getattr(config_item, "dims", None):
                raise ValueError(f"Task '{config_item.name}' requires a non-empty dims configuration.")
            if config_item.dims[0] != expected_input_dim:
                raise ValueError(
                    f"Task '{config_item.name}' expects dims[0]=={expected_input_dim}, "
                    f"but received {config_item.dims[0]}."
                )
        config_item.loss_weight = self._normalize_loss_weight(
            getattr(config_item, "loss_weight", 1.0), config_item.name
        )

    def _instantiate_task_head(self, config_item) -> nn.Module:
        """Instantiate a task head module for the provided configuration."""
        if not config_item.enabled:
            raise ValueError(f"Task '{config_item.name}' must be enabled before instantiation.")

        head_module: nn.Module
        if config_item.type == TaskType.REGRESSION:
            assert isinstance(config_item, RegressionTaskConfig)
            head_module = RegressionHead(config=config_item)
        elif config_item.type == TaskType.CLASSIFICATION:
            assert isinstance(config_item, ClassificationTaskConfig)
            head_module = ClassificationHead(config=config_item)
        elif config_item.type == TaskType.KERNEL_REGRESSION:
            assert isinstance(config_item, KernelRegressionTaskConfig)
            head_module = KernelRegressionHead(config=config_item)
        elif config_item.type == TaskType.AUTOENCODER:
            assert isinstance(config_item, _AEConfig)
            head_module = AutoEncoderHead(config=config_item)
        else:
            raise ValueError(f"Unsupported task type: {config_item.type}")

        device = self._infer_parameter_device()
        head_module.to(device)
        return head_module

    def _register_task_log_sigma(self, task_name: str):
        """Register a learnable log sigma parameter for the task if enabled."""
        if not self.enable_learnable_loss_balancer:
            return

        if task_name in self.task_log_sigmas:
            return

        device = self._infer_parameter_device()
        self.task_log_sigmas[task_name] = nn.Parameter(torch.zeros((), device=device))

    def _deregister_task_log_sigma(self, task_name: str):
        """Remove the learnable log sigma parameter for the task if present."""
        if not self.enable_learnable_loss_balancer:
            return
        if task_name in self.task_log_sigmas:
            del self.task_log_sigmas[task_name]

    def _init_task_heads(self):
        """Initialize task heads based on configurations."""
        self.task_heads = nn.ModuleDict()
        self.disabled_task_heads = nn.ModuleDict()

        for config in self.task_configs:
            if config.enabled:
                self._activate_task(config)

    def _head(self, name: str) -> BaseTaskHead:
        """The head registered under ``name``, typed.

        ``ModuleDict.__getitem__`` is annotated to return ``Module``, and ``Module.__getattr__``
        resolves to ``Tensor | Module``, so *every* method reached through ``self.task_heads[name]``
        is unresolvable to a type checker — which is why a ``# type: ignore`` had accumulated on
        one of the call sites. Every value in this dict is a ``BaseTaskHead`` by construction
        (``_activate_task`` is the only writer), so state that once here rather than at each use.

        A ``cast`` rather than an ``assert``: this is called once per task per batch from the
        training loop, and the invariant is enforced where the dict is written.
        """
        return cast(BaseTaskHead, self.task_heads[name])

    def _activate_task(self, task_config: TaskConfigType) -> nn.Module:
        """Activate (or re-activate) a task by ensuring its head and auxiliary state are registered."""
        name = task_config.name

        if name in self.task_heads:
            return self.task_heads[name]

        if name in self.disabled_task_heads:
            head_module = self.disabled_task_heads[name]
            del self.disabled_task_heads[name]
        else:
            head_module = self._instantiate_task_head(task_config)

        self.task_heads[name] = head_module
        self._register_task_log_sigma(name)

        if name in self._disabled_task_log_sigma_buffers and name in self.task_log_sigmas:
            with torch.no_grad():
                self.task_log_sigmas[name].copy_(self._disabled_task_log_sigma_buffers.pop(name))

        if task_config.freeze_parameters:
            for parameter in head_module.parameters():
                parameter.requires_grad_(False)

        return head_module

    def _deactivate_task(self, task_name: str) -> None:
        """Deactivate a task by moving its head to the disabled registry and clearing auxiliary state."""
        if task_name in self.task_heads:
            head_module = self.task_heads[task_name]
            self.disabled_task_heads[task_name] = head_module
            del self.task_heads[task_name]
        elif task_name not in self.disabled_task_heads:
            logger.warning(f"_deactivate_task: task '{task_name}' not found among active heads.")

        if task_name in self.task_log_sigmas:
            self._disabled_task_log_sigma_buffers[task_name] = self.task_log_sigmas[task_name].detach().clone()
            self._deregister_task_log_sigma(task_name)

    @staticmethod
    def _derive_ae_dims(encoder_config: BaseEncoderConfig) -> list[int]:
        """Return decoder dims that mirror the encoder: [latent_dim, ..., input_dim]."""
        if isinstance(encoder_config, MLPEncoderConfig):
            return list(reversed(encoder_config.hidden_dims))
        if isinstance(encoder_config, TransformerEncoderConfig):
            # TransformerEncoderConfig: single linear projection
            return [encoder_config.latent_dim, encoder_config.input_dim]
        raise TypeError(f"Unsupported encoder config type: {type(encoder_config).__name__}")

    def _track_task_types(self):
        """Track which types of tasks are enabled."""
        self.has_regression = any(tc.type == TaskType.REGRESSION for tc in self.task_configs if tc.enabled)
        self.has_classification = any(tc.type == TaskType.CLASSIFICATION for tc in self.task_configs if tc.enabled)
        self.has_kernel_regression = any(
            tc.type == TaskType.KERNEL_REGRESSION for tc in self.task_configs if tc.enabled
        )

    def add_task(
        self,
        *task_configs: TaskConfigType,
    ) -> "FlexibleMultiTaskModel":
        """
        Dynamically add one or more task configurations and instantiate their heads.

        Parameters
        ----------
        *task_configs : RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig
            Task configuration objects describing the new heads.
        """
        if not task_configs:
            logger.warning("add_task called without task configurations; ignoring.")
            return self

        for task_config in task_configs:
            self._validate_task_config(task_config)

        activated: list[str] = []

        for task_config in task_configs:
            # Register configuration
            self.task_configs.append(task_config)
            self.task_configs_map[task_config.name] = task_config

            # Instantiate and register task head
            if task_config.enabled:
                self._activate_task(task_config)
                activated.append(task_config.name)

        # Update task type tracking
        self._track_task_types()

        for task_config in task_configs:
            logger.info(
                f"Added task '{task_config.name}' (type={task_config.type.value}, enabled={task_config.enabled})."
            )
        if activated:
            logger.info(f"Activated tasks during add_task: {', '.join(activated)}")
        return self

    def enable_task(self, *task_names: str) -> "FlexibleMultiTaskModel":
        """
        Enable one or more existing tasks by name and ensure their heads are active.

        Parameters
        ----------
        *task_names : str
            Names of tasks to enable.
        """
        if not task_names:
            return self

        reactivated: list[str] = []

        for name in task_names:
            config = self.task_configs_map.get(name)
            if config is None:
                logger.warning(f"enable_task: task '{name}' not found; skipping.")
                continue
            if config.enabled:
                logger.debug(f"enable_task: task '{name}' already enabled; skipping.")
                continue

            config.enabled = True
            self._activate_task(config)
            reactivated.append(name)

        if reactivated:
            self._track_task_types()
            logger.info(f"Enabled tasks: {', '.join(reactivated)}")

        return self

    def disable_task(self, *task_names: str) -> "FlexibleMultiTaskModel":
        """
        Disable one or more existing tasks by name without dropping their configuration.

        Parameters
        ----------
        *task_names : str
            Names of tasks to disable.
        """
        if not task_names:
            return self

        disabled: list[str] = []

        for name in task_names:
            config = self.task_configs_map.get(name)
            if config is None:
                logger.warning(f"disable_task: task '{name}' not found; skipping.")
                continue
            if not config.enabled:
                logger.debug(f"disable_task: task '{name}' already disabled; skipping.")
                continue

            config.enabled = False
            self._deactivate_task(name)
            disabled.append(name)

        if disabled:
            self._track_task_types()
            logger.info(f"Disabled tasks: {', '.join(disabled)}")

        return self

    def remove_tasks(self, *task_names: str) -> "FlexibleMultiTaskModel":
        """
        Remove one or more tasks from the model by name.

        Parameters
        ----------
        *task_names : str
            Names of tasks to remove.
        """
        if not task_names:
            return self

        to_remove = {name for name in task_names}
        existing = {name for name in to_remove if name in self.task_configs_map}

        missing = to_remove - existing
        for name in missing:
            logger.warning(f"remove_tasks: task '{name}' not found; skipping.")

        if not existing:
            return self

        # Remove ModuleDict entries and auxiliary state
        for name in existing:
            if name in self.task_heads:
                del self.task_heads[name]
            self._deregister_task_log_sigma(name)
            if name in self.disabled_task_heads:
                del self.disabled_task_heads[name]
            self._disabled_task_log_sigma_buffers.pop(name, None)

        # Filter configurations and rebuild map
        self.task_configs = [cfg for cfg in self.task_configs if cfg.name not in existing]
        self.task_configs_map = {cfg.name: cfg for cfg in self.task_configs}

        # Refresh task-type flags
        self._track_task_types()

        logger.info(f"Removed tasks: {', '.join(sorted(existing))}")
        return self

    def _init_weights(self):
        """Initialize model weights and apply freezing based on freeze_shared_encoder config."""
        # Apply parameter freezing based on freeze_shared_encoder config
        if self.freeze_shared_encoder:
            for p in self.encoder.shared.parameters():
                p.requires_grad_(False)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        t_sequences: dict[str, List[torch.Tensor] | torch.Tensor] | None = None,  # Renamed from temps_batch
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Architecture: X → encoder → latent → Tanh → (all task heads including AE)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing formula descriptors.
        t_sequences : dict[str, torch.Tensor] | None, optional
            A dictionary where keys are KernelRegression task names and values are the
            corresponding sequence input data (e.g., temperature points, time steps)
            for the batch. Required if KernelRegression tasks are present. Defaults to None.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of task outputs, keyed by task name.
        """
        if isinstance(x, (list, tuple)):
            raise TypeError("FlexibleMultiTaskModel expects tensor inputs; received tuple/list.")
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"FlexibleMultiTaskModel expects tensor inputs; received {type(x)}.")

        # Get latent representation from encoder
        latent = self.encoder(x)

        # Apply Tanh activation - ALL task heads (including AE) receive Tanh(latent)
        # This ensures architectural consistency between training and latent space optimization
        h_task = torch.tanh(latent)

        # Apply task heads - all task heads use h_task (Tanh-activated latent)
        outputs = {}
        for name, head in self.task_heads.items():
            if isinstance(head, KernelRegressionHead):
                # Get specific sequence data for this KernelRegression head
                task_sequence_input = t_sequences.get(name) if t_sequences else None
                if task_sequence_input is not None:
                    # DOSDataset-style expansion: expand h_task and t for KernelRegressionHead
                    expanded_h_task, expanded_t = expand_for_kernel_regression(h_task, task_sequence_input)
                    outputs[name] = head(expanded_h_task, t=expanded_t)
                else:
                    # For KernelRegressionHead, t parameter is required
                    raise ValueError(
                        f"KernelRegressionHead '{name}' requires t parameter but t_sequences is missing or doesn't contain '{name}'"
                    )
            else:
                outputs[name] = head(h_task)

        return outputs

    # Lightning step hooks delegate to helper implementations for readability.

    # --- shared per-batch pipeline ------------------------------------------------------------
    #
    # training/validation/test_step were near-duplicates of ~110 lines each. They drifted: one
    # logged floats where the others logged tensors, one kept a vestigial zero accumulator, and a
    # metric-update loop once scored every task against the last task's tensors. The three stages
    # now share these helpers and differ only in what they are *supposed* to differ in — whether
    # gradients are kept, whether duplicate eval rows are masked out, and whether R² is updated.

    def _finalize_mask(
        self,
        *,
        name: str,
        stage: str,
        is_sequence: bool,
        target: torch.Tensor | list[torch.Tensor],
        mask: torch.Tensor | list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten a kernel-regression sequence and guarantee a boolean mask of the target's shape."""
        if is_sequence and isinstance(target, list):
            target = torch.cat(target, dim=0)
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"Task '{name}': expected a tensor target, got {type(target).__name__}.")

        if isinstance(mask, list):
            mask = torch.cat(mask, dim=0)
        elif mask is None:
            kind = "KernelRegression task" if is_sequence else "task"
            self._log_warning(f"Mask not found for {kind} {name} in {stage}_step. Assuming all valid.")
            mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
        return target, mask

    def _collect_batch_losses(
        self,
        *,
        stage: str,
        x: torch.Tensor,
        preds: dict[str, torch.Tensor],
        y_dict_batch: dict[str, Any],
        task_masks_batch: dict[str, Any],
        logs: dict[str, torch.Tensor],
        stage_valid: tuple[torch.Tensor | None, list[bool] | None] | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Raw loss per participating head, as ``{name: (raw_loss, pred, target, mask)}``.

        Each task's own tensors are returned rather than left in loop variables, because the
        callers iterate a second time to weight and score them — reading the loop variables there
        is how every task once got scored against whichever task happened to be processed last.
        """
        collected: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        raw_sum = torch.zeros((), device=x.device)

        for name, pred_tensor in preds.items():
            head = self._head(name)
            resolved = self._resolve_target_and_mask(
                name=name, head=head, x=x, y_dict_batch=y_dict_batch, task_masks_batch=task_masks_batch
            )
            if resolved is None:
                continue
            target, mask = resolved

            if stage_valid is not None:
                batch_valid_mask, batch_valid_list = stage_valid
                mask = self._apply_stage_valid_mask(
                    sample_mask=mask,
                    target=target,
                    batch_valid_mask=batch_valid_mask,
                    batch_valid_list=batch_valid_list,
                    is_sequence=isinstance(head, KernelRegressionHead),
                )

            target, mask = self._finalize_mask(
                name=name,
                stage=stage,
                is_sequence=isinstance(head, KernelRegressionHead),
                target=target,
                mask=mask,
            )

            raw_loss = head.compute_loss(pred_tensor, target, mask)
            if raw_loss is None:
                if not self.allow_all_missing_in_batch:
                    raise ValueError(
                        f"Task '{name}' has no valid samples in this batch and allow_all_missing_in_batch is False."
                    )
                self._log_debug(f"Task '{name}' has no valid samples in this batch. Skipping loss calculation.")
                logs[f"{stage}_{name}_all_missing"] = torch.tensor(1.0, device=x.device)
                continue

            collected[name] = (raw_loss, pred_tensor, target, mask)
            raw_sum = raw_sum + raw_loss.detach()
            logs[f"{stage}_{name}_raw_loss"] = raw_loss.detach()
            logs[f"{stage}_{name}_all_missing"] = torch.tensor(0.0, device=x.device)

        logs[f"{stage}_sum_supervised_raw_loss"] = raw_sum
        return collected

    def _weighted_total_loss(
        self,
        *,
        stage: str,
        raw_losses: dict[str, torch.Tensor],
        logs: dict[str, torch.Tensor],
        device: torch.device,
        keep_graph: bool,
    ) -> torch.Tensor:
        """Combine raw losses into the objective: static weights, then the optional balancer.

        ``keep_graph`` is the only difference between training and evaluation here — evaluation
        detaches each contribution as it accumulates so no graph is retained.
        """
        total = torch.zeros((), device=device)
        for name, raw_loss in raw_losses.items():
            static_weight = self._get_task_static_weight(name)
            if self.enable_learnable_loss_balancer and name in self.task_log_sigmas:
                log_sigma = self.task_log_sigmas[name]
                precision = torch.exp(-2 * log_sigma)
                contribution = (static_weight * 0.5 * precision * raw_loss) + log_sigma
                logs[f"{stage}_{name}_sigma_t"] = torch.exp(log_sigma).detach()
            else:
                contribution = static_weight * raw_loss
            total = total + (contribution if keep_graph else contribution.detach())
            logs[f"{stage}_{name}_final_loss_contrib"] = contribution.detach()
            logs[f"{stage}_{name}_static_weight"] = torch.tensor(static_weight, device=device)
        return total

    def _eval_step(self, batch: Any, *, stage: str) -> None:
        """Shared body of ``validation_step`` / ``test_step``.

        The two differed only in their metric namespace and their stage tracker, so they are one
        implementation now; anything that must differ goes through ``stage``.
        """
        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor inputs in {stage}_step, received {type(x)}")

        logs: dict[str, torch.Tensor] = {}
        preds = self(x, task_sequence_data_batch)
        stage_valid = self._get_batch_valid_mask(stage=stage, batch_size=x.shape[0], device=x.device) or (None, None)

        collected = self._collect_batch_losses(
            stage=stage,
            x=x,
            preds=preds,
            y_dict_batch=y_dict_batch,
            task_masks_batch=task_masks_batch,
            logs=logs,
            stage_valid=stage_valid,
        )
        total = self._weighted_total_loss(
            stage=stage,
            raw_losses={name: item[0] for name, item in collected.items()},
            logs=logs,
            device=x.device,
            keep_graph=False,
        )
        for name, (_raw, pred_tensor, target, mask) in collected.items():
            self._update_r2_metric(stage=stage, task_name=name, preds=pred_tensor, targets=target, sample_mask=mask)

        logs[f"{stage}_final_supervised_loss"] = total.detach()
        self.log_dict(logs, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_final_loss", total.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return None

    # --- Lightning hooks ----------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Supervised multi-task training step (manual optimization)."""
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor inputs in training_step, received {type(x)}")

        logs: dict[str, torch.Tensor] = {}
        preds = self(x, task_sequence_data_batch)
        collected = self._collect_batch_losses(
            stage="train",
            x=x,
            preds=preds,
            y_dict_batch=y_dict_batch,
            task_masks_batch=task_masks_batch,
            logs=logs,
        )
        total_loss = self._weighted_total_loss(
            stage="train",
            raw_losses={name: item[0] for name, item in collected.items()},
            logs=logs,
            device=x.device,
            keep_graph=True,
        )

        logs["train_final_supervised_loss"] = total_loss.detach()
        self.log_dict(logs, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_final_loss", total_loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if total_loss.requires_grad:
            self.manual_backward(total_loss)
            for opt in optimizers:
                opt.step()
            # Schedulers are stepped once per epoch in on_train_epoch_end, not here: stepping a
            # ReduceLROnPlateau per batch makes its `patience` count batches.
        else:
            # No opt.step() in this branch. It used to call it on every optimizer right after
            # logging that it was skipping the optimizer step — a no-op only because
            # zero_grad(set_to_none=True) leaves every p.grad as None and AdamW skips gradientless
            # params. With set_to_none=False the same line applies AdamW's decoupled weight decay
            # to the whole model on a batch that carried no signal.
            self._log_warning(
                f"total_loss does not require grad and has no grad_fn at batch_idx {batch_idx}. "
                "Skipping backward pass and optimizer step. "
                "This might indicate all parameters are frozen, loss contributions are zero, "
                "or an issue with the computation graph.",
            )

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Validation step — the training objective without gradients, plus R² metrics."""
        return self._eval_step(batch, stage="val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Test step — identical to validation, logged under the ``test`` namespace."""
        return self._eval_step(batch, stage="test")

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._reset_stage_metrics("val")
        self._init_stage_index_tracker("val")

    def on_train_epoch_end(self) -> None:
        """Step every ``ReduceLROnPlateau`` once, on the epoch-aggregated monitored metric.

        Under manual optimization Lightning does not drive schedulers, so the model must. Doing it
        here rather than in ``training_step`` is what makes ``patience`` count **epochs** — the
        unit every ReduceLROnPlateau tutorial assumes, and the one the config documents.

        The monitored value is read from ``trainer.callback_metrics``, so ``monitor`` selects a real
        metric instead of the current batch's loss. A missing key raises rather than silently
        skipping the step: a scheduler that never anneals is invisible in logs. The default,
        ``train_final_loss_epoch``, is the epoch aggregate of the ``train_final_loss`` that
        ``training_step`` logs with ``on_epoch=True``.
        """
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        metrics = self.trainer.callback_metrics if self.trainer is not None else {}
        for scheduler, monitor in zip(schedulers, self._scheduler_monitors):
            if scheduler is None:
                continue
            if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                continue
            if monitor not in metrics:
                raise ValueError(
                    f"LR scheduler monitor {monitor!r} is not among the logged metrics at the end "
                    f"of the training epoch. Available: {sorted(metrics)}. Set "
                    "[training.scheduler].monitor to a metric logged with on_epoch=True during "
                    "training (the default is 'train_final_loss_epoch')."
                )
            scheduler.step(metrics[monitor])

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._log_stage_r2_metrics("val")

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self._reset_stage_metrics("test")
        self._init_stage_index_tracker("test")

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self._log_stage_r2_metrics("test")

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        tasks_to_predict: Optional[List[str]] = None,
    ) -> TaskPredictions:
        """
        Prediction step that forwards inputs through the model and post-processes the outputs.

        Parameters
        ----------
        batch : tuple
            Typically contains (x_formula, _, _, task_sequence_data_batch). Only x_formula and
            task_sequence_data_batch are used.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the dataloader (if multiple).
        tasks_to_predict : list[str] | None, optional
            A list of task names to predict. If None, predicts all enabled tasks.

        Returns
        -------
        TaskPredictions
            Flat dictionary of head-specific prediction outputs. Values are NumPy arrays — one per
            prediction channel — except for kernel-regression heads, whose predictions are reshaped
            to one array per sample because their sequences have different lengths.
        """
        del dataloader_idx  # unused but kept for signature parity

        x_formula = batch[0]
        if not isinstance(x_formula, torch.Tensor):
            raise TypeError(f"Expected batch[0] to be a Tensor (x_formula), but got {type(x_formula)}")

        task_sequence_data_batch = batch[3] if len(batch) > 3 else {}

        kernel_regression_sequence_lengths = {}
        for task_name, sequence_data in task_sequence_data_batch.items():
            if task_name in self.task_heads and isinstance(self.task_heads[task_name], KernelRegressionHead):
                if isinstance(sequence_data, list):
                    kernel_regression_sequence_lengths[task_name] = [len(seq) for seq in sequence_data]
                elif isinstance(sequence_data, torch.Tensor):
                    lengths = []
                    for sample in sequence_data:
                        valid_mask = sample != 0.0
                        lengths.append(int(valid_mask.sum().item()))
                    kernel_regression_sequence_lengths[task_name] = lengths

        raw_preds = self(x_formula, task_sequence_data_batch)

        final_predictions: TaskPredictions = {}

        if tasks_to_predict is None:
            tasks_to_iterate = [(name, tensor) for name, tensor in raw_preds.items() if name in self.task_heads]
        else:
            tasks_to_iterate = []
            for task_name in tasks_to_predict:
                if task_name not in self.task_heads:
                    self._log_warning(
                        f"Task '{task_name}' requested for prediction but not found or not enabled in the model. Skipping."
                    )
                    continue
                if task_name not in raw_preds:
                    self._log_warning(
                        f"Task '{task_name}' requested for prediction, found in model heads, but not present in raw output. Skipping."
                    )
                    continue
                tasks_to_iterate.append((task_name, raw_preds[task_name]))

        for task_name, raw_pred_tensor in tasks_to_iterate:
            head = self._head(task_name)
            predictions = head.predict(raw_pred_tensor)

            # Reusing one variable for both shapes is what made this untypeable: `predict` returns
            # one array per channel, while the kernel-regression reshape returns one array per
            # sample. Keeping them separate lets each keep its own precise type, and confines the
            # union to the accumulator, which is the only thing that genuinely holds both.
            if isinstance(head, KernelRegressionHead) and task_name in kernel_regression_sequence_lengths:
                final_predictions.update(
                    reshape_kernel_regression_predictions(predictions, kernel_regression_sequence_lengths[task_name])
                )
            else:
                final_predictions.update(predictions)

        return final_predictions

    @property
    def registered_tasks_info(self) -> pd.DataFrame:
        """
        Provides information about all registered tasks in the model.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns 'name', 'type', and 'enabled', detailing each configured task.
        """
        task_info = []
        for task_config in self.task_configs:
            task_info.append(
                {
                    "name": task_config.name,
                    "type": task_config.type.value,  # Get the string value of the enum
                    "enabled": task_config.enabled,
                }
            )
        return pd.DataFrame(task_info)

    def _create_optimizer(self, params: list[torch.nn.Parameter], config: OptimizerConfig) -> torch.optim.Optimizer:
        """``AdamW`` over ``params`` with this group's hyper-parameters."""
        params = list(filter(lambda p: p.requires_grad, params))
        if not params:  # If no parameters require gradients, return a dummy optimizer or handle appropriately
            # This path should ideally not be hit if checks are done before calling _create_optimizer
            logger.warning(f"Optimizer creation called with no parameters requiring gradients for config: {config}")
            # Depending on strictness, could raise error or return a dummy. For now, let it proceed (might error in optim).
            # A more robust solution might be to return a specific dummy optimizer if PyTorch allows,
            # or ensure this function is not called with empty grad-requiring params.
            pass

        return optim.AdamW(params, lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, config: OptimizerConfig) -> LRScheduler | None:
        """``ReduceLROnPlateau`` for this parameter group, or ``None`` when the scheduler is off."""
        if not config.scheduler_enabled:
            return None
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers for all parameter groups."""

        optimizers_and_schedulers: list[Any] = []
        # Parallel to the schedulers Lightning hands back from self.lr_schedulers(), so
        # on_train_epoch_end can feed each one the metric its own group configured. Reset on
        # every call, because configure_optimizers can run more than once per model.
        self._scheduler_monitors = []

        # 1. Main parameters (Encoder + optionally task_log_sigmas)
        main_params_to_optimize = list(self.encoder.parameters())

        if self.enable_learnable_loss_balancer and hasattr(self, "task_log_sigmas") and self.task_log_sigmas:
            learnable_log_sigmas = [p for p in self.task_log_sigmas.parameters() if p.requires_grad]
            if learnable_log_sigmas:
                main_params_to_optimize.extend(learnable_log_sigmas)
                logger.info(f"Added {len(learnable_log_sigmas)} task_log_sigmas parameters to the main optimizer.")
            else:
                logger.info(
                    "No learnable task_log_sigmas parameters found to add to the main optimizer (all frozen or empty)."
                )
        elif self.enable_learnable_loss_balancer:  # task_log_sigmas might not exist or be empty
            logger.info("Learnable task uncertainty is ON, but task_log_sigmas is not populated or has no parameters.")

        # Filter main_params_to_optimize to ensure all require grad before creating optimizer
        main_params_to_optimize_filtered = [p for p in main_params_to_optimize if p.requires_grad]

        if main_params_to_optimize_filtered:
            encoder_opt = self._create_optimizer(main_params_to_optimize_filtered, self.shared_block_optimizer)
            encoder_sched = self._create_scheduler(encoder_opt, self.shared_block_optimizer)

            if encoder_sched:
                self._scheduler_monitors.append(self.shared_block_optimizer.monitor)
                optimizers_and_schedulers.append(
                    {
                        "optimizer": encoder_opt,
                        "lr_scheduler": {
                            "scheduler": encoder_sched,
                            "monitor": self.shared_block_optimizer.monitor,
                            "interval": self.shared_block_optimizer.interval,
                            "frequency": self.shared_block_optimizer.frequency,
                        },
                    }
                )
            else:
                # Always a dict, never a bare Optimizer: Lightning rejects a list that mixes the
                # two, which happens as soon as one group disables its scheduler and another
                # does not (e.g. the inference placeholder in build_model_for_checkpoint).
                optimizers_and_schedulers.append({"optimizer": encoder_opt})
        else:
            logger.info(
                "No parameters requiring gradients for the main optimizer (encoder/log_sigmas). Skipping its creation."
            )

        # 2. Task head parameters
        for name, head in self.task_heads.items():
            head_params_to_optimize = [p for p in head.parameters() if p.requires_grad]
            if not head_params_to_optimize:
                logger.info(f"No parameters requiring gradients for task head '{name}'. Skipping optimizer creation.")
                continue

            config = self.task_configs_map[name]
            task_optimizer_config = config.optimizer or OptimizerConfig()  # Use default if specific not provided

            task_opt = self._create_optimizer(head_params_to_optimize, task_optimizer_config)
            task_sched = self._create_scheduler(task_opt, task_optimizer_config)

            if task_sched:
                self._scheduler_monitors.append(task_optimizer_config.monitor)
                optimizers_and_schedulers.append(
                    {
                        "optimizer": task_opt,
                        "lr_scheduler": {
                            "scheduler": task_sched,
                            "monitor": task_optimizer_config.monitor,
                            "interval": task_optimizer_config.interval,
                            "frequency": task_optimizer_config.frequency,
                        },
                    }
                )
            else:
                optimizers_and_schedulers.append({"optimizer": task_opt})

        if not optimizers_and_schedulers:
            logger.warning(
                "No optimizers were configured. This might be due to all parameters being frozen or an issue in parameter collection."
            )
            # Lightning requires at least one optimizer if the model has trainable parameters.
            # If all parameters are frozen, this is fine. Otherwise, it's an issue.

        return optimizers_and_schedulers
