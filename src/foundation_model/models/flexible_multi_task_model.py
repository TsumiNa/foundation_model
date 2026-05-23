# Copyright 2025 TsumiNa.
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
from collections import namedtuple
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional

import lightning as L
import numpy as np
import pandas as pd  # Added
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from loguru import logger  # Replaced logging with loguru
from torch.optim.lr_scheduler import LRScheduler  # Changed from _LRScheduler
from torchmetrics.regression import R2Score

try:  # pragma: no cover - optional distributed import
    import torch.distributed as dist
except Exception:  # noqa: BLE001 - keep fallback for CPU-only environments
    dist = None

from .components.foundation_encoder import FoundationEncoder
from .model_config import (
    BaseEncoderConfig,
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    MLPEncoderConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    TaskConfigType,
    TaskType,
    _AEConfig,
    build_encoder_config,
)
from .task_head.autoencoder import AutoEncoderHead
from .task_head.classification import ClassificationHead
from .task_head.kernel_regression import KernelRegressionHead
from .task_head.regression import RegressionHead

# Named tuple for optimization results
OptimizationResult = namedtuple(
    "OptimizationResult", ["optimized_input", "optimized_target", "initial_score", "trajectory"]
)

# Composition-space optimization (gradient descent over element weights w ∈ simplex). The optimised
# w *is* the recipe (no AE-decode round-trip), so it is reported alongside the descriptor x = w @ K.
CompositionOptimizationResult = namedtuple(
    "CompositionOptimizationResult",
    ["optimized_weights", "optimized_descriptor", "optimized_target", "initial_score", "trajectory"],
)


class FlexibleMultiTaskModel(L.LightningModule):
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
        if isinstance(encoder_config, BaseEncoderConfig):
            self.encoder_config = encoder_config
        else:
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
            metric.reset()
        self._metrics_updated[stage] = set()

    def _init_stage_index_tracker(self, stage: str) -> None:
        dataset_len = None
        if self.trainer is not None and getattr(self.trainer, "datamodule", None) is not None:
            dataset = getattr(self.trainer.datamodule, f"{stage}_dataset", None)
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
        metric = metrics._modules.get(task_name)
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
            metric = metrics._modules.get(name)
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
        # TransformerEncoderConfig: single linear projection
        return [encoder_config.latent_dim, encoder_config.input_dim]

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
                    expanded_h_task, expanded_t = self._expand_for_kernel_regression(h_task, task_sequence_input)
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

    def training_step(self, batch, batch_idx):
        """Training step implementation for supervised multi-task learning."""
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        lr_schedulers = self.lr_schedulers()
        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor inputs in training_step, received {type(x)}")

        train_logs: dict[str, torch.Tensor] = {}
        supervised_loss_contribution = torch.zeros((), device=x.device)

        preds = self(x, task_sequence_data_batch)

        raw_supervised_losses = {}
        for name, pred_tensor in preds.items():
            head = self.task_heads[name]

            if isinstance(head, AutoEncoderHead):
                target = x
                sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
            elif name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue
            else:
                target = y_dict_batch[name]
                sample_mask = task_masks_batch.get(name)

            if isinstance(head, KernelRegressionHead):
                if isinstance(target, list):
                    target = torch.cat(target, dim=0)
                if sample_mask is not None and isinstance(sample_mask, list):
                    sample_mask = torch.cat(sample_mask, dim=0)
                elif sample_mask is None:
                    self._log_warning(
                        f"Mask not found for KernelRegression task {name} in training_step. Assuming all valid."
                    )
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
            else:
                if sample_mask is None:
                    self._log_warning(f"Mask not found for task {name} in training_step. Assuming all valid.")
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            raw_loss_t = head.compute_loss(pred_tensor, target, sample_mask)

            if raw_loss_t is None:
                if self.allow_all_missing_in_batch:
                    self._log_debug(f"Task '{name}' has no valid samples in this batch. Skipping loss calculation.")
                    train_logs[f"train_{name}_all_missing"] = 1.0
                    continue
                raise ValueError(
                    f"Task '{name}' has no valid samples in this batch and allow_all_missing_in_batch is False."
                )

            raw_supervised_losses[name] = raw_loss_t
            train_logs[f"train_{name}_raw_loss"] = raw_loss_t.detach()
            train_logs[f"train_{name}_all_missing"] = 0.0

        for name, raw_loss_t in raw_supervised_losses.items():
            static_weight = self._get_task_static_weight(name)
            if self.enable_learnable_loss_balancer and name in self.task_log_sigmas:
                current_log_sigma_t = self.task_log_sigmas[name]
                precision_factor_t = torch.exp(-2 * current_log_sigma_t)
                final_task_loss_component = (
                    static_weight * 0.5 * precision_factor_t * raw_loss_t
                ) + current_log_sigma_t

                supervised_loss_contribution += final_task_loss_component
                train_logs[f"train_{name}_sigma_t"] = torch.exp(current_log_sigma_t).detach()
                train_logs[f"train_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            else:
                final_task_loss_component = static_weight * raw_loss_t
                supervised_loss_contribution += final_task_loss_component
                train_logs[f"train_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            train_logs[f"train_{name}_static_weight"] = torch.tensor(static_weight, device=x.device)

        train_logs["train_final_supervised_loss"] = supervised_loss_contribution.detach()

        total_loss = supervised_loss_contribution

        self.log_dict(train_logs, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_final_loss", total_loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if total_loss.requires_grad:
            self.manual_backward(total_loss)
            for opt in optimizers:
                opt.step()

            for scheduler in lr_schedulers:
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(total_loss.detach())
                    else:
                        scheduler.step()
        else:
            self._log_warning(
                f"total_loss does not require grad and has no grad_fn at batch_idx {batch_idx}. "
                "Skipping backward pass and optimizer step. "
                "This might indicate all parameters are frozen, loss contributions are zero, "
                "or an issue with the computation graph.",
            )
            for opt in optimizers:
                opt.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step implementation mirroring training_step without gradient updates.

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict_batch, task_masks_batch, task_sequence_data_batch)
        batch_idx : int
            Index of the current batch

        Returns
        -------
        None
            This method logs metrics using self.log_dict() and does not return a value.
        """
        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor inputs in validation_step, received {type(x)}")

        val_logs: dict[str, torch.Tensor] = {}
        final_val_loss = torch.zeros((), device=x.device)
        val_supervised_loss_contribution = torch.zeros_like(final_val_loss)
        val_sum_supervised_raw_loss = torch.zeros_like(final_val_loss)

        preds = self(x, task_sequence_data_batch)
        valid_mask_info = self._get_batch_valid_mask(stage="val", batch_size=x.shape[0], device=x.device)
        if valid_mask_info is None:
            batch_valid_mask = None
            batch_valid_list: list[bool] | None = None
        else:
            batch_valid_mask, batch_valid_list = valid_mask_info

        raw_val_supervised_losses = {}
        for name, pred_tensor in preds.items():
            head = self.task_heads[name]

            if isinstance(head, AutoEncoderHead):
                target = x
                sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
            elif name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue
            else:
                target = y_dict_batch[name]
                sample_mask = task_masks_batch.get(name)

            sample_mask = self._apply_stage_valid_mask(
                sample_mask=sample_mask,
                target=target,
                batch_valid_mask=batch_valid_mask,
                batch_valid_list=batch_valid_list,
                is_sequence=isinstance(head, KernelRegressionHead),
            )

            if isinstance(head, KernelRegressionHead):
                if isinstance(target, list):
                    target = torch.cat(target, dim=0)
                if sample_mask is not None and isinstance(sample_mask, list):
                    sample_mask = torch.cat(sample_mask, dim=0)
                elif sample_mask is None:
                    self._log_warning(
                        f"Mask not found for KernelRegression task {name} in validation_step. Assuming all valid."
                    )
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
            else:
                if sample_mask is None:
                    self._log_warning(f"Mask not found for task {name} in validation_step. Assuming all valid.")
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            raw_loss_t = head.compute_loss(pred_tensor, target, sample_mask)

            if raw_loss_t is None:
                if self.allow_all_missing_in_batch:
                    self._log_debug(f"Task '{name}' has no valid samples in this batch. Skipping loss calculation.")
                    val_logs[f"val_{name}_all_missing"] = torch.tensor(1.0, device=x.device)
                    continue
                raise ValueError(
                    f"Task '{name}' has no valid samples in this batch and allow_all_missing_in_batch is False."
                )

            raw_val_supervised_losses[name] = raw_loss_t
            val_sum_supervised_raw_loss += raw_loss_t.detach()
            val_logs[f"val_{name}_raw_loss"] = raw_loss_t.detach()
            val_logs[f"val_{name}_all_missing"] = torch.tensor(0.0, device=x.device)

        val_logs["val_sum_supervised_raw_loss"] = val_sum_supervised_raw_loss

        for name, raw_loss_t in raw_val_supervised_losses.items():
            static_weight = self._get_task_static_weight(name)
            if self.enable_learnable_loss_balancer and name in self.task_log_sigmas:
                current_log_sigma_t = self.task_log_sigmas[name]
                precision_factor_t = torch.exp(-2 * current_log_sigma_t)
                final_task_loss_component = (
                    static_weight * 0.5 * precision_factor_t * raw_loss_t
                ) + current_log_sigma_t

                val_supervised_loss_contribution += final_task_loss_component.detach()
                val_logs[f"val_{name}_sigma_t"] = torch.exp(current_log_sigma_t).detach()
                val_logs[f"val_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            else:
                final_task_loss_component = static_weight * raw_loss_t
                val_supervised_loss_contribution += final_task_loss_component.detach()
                val_logs[f"val_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            val_logs[f"val_{name}_static_weight"] = torch.tensor(static_weight, device=x.device)

            self._update_r2_metric(
                stage="val",
                task_name=name,
                preds=pred_tensor,
                targets=target,
                sample_mask=sample_mask,
            )

        val_logs["val_final_supervised_loss"] = val_supervised_loss_contribution.detach()
        final_val_loss = final_val_loss + val_supervised_loss_contribution

        self.log_dict(val_logs, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_final_loss", final_val_loss.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return None

    def test_step(self, batch, batch_idx):
        """
        Test step implementation mirroring validation_step but logging to the test namespace.

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict_batch, task_masks_batch, task_sequence_data_batch)
        batch_idx : int
            Index of the current batch

        Returns
        -------
        None
        """
        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor inputs in test_step, received {type(x)}")

        test_logs: dict[str, torch.Tensor] = {}
        final_test_loss = torch.zeros((), device=x.device)
        test_supervised_loss_contribution = torch.zeros_like(final_test_loss)
        test_sum_supervised_raw_loss = torch.zeros_like(final_test_loss)

        preds = self(x, task_sequence_data_batch)
        valid_mask_info = self._get_batch_valid_mask(stage="test", batch_size=x.shape[0], device=x.device)
        if valid_mask_info is None:
            batch_valid_mask = None
            batch_valid_list: list[bool] | None = None
        else:
            batch_valid_mask, batch_valid_list = valid_mask_info

        raw_test_supervised_losses = {}
        for name, pred_tensor in preds.items():
            head = self.task_heads[name]

            if isinstance(head, AutoEncoderHead):
                target = x
                sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
            elif name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue
            else:
                target = y_dict_batch[name]
                sample_mask = task_masks_batch.get(name)

            sample_mask = self._apply_stage_valid_mask(
                sample_mask=sample_mask,
                target=target,
                batch_valid_mask=batch_valid_mask,
                batch_valid_list=batch_valid_list,
                is_sequence=isinstance(head, KernelRegressionHead),
            )

            if isinstance(head, KernelRegressionHead):
                if isinstance(target, list):
                    target = torch.cat(target, dim=0)
                if sample_mask is not None and isinstance(sample_mask, list):
                    sample_mask = torch.cat(sample_mask, dim=0)
                elif sample_mask is None:
                    self._log_warning(
                        f"Mask not found for KernelRegression task {name} in test_step. Assuming all valid."
                    )
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
            else:
                if sample_mask is None:
                    self._log_warning(f"Mask not found for task {name} in test_step. Assuming all valid.")
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            raw_loss_t = head.compute_loss(pred_tensor, target, sample_mask)

            if raw_loss_t is None:
                if self.allow_all_missing_in_batch:
                    self._log_debug(f"Task '{name}' has no valid samples in this batch. Skipping loss calculation.")
                    test_logs[f"test_{name}_all_missing"] = torch.tensor(1.0, device=x.device)
                    continue
                raise ValueError(
                    f"Task '{name}' has no valid samples in this batch and allow_all_missing_in_batch is False."
                )

            raw_test_supervised_losses[name] = raw_loss_t
            test_sum_supervised_raw_loss += raw_loss_t.detach()
            test_logs[f"test_{name}_raw_loss"] = raw_loss_t.detach()
            test_logs[f"test_{name}_all_missing"] = torch.tensor(0.0, device=x.device)

        test_logs["test_sum_supervised_raw_loss"] = test_sum_supervised_raw_loss

        for name, raw_loss_t in raw_test_supervised_losses.items():
            static_weight = self._get_task_static_weight(name)
            if self.enable_learnable_loss_balancer and name in self.task_log_sigmas:
                current_log_sigma_t = self.task_log_sigmas[name]
                precision_factor_t = torch.exp(-2 * current_log_sigma_t)
                final_task_loss_component = (
                    static_weight * 0.5 * precision_factor_t * raw_loss_t
                ) + current_log_sigma_t

                test_supervised_loss_contribution += final_task_loss_component.detach()
                test_logs[f"test_{name}_sigma_t"] = torch.exp(current_log_sigma_t).detach()
                test_logs[f"test_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            else:
                final_task_loss_component = static_weight * raw_loss_t
                test_supervised_loss_contribution += final_task_loss_component.detach()
                test_logs[f"test_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            test_logs[f"test_{name}_static_weight"] = torch.tensor(static_weight, device=x.device)

            self._update_r2_metric(
                stage="test",
                task_name=name,
                preds=pred_tensor,
                targets=target,
                sample_mask=sample_mask,
            )

        test_logs["test_final_supervised_loss"] = test_supervised_loss_contribution.detach()
        final_test_loss = final_test_loss + test_supervised_loss_contribution

        self.log_dict(test_logs, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_final_loss", final_test_loss.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        return None

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._reset_stage_metrics("val")
        self._init_stage_index_tracker("val")

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
    ) -> dict[str, torch.Tensor]:
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
        dict[str, torch.Tensor]
            Flat dictionary containing head-specific prediction outputs.
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

        final_predictions: dict[str, torch.Tensor] = {}

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
            head = self.task_heads[task_name]
            processed_pred_dict = head.predict(raw_pred_tensor)  # type: ignore

            if isinstance(head, KernelRegressionHead) and task_name in kernel_regression_sequence_lengths:
                sequence_lengths = kernel_regression_sequence_lengths[task_name]
                processed_pred_dict = self._reshape_kernel_regression_predictions(processed_pred_dict, sequence_lengths)

            final_predictions.update(processed_pred_dict)

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
        """Create an optimizer based on the configuration."""
        params = list(filter(lambda p: p.requires_grad, params))
        if not params:  # If no parameters require gradients, return a dummy optimizer or handle appropriately
            # This path should ideally not be hit if checks are done before calling _create_optimizer
            logger.warning(f"Optimizer creation called with no parameters requiring gradients for config: {config}")
            # Depending on strictness, could raise error or return a dummy. For now, let it proceed (might error in optim).
            # A more robust solution might be to return a specific dummy optimizer if PyTorch allows,
            # or ensure this function is not called with empty grad-requiring params.
            pass

        if config.optimizer_type == "AdamW":
            return optim.AdamW(
                params, lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay
            )
        elif config.optimizer_type == "Adam":
            return optim.Adam(
                params, lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay
            )
        elif config.optimizer_type == "SGD":
            return optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, config: OptimizerConfig) -> LRScheduler | None:
        """Create a learning rate scheduler based on the configuration."""
        if config.scheduler_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.mode,
                factor=config.factor,
                patience=config.patience,
                min_lr=config.min_lr,
            )
        elif config.scheduler_type == "StepLR":
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=config.factor)
        elif config.scheduler_type == "None":
            return None
        else:
            raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers for all parameter groups."""

        optimizers_and_schedulers = []

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
                optimizers_and_schedulers.append(encoder_opt)
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
                optimizers_and_schedulers.append(task_opt)

        if not optimizers_and_schedulers:
            logger.warning(
                "No optimizers were configured. This might be due to all parameters being frozen or an issue in parameter collection."
            )
            # Lightning requires at least one optimizer if the model has trainable parameters.
            # If all parameters are frozen, this is fine. Otherwise, it's an issue.

        return optimizers_and_schedulers

    def _expand_for_kernel_regression(
        self, h_task: torch.Tensor, t_sequence: List[torch.Tensor] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Expand h_task and t_sequence for KernelRegressionHead processing.

        This method does pure data expansion without any filtering or mask decisions.
        It simply concatenates all sequence data and replicates features accordingly.
        The responsibility of handling valid/invalid data lies with the mask processing
        in the loss computation steps (training/validation/test).

        FIXED: Removed harmful 0-value filtering that was incorrectly excluding
        Energy=0 data points. Now performs simple expansion only.

        Parameters
        ----------
        h_task : torch.Tensor
            Tanh-activated latent representations, shape (B, D)
        t_sequence : List[torch.Tensor] | torch.Tensor
            Sequence of t values. Can be:
            - List[torch.Tensor]: Each element is a 1D tensor of variable length
            - torch.Tensor: Legacy format, shape (B, L) with padding

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (expanded_h_task, expanded_t) where:
            - expanded_h_task: shape (N, D) where N = sum of sequence lengths across batch
            - expanded_t: shape (N,) where N = sum of sequence lengths across batch
        """
        batch_size = h_task.shape[0]
        expanded_h_list = []
        expanded_t_list = []

        if isinstance(t_sequence, list):
            # Handle List[Tensor] format from custom collate function
            if len(t_sequence) != batch_size:
                raise ValueError(
                    f"Mismatch between batch_size ({batch_size}) and t_sequence list length ({len(t_sequence)})"
                )

            for batch_idx in range(batch_size):
                t_sample = t_sequence[batch_idx]  # Shape: (seq_len,) - variable length
                h_sample = h_task[batch_idx]  # Shape: (D,)

                # Simple expansion: replicate h_sample for each t value
                seq_len = len(t_sample)
                if seq_len > 0:
                    h_replicated = h_sample.unsqueeze(0).repeat(seq_len, 1)  # Shape: (seq_len, D)
                    expanded_h_list.append(h_replicated)
                    expanded_t_list.append(t_sample)
        else:
            # Handle legacy Tensor format for backwards compatibility
            if t_sequence.dim() != 2 or t_sequence.shape[0] != batch_size:
                raise ValueError(f"Expected t_sequence tensor to have shape (B, L), got {t_sequence.shape}")

            for batch_idx in range(batch_size):
                t_sample = t_sequence[batch_idx]  # Shape: (L,)
                h_sample = h_task[batch_idx]  # Shape: (D,)

                # Simple expansion: replicate h_sample for each t value
                seq_len = len(t_sample)
                if seq_len > 0:
                    h_replicated = h_sample.unsqueeze(0).repeat(seq_len, 1)  # Shape: (seq_len, D)
                    expanded_h_list.append(h_replicated)
                    expanded_t_list.append(t_sample)

        # Concatenate all expanded samples
        if expanded_h_list:
            expanded_h_task = torch.cat(expanded_h_list, dim=0)  # Shape: (total_points, D)
            expanded_t = torch.cat(expanded_t_list, dim=0)  # Shape: (total_points,)
        else:
            # Handle empty case gracefully
            device = h_task.device
            expanded_h_task = torch.empty(0, h_task.shape[1], device=device, dtype=h_task.dtype)
            expanded_t = torch.empty(
                0, device=device, dtype=t_sequence[0].dtype if isinstance(t_sequence, list) else t_sequence.dtype
            )

        return expanded_h_task, expanded_t

    def _reshape_kernel_regression_predictions(
        self, processed_pred_dict: dict[str, np.ndarray], sequence_lengths: List[int]
    ) -> dict[str, List[np.ndarray]]:
        """
        Reshape flattened KernelRegression predictions back to List[numpy.ndarray] format.

        This method takes the flattened predictions from a KernelRegressionHead and
        reshapes them back to the original List[numpy.ndarray] format that matches the input
        structure, ensuring batch consistency with other task types and compatibility
        with PredictionDataFrameWriter.

        Parameters
        ----------
        processed_pred_dict : dict[str, np.ndarray]
            Dictionary containing flattened predictions from KernelRegressionHead.predict().
            Keys are typically prefixed with snake_case task name (e.g., "task_name_value").
            Values are already numpy arrays.
        sequence_lengths : List[int]
            List of original sequence lengths for each sample in the batch.
            Length of this list equals the batch size.

        Returns
        -------
        dict[str, List[np.ndarray]]
            Dictionary with the same keys as input, but values are reshaped to List[numpy.ndarray]
            format where each array corresponds to one sample's predictions.
        """
        reshaped_dict = {}

        for key, flattened_value in processed_pred_dict.items():
            # Handle both numpy arrays and potential torch tensors for backward compatibility
            if isinstance(flattened_value, np.ndarray):
                flattened_array = flattened_value
            else:
                # Fallback for torch tensors (should not happen with KernelRegression)
                flattened_array = flattened_value.detach().cpu().numpy()

            # Split the flattened array back into individual sample predictions
            reshaped_list = []
            start_idx = 0

            for seq_len in sequence_lengths:
                if seq_len > 0:
                    # Extract predictions for this sample
                    end_idx = start_idx + seq_len
                    sample_predictions = flattened_array[start_idx:end_idx]

                    # Squeeze to remove unnecessary dimensions (e.g., (N, 1) -> (N,))
                    # This ensures CSV output shows [1.23, 4.56] instead of [[1.23], [4.56]]
                    if sample_predictions.ndim == 2 and sample_predictions.shape[1] == 1:
                        sample_predictions = sample_predictions.squeeze(axis=1)

                    # Keep as numpy array for compatibility with PredictionDataFrameWriter
                    reshaped_list.append(sample_predictions)

                    start_idx = end_idx
                else:
                    # Handle empty sequences (though this should be rare)
                    empty_array = np.empty(0, dtype=flattened_array.dtype)
                    reshaped_list.append(empty_array)

            reshaped_dict[key] = reshaped_list

        return reshaped_dict

    def on_save_checkpoint(self, checkpoint):
        """
        Custom checkpoint saving that stores optimizer states as dict for better flexibility.

        This method converts the default list-based optimizer states to a dictionary format
        using meaningful keys (shared_encoder, task_{name}). This allows for
        robust checkpoint loading even when task configurations change.

        Special handling for task_log_sigmas:
        - Records which task_log_sigmas were saved
        - Allows for proper reconstruction during loading
        """
        super().on_save_checkpoint(checkpoint)

        if "optimizer_states" not in checkpoint:
            return

        # Convert list format to dictionary format
        optimizer_states_list = checkpoint["optimizer_states"]
        lr_schedulers_list = checkpoint.get("lr_schedulers", [])

        optimizer_states_dict = {}
        lr_schedulers_dict = {}

        # Record task_log_sigmas information for proper loading
        task_log_sigmas_info = {}
        if self.enable_learnable_loss_balancer and hasattr(self, "task_log_sigmas"):
            task_log_sigmas_info = {
                "enabled": True,
                "task_names": list(self.task_log_sigmas.keys()),
                "count": len(self.task_log_sigmas),
            }
        else:
            task_log_sigmas_info = {"enabled": False, "task_names": [], "count": 0}

        optimizer_index = 0

        # 1. Main optimizer (shared_encoder + task_log_sigmas)
        main_params = list(self.encoder.parameters())
        if self.enable_learnable_loss_balancer and hasattr(self, "task_log_sigmas"):
            main_params.extend(list(self.task_log_sigmas.parameters()))

        main_params_trainable = [p for p in main_params if p.requires_grad]
        if main_params_trainable and optimizer_index < len(optimizer_states_list):
            optimizer_states_dict["shared_encoder"] = optimizer_states_list[optimizer_index]
            if optimizer_index < len(lr_schedulers_list):
                lr_schedulers_dict["shared_encoder"] = lr_schedulers_list[optimizer_index]
            optimizer_index += 1

        # 2. Task head optimizers
        for name, head in self.task_heads.items():
            head_params_trainable = [p for p in head.parameters() if p.requires_grad]
            if head_params_trainable and optimizer_index < len(optimizer_states_list):
                optimizer_states_dict[f"task_{name}"] = optimizer_states_list[optimizer_index]
                if optimizer_index < len(lr_schedulers_list):
                    lr_schedulers_dict[f"task_{name}"] = lr_schedulers_list[optimizer_index]
                optimizer_index += 1

        # Store both formats for compatibility
        checkpoint["optimizer_states_dict"] = optimizer_states_dict
        checkpoint["lr_schedulers_dict"] = lr_schedulers_dict
        checkpoint["task_log_sigmas_info"] = task_log_sigmas_info

        logger.debug(f"Saved optimizer states in dict format: {list(optimizer_states_dict.keys())}")
        logger.debug(f"Saved task_log_sigmas info: {task_log_sigmas_info}")

    def on_load_checkpoint(self, checkpoint):
        """
        Custom checkpoint loading that handles frozen parameters using dict-based optimizer states.

        This method provides robust checkpoint loading by:
        1. Using dict-based optimizer states when available (handles task changes)
        2. Filtering out optimizer states for frozen components
        3. Falling back to graceful loading when optimizer states don't match
        """
        # Priority 1: Use new dict format if available
        if "optimizer_states_dict" in checkpoint:
            self._load_checkpoint_from_dict(checkpoint)
        # Priority 2: Use fallback logic for list format
        elif "optimizer_states" in checkpoint:
            self._load_checkpoint_with_fallback(checkpoint)

        # Always call parent method to load model state
        super().on_load_checkpoint(checkpoint)

    def _load_checkpoint_from_dict(self, checkpoint):
        """
        Load checkpoint using dict-based optimizer states with special handling for task_log_sigmas.

        This method handles the case where task_log_sigmas parameters may have changed between
        saving and loading due to different task configurations.
        """
        optimizer_states_dict = checkpoint["optimizer_states_dict"]
        lr_schedulers_dict = checkpoint.get("lr_schedulers_dict", {})
        task_log_sigmas_info = checkpoint.get("task_log_sigmas_info", {"enabled": False, "task_names": [], "count": 0})

        # Build required optimizer states list based on current configuration
        optimizer_states_list = []
        lr_schedulers_list = []

        # 1. Main optimizer (only if not frozen)
        if not self.freeze_shared_encoder:
            main_params = list(self.encoder.parameters())

            # Check task_log_sigmas compatibility
            current_task_log_sigmas_names = []
            if self.enable_learnable_loss_balancer and hasattr(self, "task_log_sigmas"):
                current_task_log_sigmas_names = list(self.task_log_sigmas.keys())
                main_params.extend(list(self.task_log_sigmas.parameters()))

            saved_task_log_sigmas_names = task_log_sigmas_info.get("task_names", [])

            # Check if task_log_sigmas configuration has changed
            task_log_sigmas_changed = set(current_task_log_sigmas_names) != set(saved_task_log_sigmas_names) or len(
                current_task_log_sigmas_names
            ) != len(saved_task_log_sigmas_names)

            main_params_trainable = [p for p in main_params if p.requires_grad]
            if main_params_trainable and "shared_encoder" in optimizer_states_dict:
                if task_log_sigmas_changed:
                    logger.warning(
                        f"task_log_sigmas configuration has changed: "
                        f"saved={saved_task_log_sigmas_names}, current={current_task_log_sigmas_names}. "
                        "Skipping main optimizer state loading to avoid parameter group mismatch."
                    )
                    # Skip loading this optimizer state due to task_log_sigmas mismatch
                else:
                    optimizer_states_list.append(optimizer_states_dict["shared_encoder"])
                    if "shared_encoder" in lr_schedulers_dict:
                        lr_schedulers_list.append(lr_schedulers_dict["shared_encoder"])
                    logger.info("Loaded shared_encoder optimizer state from checkpoint")
            elif main_params_trainable:
                logger.warning("shared_encoder has trainable params but no optimizer state found in checkpoint")

        # 2. Task head optimizers (only if not frozen)
        for name, head in self.task_heads.items():
            config = self.task_configs_map[name]
            if not config.freeze_parameters:
                head_params_trainable = [p for p in head.parameters() if p.requires_grad]
                task_key = f"task_{name}"
                if head_params_trainable and task_key in optimizer_states_dict:
                    optimizer_states_list.append(optimizer_states_dict[task_key])
                    if task_key in lr_schedulers_dict:
                        lr_schedulers_list.append(lr_schedulers_dict[task_key])
                    logger.info(f"Loaded {name} task optimizer state from checkpoint")
                elif head_params_trainable:
                    logger.warning(f"Task {name} has trainable params but no optimizer state found in checkpoint")

        # Update checkpoint with filtered states
        checkpoint["optimizer_states"] = optimizer_states_list
        checkpoint["lr_schedulers"] = lr_schedulers_list

        logger.info(f"Successfully loaded {len(optimizer_states_list)} optimizer states from dict format")

    def _load_checkpoint_with_fallback(self, checkpoint):
        """Load checkpoint with fallback logic for list-based optimizer states."""
        try:
            # Calculate expected number of optimizers based on current config
            expected_optimizers = 0

            # Count main optimizer
            main_params = list(self.encoder.parameters())
            if self.enable_learnable_loss_balancer and hasattr(self, "task_log_sigmas"):
                main_params.extend(list(self.task_log_sigmas.parameters()))
            main_params_trainable = [p for p in main_params if p.requires_grad]
            if main_params_trainable and not self.freeze_shared_encoder:
                expected_optimizers += 1

            # Count task head optimizers
            for name, head in self.task_heads.items():
                config = self.task_configs_map[name]
                head_params_trainable = [p for p in head.parameters() if p.requires_grad]
                if head_params_trainable and not config.freeze_parameters:
                    expected_optimizers += 1

            actual_optimizers = len(checkpoint["optimizer_states"])

            if expected_optimizers != actual_optimizers:
                logger.warning(
                    f"Optimizer count mismatch: expected {expected_optimizers}, "
                    f"found {actual_optimizers} in checkpoint. "
                    "This likely indicates frozen parameter configuration has changed. "
                    "Removing optimizer states to avoid parameter group size mismatch."
                )
                # Remove optimizer states to avoid mismatch errors
                checkpoint.pop("optimizer_states", None)
                checkpoint.pop("lr_schedulers", None)
            else:
                logger.info(f"Optimizer count matches: {expected_optimizers}. Loading normally.")

        except Exception as e:
            logger.warning(f"Error during optimizer state validation: {e}. Removing optimizer states as fallback.")
            checkpoint.pop("optimizer_states", None)
            checkpoint.pop("lr_schedulers", None)

    def optimize_latent(
        self,
        task_name: str | None = None,
        initial_input: torch.Tensor | None = None,
        mode: str = "max",
        steps: int = 200,
        lr: float = 0.1,
        num_restarts: int = 1,
        perturbation_std: float = 0.0,
        target_value: torch.Tensor | float | None = None,
        task_targets: Mapping[str, torch.Tensor | float] | None = None,
        class_targets: Mapping[str, int | Sequence[int]] | None = None,
        class_target_weight: float = 1.0,
        ae_align_scale: float = 0.5,
        optimize_space: str = "input",
    ) -> OptimizationResult:
        """
        Optimize inputs to drive one or multiple regression heads toward targets or extremes.

        Two strategies are available via ``optimize_space``:

        - ``"input"`` (default): gradient-descend directly on the input tensor X.
        - ``"latent"``: encode X to the latent space, optimise there, then reconstruct X
          via the built-in reconstruction head (requires ``enable_autoencoder=True`` at
          model construction time).

        Parameters
        ----------
        task_name : str | None
            Regression task to optimise (legacy single-task path). Optional — and ignored — when
            ``task_targets`` or ``class_targets`` is provided; required otherwise.
        initial_input : torch.Tensor
            Seed inputs, shape (B, input_dim). Always required (raises ``ValueError`` if ``None``).
        mode : str, optional
            ``"max"`` or ``"min"``. Ignored when ``target_value`` / ``task_targets`` is set.
        steps : int, optional
            Optimisation steps per restart. Default 200.
        lr : float, optional
            Adam learning rate. Default 0.1.
        num_restarts : int, optional
            Independent restarts (with optional perturbation). Default 1.
        perturbation_std : float, optional
            Gaussian noise std added to the starting point of each restart. Default 0.0.
        target_value : float | Tensor | None, optional
            Minimise MSE to this scalar target (single task). Overrides ``mode``.
        task_targets : Mapping[str, float | Tensor] | None, optional
            Multi-task regression targets. When provided, ``mode`` and ``target_value`` are ignored.
        class_targets : Mapping[str, int | Sequence[int]] | None, optional
            Classification objectives: maps a classification task name to the class index (or
            indices) whose combined probability should be *maximized*. Adds a ``-log P(target
            classes)`` term to the objective and may be combined with ``task_targets``.
        class_target_weight : float, optional
            Multiplier on each classification objective term relative to the regression terms.
            Use ``> 1`` to make class probability the primary objective and regression targets
            secondary. Default ``1.0``.
        ae_align_scale : float, optional
            Latent-space optimization only. How hard to pull the optimised latent ``h`` toward the
            AE's decode/encode fixed set, on a [0, 1] scale.

            * ``0.0``: **no alignment penalty** — pure unconstrained latent optimisation. This was
              shown in PR #18 to fail badly (QC drops from ~0.97 to ~0.35 after the decode/encode
              round-trip); recorded for completeness as a failure-mode baseline.
            * ``1.0``: **strong alignment penalty** — keeps ``h`` close to ``encode(decode(h))``,
              i.e. on the AE's stable manifold. Over-constraining tends to reduce target achievement.
            * ``0.5`` (default): the empirical sweet spot from PR #18 experiments.

            Implementation detail (skip if not curious): the loss gets a
            ``ae_align_scale · ‖tanh(encoder(AE.decode(h))) − h‖²`` term added. Operates in
            **latent space**; orthogonal to :meth:`optimize_composition`'s ``diversity_scale``
            which lives in composition space.
        optimize_space : str, optional
            ``"input"`` or ``"latent"``. Default ``"input"``.

        Returns
        -------
        OptimizationResult
            namedtuple with fields:
            - optimized_input  : (B, R, input_dim)
            - optimized_target : (B, R, T)
            - initial_score    : (B, R, T)
            - trajectory       : (B, R, steps, T)

        Raises
        ------
        ValueError
            If ``optimize_space="latent"`` but the model was built without
            ``enable_autoencoder=True``, or if task/mode validation fails.
        """
        _AE_TASK = "__reconstruction__"

        # Validate optimization space
        if optimize_space not in {"input", "latent"}:
            raise ValueError(f"optimize_space must be 'input' or 'latent', got '{optimize_space}'")

        # Validate regression targets
        target_tasks: dict[str, torch.Tensor | float] | None = None
        if task_targets is not None:
            if not isinstance(task_targets, Mapping) or len(task_targets) == 0:
                raise ValueError("task_targets must be a non-empty mapping of task_name -> target_value")
            target_tasks = dict(task_targets)
            if target_value is not None:
                raise ValueError("Use either task_targets (multi-task) or target_value (single task), not both.")
            for name in target_tasks:
                if name not in self.task_heads:
                    raise ValueError(
                        f"Task '{name}' not found in model. Available tasks: {list(self.task_heads.keys())}"
                    )
                cfg = self.task_configs_map[name]
                if cfg.type != TaskType.REGRESSION:
                    raise ValueError(f"Task '{name}' must be a regression task for optimization, got {cfg.type}")

        # Validate classification objectives (maximize combined class probability)
        class_target_map: dict[str, list[int]] | None = None
        if class_targets is not None:
            if not isinstance(class_targets, Mapping) or len(class_targets) == 0:
                raise ValueError("class_targets must be a non-empty mapping of task_name -> class index/indices")
            if class_target_weight <= 0:
                raise ValueError(f"class_target_weight must be > 0, got {class_target_weight}")
            class_target_map = {}
            for name, classes in class_targets.items():
                if name not in self.task_heads:
                    raise ValueError(
                        f"Task '{name}' not found in model. Available tasks: {list(self.task_heads.keys())}"
                    )
                cls_cfg = self.task_configs_map[name]
                if cls_cfg.type != TaskType.CLASSIFICATION:
                    raise ValueError(f"class_targets task '{name}' must be a classification task.")
                idxs = [int(classes)] if isinstance(classes, int) else [int(c) for c in classes]
                if not idxs:
                    raise ValueError(f"class_targets['{name}'] must specify at least one class index.")
                num_classes = getattr(cls_cfg, "num_classes", None)
                if num_classes is not None and any(not 0 <= i < num_classes for i in idxs):
                    raise ValueError(
                        f"class_targets['{name}'] indices {idxs} out of range for a "
                        f"{num_classes}-class head; valid indices are [0, {num_classes})."
                    )
                class_target_map[name] = idxs

        if not 0.0 <= ae_align_scale <= 1.0:
            raise ValueError(f"ae_align_scale must be in [0, 1], got {ae_align_scale}.")

        # Legacy single-task path (mode / target_value) only when no target maps are given
        if target_tasks is None and class_target_map is None:
            if task_name is None or task_name not in self.task_heads:
                raise ValueError(
                    f"Task '{task_name}' not found in model. Available tasks: {list(self.task_heads.keys())}"
                )
            task_config = self.task_configs_map[task_name]
            if task_config.type != TaskType.REGRESSION:
                raise ValueError(
                    f"Task '{task_name}' must be a regression task for optimization, got {task_config.type}"
                )

        # Validate autoencoder availability for latent-space mode
        if optimize_space == "latent":
            if _AE_TASK not in self.task_heads:
                raise ValueError("optimize_space='latent' requires the model to be built with enable_autoencoder=True.")
            if not isinstance(self.task_heads[_AE_TASK], AutoEncoderHead):
                raise ValueError(
                    f"Task '{_AE_TASK}' exists but is not an AutoEncoderHead; "
                    "latent-space optimization requires the built-in reconstruction head."
                )

        if target_tasks is None and class_target_map is None and mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        if num_restarts < 1:
            raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")

        # Store original training state
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        if initial_input is None:
            raise ValueError("initial_input is required and represents the inputs to optimize")

        input_tensor = initial_input
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)
        expected_dim = getattr(self.encoder, "input_dim", None)
        if expected_dim is not None and input_tensor.shape[1] != expected_dim:
            raise ValueError(
                f"initial_input feature dimension mismatch: expected {expected_dim}, got {input_tensor.shape[1]}"
            )

        # Prepare target tensor(s) if provided
        target_tensor: torch.Tensor | None = None
        target_tensor_map: dict[str, torch.Tensor] | None = None
        if target_tasks is not None:
            target_tensor_map = {
                name: torch.as_tensor(val, device=device, dtype=input_tensor.dtype)
                for name, val in target_tasks.items()
            }
        elif target_value is not None:
            target_tensor = torch.as_tensor(target_value, device=device, dtype=input_tensor.dtype)

        class_index_map: dict[str, torch.Tensor] = {}
        if class_target_map is not None:
            class_index_map = {
                name: torch.as_tensor(idxs, device=device, dtype=torch.long) for name, idxs in class_target_map.items()
            }

        # Helper to reduce predictions to scalar per task/batch
        def _reduce_pred(pred: torch.Tensor) -> torch.Tensor:
            if pred.ndim == 1:
                return pred
            return pred.mean(dim=tuple(range(1, pred.ndim)))

        def _stack_scores(vals: list[torch.Tensor]) -> torch.Tensor:
            """Stack per-task scores to (B, T); return (B, 0) when there are no regression tasks."""
            if vals:
                return torch.stack(vals, dim=-1)
            return torch.zeros((input_tensor.shape[0], 0), device=device, dtype=input_tensor.dtype)

        def _class_loss_terms(h_task: torch.Tensor) -> list[torch.Tensor]:
            """``-log P(target classes)`` per classification objective (maximize that prob)."""
            terms: list[torch.Tensor] = []
            for cname, cidx in class_index_map.items():
                logits = self.task_heads[cname](h_task)
                log_probs = F.log_softmax(logits, dim=-1)
                combined = torch.logsumexp(log_probs.index_select(-1, cidx), dim=-1)
                terms.append(-combined.mean())
            return terms

        if target_tasks is not None:
            tasks_for_optimization = list(target_tasks.keys())
        elif class_target_map is not None:
            tasks_for_optimization = []  # classification-only objective
        else:
            assert task_name is not None  # guaranteed by validation above
            tasks_for_optimization = [task_name]
        num_targets = len(tasks_for_optimization)

        optimized_inputs: list[torch.Tensor] = []
        optimized_targets: list[torch.Tensor] = []
        trajectories: list[torch.Tensor] = []
        initial_scores_list: list[torch.Tensor] = []

        for restart_idx in range(num_restarts):
            if optimize_space == "input":
                # Input space optimization: optimize X directly
                start_input = input_tensor.clone()
                if perturbation_std > 0:
                    start_input = start_input + torch.randn_like(start_input) * perturbation_std

                # Record initial score(s)
                with torch.no_grad():
                    latent = self.encoder(start_input)
                    h_task = torch.tanh(latent)
                    initial_vals = []
                    for name in tasks_for_optimization:
                        pred = self.task_heads[name](h_task)
                        initial_vals.append(_reduce_pred(pred).detach())
                    initial_vals = _stack_scores(initial_vals)  # (B, T)
                    initial_score = initial_vals
                    initial_scores_list.append(initial_score)

                # Create optimizable input
                optim_input = start_input.detach().clone().requires_grad_(True)

                # Setup optimizer
                optimizer = optim.Adam([optim_input], lr=lr)

                # Optimization loop
                step_traj: list[torch.Tensor] = []
                sign = 1.0 if mode == "max" else -1.0

                for step in range(steps):
                    optimizer.zero_grad()

                    # Forward through encoder and apply Tanh
                    latent = self.encoder(optim_input)
                    h_task = torch.tanh(latent)
                    per_task_values = []
                    loss_terms = []
                    for name in tasks_for_optimization:
                        pred = self.task_heads[name](h_task)
                        reduced = _reduce_pred(pred)
                        per_task_values.append(reduced)
                        if target_tensor_map is not None:
                            tgt = target_tensor_map[name]
                            expanded_target = tgt
                            if tgt.ndim == 0:
                                expanded_target = expanded_target.reshape([1] * pred.ndim)
                            if expanded_target.shape != pred.shape:
                                expanded_target = expanded_target.expand(pred.shape)
                            loss_terms.append(F.mse_loss(pred, expanded_target))
                        elif target_tensor is not None and name == task_name:
                            tgt = target_tensor
                            expanded_target = tgt
                            if tgt.ndim == 0:
                                expanded_target = expanded_target.reshape([1] * pred.ndim)
                            if expanded_target.shape != pred.shape:
                                expanded_target = expanded_target.expand(pred.shape)
                            loss_terms.append(F.mse_loss(pred, expanded_target))

                    loss_terms.extend(class_target_weight * term for term in _class_loss_terms(h_task))
                    per_task_values_tensor = _stack_scores(per_task_values)  # (B, T)

                    if loss_terms:
                        loss = torch.stack(loss_terms).mean()
                        score_for_history = per_task_values_tensor.detach()
                    else:
                        aggregate = per_task_values_tensor.mean(dim=-1)  # (B,)
                        loss = -sign * aggregate.mean()
                        score_for_history = per_task_values_tensor.detach()

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    # Record history
                    step_traj.append(score_for_history)

                # Get final optimized values
                with torch.no_grad():
                    latent = self.encoder(optim_input)
                    h_task = torch.tanh(latent)
                    per_task_final = []
                    for name in tasks_for_optimization:
                        pred = self.task_heads[name](h_task)
                        per_task_final.append(_reduce_pred(pred).detach())
                    per_task_final_tensor = _stack_scores(per_task_final)  # (B, T)
                    optimized_input = optim_input.detach()

                optimized_inputs.append(optimized_input.detach())  # (B, D)
                optimized_targets.append(per_task_final_tensor)  # (B, T)
                traj_tensor = torch.stack(step_traj, dim=0)  # (steps, B, T)
                trajectories.append(traj_tensor)

            else:  # optimize_space == "latent"
                # Latent space optimization: encode X -> optimize latent -> decode via AE
                with torch.no_grad():
                    initial_latent = self.encoder(input_tensor)

                start_latent = initial_latent.clone()
                if perturbation_std > 0:
                    start_latent = start_latent + torch.randn_like(start_latent) * perturbation_std

                # Record initial score(s)
                # Apply Tanh to get task representation (consistent with forward())
                with torch.no_grad():
                    h_task = torch.tanh(start_latent)
                    initial_vals = []
                    for name in tasks_for_optimization:
                        pred = self.task_heads[name](h_task)
                        initial_vals.append(_reduce_pred(pred).detach())
                    initial_vals = _stack_scores(initial_vals)  # (B, T)
                    initial_score = initial_vals
                    initial_scores_list.append(initial_score)

                # Create optimizable latent
                optim_latent = start_latent.detach().clone().requires_grad_(True)

                # Setup optimizer
                optimizer = optim.Adam([optim_latent], lr=lr)

                # Optimization loop
                step_traj: list[torch.Tensor] = []
                sign = 1.0 if mode == "max" else -1.0

                for step in range(steps):
                    optimizer.zero_grad()

                    # Apply Tanh to get task representation (consistent with forward())
                    # This ensures architectural consistency on every optimization step
                    h_task = torch.tanh(optim_latent)

                    # Forward through task heads using h_task
                    per_task_values = []
                    loss_terms = []
                    for name in tasks_for_optimization:
                        pred = self.task_heads[name](h_task)
                        reduced = _reduce_pred(pred)
                        per_task_values.append(reduced)
                        if target_tensor_map is not None:
                            tgt = target_tensor_map[name]
                            expanded_target = tgt
                            if tgt.ndim == 0:
                                expanded_target = expanded_target.reshape([1] * pred.ndim)
                            if expanded_target.shape != pred.shape:
                                expanded_target = expanded_target.expand(pred.shape)
                            loss_terms.append(F.mse_loss(pred, expanded_target))
                        elif target_tensor is not None and name == task_name:
                            tgt = target_tensor
                            expanded_target = tgt
                            if tgt.ndim == 0:
                                expanded_target = expanded_target.reshape([1] * pred.ndim)
                            if expanded_target.shape != pred.shape:
                                expanded_target = expanded_target.expand(pred.shape)
                            loss_terms.append(F.mse_loss(pred, expanded_target))

                    loss_terms.extend(class_target_weight * term for term in _class_loss_terms(h_task))
                    if ae_align_scale > 0:
                        # Pull the optimised latent toward what the AE faithfully reconstructs:
                        # decode it to a descriptor, re-encode, and penalise the drift in h_task.
                        # The user-facing knob is [0, 1] with 0 = no penalty / 1 = strong penalty.
                        re_h_task = torch.tanh(self.encoder(self.task_heads[_AE_TASK](h_task)))
                        loss_terms.append(ae_align_scale * F.mse_loss(re_h_task, h_task))
                    per_task_values_tensor = _stack_scores(per_task_values)  # (B, T)

                    if loss_terms:
                        loss = torch.stack(loss_terms).mean()
                        score_for_history = per_task_values_tensor.detach()
                    else:
                        aggregate = per_task_values_tensor.mean(dim=-1)  # (B,)
                        loss = -sign * aggregate.mean()
                        score_for_history = per_task_values_tensor.detach()

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    # Record history
                    step_traj.append(score_for_history)

                # Get final optimized values and reconstruct via AE
                with torch.no_grad():
                    # Apply Tanh to get final task representation (consistent with forward())
                    final_h_task = torch.tanh(optim_latent)
                    per_task_final = []
                    for name in tasks_for_optimization:
                        pred = self.task_heads[name](final_h_task)
                        per_task_final.append(_reduce_pred(pred).detach())
                    per_task_final_tensor = _stack_scores(per_task_final)  # (B, T)

                    # Reconstruct input via the built-in reconstruction head
                    reconstructed_input = self.task_heads[_AE_TASK](final_h_task)

                optimized_inputs.append(reconstructed_input.detach())  # (B, D)
                optimized_targets.append(per_task_final_tensor)  # (B, T)
                traj_tensor = torch.stack(step_traj, dim=0)  # (steps, B, T)
                trajectories.append(traj_tensor)

        # Restore training state
        self.train(was_training)

        # Stack outputs
        opt_input_tensor = torch.stack(optimized_inputs, dim=1)  # (B, R, D)
        opt_target_tensor = torch.stack(optimized_targets, dim=1)  # (B, R, T)
        traj_tensor = torch.stack(trajectories, dim=0)  # (R, steps, B, T)
        traj_tensor = traj_tensor.permute(2, 0, 1, 3)  # (B, R, steps, T)
        initial_score_tensor = torch.stack(initial_scores_list, dim=0)  # (R, B, T)
        initial_score_tensor = initial_score_tensor.permute(1, 0, 2)  # (B, R, T)

        return OptimizationResult(
            optimized_input=opt_input_tensor,
            optimized_target=opt_target_tensor,
            initial_score=initial_score_tensor,
            trajectory=traj_tensor,
        )

    def optimize_composition(
        self,
        kmd_kernel: torch.Tensor,
        *,
        initial_weights: torch.Tensor | None = None,
        n_starts: int = 16,
        task_targets: Mapping[str, torch.Tensor | float] | None = None,
        class_targets: Mapping[str, int | Sequence[int]] | None = None,
        class_target_weight: float = 1.0,
        diversity_scale: float = 1.0,
        allowed_elements: str | list[str] = "all",
        element_step_scale: float | Mapping[str, float] = 1.0,
        seed_blend: float = 0.95,
        steps: int = 300,
        lr: float = 0.05,
    ) -> CompositionOptimizationResult:
        """Gradient-based inverse design in **composition space**.

        Optimises a simplex-constrained element-weight vector ``w`` directly through the
        differentiable KMD transform ``x = w @ K`` and the supervised heads:

        ``logits → w = softmax(logits) → x = w @ kmd_kernel → encoder → tanh → heads → loss``

        Because the optimisation variable *is* the recipe, there is **no AE-decode round-trip**:
        the optimised ``w`` is the composition you would report. Compared to :meth:`optimize_latent`
        in ``"latent"`` mode, this method (a) eliminates the round-trip fidelity drop, (b) keeps the
        solution on the legitimate composition simplex by construction, and (c) makes ``w`` itself
        the output.

        Parameters
        ----------
        kmd_kernel : torch.Tensor
            The precomputed KMD kernel matrix, shape ``(n_components, x_dim)`` — typically obtained
            from :meth:`foundation_model.utils.kmd_plus.KMD.kernel_torch`. ``x_dim`` must match the
            encoder's input dim.
        initial_weights : torch.Tensor | None
            Seed weights, shape ``(B, n_components)``. If ``None``, ``n_starts`` random starts are
            sampled from a Gaussian over the logits (mildly diverse simplex starting points).
        n_starts : int
            Batch size when ``initial_weights is None``. Default 16.
        task_targets, class_targets, class_target_weight :
            Same semantics as :meth:`optimize_latent`. Regression targets are matched by MSE;
            classification objectives add ``-log P(target classes)`` (scaled by
            ``class_target_weight``).
        diversity_scale : float, optional
            How spread-out the per-output element mixture is allowed to be, on a [0, 1] scale.
            Bigger = more diverse / multi-element per output.

            * ``1.0`` (default): **no penalty** on having many elements — the optimiser is free
              to land on a many-element recipe if the main objective likes it.
            * ``0.0``: **strong penalty** on having many elements — the optimiser is pushed
              toward peaky few-element recipes (e.g. binary alloys).
            * ``0.5`` etc.: linearly interpolates between the two.

            The point is to give users a simple [0, 1] knob without needing to know the underlying
            math. **Implementation detail** (skip if not curious): the loss gets a
            ``(1 − diversity_scale) · H(w)`` term added, where ``H(w) = −Σ w_i log w_i`` is the
            Shannon entropy of the per-row weight vector. ``diversity_scale = 1`` zeros that
            coefficient (no penalty); ``diversity_scale = 0`` applies the full entropy penalty.

            Important: this is a **per-output complexity** knob, not a diversity-*between*-outputs
            knob. Increasing it lets each of the ``B`` outputs individually use more elements;
            whether the ``B`` outputs are different from each other (pairwise L1) depends on the
            optimisation landscape, not on this knob.
        allowed_elements : str | list[str], optional
            Element whitelist for the optimisation. ``"all"`` (default) imposes no constraint.
            A non-empty list of element symbols (e.g. ``["Mg", "Al", "Cu", "Ni"]``) restricts the
            optimisation to those elements only — disallowed elements are forced to ``w = 0`` at
            every step (their logits are masked to ``-inf`` inside the softmax), so no gradient
            ever lifts them. Symbols are resolved against
            :data:`~foundation_model.utils.kmd_plus.DEFAULT_ELEMENTS`; the kernel must therefore
            have ``n_components == len(DEFAULT_ELEMENTS)`` when symbols are used.
        element_step_scale : float | Mapping[str, float], optional
            Per-element constraint on how fast each element's weight can move during optimisation.
            A scalar applies uniformly to every element (default ``1.0`` = no constraint). A
            symbol→float mapping overrides specific elements while leaving the rest at ``1.0``.

            Two regimes with different mechanics:

            * **Hard lock (value = 0):** ``{"Mg": 0.0, "Al": 0.0}`` pins those elements' weights
              at their un-blended ``initial_weights`` values for the entire optimisation. The
              implementation rewrites the softmax output to paste seed values back at locked
              positions and renormalises the unlocked positions over the remaining
              ``1 − Σ_locked seed`` mass — so the locked weights truly do not drift, even when
              other (unlocked) logits move. Requires ``initial_weights`` (no seed → nothing to
              lock to) and the locked elements must be in ``allowed_elements`` if a whitelist
              is set.
            * **Soft constraint (0 < value < 1):** the element's logit gradient is multiplied by
              the scale before each Adam step, slowing (but not freezing) its drift. ``0.1`` lets
              an element move at 10 % of the normal speed. The softmax denominator still couples
              it to the rest of the row, so this is a soft preference, not a hard guarantee.

            Symbols are resolved against ``DEFAULT_ELEMENTS`` (kernel alignment required, as above).
        seed_blend : float, optional
            How much of the (per-row) seed prior to keep when ``initial_weights`` is given;
            ``w0 ← seed_blend · seed + (1 − seed_blend) · uniform_over_allowed``. Default ``0.95``
            (5 % uniform mass spread over the allowed elements). The blend lifts non-seed-element
            logits from ``log(1e-12) ≈ −27.6`` (effectively unreachable by Adam in a few hundred
            steps) to ``log(0.05 / |allowed|) ≈ −7.6``, so the optimiser can introduce new elements
            when they help the objective. Set to ``1.0`` to reproduce the strict seed-only behaviour
            (no new elements can enter the support set); ``0.0`` makes the seed irrelevant and
            starts from uniform. Ignored when ``initial_weights is None``.
        steps : int
            Adam optimisation steps. Default 300.
        lr : float
            Adam learning rate over the logits. Default 0.05.

        Returns
        -------
        CompositionOptimizationResult
            with fields:
            - ``optimized_weights``    : (B, n_components), each row a simplex point — the recipe.
            - ``optimized_descriptor`` : (B, x_dim), equals ``optimized_weights @ kmd_kernel``.
            - ``optimized_target``     : (B, T), final per-regression-task predicted values.
            - ``initial_score``        : (B, T), same shape, predicted at step 0.
            - ``trajectory``           : (steps, B, T), per-task values across optimisation.
        """
        # --- Validate the kernel ----------------------------------------------------------------
        if not isinstance(kmd_kernel, torch.Tensor) or kmd_kernel.ndim != 2:
            raise ValueError("kmd_kernel must be a 2D torch.Tensor of shape (n_components, x_dim).")
        n_components, x_dim = kmd_kernel.shape
        expected_dim = getattr(self.encoder, "input_dim", None)
        if expected_dim is not None and x_dim != expected_dim:
            raise ValueError(f"kmd_kernel.shape[1]={x_dim} does not match encoder.input_dim={expected_dim}.")

        # --- Validate regression / classification objectives (mirrors optimize_latent) ----------
        target_tasks: dict[str, torch.Tensor | float] | None = None
        if task_targets is not None:
            if not isinstance(task_targets, Mapping) or len(task_targets) == 0:
                raise ValueError("task_targets must be a non-empty mapping of task_name -> target_value")
            target_tasks = dict(task_targets)
            for name in target_tasks:
                if name not in self.task_heads:
                    raise ValueError(
                        f"Task '{name}' not found in model. Available tasks: {list(self.task_heads.keys())}"
                    )
                if self.task_configs_map[name].type != TaskType.REGRESSION:
                    raise ValueError(f"Task '{name}' must be a regression task for optimization.")

        class_target_map: dict[str, list[int]] | None = None
        if class_targets is not None:
            if not isinstance(class_targets, Mapping) or len(class_targets) == 0:
                raise ValueError("class_targets must be a non-empty mapping of task_name -> class index/indices")
            if class_target_weight <= 0:
                raise ValueError(f"class_target_weight must be > 0, got {class_target_weight}")
            class_target_map = {}
            for name, classes in class_targets.items():
                if name not in self.task_heads:
                    raise ValueError(
                        f"Task '{name}' not found in model. Available tasks: {list(self.task_heads.keys())}"
                    )
                cls_cfg = self.task_configs_map[name]
                if cls_cfg.type != TaskType.CLASSIFICATION:
                    raise ValueError(f"class_targets task '{name}' must be a classification task.")
                idxs = [int(classes)] if isinstance(classes, int) else [int(c) for c in classes]
                if not idxs:
                    raise ValueError(f"class_targets['{name}'] must specify at least one class index.")
                num_classes = getattr(cls_cfg, "num_classes", None)
                if num_classes is not None and any(not 0 <= i < num_classes for i in idxs):
                    raise ValueError(
                        f"class_targets['{name}'] indices {idxs} out of range for a "
                        f"{num_classes}-class head; valid indices are [0, {num_classes})."
                    )
                class_target_map[name] = idxs

        if target_tasks is None and class_target_map is None:
            raise ValueError("Provide at least one of task_targets / class_targets.")
        if not 0.0 <= diversity_scale <= 1.0:
            raise ValueError(f"diversity_scale must be in [0, 1], got {diversity_scale}.")
        if not 0.0 <= seed_blend <= 1.0:
            raise ValueError(f"seed_blend must be in [0, 1], got {seed_blend}")

        # --- Per-element constraints (symbol-based) -----------------------------------------------
        # ``allowed_elements`` is a hard whitelist; ``element_step_scale`` is a soft per-element
        # learning-rate multiplier (0 = frozen). Symbol-based inputs are resolved against the
        # bundled :data:`DEFAULT_ELEMENTS` registry — see argument docs above.
        from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS  # local import; small list

        elem_mask_arg: torch.Tensor | None = None
        if isinstance(allowed_elements, str):
            if allowed_elements != "all":
                raise ValueError(f"allowed_elements as a string must be 'all'; got {allowed_elements!r}.")
            # "all": no constraint, leave elem_mask_arg as None.
        elif isinstance(allowed_elements, (list, tuple)):
            if len(allowed_elements) == 0:
                raise ValueError("allowed_elements list must be non-empty.")
            sym_to_idx = {s: i for i, s in enumerate(DEFAULT_ELEMENTS)}
            bad = [s for s in allowed_elements if s not in sym_to_idx]
            if bad:
                raise ValueError(f"Unknown element symbol(s) in allowed_elements: {bad}.")
            if n_components != len(DEFAULT_ELEMENTS):
                raise ValueError(
                    f"allowed_elements as element symbols requires the kernel to align with "
                    f"DEFAULT_ELEMENTS (n_components={n_components}, expected {len(DEFAULT_ELEMENTS)})."
                )
            elem_mask_arg = torch.zeros(n_components, dtype=torch.bool)
            for sym in allowed_elements:
                elem_mask_arg[sym_to_idx[sym]] = True
        else:
            raise TypeError(
                f"allowed_elements must be 'all' or a non-empty list of element symbols; got {type(allowed_elements).__name__}."
            )

        step_scale_arg: torch.Tensor | None = None
        if isinstance(element_step_scale, (int, float)) and not isinstance(element_step_scale, bool):
            if element_step_scale < 0:
                raise ValueError(f"element_step_scale must be >= 0; got {element_step_scale}.")
            if float(element_step_scale) != 1.0:
                step_scale_arg = torch.full((n_components,), float(element_step_scale))
            # else: 1.0 means "no scaling"; keep step_scale_arg = None for the fast path.
        elif isinstance(element_step_scale, Mapping):
            sym_to_idx = {s: i for i, s in enumerate(DEFAULT_ELEMENTS)}
            bad = [s for s in element_step_scale if s not in sym_to_idx]
            if bad:
                raise ValueError(f"Unknown element symbol(s) in element_step_scale: {bad}.")
            if any(float(v) < 0 for v in element_step_scale.values()):
                raise ValueError("element_step_scale values must be >= 0.")
            if n_components != len(DEFAULT_ELEMENTS):
                raise ValueError(
                    f"element_step_scale as a symbol dict requires the kernel to align with "
                    f"DEFAULT_ELEMENTS (n_components={n_components}, expected {len(DEFAULT_ELEMENTS)})."
                )
            step_scale_arg = torch.ones(n_components)
            for sym, val in element_step_scale.items():
                step_scale_arg[sym_to_idx[sym]] = float(val)
        else:
            raise TypeError(
                f"element_step_scale must be a non-negative float or a mapping of "
                f"element_symbol → float; got {type(element_step_scale).__name__}."
            )

        # --- Validate the seed (BEFORE touching model state, so a bad input doesn't leave the
        #     model in eval() / with params switched off). ---------------------------------------
        if initial_weights is None:
            if n_starts < 1:
                raise ValueError("n_starts must be >= 1 when initial_weights is None.")
        else:
            if initial_weights.ndim != 2 or initial_weights.shape[1] != n_components:
                raise ValueError(
                    f"initial_weights must have shape (B, {n_components}); got {tuple(initial_weights.shape)}."
                )
            if (initial_weights < 0).any():
                raise ValueError("initial_weights must be non-negative (no silent clamping).")
            if (initial_weights.sum(dim=-1) <= 0).any():
                raise ValueError("initial_weights rows must have a positive sum.")

        # --- Save / restore model state ------------------------------------------------------------
        # Wrap the optimisation in try/finally so a later raise (e.g. a head failure) still
        # restores training mode and parameter requires_grad flags. During the call we also turn
        # off requires_grad on every parameter — only ``logits`` is being optimised, so
        # ``loss.backward()`` would otherwise populate stale ``.grad`` on every encoder/head
        # parameter for no benefit.
        was_training = self.training
        saved_req_grad: list[tuple[torch.nn.Parameter, bool]] = [(p, p.requires_grad) for p in self.parameters()]
        self.eval()
        for p, _ in saved_req_grad:
            p.requires_grad_(False)
        try:
            ref_param = next(self.parameters())
            device, dtype = ref_param.device, ref_param.dtype  # match the model's precision
            kmd_kernel = kmd_kernel.to(device=device, dtype=dtype)

            # --- Build logits over n_components ---------------------------------------------------
            # We additionally capture the *un-blended* normalised seed (``w0_seed``) — the
            # locked-element hard-lock below uses these values, not the post-blend ones, so a
            # user who writes ``element_step_scale={"Mg": 0.0}`` with ``initial_weights`` placing
            # Mg at 0.30 sees Mg held at exactly 0.30 (not the slightly blended 0.286).
            w0_seed: torch.Tensor | None = None
            if initial_weights is None:
                # Use the caller's existing global RNG state — don't reseed here (would defeat
                # the intended diversity across repeated calls and would leak state outward).
                logits = torch.randn(n_starts, n_components, device=device, dtype=dtype) * 0.5
                if elem_mask_arg is not None:
                    # Push disallowed elements to a deep negative logit so softmax mask works
                    # consistently for both the random and seeded branches (the per-step mask
                    # below also enforces this; we mirror it here for the t=0 score).
                    logits = logits.masked_fill(~elem_mask_arg.to(device=device), -1e9)
            else:
                w0 = initial_weights.to(device=device, dtype=dtype)
                w0 = w0 / w0.sum(dim=-1, keepdim=True)
                w0_seed = w0.detach().clone()  # un-blended; used as the lock reference below
                # Blend in a uniform prior so non-seed-element logits are reachable by Adam.
                # Without this, log(0) → −∞ (clamped to log(1e-12) ≈ −27.6); the softmax Jacobian
                # is proportional to w_i, so the per-step gradient on those logits is ≈ 1e-12 and
                # Adam cannot lift them within a few hundred steps — the support set is frozen to
                # the seed's nonzero elements. ``seed_blend < 1`` spreads a small uniform mass
                # over the allowed elements so every reachable element starts at a workable logit.
                if seed_blend < 1.0:
                    if elem_mask_arg is not None:
                        uniform_row = elem_mask_arg.to(device=device, dtype=dtype)
                        uniform_row = uniform_row / uniform_row.sum()
                    else:
                        uniform_row = torch.full((n_components,), 1.0 / n_components, device=device, dtype=dtype)
                    w0 = seed_blend * w0 + (1.0 - seed_blend) * uniform_row
                    w0 = w0 / w0.sum(dim=-1, keepdim=True)
                # Tiny floor only to avoid log(0) when an element is both disallowed AND not in
                # the uniform support (i.e. seed_blend == 1.0 with sparse seeds).
                logits = torch.log(w0.clamp(min=1e-12)).detach().clone()
            logits = logits.requires_grad_(True)
            optimizer = optim.Adam([logits], lr=lr)

            tasks_for_optimization: list[str] = list(target_tasks.keys()) if target_tasks is not None else []
            target_tensor_map: dict[str, torch.Tensor] = {}
            if target_tasks is not None:
                target_tensor_map = {
                    name: torch.as_tensor(val, device=device, dtype=dtype) for name, val in target_tasks.items()
                }
            class_index_map: dict[str, torch.Tensor] = {}
            if class_target_map is not None:
                class_index_map = {
                    name: torch.as_tensor(idxs, device=device, dtype=torch.long)
                    for name, idxs in class_target_map.items()
                }

            # Move the element-constraint tensors onto the right device (validated above).
            elem_mask = elem_mask_arg.to(device=device) if elem_mask_arg is not None else None
            step_scale = step_scale_arg.to(device=device, dtype=dtype) if step_scale_arg is not None else None

            # --- Hard-lock setup for elements with step_scale == 0 ----------------------------------
            # Zeroing ``logit_i.grad`` keeps that logit constant but does NOT keep ``w_i`` constant,
            # because softmax renormalises across all logits — when other (unlocked) logits move, the
            # softmax denominator changes and so does the locked weight. To truly honour the docstring
            # promise "freezes those elements at their seed values", we (a) detect locked indices, (b)
            # capture their per-row seed weights, and (c) inside ``_w_from_logits`` paste those seed
            # values back over the softmax output and renormalise the unlocked positions to fill the
            # remaining ``1 − Σ locked_w`` mass per row. The gradient through the locked indices is
            # automatically zero (the lock branch uses a constant), so we no longer need the
            # ``step_scale.mul_`` zeroing for them — but we leave that path active for the genuinely
            # soft case ``0 < step_scale < 1``.
            locked_mask: torch.Tensor | None = None
            locked_w0: torch.Tensor | None = None
            if step_scale is not None:
                locked_idx_mask = step_scale == 0
                if locked_idx_mask.any():
                    if w0_seed is None:
                        raise ValueError(
                            "element_step_scale = 0 (hard lock) requires initial_weights — there's no "
                            "per-row seed to lock to when initial_weights=None."
                        )
                    if elem_mask is not None and (~elem_mask[locked_idx_mask]).any():
                        raise ValueError(
                            "Locked elements (element_step_scale = 0) must also be in allowed_elements; "
                            "locking a disallowed element is contradictory."
                        )
                    locked_mask = locked_idx_mask  # (n_components,) bool, on device
                    # (B, n_components): seed values at locked positions, 0 elsewhere — constant.
                    locked_w0 = (w0_seed * locked_mask.to(dtype)).detach()

            def _w_from_logits(lg: torch.Tensor) -> torch.Tensor:
                """Softmax over logits; mask disallowed elements; hard-lock the chosen ones at seed."""
                if elem_mask is not None:
                    lg = lg.masked_fill(~elem_mask, float("-inf"))
                w = torch.softmax(lg, dim=-1)
                if locked_mask is None:
                    return w
                # Locked rows hold their seed values; unlocked rows are renormalised to fill the
                # remaining mass ``1 − Σ_locked seed``. Differentiable: the lock branch is a constant
                # so its gradient is 0; the unlocked branch's gradient flows through the rescale.
                free_mask_f = (~locked_mask).to(w.dtype)  # (n_components,)
                w_unlocked = w * free_mask_f  # zero at locked positions
                # type: ignore[union-attr] — locked_w0 is set together with locked_mask above.
                free_mass = (1.0 - locked_w0.sum(dim=-1, keepdim=True)).clamp(min=0.0)
                w_unlocked = w_unlocked / w_unlocked.sum(dim=-1, keepdim=True).clamp(min=1e-12) * free_mass
                return w_unlocked + locked_w0

            def _heads_forward(h_task: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
                """Run regression heads, return (per-task predictions, loss terms)."""
                preds, terms = [], []
                for name in tasks_for_optimization:
                    pred = self.task_heads[name](h_task)
                    reduced = pred if pred.ndim == 1 else pred.mean(dim=tuple(range(1, pred.ndim)))
                    preds.append(reduced)
                    tgt = target_tensor_map[name]
                    if tgt.ndim == 0:
                        tgt = tgt.reshape([1] * pred.ndim)
                    if tgt.shape != pred.shape:
                        tgt = tgt.expand(pred.shape)
                    terms.append(F.mse_loss(pred, tgt))
                for cname, cidx in class_index_map.items():
                    cls_logits = self.task_heads[cname](h_task)
                    log_probs = F.log_softmax(cls_logits, dim=-1)
                    combined = torch.logsumexp(log_probs.index_select(-1, cidx), dim=-1)
                    terms.append(class_target_weight * (-combined.mean()))
                return preds, terms

            n_reg_tracked = len(tasks_for_optimization)

            def _stack(values: list[torch.Tensor], B: int) -> torch.Tensor:
                return torch.stack(values, dim=-1) if values else torch.zeros((B, 0), device=device, dtype=dtype)

            # --- Record initial scores --------------------------------------------------------------
            with torch.no_grad():
                w0_tensor = _w_from_logits(logits)
                h0 = torch.tanh(self.encoder(w0_tensor @ kmd_kernel))
                initial_preds, _ = _heads_forward(h0)
                initial_score = _stack([p.detach() for p in initial_preds], logits.shape[0])

            # --- Optimisation loop ------------------------------------------------------------------
            # With every model parameter at ``requires_grad=False``, ``loss.backward()`` populates
            # gradient only on ``logits`` — no stale grads accumulate on encoder/heads.
            trajectory: list[torch.Tensor] = []
            for _ in range(steps):
                optimizer.zero_grad()
                w = _w_from_logits(logits)
                x = w @ kmd_kernel
                h_task = torch.tanh(self.encoder(x))
                preds, terms = _heads_forward(h_task)
                if diversity_scale < 1.0:
                    # The penalty strength is (1 − diversity_scale): user sees a [0, 1] knob
                    # where 1 means "no penalty / most diverse" and 0 means "max penalty / most
                    # peaky". The internal term is `(1 − diversity_scale) · H(w)` added to loss.
                    entropy = -(w * w.clamp(min=1e-12).log()).sum(dim=-1).mean()
                    terms.append((1.0 - diversity_scale) * entropy)
                loss = torch.stack(terms).mean()
                loss.backward()
                if step_scale is not None and logits.grad is not None:
                    # Soft per-element constraint: scale each element's logit gradient (0 = frozen).
                    logits.grad.mul_(step_scale)
                optimizer.step()
                trajectory.append(_stack([p.detach() for p in preds], logits.shape[0]))

            # --- Final state ------------------------------------------------------------------------
            with torch.no_grad():
                w_final = _w_from_logits(logits)
                x_final = w_final @ kmd_kernel
                h_final = torch.tanh(self.encoder(x_final))
                final_preds, _ = _heads_forward(h_final)
                final_target = _stack([p.detach() for p in final_preds], logits.shape[0])

            return CompositionOptimizationResult(
                optimized_weights=w_final.detach(),
                optimized_descriptor=x_final.detach(),
                optimized_target=final_target,
                initial_score=initial_score,
                # Preserve the (steps, B, T) shape contract even when steps == 0.
                trajectory=torch.stack(trajectory, dim=0)
                if trajectory
                else torch.empty((0, logits.shape[0], n_reg_tracked), device=device, dtype=dtype),
            )
        finally:
            if was_training:
                self.train()
            for p, prev in saved_req_grad:
                p.requires_grad_(prev)
