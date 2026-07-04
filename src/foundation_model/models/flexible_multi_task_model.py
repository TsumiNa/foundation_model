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
from dataclasses import dataclass, field
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

# Named tuple for optimization results. ``input_trajectory`` is None unless the caller passes
# ``record_input_trajectory=True`` to :meth:`optimize_latent` (gated because storing it costs
# O(B·R·steps·input_dim) memory and per-step latent-→-input decodes); when present it has shape
# ``(B, R, steps, input_dim)`` — used by the inverse-design trajectory animations to decode the
# per-step composition without rerunning the optimisation.
OptimizationResult = namedtuple(
    "OptimizationResult",
    ["optimized_input", "optimized_target", "initial_score", "trajectory", "input_trajectory"],
    defaults=[None],
)

# Composition-space optimization (gradient descent over element weights w ∈ simplex). The optimised
# w *is* the recipe (no AE-decode round-trip), so it is reported alongside the descriptor x = w @ K.
# ``weights_trajectory`` is None unless the caller passes ``record_weights_trajectory=True`` to
# :meth:`optimize_composition`; when present it has shape ``(steps, B, n_components)``.
CompositionOptimizationResult = namedtuple(
    "CompositionOptimizationResult",
    [
        "optimized_weights",
        "optimized_descriptor",
        "optimized_target",
        "initial_score",
        "trajectory",
        "weights_trajectory",
    ],
    defaults=[None],
)


@dataclass(frozen=True, kw_only=True)
class OptimizationTarget:
    """One user-specified inverse-design objective term.

    The target kind is derived from the task's head type (never guessed from the fields):

    - **regression** — exactly one of ``value`` (minimise ``(ŷ − value)²``) or ``direction``
      (``"high"`` maximises ``ŷ``, ``"low"`` minimises it; unbounded — there is no stationary
      point, so the achieved magnitude scales with ``steps × lr``).
    - **kernel_regression** — ``points`` = a target curve ``[[t, y], ...]``; minimises the MSE of
      the head evaluated at the given ``t`` values against the given ``y`` values.
    - **classification** — ``classes`` = label indices whose combined probability is pushed
      ``"high"`` (default) or ``"low"``. Must be a strict subset of the head's classes ("low" on
      the full set has an empty complement; "high" on the full set is a constant).

    ``weight`` scales this term relative to the others (all kinds; must be > 0).
    """

    task: str
    value: float | None = None
    direction: str | None = None  # "high" | "low"
    points: Sequence[Sequence[float]] | None = None  # [[t, y], ...]
    classes: Sequence[int] | None = None
    weight: float = 1.0


@dataclass(kw_only=True)
class _PreparedTarget:
    """Validated, tensor-ready form of one :class:`OptimizationTarget` (internal)."""

    task: str
    kind: str  # "value" | "direction" | "curve" | "class"
    weight: float
    value: torch.Tensor | None = None  # 0-d, value kind
    sign: float = 0.0  # direction kind: -1.0 high (maximize), +1.0 low
    t: torch.Tensor | None = None  # (K,), curve kind
    y: torch.Tensor | None = None  # (K,), curve kind
    classes: torch.Tensor | None = None  # (C_sel,) long, class kind
    complement: torch.Tensor | None = None  # (C_rest,) long, class kind ("low" objective)
    class_high: bool = True
    class_indices: list[int] = field(default_factory=list)


def _reduce_pred(pred: torch.Tensor) -> torch.Tensor:
    """Reduce a head prediction to one scalar per batch row: mean over all non-batch dims."""
    if pred.ndim == 1:
        return pred
    return pred.mean(dim=tuple(range(1, pred.ndim)))


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

    def _prepare_optimization_targets(
        self, targets: Sequence[OptimizationTarget], *, device: torch.device, dtype: torch.dtype
    ) -> list[_PreparedTarget]:
        """Validate a target list against the model's heads and build the per-term tensors.

        The target *kind* comes from the head type in ``task_configs_map`` — the caller never
        declares it. Raises ``ValueError`` on any field/kind mismatch so config errors surface
        before the optimisation loop starts.
        """
        if not targets:
            raise ValueError("targets must be a non-empty sequence of OptimizationTarget.")
        prepared: list[_PreparedTarget] = []
        seen: set[str] = set()
        for spec in targets:
            name = spec.task
            if name in seen:
                raise ValueError(f"Duplicate optimization target for task '{name}'.")
            seen.add(name)
            if name not in self.task_heads:
                raise ValueError(f"Task '{name}' not found in model. Available tasks: {list(self.task_heads.keys())}")
            if spec.weight is None or float(spec.weight) <= 0:
                raise ValueError(f"targets['{name}'].weight must be > 0, got {spec.weight}.")
            weight = float(spec.weight)
            cfg = self.task_configs_map[name]
            if cfg.type == TaskType.REGRESSION:
                if spec.points is not None or spec.classes is not None:
                    raise ValueError(f"Regression target '{name}' accepts value/direction, not points/classes.")
                if (spec.value is None) == (spec.direction is None):
                    raise ValueError(f"Regression target '{name}' needs exactly one of value or direction.")
                if spec.value is not None:
                    prepared.append(
                        _PreparedTarget(
                            task=name,
                            kind="value",
                            weight=weight,
                            value=torch.as_tensor(float(spec.value), device=device, dtype=dtype),
                        )
                    )
                else:
                    if spec.direction not in {"high", "low"}:
                        raise ValueError(
                            f"targets['{name}'].direction must be 'high' or 'low', got {spec.direction!r}."
                        )
                    prepared.append(
                        _PreparedTarget(
                            task=name,
                            kind="direction",
                            weight=weight,
                            sign=-1.0 if spec.direction == "high" else 1.0,
                        )
                    )
            elif cfg.type == TaskType.KERNEL_REGRESSION:
                if spec.value is not None or spec.direction is not None or spec.classes is not None:
                    raise ValueError(f"Kernel-regression target '{name}' accepts points only.")
                if not spec.points:
                    raise ValueError(
                        f"Kernel-regression target '{name}' needs a non-empty points list of [t, y] pairs."
                    )
                pairs = []
                for p in spec.points:
                    pair = list(p)
                    if len(pair) != 2:
                        raise ValueError(f"targets['{name}'].points entries must be [t, y] pairs, got {p!r}.")
                    pairs.append((float(pair[0]), float(pair[1])))
                prepared.append(
                    _PreparedTarget(
                        task=name,
                        kind="curve",
                        weight=weight,
                        t=torch.as_tensor([p[0] for p in pairs], device=device, dtype=dtype),
                        y=torch.as_tensor([p[1] for p in pairs], device=device, dtype=dtype),
                    )
                )
            elif cfg.type == TaskType.CLASSIFICATION:
                if spec.value is not None or spec.points is not None:
                    raise ValueError(f"Classification target '{name}' accepts classes (+ direction) only.")
                if not spec.classes:
                    raise ValueError(f"Classification target '{name}' needs a non-empty classes list.")
                direction = spec.direction if spec.direction is not None else "high"
                if direction not in {"high", "low"}:
                    raise ValueError(f"targets['{name}'].direction must be 'high' or 'low', got {spec.direction!r}.")
                idxs = sorted({int(c) for c in spec.classes})
                num_classes = getattr(cfg, "num_classes", None)
                if num_classes is None:
                    raise ValueError(f"Classification task '{name}' has no num_classes; cannot build a class target.")
                if any(not 0 <= i < num_classes for i in idxs):
                    raise ValueError(
                        f"targets['{name}'].classes {idxs} out of range for a {num_classes}-class head; "
                        f"valid indices are [0, {num_classes})."
                    )
                if len(idxs) >= num_classes:
                    raise ValueError(
                        f"targets['{name}'].classes {idxs} covers every class of a {num_classes}-class head; "
                        "the objective would be constant ('high') or undefined ('low'). Use a strict subset."
                    )
                complement = [i for i in range(num_classes) if i not in set(idxs)]
                prepared.append(
                    _PreparedTarget(
                        task=name,
                        kind="class",
                        weight=weight,
                        classes=torch.as_tensor(idxs, device=device, dtype=torch.long),
                        complement=torch.as_tensor(complement, device=device, dtype=torch.long),
                        class_high=direction == "high",
                        class_indices=idxs,
                    )
                )
            else:
                raise ValueError(f"Task '{name}' has unsupported head type {cfg.type} for optimization targets.")
        return prepared

    def _optimization_objective(
        self, h_task: torch.Tensor, prepared: Sequence[_PreparedTarget]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate every target term at ``h_task``.

        Returns ``(channels, losses)``, both ``(B, T)`` with one column per target in declaration
        order. ``channels`` is the human-readable per-target scalar (regression: ŷ; curve:
        RMSE-to-curve; class: P(classes)); ``losses`` is the weighted per-sample loss whose sum is
        the optimisation objective (lower = better; direction terms make it sign-indefinite).
        """
        channel_cols: list[torch.Tensor] = []
        loss_cols: list[torch.Tensor] = []
        for tgt in prepared:
            head = self.task_heads[tgt.task]
            if tgt.kind == "curve":
                assert tgt.t is not None and tgt.y is not None
                k = tgt.t.shape[0]
                t_batch = tgt.t.unsqueeze(0).expand(h_task.shape[0], k)
                h_rep, t_rep = self._expand_for_kernel_regression(h_task, t_batch)
                pred = head(h_rep, t=t_rep).view(h_task.shape[0], k)
                sq = (pred - tgt.y.unsqueeze(0)) ** 2  # (B, K)
                per_sample = sq.mean(dim=1)
                channel_cols.append(per_sample.sqrt())  # RMSE-to-curve; 0 = perfect fit
                loss_cols.append(tgt.weight * per_sample)
            elif tgt.kind == "class":
                assert tgt.classes is not None and tgt.complement is not None
                log_probs = F.log_softmax(head(h_task), dim=-1)
                lp_sel = torch.logsumexp(log_probs.index_select(-1, tgt.classes), dim=-1)  # (B,)
                channel_cols.append(lp_sel.exp())  # P(classes) regardless of direction
                if tgt.class_high:
                    loss_cols.append(tgt.weight * (-lp_sel))
                else:
                    # "low" = maximize the complement's probability: numerically clean near
                    # P(classes) → 1 (a direct -log(1 - P) would blow up) and reuses the same
                    # logsumexp machinery.
                    lp_rest = torch.logsumexp(log_probs.index_select(-1, tgt.complement), dim=-1)
                    loss_cols.append(tgt.weight * (-lp_rest))
            else:
                pred = head(h_task)
                reduced = _reduce_pred(pred)
                channel_cols.append(reduced)
                if tgt.kind == "value":
                    assert tgt.value is not None
                    expanded = tgt.value.reshape([1] * pred.ndim).expand(pred.shape)
                    per_sample = (pred - expanded) ** 2
                    per_sample = _reduce_pred(per_sample)
                    loss_cols.append(tgt.weight * per_sample)
                else:  # direction
                    loss_cols.append(tgt.weight * tgt.sign * reduced)
        return torch.stack(channel_cols, dim=-1), torch.stack(loss_cols, dim=-1)

    @torch.no_grad()
    def evaluate_targets(
        self, x: torch.Tensor, targets: Sequence[OptimizationTarget]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score descriptors against a target list without optimising.

        Parameters
        ----------
        x : torch.Tensor
            Descriptors, shape ``(B, input_dim)`` (a 1-d tensor is treated as a single row).
        targets : Sequence[OptimizationTarget]
            The objective terms (same specs :meth:`optimize_latent` / :meth:`optimize_composition`
            accept via ``targets=``).

        Returns
        -------
        (channels, objective)
            ``channels`` — ``(B, T)`` per-target scalars (see :meth:`_optimization_objective`);
            ``objective`` — ``(B,)`` summed weighted loss, lower = better (sign-indefinite when
            direction targets are present). This is the exact quantity the optimisers minimise
            (minus the space-specific extras), so seed ranking and optimisation cannot drift.
        """
        ref = next(self.parameters())
        device, dtype = ref.device, ref.dtype
        prepared = self._prepare_optimization_targets(targets, device=device, dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(device=device, dtype=dtype)
        was_training = self.training
        self.eval()
        try:
            h_task = torch.tanh(self.encoder(x))
            channels, losses = self._optimization_objective(h_task, prepared)
        finally:
            self.train(was_training)
        return channels, losses.sum(dim=1)

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
        targets: Sequence[OptimizationTarget] | None = None,
        task_targets: Mapping[str, torch.Tensor | float] | None = None,
        class_targets: Mapping[str, int | Sequence[int]] | None = None,
        ae_align_scale: float = 0.5,
        optimize_space: str = "input",
        record_input_trajectory: bool = False,
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
        targets : Sequence[OptimizationTarget] | None, optional
            The primary multi-objective interface: one :class:`OptimizationTarget` per term —
            regression value/direction, kernel-regression target curves, classification
            probability high/low, each with its own ``weight``. Mutually exclusive with
            ``task_targets`` / ``class_targets`` / ``target_value``.
        task_targets : Mapping[str, float | Tensor] | None, optional
            Sugar for value-mode regression targets (each entry becomes an
            ``OptimizationTarget(task=..., value=..., weight=1.0)``). When provided, ``mode`` and
            ``target_value`` are ignored.
        class_targets : Mapping[str, int | Sequence[int]] | None, optional
            Sugar for "high"-direction classification objectives with ``weight=1.0`` (use
            ``targets=`` for "low" direction or a non-default weight). May be combined with
            ``task_targets``.
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
            namedtuple with fields (``T`` = number of targets, one channel per target in
            declaration order — regression: ŷ, curve: RMSE-to-curve, classification: P(classes)):
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

        # Resolve every input style into one list of OptimizationTargets.
        resolved_targets: list[OptimizationTarget]
        if targets is not None:
            if task_targets is not None or class_targets is not None or target_value is not None:
                raise ValueError("targets is mutually exclusive with task_targets/class_targets/target_value.")
            resolved_targets = list(targets)
        elif task_targets is not None or class_targets is not None:
            if target_value is not None:
                raise ValueError("Use either task_targets (multi-task) or target_value (single task), not both.")
            if task_targets is not None and (not isinstance(task_targets, Mapping) or len(task_targets) == 0):
                raise ValueError("task_targets must be a non-empty mapping of task_name -> target_value")
            if class_targets is not None and (not isinstance(class_targets, Mapping) or len(class_targets) == 0):
                raise ValueError("class_targets must be a non-empty mapping of task_name -> class index/indices")
            resolved_targets = [
                OptimizationTarget(task=name, value=float(torch.as_tensor(val).reshape(-1)[0]))
                for name, val in (task_targets or {}).items()
            ]
            resolved_targets += [
                OptimizationTarget(task=name, classes=[int(cls)] if isinstance(cls, int) else [int(c) for c in cls])
                for name, cls in (class_targets or {}).items()
            ]
        else:
            # Legacy single-task path (task_name + mode / target_value).
            if task_name is None or task_name not in self.task_heads:
                raise ValueError(
                    f"Task '{task_name}' not found in model. Available tasks: {list(self.task_heads.keys())}"
                )
            if target_value is None and mode not in {"max", "min"}:
                raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")
            if target_value is not None:
                resolved_targets = [
                    OptimizationTarget(task=task_name, value=float(torch.as_tensor(target_value).reshape(-1)[0]))
                ]
            else:
                resolved_targets = [OptimizationTarget(task=task_name, direction="high" if mode == "max" else "low")]

        if not 0.0 <= ae_align_scale <= 1.0:
            raise ValueError(f"ae_align_scale must be in [0, 1], got {ae_align_scale}.")

        # Validate autoencoder availability for latent-space mode
        if optimize_space == "latent":
            if _AE_TASK not in self.task_heads:
                raise ValueError("optimize_space='latent' requires the model to be built with enable_autoencoder=True.")
            if not isinstance(self.task_heads[_AE_TASK], AutoEncoderHead):
                raise ValueError(
                    f"Task '{_AE_TASK}' exists but is not an AutoEncoderHead; "
                    "latent-space optimization requires the built-in reconstruction head."
                )

        if num_restarts < 1:
            raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")

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

        # Validate the targets against the heads and build the per-term tensors once — BEFORE the
        # requires_grad freeze below, so a validation error cannot leave the model frozen.
        prepared = self._prepare_optimization_targets(resolved_targets, device=device, dtype=input_tensor.dtype)

        # Store original training state. We also snapshot every parameter's ``requires_grad``
        # because the optimisation only differentiates through ``optim_input`` / ``optim_latent``
        # — leaving ``requires_grad=True`` on the model parameters would let ``loss.backward()``
        # populate stale ``.grad`` tensors on the encoder / heads. Mirrors the same pattern used
        # by :meth:`optimize_composition` so a later ``model.fit(...)`` works as expected.
        was_training = self.training
        saved_req_grad: list[tuple[torch.nn.Parameter, bool]] = [(p, p.requires_grad) for p in self.parameters()]
        self.eval()
        for p, _ in saved_req_grad:
            p.requires_grad_(False)

        optimized_inputs: list[torch.Tensor] = []
        optimized_targets: list[torch.Tensor] = []
        trajectories: list[torch.Tensor] = []
        # When ``record_input_trajectory=True`` we snapshot the per-step input every iteration
        # (input-space: ``optim_input`` directly; latent-space: ``AE.decode(tanh(h))``). Stored on
        # CPU to keep GPU memory flat on long trajectories. One per restart, stacked at the end.
        input_trajectories: list[torch.Tensor] = []
        initial_scores_list: list[torch.Tensor] = []

        for restart_idx in range(num_restarts):
            if optimize_space == "input":
                # Input space optimization: optimize X directly
                start_input = input_tensor.clone()
                if perturbation_std > 0:
                    start_input = start_input + torch.randn_like(start_input) * perturbation_std

                # Record initial score(s)
                with torch.no_grad():
                    h_task = torch.tanh(self.encoder(start_input))
                    channels, _ = self._optimization_objective(h_task, prepared)
                    initial_scores_list.append(channels.detach())  # (B, T)

                # Create optimizable input
                optim_input = start_input.detach().clone().requires_grad_(True)

                # Setup optimizer
                optimizer = optim.Adam([optim_input], lr=lr)

                # Optimization loop
                step_traj: list[torch.Tensor] = []
                step_input_traj: list[torch.Tensor] = []

                for step in range(steps):
                    optimizer.zero_grad()

                    # Forward through encoder and apply Tanh
                    h_task = torch.tanh(self.encoder(optim_input))
                    channels, per_sample_losses = self._optimization_objective(h_task, prepared)
                    loss = per_sample_losses.mean(dim=0).mean()

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    # Record history
                    step_traj.append(channels.detach())
                    if record_input_trajectory:
                        # Input-space optim variable IS the input — just snapshot it.
                        step_input_traj.append(optim_input.detach().cpu())

                # Get final optimized values
                with torch.no_grad():
                    h_task = torch.tanh(self.encoder(optim_input))
                    per_task_final_tensor, _ = self._optimization_objective(h_task, prepared)
                    per_task_final_tensor = per_task_final_tensor.detach()  # (B, T)
                    optimized_input = optim_input.detach()

                optimized_inputs.append(optimized_input.detach())  # (B, D)
                optimized_targets.append(per_task_final_tensor)  # (B, T)
                traj_tensor = torch.stack(step_traj, dim=0)  # (steps, B, T)
                trajectories.append(traj_tensor)
                if record_input_trajectory:
                    input_trajectories.append(torch.stack(step_input_traj, dim=0))  # (steps, B, D)

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
                    channels, _ = self._optimization_objective(h_task, prepared)
                    initial_scores_list.append(channels.detach())  # (B, T)

                # Create optimizable latent
                optim_latent = start_latent.detach().clone().requires_grad_(True)

                # Setup optimizer
                optimizer = optim.Adam([optim_latent], lr=lr)

                # Optimization loop
                step_traj: list[torch.Tensor] = []
                step_input_traj: list[torch.Tensor] = []

                for step in range(steps):
                    optimizer.zero_grad()

                    # Apply Tanh to get task representation (consistent with forward())
                    # This ensures architectural consistency on every optimization step
                    h_task = torch.tanh(optim_latent)
                    channels, per_sample_losses = self._optimization_objective(h_task, prepared)
                    loss_terms = list(per_sample_losses.mean(dim=0))
                    if ae_align_scale > 0:
                        # Pull the optimised latent toward what the AE faithfully reconstructs:
                        # decode it to a descriptor, re-encode, and penalise the drift in h_task.
                        # The user-facing knob is [0, 1] with 0 = no penalty / 1 = strong penalty.
                        re_h_task = torch.tanh(self.encoder(self.task_heads[_AE_TASK](h_task)))
                        loss_terms.append(ae_align_scale * F.mse_loss(re_h_task, h_task))
                    loss = torch.stack(loss_terms).mean()

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    # Record history
                    step_traj.append(channels.detach())
                    if record_input_trajectory:
                        # Latent-space optim: decode the current h via the AE head to recover the
                        # per-step input. ``no_grad`` keeps this from polluting the optim graph.
                        with torch.no_grad():
                            step_input = self.task_heads[_AE_TASK](torch.tanh(optim_latent))
                        step_input_traj.append(step_input.detach().cpu())

                # Get final optimized values and reconstruct via AE
                with torch.no_grad():
                    # Apply Tanh to get final task representation (consistent with forward())
                    final_h_task = torch.tanh(optim_latent)
                    per_task_final_tensor, _ = self._optimization_objective(final_h_task, prepared)
                    per_task_final_tensor = per_task_final_tensor.detach()  # (B, T)

                    # Reconstruct input via the built-in reconstruction head
                    reconstructed_input = self.task_heads[_AE_TASK](final_h_task)

                optimized_inputs.append(reconstructed_input.detach())  # (B, D)
                optimized_targets.append(per_task_final_tensor)  # (B, T)
                traj_tensor = torch.stack(step_traj, dim=0)  # (steps, B, T)
                trajectories.append(traj_tensor)
                if record_input_trajectory:
                    input_trajectories.append(torch.stack(step_input_traj, dim=0))  # (steps, B, D)

        # Restore training state + per-parameter ``requires_grad``. Without the latter, every
        # encoder / head parameter would be left frozen for any later ``.fit()`` in the same
        # Python session — the symptom is "training silently stops moving the encoder" which
        # is annoying to bisect.
        self.train(was_training)
        for p, prev in saved_req_grad:
            p.requires_grad_(prev)

        # Stack outputs
        opt_input_tensor = torch.stack(optimized_inputs, dim=1)  # (B, R, D)
        opt_target_tensor = torch.stack(optimized_targets, dim=1)  # (B, R, T)
        traj_tensor = torch.stack(trajectories, dim=0)  # (R, steps, B, T)
        traj_tensor = traj_tensor.permute(2, 0, 1, 3)  # (B, R, steps, T)
        initial_score_tensor = torch.stack(initial_scores_list, dim=0)  # (R, B, T)
        initial_score_tensor = initial_score_tensor.permute(1, 0, 2)  # (B, R, T)

        input_traj_tensor: torch.Tensor | None = None
        if record_input_trajectory and input_trajectories:
            input_traj_tensor = torch.stack(input_trajectories, dim=0)  # (R, steps, B, D)
            input_traj_tensor = input_traj_tensor.permute(2, 0, 1, 3)  # (B, R, steps, D)

        return OptimizationResult(
            optimized_input=opt_input_tensor,
            optimized_target=opt_target_tensor,
            initial_score=initial_score_tensor,
            trajectory=traj_tensor,
            input_trajectory=input_traj_tensor,
        )

    def optimize_composition(
        self,
        kmd_kernel: torch.Tensor,
        *,
        initial_weights: torch.Tensor | None = None,
        n_starts: int = 16,
        targets: Sequence[OptimizationTarget] | None = None,
        task_targets: Mapping[str, torch.Tensor | float] | None = None,
        class_targets: Mapping[str, int | Sequence[int]] | None = None,
        diversity_scale: float = 1.0,
        allowed_elements: str | list[str] = "all",
        element_step_scale: float | Mapping[str, float] = 1.0,
        fixed_amounts: Mapping[str, float] | None = None,
        min_nonzero_weight: float = 0.0,
        seed_blend: float = 0.95,
        max_elements: int | None = None,
        annealing_scale: float = 0.5,
        annealing_schedule: Mapping[str, Any] | None = None,
        steps: int = 300,
        lr: float = 0.05,
        record_weights_trajectory: bool = False,
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
        targets, task_targets, class_targets :
            Same semantics as :meth:`optimize_latent`: ``targets`` is the primary
            :class:`OptimizationTarget` interface (regression value/direction, kernel-regression
            curves, classification high/low, per-target ``weight``); ``task_targets`` /
            ``class_targets`` are sugar for value-mode regression / "high" classification terms.
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
        fixed_amounts : Mapping[str, float] | None, optional
            Pin specific elements at user-specified weights for the entire optimisation; the
            optimiser distributes the remaining mass ``1 − Σ fixed_amounts.values()`` across
            the unfixed elements freely.

            Example: ``{"Au": 0.65, "Ga": 0.20}`` produces recipes with Au exactly 65 % and
            Ga exactly 20 %; the remaining 15 % is split among other allowed elements as the
            objective prefers.

            Implementation reuses the same lock-paste machinery as ``element_step_scale = 0``:
            a per-row tensor ``locked_w0`` is built with the user's amounts at the named
            positions; ``_w_from_logits`` overwrites those positions every step and
            renormalises the unlocked positions over ``1 − Σ locked``.

            Constraints:
              * Each symbol must be in :data:`DEFAULT_ELEMENTS` (kernel alignment required).
              * Each amount must be in ``(0, 1)``; ``Σ values < 1.0`` (need free mass).
              * If ``allowed_elements`` is set, every fixed element must also be in the
                whitelist (locking outside the whitelist is contradictory).
              * If ``element_step_scale = 0`` is also used, the two sets of locked symbols
                **must not overlap** — use one mechanism per element.
              * If ``max_elements`` is also set, fixed elements count toward K (they're
                always in the selection); strict inequality ``max_elements > n_locked_total``
                is enforced.

            Unlike ``element_step_scale = 0``'s hard lock, ``fixed_amounts`` does **not**
            require ``initial_weights`` — the lock values come straight from this kwarg.
        min_nonzero_weight : float, optional
            Lower bound on every unlocked element's final weight: positions with
            ``0 < w < min_nonzero_weight`` are zeroed out and their mass is redistributed across
            the remaining unlocked positions. Default ``0.0`` (no floor).

            Use case: avoid trace-amount appearances (e.g. ``Pt = 0.5%``) that are not
            synthesisable — "if you use it, use ≥ 10%".

            Implementation: applied as the *last* step in ``_w_from_logits`` (after soft top-K
            and lock-paste) and again after the final hard top-K projection. Locked elements
            (from ``element_step_scale = 0`` or ``fixed_amounts``) are **not** subject to the
            floor — their values are set explicitly by the user.

            Constraints:
              * ``0 ≤ min_nonzero_weight ≤ 1``.
              * If ``max_elements`` is set: ``min_nonzero_weight ≤ 1 / max_elements`` (otherwise
                ``K`` elements each ≥ floor can't sum to ≤ 1).
              * If ``fixed_amounts`` is set: every fixed value must be ≥ floor (else
                contradiction).
              * If ``element_step_scale = 0`` locks with ``initial_weights`` are present: every
                locked seed value must be ≥ floor (checked at runtime once the seed is
                normalised).

            Edge case: if dropping every below-floor position would leave a row with zero
            unlocked mass (no element survives), the floor is skipped *for that row only* —
            preserving the simplex (rows always sum to 1). When this happens, the row will
            contain unlocked positions below ``min_nonzero_weight``; if you see this in
            practice your floor is too aggressive for the model's preferred subset.

            Practical note: when ``max_elements`` is not set, no upper bound on the floor is
            enforced beyond ``floor ≤ 1``. A very large floor (e.g. 0.5 with 94 components) will
            silently trigger the per-row fallback on almost every row — the result is a valid
            simplex but the floor is effectively ignored. Pair the floor with ``max_elements``
            (which enforces ``floor ≤ 1 / max_elements``) when you want a hard guarantee.

            "At most K" implication: when combined with ``max_elements``, the floor can drop
            below-floor positions in the K-subset, so the final non-zero count can be **less
            than K** (still ≤ K — the user-facing promise is unchanged).
        seed_blend : float, optional
            How much of the (per-row) seed prior to keep when ``initial_weights`` is given;
            ``w0 ← seed_blend · seed + (1 − seed_blend) · uniform_over_allowed``. Default ``0.95``
            (5 % uniform mass spread over the allowed elements). The blend lifts non-seed-element
            logits from ``log(1e-12) ≈ −27.6`` (effectively unreachable by Adam in a few hundred
            steps) to ``log(0.05 / |allowed|) ≈ −7.6``, so the optimiser can introduce new elements
            when they help the objective. Set to ``1.0`` to reproduce the strict seed-only behaviour
            (no new elements can enter the support set); ``0.0`` makes the seed irrelevant and
            starts from uniform. Ignored when ``initial_weights is None``.
        max_elements : int | None, optional
            If set, restricts the final composition to at most this many non-zero elements.
            Unlike a naive post-hoc top-K projection, the constraint **participates in
            optimisation throughout** via a differentiable iterative-softmax K-hot mask
            (Plötz–Roth, NeurIPS 2018) coupled with a temperature-annealing schedule.

            How it works in one paragraph: at each step we compute a soft K-hot mask
            ``m ∈ [0,1]^n`` with ``Σm = K`` from the same logits the softmax uses, then form
            ``w = (softmax(lg) · m) / Σ(softmax(lg) · m)``. Temperature ``τ`` controls how
            "K-hot" ``m`` is: large τ → uniform-ish (the constraint is soft, gradient can flow
            between candidate subsets), small τ → near one-hot per iteration (constraint is hard).
            τ is driven by the ``annealing_scale`` / ``annealing_schedule`` kwargs below — by
            default a geometric schedule from ``25**annealing_scale`` down to a fixed
            ``τ_end = 0.01``. The annealing doubles as a continuation method that helps escape
            local optima.

            After the loop, a final hard top-K projection is applied so the returned
            ``optimized_weights`` has **at most** ``max_elements`` non-zero positions (subject
            to any locked elements, which are always counted toward K — see below). The
            count saturates at K when the optimiser left at least K positions with positive
            ``w_soft`` mass; if it drove some logits all the way to zero, the row can land
            below K — this is by design, not a bug ("at most K" is the user-facing promise).

            Constraints:
              * ``1 ≤ max_elements ≤ n_components``.
              * If any element is hard-locked via ``element_step_scale=0``, the lock counts
                toward K; require ``max_elements ≥ n_locked``.
              * If ``allowed_elements`` restricts the support, require ``max_elements ≤ |allowed|``.

            ``None`` (default) or ``max_elements == n_components`` disables the constraint.
        annealing_scale : float, optional
            Single-knob "softness" of the annealing schedule, normalised to ``[0, 1]``.
            Default ``0.5``. Maps internally to raw temperature via ``τ_start = 25**scale``:

              * ``0.0`` → ``τ_start = 1.0``    (no exploration; constraint hard from the start)
              * ``0.5`` → ``τ_start = 5.0``    (default; safe choice — QC stable, decent targets)
              * ``1.0`` → ``τ_start = 25.0``   (max exploration; longer soft phase)

            The full schedule is geometric from ``τ_start(scale)`` down to ``τ_end = 0.01``.
            Ignored when ``max_elements`` is None.

            **Calibration**: the 0.5 default was picked from a sweep on the inverse-design
            fine-tuned model (300 steps, K∈{3, 5}; see ``logs/sweep_tau_schedule.png``). Across
            the 3 paper scenarios it keeps QC within ±0.02 of the unconstrained baseline while
            hitting K=3/5 cardinality. For aggressive target chasing, raise toward 0.8-1.0
            (and consider an advanced schedule with ``annealing_func="linear"`` to hold the
            soft phase longer). For QC priority, leave at 0.5.
        annealing_schedule : dict | None, optional
            Advanced piecewise schedule. **Overrides the front of the simple schedule.**
            When supplied, this dict takes precedence over ``annealing_scale``'s implicit
            schedule for the steps it covers. The format is three parallel lists of length N:

            .. code-block:: python

                {
                    "step":           [0.2, 0.5, 1.0],         # fractional step boundaries (0,1]
                    "scale":          [0.8, 0.5, 0.5],         # normalised scale [0,1] at each boundary
                    "annealing_func": ["geometric", "geometric", "geometric"],   # interpolation in each segment
                }

            **Reading the dict**: the schedule starts at step=0 from the value given by
            ``annealing_scale``. Segment ``i`` covers ``(step[i-1], step[i]]`` (with
            ``step[-1] := 0``); within that segment, the normalised scale interpolates from the
            previous segment's endpoint (or ``annealing_scale`` for segment 0) to ``scale[i]``
            using ``annealing_func[i]``. The interpolated scale is then mapped to raw τ via the
            same ``25**scale`` formula used by ``annealing_scale``.

            **If ``step[-1] < 1.0``**, the remaining ``(step[-1], 1.0]`` portion continues with
            a default geometric tail: from the raw τ value at ``step[-1]`` (i.e.
            ``25**scale[-1]``) down to ``τ_end = 0.01``. This guarantees the schedule always
            reaches the hard end inside the loop (the final hard-projection cleans up K-hot
            either way).

            **Allowed annealing_func values**: ``"geometric"``, ``"linear"``, ``"cosine"``,
            ``"constant"``. ``"constant"`` holds the segment's starting value (``scale[i]`` is
            ignored — useful for warm-up phases).
        steps : int
            Adam optimisation steps. Default 300.
        lr : float
            Adam learning rate over the logits. Default 0.05.

        Returns
        -------
        CompositionOptimizationResult
            with fields (``T`` = number of targets, one channel per target in declaration order —
            regression: ŷ, curve: RMSE-to-curve, classification: P(classes)):
            - ``optimized_weights``    : (B, n_components), each row a simplex point — the recipe.
            - ``optimized_descriptor`` : (B, x_dim), equals ``optimized_weights @ kmd_kernel``.
            - ``optimized_target``     : (B, T), final per-target channel values.
            - ``initial_score``        : (B, T), same shape, evaluated at step 0.
            - ``trajectory``           : (steps, B, T), per-target channels across optimisation.
        """
        # --- Validate the kernel ----------------------------------------------------------------
        if not isinstance(kmd_kernel, torch.Tensor) or kmd_kernel.ndim != 2:
            raise ValueError("kmd_kernel must be a 2D torch.Tensor of shape (n_components, x_dim).")
        n_components, x_dim = kmd_kernel.shape
        expected_dim = getattr(self.encoder, "input_dim", None)
        if expected_dim is not None and x_dim != expected_dim:
            raise ValueError(f"kmd_kernel.shape[1]={x_dim} does not match encoder.input_dim={expected_dim}.")

        # --- Resolve the objective into one list of OptimizationTargets (mirrors optimize_latent)
        resolved_targets: list[OptimizationTarget]
        if targets is not None:
            if task_targets is not None or class_targets is not None:
                raise ValueError("targets is mutually exclusive with task_targets/class_targets.")
            resolved_targets = list(targets)
        else:
            if task_targets is not None and (not isinstance(task_targets, Mapping) or len(task_targets) == 0):
                raise ValueError("task_targets must be a non-empty mapping of task_name -> target_value")
            if class_targets is not None and (not isinstance(class_targets, Mapping) or len(class_targets) == 0):
                raise ValueError("class_targets must be a non-empty mapping of task_name -> class index/indices")
            resolved_targets = [
                OptimizationTarget(task=name, value=float(torch.as_tensor(val).reshape(-1)[0]))
                for name, val in (task_targets or {}).items()
            ]
            resolved_targets += [
                OptimizationTarget(task=name, classes=[int(cls)] if isinstance(cls, int) else [int(c) for c in cls])
                for name, cls in (class_targets or {}).items()
            ]
            if not resolved_targets:
                raise ValueError("Provide at least one of targets / task_targets / class_targets.")
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

        # --- Validate fixed_amounts (per-element explicit pinning) -------------------------------
        # Build the (n_components,) tensors lazily: ``fixed_w0_vec`` (per-element pinned value,
        # zero elsewhere) and ``fixed_mask_vec`` (bool: True at pinned positions). The actual
        # batch-shaped ``locked_w0`` is materialised later (alongside step_scale=0 locks) once we
        # know the batch size.
        fixed_w0_vec: torch.Tensor | None = None
        fixed_mask_vec: torch.Tensor | None = None
        if fixed_amounts is not None:
            if not isinstance(fixed_amounts, Mapping):
                raise TypeError(
                    f"fixed_amounts must be a mapping of element_symbol → float or None; "
                    f"got {type(fixed_amounts).__name__}."
                )
            if len(fixed_amounts) == 0:
                raise ValueError("fixed_amounts must be non-empty when provided.")
            sym_to_idx = {s: i for i, s in enumerate(DEFAULT_ELEMENTS)}
            bad_syms = [s for s in fixed_amounts if s not in sym_to_idx]
            if bad_syms:
                raise ValueError(f"Unknown element symbol(s) in fixed_amounts: {bad_syms}.")
            if n_components != len(DEFAULT_ELEMENTS):
                raise ValueError(
                    f"fixed_amounts requires the kernel to align with DEFAULT_ELEMENTS "
                    f"(n_components={n_components}, expected {len(DEFAULT_ELEMENTS)})."
                )
            for sym, amt in fixed_amounts.items():
                if not 0.0 < float(amt) < 1.0:
                    raise ValueError(f"fixed_amounts['{sym}']={amt} must be strictly between 0 and 1.")
            total = float(sum(fixed_amounts.values()))
            if total >= 1.0:
                raise ValueError(
                    f"sum(fixed_amounts.values())={total:.4f} must be strictly less than 1.0 "
                    "(the optimiser needs unfixed mass to allocate)."
                )
            # Allowed-list compatibility — pinning outside the whitelist is contradictory.
            if elem_mask_arg is not None:
                bad_against_allowed = [s for s in fixed_amounts if not elem_mask_arg[sym_to_idx[s]]]
                if bad_against_allowed:
                    raise ValueError(
                        f"fixed_amounts symbols {bad_against_allowed} are not in allowed_elements — "
                        "pinning a disallowed element is contradictory."
                    )
            # Mutual exclusion with element_step_scale = 0 (the other hard-lock path).
            if step_scale_arg is not None:
                overlap = [s for s in fixed_amounts if float(step_scale_arg[sym_to_idx[s]]) == 0.0]
                if overlap:
                    raise ValueError(
                        f"Symbols {overlap} appear in both element_step_scale=0 and "
                        "fixed_amounts. Use one mechanism per element."
                    )
            fixed_w0_vec = torch.zeros(n_components)
            fixed_mask_vec = torch.zeros(n_components, dtype=torch.bool)
            for sym, amt in fixed_amounts.items():
                idx = sym_to_idx[sym]
                fixed_w0_vec[idx] = float(amt)
                fixed_mask_vec[idx] = True

        # --- Validate min_nonzero_weight (per-element floor) -------------------------------------
        if not 0.0 <= min_nonzero_weight <= 1.0:
            raise ValueError(f"min_nonzero_weight must be in [0, 1]; got {min_nonzero_weight}.")
        if min_nonzero_weight > 0.0:
            # If max_elements is set, the floor must be feasible: K elements ≥ floor summing to 1
            # implies K * floor ≤ 1.
            if max_elements is not None and min_nonzero_weight > 1.0 / max_elements:
                raise ValueError(
                    f"min_nonzero_weight={min_nonzero_weight} exceeds 1 / max_elements="
                    f"{1.0 / max_elements:.4f}. With at most {max_elements} non-zero positions, "
                    "no row can have every weight ≥ floor and still sum to 1."
                )
            # Fixed amounts must themselves be ≥ the floor (else contradiction).
            if fixed_amounts is not None:
                bad = sorted((s, v) for s, v in fixed_amounts.items() if float(v) < min_nonzero_weight)
                if bad:
                    raise ValueError(
                        f"fixed_amounts entries {bad} are below min_nonzero_weight="
                        f"{min_nonzero_weight}. The floor cannot override an explicit pin."
                    )

        # --- Validate cardinality constraint (max_elements + annealing knobs) -----------------------
        if max_elements is not None:
            if not isinstance(max_elements, int) or isinstance(max_elements, bool):
                raise TypeError(f"max_elements must be an int or None; got {type(max_elements).__name__}.")
            if not 1 <= max_elements <= n_components:
                raise ValueError(f"max_elements must be in [1, n_components={n_components}]; got {max_elements}.")
            if elem_mask_arg is not None:
                n_allowed = int(elem_mask_arg.sum().item())
                if max_elements > n_allowed:
                    raise ValueError(
                        f"max_elements={max_elements} exceeds the number of allowed elements "
                        f"({n_allowed}). Widen ``allowed_elements`` or lower ``max_elements``."
                    )
            # Lock-vs-K check: locked positions (element_step_scale=0 ∪ fixed_amounts) all count
            # toward K. We require *strict* ``max_elements > n_locked`` for both lock paths:
            # equality leaves the lock-paste with no unlocked slot to absorb the leftover mass
            # (1 − Σ locked) and produces rows that sum to < 1 — silently breaking the simplex.
            # For ``fixed_amounts`` this is definite (``Σ < 1`` enforced at kwarg time); for
            # ``element_step_scale=0`` the seed values *could* sum to exactly 1, but K-constrained
            # all-locked recipes have no degrees of freedom anyway, so rejecting equality is
            # both safe and clearer.
            n_locked_pre = 0
            if step_scale_arg is not None:
                n_locked_pre += int((step_scale_arg == 0).sum().item())
            if fixed_mask_vec is not None:
                n_locked_pre += int(fixed_mask_vec.sum().item())
            if n_locked_pre >= max_elements:
                raise ValueError(
                    f"max_elements={max_elements} must be > total locked elements ({n_locked_pre}, "
                    "counting element_step_scale=0 ∪ fixed_amounts) — the lock-paste needs at "
                    "least one unlocked slot to absorb the leftover mass (1 − Σ locked); equality "
                    "would silently produce row sums < 1. Raise max_elements or unlock some."
                )
            if not 0.0 <= annealing_scale <= 1.0:
                raise ValueError(f"annealing_scale must be in [0, 1]; got {annealing_scale}.")
            if annealing_schedule is not None:
                if not isinstance(annealing_schedule, Mapping):
                    raise TypeError(f"annealing_schedule must be a mapping; got {type(annealing_schedule).__name__}.")
                missing = {"step", "scale", "annealing_func"} - set(annealing_schedule)
                if missing:
                    raise ValueError(
                        f"annealing_schedule missing required keys {sorted(missing)}. "
                        "Required: step, scale, annealing_func — all parallel lists."
                    )
                sched_steps = list(annealing_schedule["step"])
                sched_scales = list(annealing_schedule["scale"])
                sched_funcs = list(annealing_schedule["annealing_func"])
                if not (len(sched_steps) == len(sched_scales) == len(sched_funcs)):
                    raise ValueError(
                        f"annealing_schedule lists must be the same length; got "
                        f"step={len(sched_steps)}, scale={len(sched_scales)}, "
                        f"annealing_func={len(sched_funcs)}."
                    )
                if len(sched_steps) == 0:
                    raise ValueError("annealing_schedule lists must be non-empty.")
                prev_s = 0.0
                for s in sched_steps:
                    if not 0.0 < float(s) <= 1.0:
                        raise ValueError(f"annealing_schedule['step'] entries must be in (0, 1]; got {s}.")
                    if float(s) <= prev_s:
                        raise ValueError(f"annealing_schedule['step'] must be strictly increasing; got {sched_steps}.")
                    prev_s = float(s)
                for t in sched_scales:
                    if not 0.0 <= float(t) <= 1.0:
                        raise ValueError(f"annealing_schedule['scale'] entries must be in [0, 1]; got {t}.")
                allowed_funcs = ("geometric", "linear", "cosine", "constant")
                for f in sched_funcs:
                    if f not in allowed_funcs:
                        raise ValueError(
                            f"annealing_schedule['annealing_func'] entries must be one of {allowed_funcs}; got {f!r}."
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

            # Validate the targets against the heads and build the per-term tensors once.
            prepared = self._prepare_optimization_targets(resolved_targets, device=device, dtype=dtype)

            # Move the element-constraint tensors onto the right device (validated above).
            elem_mask = elem_mask_arg.to(device=device) if elem_mask_arg is not None else None
            step_scale = step_scale_arg.to(device=device, dtype=dtype) if step_scale_arg is not None else None
            fixed_w0_dev = fixed_w0_vec.to(device=device, dtype=dtype) if fixed_w0_vec is not None else None
            fixed_mask_dev = fixed_mask_vec.to(device=device) if fixed_mask_vec is not None else None

            # --- Hard-lock setup ----------------------------------------------------------------------
            # Two hard-lock sources both end up in the same ``(locked_mask, locked_w0)`` pair so the
            # downstream ``_w_from_logits`` / ``_apply_lock_paste`` logic is unchanged:
            #
            #   1. ``element_step_scale = 0``: pins the listed elements at their (un-blended)
            #      ``initial_weights`` values. Requires ``initial_weights`` because there's no other
            #      source for per-row seed values.
            #   2. ``fixed_amounts``: pins the listed elements at user-given absolute amounts. No
            #      ``initial_weights`` required — the lock values come straight from the kwarg.
            #
            # The two paths must not overlap (validated above). When both are present, we just
            # OR the masks and add the value tensors (disjoint by construction).
            #
            # Why this matters: zeroing ``logit_i.grad`` keeps that logit constant but does NOT keep
            # ``w_i`` constant — softmax renormalises across all logits, so when other (unlocked)
            # logits move, the softmax denominator changes and so does the locked weight. The fix
            # is to (a) detect locked indices, (b) capture their per-row target weights, and (c)
            # inside ``_w_from_logits`` paste those values back over the softmax output and
            # renormalise the unlocked positions to fill the remaining ``1 − Σ locked_w`` mass per
            # row. The gradient through the locked indices is automatically zero (the lock branch
            # uses a constant), so we no longer need the ``step_scale.mul_`` zeroing for them —
            # but we leave that path active for the genuinely soft case ``0 < step_scale < 1``.
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
            if fixed_mask_dev is not None:
                # Broadcast the per-element fixed values to every row in the batch.
                B = logits.shape[0]
                fixed_w0_batch = fixed_w0_dev.unsqueeze(0).expand(B, -1).detach()
                if locked_mask is None:
                    locked_mask = fixed_mask_dev
                    locked_w0 = fixed_w0_batch
                else:
                    locked_mask = locked_mask | fixed_mask_dev  # validated disjoint
                    locked_w0 = locked_w0 + fixed_w0_batch

            # Runtime sanity: combined lock sum must leave room (or fit exactly) for the simplex.
            # ``fixed_amounts`` enforces ``Σ < 1`` at kwarg time, and ``element_step_scale=0``
            # locks at seed values which sum to ≤ 1 per row — but the *combined* total could
            # exceed 1 (e.g. seed-lock Mg=0.50 + fix Au=0.65). Check here, with a tiny tolerance
            # for float noise.
            if locked_w0 is not None:
                lock_sums = locked_w0.sum(dim=-1)
                if (lock_sums > 1.0 + 1e-5).any():
                    raise ValueError(
                        f"Combined locked mass exceeds 1.0 on at least one row "
                        f"(max row-sum = {float(lock_sums.max()):.4f}). Likely cause: "
                        "``element_step_scale=0`` locks plus ``fixed_amounts`` together claim more "
                        "than 100% of the simplex. Lower one set of values or drop a lock."
                    )

            # Runtime sanity: floored elements must not contradict the lock-paste targets.
            # ``fixed_amounts`` was checked at kwarg time; ``element_step_scale=0`` locks have
            # per-row seed values we couldn't see earlier — verify them now.
            if min_nonzero_weight > 0.0 and locked_mask is not None and locked_w0 is not None:
                locked_below_floor = (locked_w0 > 0) & (locked_w0 < min_nonzero_weight)
                if locked_below_floor.any():
                    raise ValueError(
                        f"At least one locked element's value falls below min_nonzero_weight="
                        f"{min_nonzero_weight}. Likely cause: an element_step_scale=0 lock points "
                        "at a seed value below the floor (raise the seed, lower the floor, or "
                        "drop the lock)."
                    )

            # --- Soft top-K (cardinality constraint) helpers ----------------------------------------
            # Schedule shape (controlled by ``annealing_scale`` and optionally ``annealing_schedule``):
            #
            #   * Normalised scale ∈ [0, 1] is the user-facing knob; raw τ is derived via
            #     ``τ = 25**scale``  (so scale=0 → τ=1, scale=0.5 → τ=5, scale=1 → τ=25).
            #   * Default schedule when no dict is given: geometric from ``τ_start=25**annealing_scale``
            #     at fractional step 0 down to ``_TAU_END=0.01`` at fractional step 1.
            #   * When ``annealing_schedule`` dict is provided, its segments override the front of
            #     the schedule; the segment from ``step[-1]`` to 1.0 (if not already at 1.0) falls
            #     back to the geometric tail from ``25**scale[-1]`` down to ``_TAU_END``.
            #
            # ``current_tau`` lives in a list so the optimisation loop can mutate it each step
            # without rebuilding the ``_w_from_logits`` closure that reads it.
            _TAU_FLOOR = 1e-3  # numerical lower bound; below this softmax(lg/τ) loses precision
            _TAU_END = 0.01  # fixed final hardness for the default schedule's tail
            _SCALE_TAU_BASE = 25.0  # τ = _SCALE_TAU_BASE**scale → 0→1, 0.5→5, 1→25

            def _scale_to_tau(scale: float) -> float:
                return float(_SCALE_TAU_BASE ** max(0.0, min(1.0, scale)))

            def _interp_scalar(a: float, b: float, t: float, func: str) -> float:
                """Interpolate from ``a`` to ``b`` at local-time ``t`` ∈ [0, 1]."""
                if func == "constant":
                    return a
                if func == "linear":
                    return a + (b - a) * t
                if func == "cosine":
                    return b + 0.5 * (a - b) * (1.0 + math.cos(math.pi * t))
                # geometric — guard against zero/sign issues by working in log space when both >0.
                if a > 0.0 and b > 0.0:
                    return a * (b / a) ** t
                # Fall back to linear for degenerate cases (shouldn't trigger in normal use).
                return a + (b - a) * t

            # Materialise schedule arrays once (validated above), so the per-step lookup is light.
            _sched_steps: list[float] = (
                [float(s) for s in annealing_schedule["step"]] if annealing_schedule is not None else []
            )
            _sched_scales: list[float] = (
                [float(t) for t in annealing_schedule["scale"]] if annealing_schedule is not None else []
            )
            _sched_funcs: list[str] = (
                list(annealing_schedule["annealing_func"]) if annealing_schedule is not None else []
            )

            def _tau_for_step(step: int) -> float:
                """Return the raw τ for integer optimisation step ``step``."""
                if max_elements is None or steps <= 1:
                    return float(max(_TAU_END, _TAU_FLOOR))
                # Fractional progress in [0, 1].
                s = step / (steps - 1)
                # Default schedule (used directly when no dict, or for the tail when dict ends < 1.0).
                default_tau_start = _scale_to_tau(annealing_scale)
                default_tau_end = _TAU_END

                if _sched_steps:
                    # Walk through dict segments to find the one containing ``s``.
                    prev_step = 0.0
                    prev_scale = annealing_scale  # segment 0 starts at the simple knob's value
                    for i, seg_end in enumerate(_sched_steps):
                        if s <= seg_end:
                            local_t = (s - prev_step) / max(seg_end - prev_step, 1e-12)
                            scale_now = _interp_scalar(prev_scale, _sched_scales[i], local_t, _sched_funcs[i])
                            return float(max(_scale_to_tau(scale_now), _TAU_FLOOR))
                        prev_step = seg_end
                        prev_scale = _sched_scales[i]
                    # ``s`` is past the dict's last step → use the geometric tail from
                    # ``25**scale[-1]`` at ``step[-1]`` down to ``_TAU_END`` at 1.0.
                    tail_start_tau = _scale_to_tau(_sched_scales[-1])
                    tail_end_step = 1.0
                    tail_local_t = (s - _sched_steps[-1]) / max(tail_end_step - _sched_steps[-1], 1e-12)
                    val = tail_start_tau * (default_tau_end / tail_start_tau) ** tail_local_t
                    return float(max(val, _TAU_FLOOR))

                # No dict — default geometric schedule from τ_start(annealing_scale) to _TAU_END.
                val = default_tau_start * (default_tau_end / default_tau_start) ** s
                return float(max(val, _TAU_FLOOR))

            current_tau = [_tau_for_step(0)]

            def _soft_topk_mask(
                lg: torch.Tensor, K: int, tau: float, *, force_select: torch.Tensor | None = None
            ) -> torch.Tensor:
                """Plötz–Roth iterative softmax. Returns m ∈ [0,1]^(B, n) with Σm = K.

                ``force_select`` (n_components,) bool marks positions that must be in the K
                selection (e.g. hard-locked elements). Instead of boosting those logits — which
                would make the iterative softmax pick them K times in a row, never moving on —
                we **pre-seed** the mask with 1.0 at those positions and run only ``K - n_locked``
                iterations on the *unlocked* positions (their logits are masked to ``-inf``
                inside the iteration so they never compete).
                """
                if force_select is None:
                    alpha = lg
                    m = torch.zeros_like(lg)
                    n_iter = K
                else:
                    # Pre-mark locked positions as fully selected; iterate only on the rest.
                    n_locked = int(force_select.sum().item())
                    n_iter = K - n_locked
                    locked_row = force_select.to(lg.dtype).unsqueeze(0).expand_as(lg)
                    m = locked_row.clone()
                    alpha = lg.masked_fill(force_select, float("-inf"))
                for _ in range(n_iter):
                    p = torch.softmax(alpha / tau, dim=-1)
                    m = m + p
                    # The shift in scaled-logit space at the selected position is
                    # ``log(1−p)/τ`` — at small τ this is enormously negative, so the next
                    # iteration cannot re-pick the same position. (We must NOT multiply by τ here.)
                    alpha = alpha + torch.log((1.0 - p).clamp(min=1e-12))
                return m

            def _hard_topk_project(w: torch.Tensor, K: int) -> torch.Tensor:
                """Hard top-K projection: keep K largest per row, zero rest, renormalise.

                If ``locked_mask`` is set, every locked position is forced into the kept set
                (so the lock-paste below still has a place to write its seed values); the
                remaining ``K − n_locked`` slots are filled by the largest unlocked weights.
                """
                if locked_mask is None:
                    _, idx = w.topk(K, dim=-1)
                    keep = torch.zeros_like(w).scatter_(-1, idx, 1.0)
                else:
                    n_locked = int(locked_mask.sum().item())
                    n_free = K - n_locked
                    locked_row = locked_mask.to(w.dtype).unsqueeze(0).expand_as(w)
                    if n_free > 0:
                        # Exclude locked positions from the unlocked competition by sending them
                        # to ``-inf`` before topk; locked positions are added back via ``locked_row``.
                        w_for_free = w.masked_fill(locked_mask.unsqueeze(0), float("-inf"))
                        _, idx = w_for_free.topk(n_free, dim=-1)
                        free_keep = torch.zeros_like(w).scatter_(-1, idx, 1.0)
                        keep = (locked_row + free_keep).clamp(max=1.0)
                    else:
                        keep = locked_row
                w = w * keep
                return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)

            def _apply_lock_paste(w: torch.Tensor) -> torch.Tensor:
                """Paste locked seed values onto ``w`` and renormalise unlocked positions."""
                if locked_mask is None:
                    return w
                free_mask_f = (~locked_mask).to(w.dtype)
                w_unlocked = w * free_mask_f
                free_mass = (1.0 - locked_w0.sum(dim=-1, keepdim=True)).clamp(min=0.0)
                w_unlocked = w_unlocked / w_unlocked.sum(dim=-1, keepdim=True).clamp(min=1e-12) * free_mass
                return w_unlocked + locked_w0

            def _apply_min_floor(w: torch.Tensor) -> torch.Tensor:
                """Drop unlocked positions below ``min_nonzero_weight`` and re-fill free mass.

                Locked positions are exempt (their values are user-set). If dropping below-floor
                positions would leave a row with zero unlocked mass, the floor is skipped for
                that row — preserving the simplex invariant. The "at most K" guarantee still
                holds; some rows may end up with fewer than K non-zero positions.
                """
                if min_nonzero_weight <= 0.0:
                    return w
                if locked_mask is not None:
                    unlocked_f = (~locked_mask).to(w.dtype)
                    free_mass = (1.0 - locked_w0.sum(dim=-1, keepdim=True)).clamp(min=0.0)
                    unlocked_bool = (~locked_mask).unsqueeze(0).expand_as(w)
                else:
                    unlocked_f = torch.ones_like(w[0])
                    free_mass = torch.ones(w.shape[0], 1, dtype=w.dtype, device=w.device)
                    unlocked_bool = torch.ones_like(w, dtype=torch.bool)
                below = (w > 0) & (w < min_nonzero_weight) & unlocked_bool
                if not below.any():
                    return w
                w_drop = w.masked_fill(below, 0.0)
                # Per-row unlocked sum after the tentative drop.
                unlocked_after = w_drop * unlocked_f
                unlocked_sum = unlocked_after.sum(dim=-1, keepdim=True)
                # Rows where the drop is safe — at least one unlocked position survives.
                can_drop = unlocked_sum > 1e-12
                # Renormalise unlocked portion to fit the free mass; locked stays as-is.
                safe_sum = unlocked_sum.clamp(min=1e-12)
                if locked_mask is not None:
                    locked_part = w_drop * locked_mask.to(w.dtype)
                    w_renorm = locked_part + unlocked_after * (free_mass / safe_sum)
                else:
                    w_renorm = w_drop / safe_sum
                return torch.where(can_drop.expand_as(w), w_renorm, w)

            def _w_from_logits(lg: torch.Tensor) -> torch.Tensor:
                """Softmax → optional soft top-K → optional hard-lock paste → optional min-floor.

                Reads ``current_tau[0]`` (set by the outer loop) for the soft top-K temperature.
                """
                if elem_mask is not None:
                    lg = lg.masked_fill(~elem_mask, float("-inf"))
                w_soft = torch.softmax(lg, dim=-1)
                if max_elements is not None and max_elements < n_components:
                    # Force locked positions to always sit in the K-hot mask so the lock-paste
                    # below has somewhere to write. ``w_soft`` itself is computed from the
                    # *unboosted* logits, so the within-K ratios reflect the optimisation state.
                    m_topk = _soft_topk_mask(lg, max_elements, current_tau[0], force_select=locked_mask)
                    w = w_soft * m_topk
                    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                else:
                    w = w_soft
                return _apply_min_floor(_apply_lock_paste(w))

            def _heads_forward(h_task: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
                """Evaluate the target terms; return (per-target channels (B, T), scalar loss terms)."""
                channels, per_sample_losses = self._optimization_objective(h_task, prepared)
                return channels, list(per_sample_losses.mean(dim=0))

            n_channels = len(prepared)

            # --- Record initial scores --------------------------------------------------------------
            # Initial scoring uses τ at step 0 of the annealing schedule — i.e. the softest end
            # of the (annealing_scale + annealing_schedule)-derived τ curve, where the optimisation
            # actually begins.
            current_tau[0] = _tau_for_step(0)
            with torch.no_grad():
                w0_tensor = _w_from_logits(logits)
                h0 = torch.tanh(self.encoder(w0_tensor @ kmd_kernel))
                initial_channels, _ = _heads_forward(h0)
                initial_score = initial_channels.detach()

            # --- Optimisation loop ------------------------------------------------------------------
            # With every model parameter at ``requires_grad=False``, ``loss.backward()`` populates
            # gradient only on ``logits`` — no stale grads accumulate on encoder/heads.
            trajectory: list[torch.Tensor] = []
            weights_trajectory: list[torch.Tensor] = [] if record_weights_trajectory else []
            for step in range(steps):
                current_tau[0] = _tau_for_step(step)
                optimizer.zero_grad()
                w = _w_from_logits(logits)
                x = w @ kmd_kernel
                h_task = torch.tanh(self.encoder(x))
                channels, terms = _heads_forward(h_task)
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
                trajectory.append(channels.detach())
                if record_weights_trajectory:
                    # Snapshot the post-step weights at the *current* (still-soft) τ — the
                    # trajectory thus reflects the annealing schedule, not the hard projection.
                    # Stored on CPU to keep GPU memory flat for long trajectories on large B.
                    with torch.no_grad():
                        weights_trajectory.append(_w_from_logits(logits).detach().cpu())

            # --- Final state ------------------------------------------------------------------------
            # Use the hardest τ for the final readout, then (if ``max_elements`` is active) apply
            # a hard top-K projection so the returned ``optimized_weights`` has **at most** K
            # non-zero positions (the floor below may reduce that further) — at τ_end ≈ 0.01 the
            # soft mask is already near-K-hot, so the projection just cleans up residual
            # sub-threshold weights.
            current_tau[0] = float(max(_TAU_END, _TAU_FLOOR))
            with torch.no_grad():
                w_final = _w_from_logits(logits)
                if max_elements is not None and max_elements < n_components:
                    w_final = _hard_topk_project(w_final, max_elements)
                    # Re-apply lock-paste — the projection may have re-distributed mass across
                    # unlocked positions, and lock-paste's "free mass" renormalisation needs to
                    # be re-run so the row still sums to exactly 1. Then re-floor: the projection
                    # may have promoted a previously-zeroed below-floor position back in.
                    w_final = _apply_lock_paste(w_final)
                    w_final = _apply_min_floor(w_final)
                x_final = w_final @ kmd_kernel
                h_final = torch.tanh(self.encoder(x_final))
                final_channels, _ = _heads_forward(h_final)
                final_target = final_channels.detach()

            weights_traj_tensor: torch.Tensor | None = None
            if record_weights_trajectory:
                # (steps, B, n_components). Same empty-steps fallback as ``trajectory`` so the
                # downstream code can rely on the shape contract without a None branch.
                weights_traj_tensor = (
                    torch.stack(weights_trajectory, dim=0)
                    if weights_trajectory
                    else torch.empty((0, logits.shape[0], n_components), dtype=torch.float32)
                )

            return CompositionOptimizationResult(
                optimized_weights=w_final.detach(),
                optimized_descriptor=x_final.detach(),
                optimized_target=final_target,
                initial_score=initial_score,
                # Preserve the (steps, B, T) shape contract even when steps == 0.
                trajectory=torch.stack(trajectory, dim=0)
                if trajectory
                else torch.empty((0, logits.shape[0], n_channels), device=device, dtype=dtype),
                weights_trajectory=weights_traj_tensor,
            )
        finally:
            if was_training:
                self.train()
            for p, prev in saved_req_grad:
                p.requires_grad_(prev)
