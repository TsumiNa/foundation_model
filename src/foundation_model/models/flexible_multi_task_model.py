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

from typing import List, Optional, Sequence

import lightning as L
import numpy as np
import pandas as pd  # Added
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from loguru import logger  # Replaced logging with loguru
from torch.optim.lr_scheduler import LRScheduler  # Changed from _LRScheduler

from .components.foundation_encoder import FoundationEncoder
from .model_config import (
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    TaskType,
)
from .task_head.classification import ClassificationHead
from .task_head.kernel_regression import KernelRegressionHead
from .task_head.regression import RegressionHead


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
    4. Continual Learning: Support model updates via deposit layer design

    Parameters
    ----------
    shared_block_dims : list[int]
        Widths of shared MLP layers (foundation encoder). The first element is the input dimension,
        and the last element is the latent representation dimension.
        Example: [128, 256, 512, 256] represents a 3-layer MLP with input dimension 128 and latent dimension 256.
    task_configs : list[RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig]
        List of task configurations, each defining a prediction task. Each configuration must specify
        task type, name, dimensions, etc. Regression and classification task heads receive the deposit
        layer output, while KernelRegression task heads receive both deposit layer output and sequence points.
        A task-specific `loss_weight` (defaults to 1.0) can be set in each configuration to scale its loss.
    norm_shared : bool
        Whether to apply layer normalization in shared layers.
    residual_shared : bool
        Whether to use residual connections in shared layers.
    shared_block_optimizer : OptimizerConfig | None
        Optimizer configuration for the shared foundation encoder and deposit layer.
    enable_learnable_loss_balancer : bool
        Whether to use learnable log_sigma_t parameters for each supervised task to weight their losses.
    """

    def __init__(
        self,
        shared_block_dims: list[int],
        task_configs: Sequence[RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig],
        *,
        # Normalization/residual options
        norm_shared: bool = True,
        residual_shared: bool = False,
        # Freezing parameters
        freeze_shared_encoder: bool = False,
        # Optimization parameters
        shared_block_optimizer: OptimizerConfig | None = None,
        enable_learnable_loss_balancer: bool = False,  # New parameter
        # Loss calculation behavior
        allow_all_missing_in_batch: bool = True,  # New parameter for handling all-missing batches
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store the new parameters
        self.enable_learnable_loss_balancer = enable_learnable_loss_balancer
        self.allow_all_missing_in_batch = allow_all_missing_in_batch

        # Validate inputs
        if len(shared_block_dims) < 2:
            raise ValueError("shared_block_dims must have at least 2 elements")

        if not task_configs:
            raise ValueError("At least one task configuration must be provided")

        # Store configuration parameters
        self.shared_block_dims = shared_block_dims
        self.deposit_dim = self.shared_block_dims[-1]  # Define deposit_dim consistently
        self.task_configs = task_configs
        self.task_configs_map = {cfg.name: cfg for cfg in self.task_configs}
        for cfg in self.task_configs:
            cfg.loss_weight = self._normalize_loss_weight(getattr(cfg, "loss_weight", 1.0), cfg.name)

        # Normalization/residual options
        self.norm_shared = norm_shared
        self.residual_shared = residual_shared

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
        # deposit_dim is now self.deposit_dim, defined in __init__

        self.encoder = FoundationEncoder(
            input_dim=self.shared_block_dims[0],
            hidden_dims=self.shared_block_dims[1:],
            deposit_dim=self.deposit_dim,
            norm=self.norm_shared,
            residual=self.residual_shared,
        )

        # Keep references to original components for backward compatibility
        self.shared = self.encoder.shared
        self.deposit = self.encoder.deposit

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
        self, config_item: RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig
    ):
        """Validate that the task configuration is compatible with the shared encoder."""
        if not config_item.name:
            raise ValueError("Task config must have a non-empty name.")
        if config_item.name in self.task_configs_map:
            raise ValueError(f"Task '{config_item.name}' already exists in the model.")

        expected_input_dim = self.deposit_dim

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

    def _instantiate_task_head(
        self, config_item: RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig
    ) -> nn.Module:
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

    def _activate_task(
        self,
        task_config: RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig,
    ) -> nn.Module:
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

    def _track_task_types(self):
        """Track which types of tasks are enabled."""
        self.has_regression = any(tc.type == TaskType.REGRESSION for tc in self.task_configs if tc.enabled)
        self.has_classification = any(tc.type == TaskType.CLASSIFICATION for tc in self.task_configs if tc.enabled)
        self.has_kernel_regression = any(
            tc.type == TaskType.KERNEL_REGRESSION for tc in self.task_configs if tc.enabled
        )

    def add_task(
        self,
        *task_configs: RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig,
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
            for p in self.shared.parameters():
                p.requires_grad_(False)
            for p in self.deposit.parameters():
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

        _, h_task = self.encoder(x)

        # Apply task heads - all task heads use h_task (deposit layer output)
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
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue

            head = self.task_heads[name]
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

        raw_val_supervised_losses = {}
        for name, pred_tensor in preds.items():
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue

            head = self.task_heads[name]
            target = y_dict_batch[name]
            sample_mask = task_masks_batch.get(name)

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
                    val_logs[f"val_{name}_all_missing"] = 1.0
                    continue
                raise ValueError(
                    f"Task '{name}' has no valid samples in this batch and allow_all_missing_in_batch is False."
                )

            raw_val_supervised_losses[name] = raw_loss_t
            val_sum_supervised_raw_loss += raw_loss_t.detach()
            val_logs[f"val_{name}_raw_loss"] = raw_loss_t.detach()
            val_logs[f"val_{name}_all_missing"] = 0.0

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

        raw_test_supervised_losses = {}
        for name, pred_tensor in preds.items():
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue

            head = self.task_heads[name]
            target = y_dict_batch[name]
            sample_mask = task_masks_batch.get(name)

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
                    test_logs[f"test_{name}_all_missing"] = 1.0
                    continue
                raise ValueError(
                    f"Task '{name}' has no valid samples in this batch and allow_all_missing_in_batch is False."
                )

            raw_test_supervised_losses[name] = raw_loss_t
            test_sum_supervised_raw_loss += raw_loss_t.detach()
            test_logs[f"test_{name}_raw_loss"] = raw_loss_t.detach()
            test_logs[f"test_{name}_all_missing"] = 0.0

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

        test_logs["test_final_supervised_loss"] = test_supervised_loss_contribution.detach()
        final_test_loss = final_test_loss + test_supervised_loss_contribution

        self.log_dict(test_logs, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_final_loss", final_test_loss.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        return None

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
            Task representations from deposit layer, shape (B, D)
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
