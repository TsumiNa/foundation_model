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

from typing import Any, List, Optional  # Added List, Optional

import lightning as L
import pandas as pd  # Added
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger  # Replaced logging with loguru
from torch.optim.lr_scheduler import _LRScheduler

from foundation_model.configs.model_config import (
    ClassificationTaskConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    SequenceTaskConfig,
    TaskType,
)

from .components.foundation_encoder import FoundationEncoder, MultiModalFoundationEncoder
from .components.self_supervised import SelfSupervisedModule
from .task_head.base import SequenceBaseHead
from .task_head.classification import ClassificationHead
from .task_head.regression import RegressionHead
from .task_head.sequence import create_sequence_head


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
       - Sequence tasks: Predict time-series data (e.g., temperature curves)

    4. Structure Fusion (optional):
       When with_structure=True, can fuse information from different modalities (e.g., formula and structure).

    5. Self-supervised Training (optional):
       When enable_self_supervised_training=True, enables self-supervised learning objectives:
       - Masked Feature Modeling (MFM): Similar to BERT's masked language modeling
       - Contrastive Learning: Aligns representations from different modalities
       - Cross-reconstruction: Reconstructs one modality from another

    Training Process:
    - Each batch's loss includes task-specific losses and optional self-supervised learning losses
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
    task_configs : list[RegressionTaskConfig | ClassificationTaskConfig | SequenceTaskConfig]
        List of task configurations, each defining a prediction task. Each configuration must specify
        task type, name, dimensions, etc. Regression and classification task heads receive the deposit
        layer output, while sequence task heads receive both deposit layer output and sequence points.
    norm_shared : bool
        Whether to apply layer normalization in shared layers.
    residual_shared : bool
        Whether to use residual connections in shared layers.
    shared_block_optimizer : OptimizerConfig | None
        Optimizer configuration for shared foundation encoder and deposit layer.
    with_structure : bool
        Whether to enable structure encoding and modality fusion. When set to True,
        the model expects inputs from two modalities.
    struct_block_dims : list[int] | None
        Dimensions for structure encoder MLP layers. The first element is the input dimension
        of the structure features, and the last element must match shared_block_dims[-1] to ensure
        both modalities have the same dimension before fusion. Only used when with_structure=True.
    modality_dropout_p : float
        Probability of modality dropout. During self-supervised training, there's this probability of
        randomly dropping the structure modality, forcing the model to learn to handle
        single-modality cases. Only relevant when with_structure=True and enable_self_supervised_training=True.
    enable_self_supervised_training : bool
        Whether to enable additional self-supervised training objectives (masked feature modeling,
        contrastive learning, and cross-reconstruction). When True, these additional losses are
        added to task-specific losses to improve feature representations.
    loss_weights : dict[str, float] | None
        Weight coefficients dictionary for balancing different loss components in the total loss.
        Includes the following keys:
        - Task names: Weight for each task's loss, default 1.0
        - "mfm": Weight for masked feature modeling loss, default 1.0
        - "contrastive": Weight for contrastive learning loss, default 1.0
        - "cross_recon": Weight for cross-reconstruction loss, default 1.0
        Example: {"band_gap": 1.0, "formation_energy": 0.5, "mfm": 0.2, "contrastive": 0.1, "cross_recon": 0.1}
    mask_ratio : float
        Ratio of features to be randomly masked in masked feature modeling.
        Typical value is 0.15. Only used when enable_self_supervised_training=True.
    temperature : float
        Temperature coefficient in contrastive learning. Controls the smoothness of
        the similarity distribution, with smaller values increasing contrast.
        Typical value is 0.07. Only used when enable_self_supervised_training=True and with_structure=True.
    """

    def __init__(
        self,
        shared_block_dims: list[int],
        task_configs: list[RegressionTaskConfig | ClassificationTaskConfig | SequenceTaskConfig],
        *,
        # Normalization/residual options
        norm_shared: bool = True,
        residual_shared: bool = False,
        # Optimization parameters
        shared_block_optimizer: OptimizerConfig | None = None,
        # Structure fusion options
        with_structure: bool = False,
        struct_block_dims: list[int] | None = None,
        modality_dropout_p: float = 0.3,
        # Self-supervised training options
        enable_self_supervised_training: bool = False,
        loss_weights: dict[str, float] | None = None,
        mask_ratio: float = 0.15,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Validate inputs
        if len(shared_block_dims) < 2:
            raise ValueError("shared_block_dims must have at least 2 elements")

        if not task_configs:
            raise ValueError("At least one task configuration must be provided")

        # Store configuration parameters
        self.shared_block_dims = shared_block_dims
        self.task_configs = task_configs
        self.task_configs_map = {cfg.name: cfg for cfg in self.task_configs}

        # Normalization/residual options
        self.norm_shared = norm_shared
        self.residual_shared = residual_shared

        # Structure fusion parameters
        self.with_structure = with_structure
        self.struct_block_dims = struct_block_dims
        self.mod_dropout_p = modality_dropout_p

        # Self-supervised training parameters
        self.enable_self_supervised_training = enable_self_supervised_training
        self.mask_ratio = mask_ratio
        self.temperature = temperature

        # Optimizer configurations
        self.shared_block_optimizer = shared_block_optimizer or OptimizerConfig(weight_decay=1e-2)

        # Initialize loss weights
        self._init_loss_weights(loss_weights)

        # Initialize learnable uncertainty parameters (log(sigma_t) for each supervised task)
        self.task_log_sigmas = nn.ParameterDict(
            {
                cfg.name: nn.Parameter(torch.zeros((), device=self.device))  # Ensure device matches
                for cfg in self.task_configs
                if cfg.enabled
                # SSL tasks like "mfm", "contrastive", "cross_recon" are handled separately by self.w
                # and do not get their own learnable sigmas here unless explicitly designed.
            }
        )

        # Initialize model components
        self._init_foundation_encoder()
        self._init_task_heads()
        if enable_self_supervised_training:
            self._init_self_supervised_module()

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

    def _init_loss_weights(self, loss_weights: dict[str, float] | None = None):
        """Initialize loss weights for different components."""
        # Start with default weights for all enabled tasks
        self.w = {cfg.name: 1.0 for cfg in self.task_configs if cfg.enabled}

        # Add weights for self-supervised objectives if enabled
        if self.enable_self_supervised_training:
            self.w.update(
                {
                    "mfm": 1.0,  # Masked Feature Modeling
                    "contrastive": 1.0,  # Contrastive Learning
                    "cross_recon": 1.0,  # Cross-Reconstruction
                }
            )

        # Apply user-provided weights
        if loss_weights:
            self.w.update(loss_weights)

    def _init_foundation_encoder(self):
        """Initialize the foundation encoder."""
        # Get deposit dimension from first task config
        deposit_dim = next(
            (c.dims[0] for c in self.task_configs if hasattr(c, "dims")),
            self.shared_block_dims[-1],  # Default to latent dimension
        )

        # Initialize appropriate encoder based on structure usage
        if self.with_structure:
            self.encoder = MultiModalFoundationEncoder(
                formula_input_dim=self.shared_block_dims[0],
                formula_hidden_dims=self.shared_block_dims[1:],
                structure_input_dim=self.struct_block_dims[0],
                structure_hidden_dims=self.struct_block_dims[1:],
                deposit_dim=deposit_dim,
                norm=self.norm_shared,
                residual=self.residual_shared,
            )

            # Keep references to original components for backward compatibility
            self.shared = self.encoder.formula_encoder
            self.deposit = self.encoder.deposit
            self.struct_enc = self.encoder.structure_encoder
            self.fusion = self.encoder.fusion
        else:
            self.encoder = FoundationEncoder(
                input_dim=self.shared_block_dims[0],
                hidden_dims=self.shared_block_dims[1:],
                deposit_dim=deposit_dim,
                norm=self.norm_shared,
                residual=self.residual_shared,
            )

            # Keep references to original components for backward compatibility
            self.shared = self.encoder.shared
            self.deposit = self.encoder.deposit

    def _build_task_heads(self) -> nn.ModuleDict:
        """
        Create task heads based on configurations.
        This method is called during model initialization.
        """
        task_heads_dict = nn.ModuleDict()

        for config_item in self.task_configs:
            if not config_item.enabled:
                continue

            if config_item.type == TaskType.REGRESSION:
                assert isinstance(config_item, RegressionTaskConfig)
                task_heads_dict[config_item.name] = RegressionHead(config=config_item)
            elif config_item.type == TaskType.CLASSIFICATION:
                assert isinstance(config_item, ClassificationTaskConfig)
                task_heads_dict[config_item.name] = ClassificationHead(config=config_item)
            elif config_item.type == TaskType.SEQUENCE:
                assert isinstance(config_item, SequenceTaskConfig)
                task_heads_dict[config_item.name] = create_sequence_head(config=config_item)
        return task_heads_dict

    def _init_task_heads(self):
        """Initialize task heads based on configurations."""
        self.task_heads = self._build_task_heads()

        # Apply optimizer freeze_parameters to task heads
        for name, head in self.task_heads.items():
            config = self.task_configs_map[name]
            if config.optimizer and config.optimizer.freeze_parameters:
                for p in head.parameters():
                    p.requires_grad_(False)

    def _init_self_supervised_module(self):
        """Initialize self-supervised training module if enabled."""
        structure_dim = self.struct_block_dims[0] if self.with_structure else None

        self.ssl_module = SelfSupervisedModule(
            latent_dim=self.shared_block_dims[-1],
            formula_dim=self.shared_block_dims[0],
            structure_dim=structure_dim,
            mask_ratio=self.mask_ratio,
            temperature=self.temperature,
        )

        # Keep references to original components for backward compatibility
        self.dec_formula = self.ssl_module.formula_decoder
        if self.with_structure:
            self.dec_struct = self.ssl_module.structure_decoder

    def _track_task_types(self):
        """Track which types of tasks are enabled."""
        self.has_regression = any(tc.type == TaskType.REGRESSION for tc in self.task_configs if tc.enabled)
        self.has_classification = any(tc.type == TaskType.CLASSIFICATION for tc in self.task_configs if tc.enabled)
        self.has_sequence = any(tc.type == TaskType.SEQUENCE for tc in self.task_configs if tc.enabled)

    def _init_weights(self):
        """Initialize model weights and apply freezing based on optimizer configs."""
        # Apply parameter freezing based on optimizer config
        if self.shared_block_optimizer and self.shared_block_optimizer.freeze_parameters:
            for p in self.shared.parameters():
                p.requires_grad_(False)
            for p in self.deposit.parameters():
                p.requires_grad_(False)

            # If structure fusion is enabled, freeze those parameters too
            if self.with_structure:
                for p in self.struct_enc.parameters():
                    p.requires_grad_(False)
                for p in self.fusion.parameters():
                    p.requires_grad_(False)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(
        self, x_formula: torch.Tensor, x_struct: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode inputs through the foundation encoder.

        This method is kept for backward compatibility.

        Parameters
        ----------
        x_formula : torch.Tensor
            Formula input tensor.
        x_struct : torch.Tensor | None
            Structure input tensor, if using structure fusion.

        Returns
        -------
        h_latent : torch.Tensor
            Latent representation (B, D_latent).
        h_task : torch.Tensor
            Task input representation (B, deposit_dim) after deposit layer.
        """
        if self.with_structure:
            h_formula, h_structure, h_fused, h_task = self.encoder(x_formula, x_struct)
            return h_fused, h_task
        else:
            h_latent, h_task = self.encoder(x_formula)
            return h_latent, h_task

    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        task_sequence_data_batch: dict[str, torch.Tensor] | None = None,  # Renamed from temps_batch
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]
            Input tensor(s). If structure fusion is enabled, this should be a tuple
            of (formula_tensor, structure_tensor).
        task_sequence_data_batch : dict[str, torch.Tensor] | None, optional
            A dictionary where keys are sequence task names and values are the
            corresponding sequence input data (e.g., temperature points, time steps)
            for the batch. Required if sequence tasks are present. Defaults to None.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of task outputs, keyed by task name.
        """
        # Unpack inputs
        if self.with_structure and isinstance(x, (list, tuple)):
            x_formula, x_struct = x
        else:
            x_formula, x_struct = x, None

        # Get latent and task-specific representations
        if self.with_structure:
            _, _, _, h_task = self.encoder(x_formula, x_struct)
        else:
            _, h_task = self.encoder(x_formula)

        # Apply task heads - all task heads use h_task (deposit layer output)
        outputs = {}
        for name, head in self.task_heads.items():
            if isinstance(head, SequenceBaseHead):
                # Get specific sequence data for this sequence head
                task_sequence_input = task_sequence_data_batch.get(name) if task_sequence_data_batch else None
                if task_sequence_input is not None:
                    outputs[name] = head(h_task, task_sequence_input)
                # else: # Decide how to handle if a sequence task head doesn't get its temps
                # Could raise error, or head itself might have default behavior
                # For now, assume temps are provided if head is sequence type
            else:
                outputs[name] = head(h_task)

        return outputs

    def training_step(self, batch, batch_idx):
        """
        Training step implementation with multi-component loss calculation,
        handling self-supervised learning and modality dropout.

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict_batch, task_masks_batch, task_sequence_data_batch)
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Total weighted loss for optimization.
        """
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        lr_schedulers = self.lr_schedulers()
        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        # 1. Unpack batch data
        # y_dict_batch, task_masks_batch, task_sequence_data_batch are now dictionaries keyed by task_name
        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        train_logs = {}  # For detailed logging, not on progress bar

        # 2. Determine input modalities based on configuration and batch data
        x_formula = None
        original_x_struct = None  # Keep original structure input for potential cross-recon target
        if self.with_structure and isinstance(x, (list, tuple)):
            # Multi-modal input expected and received
            x_formula, original_x_struct = x
            if x_formula is None:
                raise ValueError("Formula input (x_formula) cannot be None in multi-modal mode.")
        elif not self.with_structure and isinstance(x, torch.Tensor):
            # Single-modal input expected and received
            x_formula = x
        elif self.with_structure and isinstance(x, torch.Tensor):
            # Multi-modal expected, but only single-modal received (treat as formula only)
            x_formula = x
            # original_x_struct remains None
        else:
            raise TypeError(
                f"Unexpected input type/combination. with_structure={self.with_structure}, type(x)={type(x)}"
            )

        zero = torch.zeros([], device=x_formula.device if x_formula is not None else "cpu")
        total_loss = zero + 0.0  # non-leaf tensor placeholder for loss accumulation

        # 3. Handle Modality Dropout (only during SSL training)
        x_struct_for_processing = original_x_struct  # Start with the original structure input
        if (
            self.enable_self_supervised_training
            and self.with_structure
            and original_x_struct is not None
            and torch.rand(1).item() < self.mod_dropout_p
        ):
            # Apply dropout: use None for structure in subsequent processing
            x_struct_for_processing = None
            train_logs["train_modality_dropout_applied"] = 1.0  # Use float for logging
        elif self.enable_self_supervised_training and self.with_structure:
            train_logs["train_modality_dropout_applied"] = 0.0  # Use float for logging

        # --- Self-Supervised Learning (SSL) Calculations ---
        ssl_loss_contribution = torch.tensor(0.0, device=total_loss.device)
        sum_ssl_raw_loss = torch.tensor(0.0, device=total_loss.device)

        if self.enable_self_supervised_training:
            # We need non-masked embeddings for contrastive/cross-recon later
            # Get these based on potentially dropped structure input
            if self.with_structure:
                h_formula_ssl, h_structure_ssl, _, _ = self.encoder(x_formula, x_struct_for_processing)
            else:
                # When not using structure, FoundationEncoder takes only x_formula
                # and returns (latent, task_representation)
                h_formula_ssl, _ = self.encoder(x_formula)  # x_struct_for_processing is None or not used
                h_structure_ssl = None

            # 4a. Masked Feature Modeling (MFM)
            if self.w.get("mfm", 0) > 0:
                # Pass the encoder function that handles masking internally
                encoder_fn = self.encoder.encode_masked
                # Compute MFM loss using the potentially dropped structure input
                mfm_loss, mfm_logs = self.ssl_module.compute_masked_feature_loss(
                    encoder_fn, x_formula, x_struct_for_processing
                )
                # Log MFM specific raw losses (e.g., formula_mfm, struct_mfm if applicable)
                for k, v_loss in mfm_logs.items():
                    train_logs[f"train_raw_{k}"] = v_loss.detach()  # e.g. train_raw_formula_mfm_loss

                # Use self.w for SSL task weighting
                static_weight_mfm = self.w.get("mfm", 1.0)
                weighted_mfm_loss = static_weight_mfm * mfm_loss
                ssl_loss_contribution += weighted_mfm_loss
                sum_ssl_raw_loss += mfm_loss.detach()
                train_logs["train_mfm_raw_loss"] = mfm_loss.detach()
                train_logs["train_mfm_final_loss_contrib"] = weighted_mfm_loss.detach()

            # 4b. Contrastive Loss
            if (
                self.with_structure
                and h_structure_ssl is not None  # Only if structure wasn't dropped/missing
                and self.w.get("contrastive", 0) > 0
            ):
                contrastive_loss = self.ssl_module.compute_contrastive_loss(h_formula_ssl, h_structure_ssl)
                static_weight_contrastive = self.w.get("contrastive", 1.0)
                weighted_contrastive_loss = static_weight_contrastive * contrastive_loss
                ssl_loss_contribution += weighted_contrastive_loss
                sum_ssl_raw_loss += contrastive_loss.detach()
                train_logs["train_contrastive_raw_loss"] = contrastive_loss.detach()
                train_logs["train_contrastive_final_loss_contrib"] = weighted_contrastive_loss.detach()
            elif self.with_structure and self.w.get("contrastive", 0) > 0:
                # Log zero if structure was dropped or missing
                train_logs["train_contrastive_raw_loss"] = 0.0
                train_logs["train_contrastive_final_loss_contrib"] = 0.0

            # 4c. Cross-Reconstruction Loss
            if (
                self.with_structure
                and h_structure_ssl is not None  # Only if structure wasn't dropped/missing for encoding
                and original_x_struct is not None  # Only if original structure existed for target
                and self.w.get("cross_recon", 0) > 0
            ):
                # Use non-masked embeddings derived from potentially dropped input,
                # but reconstruct the *original* structure input.
                cross_recon_loss = self.ssl_module.compute_cross_reconstruction_loss(
                    h_formula_ssl, h_structure_ssl, x_formula, original_x_struct
                )
                static_weight_cross_recon = self.w.get("cross_recon", 1.0)
                weighted_cross_recon_loss = static_weight_cross_recon * cross_recon_loss
                ssl_loss_contribution += weighted_cross_recon_loss
                sum_ssl_raw_loss += cross_recon_loss.detach()
                train_logs["train_cross_recon_raw_loss"] = cross_recon_loss.detach()
                train_logs["train_cross_recon_final_loss_contrib"] = weighted_cross_recon_loss.detach()
            elif self.with_structure and self.w.get("cross_recon", 0) > 0:
                # Log zero if structure was dropped or missing
                train_logs["train_cross_recon_raw_loss"] = 0.0
                train_logs["train_cross_recon_final_loss_contrib"] = 0.0

        if self.enable_self_supervised_training:
            train_logs["train_sum_ssl_raw_loss"] = sum_ssl_raw_loss
            train_logs["train_final_ssl_loss"] = ssl_loss_contribution.detach()

        # --- Supervised Task Calculations ---
        supervised_loss_contribution = torch.tensor(0.0, device=total_loss.device)

        # 5. Prepare input for the standard forward pass
        # Use the structure input *after* potential modality dropout for consistent forward pass
        if self.with_structure:
            forward_input = (x_formula, x_struct_for_processing)
        else:
            forward_input = x_formula

        # 6. Get predictions from the forward method
        preds = self(forward_input, task_sequence_data_batch)  # Pass task_sequence_data_batch dictionary

        # 7. Calculate supervised task losses
        raw_supervised_losses = {}
        for name, pred_tensor in preds.items():
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue

            head = self.task_heads[name]
            target = y_dict_batch[name]
            sample_mask = task_masks_batch.get(name)
            if sample_mask is None:
                logger.warning(f"Mask not found for task {name} in training_step. Assuming all valid.")
                sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            raw_loss_t, _ = head.compute_loss(pred_tensor, target, sample_mask)
            raw_supervised_losses[name] = raw_loss_t
            train_logs[f"train_{name}_raw_loss"] = raw_loss_t.detach()

        # Apply uncertainty weighting for supervised tasks
        for name, raw_loss_t in raw_supervised_losses.items():
            if name in self.task_log_sigmas:  # This task uses learnable uncertainty
                current_log_sigma_t = self.task_log_sigmas[name]
                static_weight = self.w.get(name, 1.0)  # User-defined static weight

                precision_factor_t = torch.exp(-2 * current_log_sigma_t)  # 1 / sigma_t^2

                # final_task_loss_component = static_weight * 0.5 * precision_factor_t * raw_loss_t + current_log_sigma_t
                # Using the formulation: static_w * (1/(2*sigma^2)) * raw_loss + log(sigma)
                task_loss_data_term = static_weight * 0.5 * precision_factor_t * raw_loss_t
                task_loss_reg_term = current_log_sigma_t
                final_task_loss_component = task_loss_data_term + task_loss_reg_term

                supervised_loss_contribution += final_task_loss_component

                train_logs[f"train_{name}_sigma_t"] = torch.exp(current_log_sigma_t).detach()
                train_logs[f"train_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            else:  # Should not happen if task_log_sigmas is created for all enabled supervised tasks
                logger.warning(
                    f"Task {name} found in predictions but not in task_log_sigmas. Using static weight only."
                )
                static_weight = self.w.get(name, 1.0)
                final_task_loss_component = static_weight * raw_loss_t
                supervised_loss_contribution += final_task_loss_component
                train_logs[f"train_{name}_final_loss_contrib"] = final_task_loss_component.detach()

        train_logs["train_final_supervised_loss"] = supervised_loss_contribution.detach()

        # Combine supervised and SSL contributions for total loss
        total_loss = supervised_loss_contribution + ssl_loss_contribution

        # 8. Log metrics
        # Log detailed metrics without progress bar
        self.log_dict(train_logs, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        # Log the main training loss with progress bar
        self.log("train_final_loss", total_loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # Manual optimization
        if total_loss.requires_grad:  # Ensure there's something to backpropagate
            self.manual_backward(total_loss)
            for opt in optimizers:
                opt.step()
            for scheduler in lr_schedulers:
                if scheduler is not None:  # Add check for None scheduler
                    scheduler.step(total_loss)
        else:
            logger.warning(
                f"total_loss does not require grad and has no grad_fn at batch_idx {batch_idx}. "
                "Skipping backward pass and optimizer step. "
                "This might indicate all parameters are frozen, loss contributions are zero, "
                "or an issue with the computation graph.",
            )
            # It's good practice to still zero_grad optimizers to clear any stale grads from previous iterations
            for opt in optimizers:
                opt.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step implementation. Performs computations similar to training_step
        (including SSL if enabled) but without modality dropout or gradient updates.
        Logs all relevant unweighted losses.

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
        # 1. Unpack batch data
        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        val_logs = {}  # For detailed logging

        # 2. Determine input modalities based on configuration and batch data
        x_formula = None
        original_x_struct = None  # Structure input from the batch
        if self.with_structure and isinstance(x, (list, tuple)):
            x_formula, original_x_struct = x
            if x_formula is None:
                raise ValueError("Formula input (x_formula) cannot be None in multi-modal mode during validation.")
        elif not self.with_structure and isinstance(x, torch.Tensor):
            x_formula = x
        elif self.with_structure and isinstance(x, torch.Tensor):
            x_formula = x
        else:
            raise TypeError(
                f"Unexpected input type/combination during validation. with_structure={self.with_structure}, type(x)={type(x)}"
            )

        # Initialize accumulators for validation losses
        sum_val_raw_loss = torch.tensor(0.0, device=x_formula.device if x_formula is not None else "cpu")
        final_val_loss = torch.tensor(
            0.0, device=x_formula.device if x_formula is not None else "cpu"
        )  # This will be val_final_loss

        # For validation, x_struct_for_processing is always the original_x_struct (no dropout)
        x_struct_for_processing = original_x_struct

        # --- Self-Supervised Learning (SSL) Calculations (if enabled) ---
        val_ssl_loss_contribution = torch.tensor(0.0, device=final_val_loss.device)
        val_sum_ssl_raw_loss = torch.tensor(0.0, device=final_val_loss.device)

        if self.enable_self_supervised_training:
            # Get SSL-specific embeddings using the (non-dropped) structure input
            if self.with_structure:
                h_formula_ssl, h_structure_ssl, _, _ = self.encoder(x_formula, x_struct_for_processing)
            else:
                # When not using structure, FoundationEncoder takes only x_formula
                # and returns (latent, task_representation)
                h_formula_ssl, _ = self.encoder(x_formula)  # x_struct_for_processing is None or not used
                h_structure_ssl = None

            # 3a. Masked Feature Modeling (MFM)
            if self.w.get("mfm", 0) > 0:
                encoder_fn = self.encoder.encode_masked
                mfm_loss, mfm_logs = self.ssl_module.compute_masked_feature_loss(
                    encoder_fn, x_formula, x_struct_for_processing
                )
                for k, v_loss in mfm_logs.items():
                    val_logs[f"val_raw_{k}"] = v_loss.detach()

                static_weight_mfm = self.w.get("mfm", 1.0)
                weighted_mfm_loss = static_weight_mfm * mfm_loss
                val_ssl_loss_contribution += weighted_mfm_loss.detach()
                val_sum_ssl_raw_loss += mfm_loss.detach()
                val_logs["val_mfm_raw_loss"] = mfm_loss.detach()
                val_logs["val_mfm_final_loss_contrib"] = weighted_mfm_loss.detach()

            if self.with_structure and h_structure_ssl is not None and self.w.get("contrastive", 0) > 0:
                contrastive_loss = self.ssl_module.compute_contrastive_loss(h_formula_ssl, h_structure_ssl)
                static_weight_contrastive = self.w.get("contrastive", 1.0)
                weighted_contrastive_loss = static_weight_contrastive * contrastive_loss
                val_ssl_loss_contribution += weighted_contrastive_loss.detach()
                val_sum_ssl_raw_loss += contrastive_loss.detach()
                val_logs["val_contrastive_raw_loss"] = contrastive_loss.detach()
                val_logs["val_contrastive_final_loss_contrib"] = weighted_contrastive_loss.detach()
            elif self.with_structure and self.w.get("contrastive", 0) > 0:
                val_logs["val_contrastive_raw_loss"] = 0.0
                val_logs["val_contrastive_final_loss_contrib"] = 0.0

            if (
                self.with_structure
                and h_structure_ssl is not None
                and original_x_struct is not None
                and self.w.get("cross_recon", 0) > 0
            ):
                cross_recon_loss = self.ssl_module.compute_cross_reconstruction_loss(
                    h_formula_ssl, h_structure_ssl, x_formula, original_x_struct
                )
                static_weight_cross_recon = self.w.get("cross_recon", 1.0)
                weighted_cross_recon_loss = static_weight_cross_recon * cross_recon_loss
                val_ssl_loss_contribution += weighted_cross_recon_loss.detach()
                val_sum_ssl_raw_loss += cross_recon_loss.detach()
                val_logs["val_cross_recon_raw_loss"] = cross_recon_loss.detach()
                val_logs["val_cross_recon_final_loss_contrib"] = weighted_cross_recon_loss.detach()
            elif self.with_structure and self.w.get("cross_recon", 0) > 0:
                val_logs["val_cross_recon_raw_loss"] = 0.0
                val_logs["val_cross_recon_final_loss_contrib"] = 0.0

        if self.enable_self_supervised_training:
            val_logs["val_sum_ssl_raw_loss"] = val_sum_ssl_raw_loss
            val_logs["val_final_ssl_loss"] = val_ssl_loss_contribution.detach()
            sum_val_raw_loss += val_sum_ssl_raw_loss  # Add to total raw sum
            final_val_loss += val_ssl_loss_contribution  # Add to total final sum

        # --- Supervised Task Calculations ---
        val_supervised_loss_contribution = torch.tensor(0.0, device=final_val_loss.device)
        val_sum_supervised_raw_loss = torch.tensor(0.0, device=final_val_loss.device)

        # 4. Prepare input for the standard forward pass
        if self.with_structure:
            forward_input = (x_formula, x_struct_for_processing)
        else:
            forward_input = x_formula

        # 5. Get predictions from the forward method
        preds = self(forward_input, task_sequence_data_batch)  # Pass task_sequence_data_batch dictionary

        # 6. Calculate supervised task losses
        raw_val_supervised_losses = {}
        for name, pred_tensor in preds.items():
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue

            head = self.task_heads[name]
            target = y_dict_batch[name]
            sample_mask = task_masks_batch.get(name)
            if sample_mask is None:
                logger.warning(f"Mask not found for task {name} in validation_step. Assuming all valid.")
                sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            raw_loss_t, _ = head.compute_loss(pred_tensor, target, sample_mask)
            raw_val_supervised_losses[name] = raw_loss_t
            val_sum_supervised_raw_loss += raw_loss_t.detach()
            val_logs[f"val_{name}_raw_loss"] = raw_loss_t.detach()

        val_logs["val_sum_supervised_raw_loss"] = val_sum_supervised_raw_loss
        sum_val_raw_loss += val_sum_supervised_raw_loss  # Add to total raw sum

        # Apply uncertainty weighting for supervised tasks
        for name, raw_loss_t in raw_val_supervised_losses.items():
            if name in self.task_log_sigmas:  # This task uses learnable uncertainty
                current_log_sigma_t = self.task_log_sigmas[name]  # Use learned value
                static_weight = self.w.get(name, 1.0)

                precision_factor_t = torch.exp(-2 * current_log_sigma_t)

                task_loss_data_term = static_weight * 0.5 * precision_factor_t * raw_loss_t
                task_loss_reg_term = current_log_sigma_t
                final_task_loss_component = task_loss_data_term + task_loss_reg_term

                val_supervised_loss_contribution += final_task_loss_component.detach()

                val_logs[f"val_{name}_sigma_t"] = torch.exp(current_log_sigma_t).detach()
                val_logs[f"val_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            else:  # Should not happen
                logger.warning(
                    f"Task {name} found in validation predictions but not in task_log_sigmas. Using static weight only."
                )
                static_weight = self.w.get(name, 1.0)
                final_task_loss_component = static_weight * raw_loss_t
                val_supervised_loss_contribution += final_task_loss_component.detach()
                val_logs[f"val_{name}_final_loss_contrib"] = final_task_loss_component.detach()

        val_logs["val_final_supervised_loss"] = val_supervised_loss_contribution.detach()
        final_val_loss += val_supervised_loss_contribution  # Add to total final sum

        # 7. Log metrics
        val_logs["val_sum_raw_loss"] = sum_val_raw_loss  # Total raw loss (supervised + SSL)

        # Log detailed metrics without progress bar
        self.log_dict(val_logs, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # Log the main validation loss with progress bar (this is the one for callbacks)
        self.log("val_final_loss", final_val_loss.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return None  # As per Lightning best practices when using self.log_dict

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
        additional_output: bool = False,
        tasks_to_predict: Optional[List[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Prediction step. Performs standard forward pass assuming only formula input
        and then processes raw outputs using each task head's `predict` method.

        Even if the model is configured with `with_structure=True`, this step
        simulates the deployment scenario where only formula information (`x_formula`)
        is available. It explicitly passes `None` for the structure input to the
        forward method.

        Parameters
        ----------
        batch : tuple
            Typically contains (x_formula, [ignored targets], [ignored masks], task_sequence_data_batch)
            or just (x_formula, task_sequence_data_batch) or similar variations. `batch[0]` is always
            assumed to be `x_formula`. `batch[3]` is task_sequence_data_batch.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the dataloader (if multiple).
        additional_output : bool, optional
            If True, task heads will return additional prediction information
            (e.g., probabilities for classification). Defaults to False.
        tasks_to_predict : list[str] | None, optional
            A list of task names to predict. If None (default), predicts all enabled tasks.
            If a task in the list is not found or not enabled, a warning is logged and it's skipped.

        Returns
        -------
        dict[str, torch.Tensor]
            A flat dictionary where keys are prefixed with the snake_case task name
            (e.g., "task_name_labels", "task_name_probabilities") and values are
            the corresponding predictions.
        """
        # 1. Unpack batch - Assume batch[0] is always x_formula for prediction
        # Batch structure from predict_dataloader: model_input_x, sample_y_dict, sample_task_masks_dict, sample_task_sequence_data_dict
        # model_input_x is x_formula for predict_set=True in dataset
        x_formula = batch[0]
        if not isinstance(x_formula, torch.Tensor):  # x_formula should not be a tuple here
            # If model is with_structure, x_formula might be (formula_tensor, None) from dataset
            # but predict_step expects just formula_tensor from batch[0]
            if isinstance(x_formula, tuple) and x_formula[1] is None:
                x_formula = x_formula[0]
            else:
                raise TypeError(f"Expected batch[0] to be a Tensor (x_formula), but got {type(x_formula)}")

        # Sequence input data is now a dictionary
        task_sequence_data_batch = batch[3] if len(batch) > 3 else {}  # Default to empty dict if not provided

        # 2. Prepare input for the raw forward pass, always treating structure as None for predict_step
        if self.with_structure:
            # If model uses structure, pass None explicitly during prediction
            raw_forward_input = (x_formula, None)
        else:
            # If model doesn't use structure, just pass formula
            raw_forward_input = x_formula

        # 3. Get raw predictions from the model's forward method
        raw_preds = self(raw_forward_input, task_sequence_data_batch)  # Pass task_sequence_data_batch dictionary

        # 4. Process raw predictions using each task head's `predict` method
        final_predictions = {}

        tasks_to_iterate = []
        if tasks_to_predict is None:
            # Predict all tasks present in raw_preds that have a corresponding head
            tasks_to_iterate = [(name, tensor) for name, tensor in raw_preds.items() if name in self.task_heads]
        else:
            # Predict only specified tasks, after validation
            for task_name in tasks_to_predict:
                if task_name not in self.task_heads:
                    logger.warning(
                        f"Task '{task_name}' requested for prediction but not found or not enabled in the model. Skipping."
                    )
                    continue
                if task_name not in raw_preds:
                    logger.warning(
                        f"Task '{task_name}' requested for prediction, found in model heads, "
                        f"but not present in raw model output. Skipping."
                    )
                    continue
                tasks_to_iterate.append((task_name, raw_preds[task_name]))

        for task_name, raw_pred_tensor in tasks_to_iterate:
            head = self.task_heads[task_name]
            # The head.predict() method now handles snake_case prefixing
            processed_pred_dict = head.predict(raw_pred_tensor, additional=additional_output)
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

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, config: OptimizerConfig) -> _LRScheduler | None:
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

    def configure_optimizers(self) -> list[dict[str, Any] | torch.optim.Optimizer]:
        """Configure optimizers for all parameter groups."""
        optimizers_and_schedulers = []

        # 1. Encoder parameters
        if self.with_structure:
            encoder_params = list(self.encoder.parameters())
        else:
            encoder_params = list(self.encoder.parameters())

        encoder_opt = self._create_optimizer(encoder_params, self.shared_block_optimizer)
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

        # 2. Task head parameters
        for name, head in self.task_heads.items():
            config = self.task_configs_map[name]

            # If the task has a specific optimizer config, use it
            if config.optimizer:
                task_opt = self._create_optimizer(head.parameters(), config.optimizer)
                task_sched = self._create_scheduler(task_opt, config.optimizer)

                if task_sched:
                    optimizers_and_schedulers.append(
                        {
                            "optimizer": task_opt,
                            "lr_scheduler": {
                                "scheduler": task_sched,
                                "monitor": config.optimizer.monitor,
                                "interval": config.optimizer.interval,
                                "frequency": config.optimizer.frequency,
                            },
                        }
                    )
                else:
                    optimizers_and_schedulers.append(task_opt)
            else:
                # Use default optimizer with task-specific settings
                default_config = OptimizerConfig()  # Use default settings
                task_opt = self._create_optimizer(head.parameters(), default_config)
                task_sched = self._create_scheduler(task_opt, default_config)

                if task_sched:
                    optimizers_and_schedulers.append(
                        {
                            "optimizer": task_opt,
                            "lr_scheduler": {
                                "scheduler": task_sched,
                                "monitor": default_config.monitor,
                                "interval": default_config.interval,
                                "frequency": default_config.frequency,
                            },
                        }
                    )
                else:
                    optimizers_and_schedulers.append(task_opt)

        # 3. Self-supervised module parameters (if enabled)
        if self.enable_self_supervised_training and hasattr(self, "ssl_module"):
            ssl_opt = self._create_optimizer(self.ssl_module.parameters(), self.shared_block_optimizer)
            ssl_sched = self._create_scheduler(ssl_opt, self.shared_block_optimizer)

            if ssl_sched:
                optimizers_and_schedulers.append(
                    {
                        "optimizer": ssl_opt,
                        "lr_scheduler": {
                            "scheduler": ssl_sched,
                            "monitor": self.shared_block_optimizer.monitor,
                            "interval": self.shared_block_optimizer.interval,
                            "frequency": self.shared_block_optimizer.frequency,
                        },
                    }
                )
            else:
                optimizers_and_schedulers.append(ssl_opt)

        return optimizers_and_schedulers
