# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


"""
Module: flexible_multi_task_model
---------------------------------

A flexible multi-task model with foundation model capabilities.

Tensor shape legend (used across all docstrings):
* **B** – batch size
* **L** – sequence length (e.g. number of temperature points)
* **D** – latent / embedding feature dimension
"""

import logging  # Added
from typing import Any, List, Optional  # Added List, Optional

import lightning as L
import pandas as pd  # Added
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from .components.foundation_encoder import FoundationEncoder, MultiModalFoundationEncoder
from .components.self_supervised import SelfSupervisedModule
from .model_config import (
    ClassificationTaskConfig,
    OptimizerConfig,
    RegressionTaskConfig,
    SequenceTaskConfig,
    TaskType,
)
from .task_head.classification import ClassificationHead
from .task_head.regression import RegressionHead
from .task_head.sequence import create_sequence_head
from .task_head.sequence.base import SequenceBaseHead

logger = logging.getLogger(__name__)  # Added


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
        # 1. Unpack batch data
        # y_dict_batch, task_masks_batch, task_sequence_data_batch are now dictionaries keyed by task_name
        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        total_loss = 0.0
        logs = {}

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
            logs["train_modality_dropout_applied"] = True  # CHANGED
        elif self.enable_self_supervised_training and self.with_structure:
            logs["train_modality_dropout_applied"] = False  # CHANGED

        # --- Self-Supervised Learning (SSL) Calculations ---
        if self.enable_self_supervised_training:
            # We need non-masked embeddings for contrastive/cross-recon later
            # Get these based on potentially dropped structure input
            h_formula_ssl, h_structure_ssl, _, _ = self.encoder(x_formula, x_struct_for_processing)

            # 4a. Masked Feature Modeling (MFM)
            if self.w.get("mfm", 0) > 0:
                # Pass the encoder function that handles masking internally
                encoder_fn = self.encoder.encode_masked
                # Compute MFM loss using the potentially dropped structure input
                mfm_loss, mfm_logs = self.ssl_module.compute_masked_feature_loss(
                    encoder_fn, x_formula, x_struct_for_processing
                )
                weighted_mfm_loss = self.w["mfm"] * mfm_loss
                total_loss += weighted_mfm_loss
                # Log MFM specific losses (e.g., formula_mfm, struct_mfm if applicable)
                logs.update({f"train_{k}": v.detach() for k, v in mfm_logs.items()})
                logs["train_mfm_loss_weighted"] = weighted_mfm_loss.detach()

            # 4b. Contrastive Loss
            if (
                self.with_structure
                and h_structure_ssl is not None  # Only if structure wasn't dropped/missing
                and self.w.get("contrastive", 0) > 0
            ):
                contrastive_loss = self.ssl_module.compute_contrastive_loss(h_formula_ssl, h_structure_ssl)
                weighted_contrastive_loss = self.w["contrastive"] * contrastive_loss
                total_loss += weighted_contrastive_loss
                logs["train_contrastive_loss"] = contrastive_loss.detach()
                logs["train_contrastive_loss_weighted"] = weighted_contrastive_loss.detach()
            elif self.with_structure and self.w.get("contrastive", 0) > 0:
                # Log zero if structure was dropped or missing
                logs["train_contrastive_loss"] = 0.0
                logs["train_contrastive_loss_weighted"] = 0.0

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
                weighted_cross_recon_loss = self.w["cross_recon"] * cross_recon_loss
                total_loss += weighted_cross_recon_loss
                logs["train_cross_recon_loss"] = cross_recon_loss.detach()
                logs["train_cross_recon_loss_weighted"] = weighted_cross_recon_loss.detach()
            elif self.with_structure and self.w.get("cross_recon", 0) > 0:
                # Log zero if structure was dropped or missing
                logs["train_cross_recon_loss"] = 0.0
                logs["train_cross_recon_loss_weighted"] = 0.0

        # --- Supervised Task Calculations ---
        # 5. Prepare input for the standard forward pass
        # Use the structure input *after* potential modality dropout for consistent forward pass
        if self.with_structure:
            forward_input = (x_formula, x_struct_for_processing)
        else:
            forward_input = x_formula

        # 6. Get predictions from the forward method
        preds = self(forward_input, task_sequence_data_batch)  # Pass task_sequence_data_batch dictionary

        # 7. Calculate supervised task losses
        # task_mask_indices = {name: i for i, name in enumerate(self.task_heads.keys())} # Not needed anymore
        for name, pred_tensor in preds.items():
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue  # Skip tasks without targets or disabled tasks

            head = self.task_heads[name]
            target = y_dict_batch[name]

            # Get mask for the current task from the task_masks_batch dictionary
            sample_mask = task_masks_batch.get(name)
            if sample_mask is None:  # Should not happen if dataset prepares it
                # Default: all samples are valid if no specific mask provided
                logger.warning(f"Mask not found for task {name} in training_step. Assuming all valid.")
                sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            # Ensure mask has a compatible shape for broadcasting, e.g. [B, 1] or [B, L] or [B, L, 1]
            # The mask from dataset is already [B,1] (bool)
            # For sequence tasks, compute_loss in SequenceBaseHead handles mask expansion if needed.

            # Compute loss using the head's method
            loss, _ = head.compute_loss(pred_tensor, target, sample_mask)  # Changed pred to pred_tensor

            # Apply task-specific weight and accumulate
            task_weight = self.w.get(name, 1.0)
            weighted_loss = task_weight * loss
            total_loss += weighted_loss

            # Log individual task losses (unweighted and weighted)
            logs[f"train_{name}_loss"] = loss.detach()
            logs[f"train_{name}_loss_weighted"] = weighted_loss.detach()

        # 8. Log total loss and other aggregated metrics
        logs["train_total_loss"] = total_loss.detach()
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # Manual optimization
        self.manual_backward(total_loss)
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            for opt in optimizers:
                opt.step()
                opt.zero_grad()
        else:  # Single optimizer case
            optimizers.step()
            optimizers.zero_grad()

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
        total_val_loss = 0.0  # Accumulates unweighted losses
        logs = {}

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

        # For validation, x_struct_for_processing is always the original_x_struct (no dropout)
        x_struct_for_processing = original_x_struct

        # --- Self-Supervised Learning (SSL) Calculations (if enabled) ---
        if self.enable_self_supervised_training:
            # Get SSL-specific embeddings using the (non-dropped) structure input
            h_formula_ssl, h_structure_ssl, _, _ = self.encoder(x_formula, x_struct_for_processing)

            # 3a. Masked Feature Modeling (MFM)
            if self.w.get("mfm", 0) > 0:  # Check if MFM is weighted, implying it's active
                encoder_fn = self.encoder.encode_masked
                mfm_loss, mfm_logs = self.ssl_module.compute_masked_feature_loss(
                    encoder_fn, x_formula, x_struct_for_processing
                )
                total_val_loss += mfm_loss.detach()  # Add unweighted MFM loss
                logs["val_mfm_loss"] = mfm_loss.detach()
                logs.update({f"val_{k}": v.detach() for k, v in mfm_logs.items()})

            # 3b. Contrastive Loss
            if (
                self.with_structure
                and h_structure_ssl is not None
                and self.w.get("contrastive", 0) > 0  # Check if contrastive is weighted
            ):
                contrastive_loss = self.ssl_module.compute_contrastive_loss(h_formula_ssl, h_structure_ssl)
                total_val_loss += contrastive_loss.detach()  # Add unweighted contrastive loss
                logs["val_contrastive_loss"] = contrastive_loss.detach()
            elif self.with_structure and self.w.get("contrastive", 0) > 0:
                logs["val_contrastive_loss"] = 0.0  # Log zero if not applicable

            # 3c. Cross-Reconstruction Loss
            if (
                self.with_structure
                and h_structure_ssl is not None
                and original_x_struct is not None  # Target for reconstruction must exist
                and self.w.get("cross_recon", 0) > 0  # Check if cross-recon is weighted
            ):
                cross_recon_loss = self.ssl_module.compute_cross_reconstruction_loss(
                    h_formula_ssl, h_structure_ssl, x_formula, original_x_struct
                )
                total_val_loss += cross_recon_loss.detach()  # Add unweighted cross-recon loss
                logs["val_cross_recon_loss"] = cross_recon_loss.detach()
            elif self.with_structure and self.w.get("cross_recon", 0) > 0:
                logs["val_cross_recon_loss"] = 0.0  # Log zero if not applicable

        # --- Supervised Task Calculations ---
        # 4. Prepare input for the standard forward pass
        if self.with_structure:
            forward_input = (x_formula, x_struct_for_processing)
        else:
            forward_input = x_formula

        # 5. Get predictions from the forward method
        preds = self(forward_input, task_sequence_data_batch)  # Pass task_sequence_data_batch dictionary

        # 6. Calculate supervised task losses (unweighted for validation)
        # task_mask_indices = {name: i for i, name in enumerate(self.task_heads.keys())} # Not needed
        for name, pred_tensor in preds.items():
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue

            head = self.task_heads[name]
            target = y_dict_batch[name]

            # Get mask for the current task
            sample_mask = task_masks_batch.get(name)
            if sample_mask is None:
                logger.warning(f"Mask not found for task {name} in validation_step. Assuming all valid.")
                sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            # Compute unweighted loss
            loss, _ = head.compute_loss(pred_tensor, target, sample_mask)  # Changed pred to pred_tensor
            total_val_loss += loss.detach()  # Accumulate unweighted task loss

            # Log individual task validation losses (unweighted)
            logs[f"val_{name}_loss"] = loss.detach()

        # 7. Log total validation loss and other metrics
        logs["val_total_loss"] = total_val_loss  # total_val_loss is already detached
        # Log on epoch end, disable progress bar for validation
        self.log_dict(logs, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        # As per Lightning best practices when using self.log_dict, return None
        return None

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
