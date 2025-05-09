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

from typing import Any

import lightning as L
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
from .task_head import create_task_heads
from .task_head.sequence.base import SequenceBaseHead


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

    def _init_task_heads(self):
        """Initialize task heads based on configurations."""
        deposit_dim = next(
            (c.dims[0] for c in self.task_configs if hasattr(c, "dims")),
            self.shared_block_dims[-1],  # Default to latent dimension
        )
        latent_dim = self.shared_block_dims[-1]

        self.task_heads = create_task_heads(
            task_configs=self.task_configs,
            deposit_dim=deposit_dim,
            latent_dim=latent_dim,
        )

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
        temps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]
            Input tensor(s). If structure fusion is enabled, this should be a tuple
            of (formula_tensor, structure_tensor).
        temps : torch.Tensor | None
            Temperature or sequence points, required for sequence tasks.

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
                if temps is not None:
                    outputs[name] = head(h_task, temps)
            else:
                outputs[name] = head(h_task)

        return outputs

    def training_step(self, batch, batch_idx):
        """
        Training step implementation with multi-component loss calculation.

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict, task_masks, temps)
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Total weighted loss
        """
        # Unpack batch
        x, y_dict, task_masks, temps = batch

        # Unpack inputs
        if self.with_structure and isinstance(x, (list, tuple)):
            x_formula, x_struct = x
        else:
            x_formula, x_struct = x, None

        # Get encodings for supervised and self-supervised learning
        if self.with_structure:
            # For multi-modal, we need all intermediate representations for self-supervised learning
            h_formula, h_structure, h_fused, h_task = self.encoder(x_formula, x_struct)

            # Apply modality dropout during self-supervised training
            if (
                self.enable_self_supervised_training
                and h_structure is not None
                and torch.rand(1).item() < self.mod_dropout_p
            ):
                h_structure, x_struct = None, None
                # Update input x to reflect modality dropout
                if isinstance(x, (list, tuple)):
                    x = (x_formula, None)
        else:
            # For single modality
            h_latent, h_task = self.encoder(x_formula)
            h_formula, h_structure = h_latent, None

        # Forward pass for task predictions
        preds = self(x, temps)

        # Calculate task losses
        total_loss = 0.0
        logs = {}

        # Get task mask indices mapping
        task_mask_indices = {name: i for i, name in enumerate(self.task_heads.keys())}

        # Process each prediction
        for name, pred in preds.items():
            if name not in y_dict:
                continue  # Skip tasks without targets

            head = self.task_heads[name]
            target = y_dict[name]

            # Get mask for this task
            mask_idx = task_mask_indices.get(name)
            if mask_idx is not None and mask_idx < task_masks.shape[1]:
                # Create sample mask tensor matching target shape
                sample_mask = task_masks[:, mask_idx].view(-1, 1)

                # Handle sequence task masks
                if isinstance(head, SequenceBaseHead) and len(target.shape) > 2:
                    sample_mask = sample_mask.unsqueeze(1).expand(-1, target.shape[1], -1)
            else:
                # Default: all samples valid
                sample_mask = torch.ones_like(target)

            # Compute loss
            loss, per_dim_loss = head.compute_loss(pred, target, sample_mask)

            # Apply task-specific weight and accumulate
            task_weight = self.w.get(name, 1.0)
            weighted_loss = task_weight * loss
            total_loss += weighted_loss

            # Log task losses
            logs[f"train_{name}_loss"] = loss.detach()

        # Self-supervised losses
        if self.enable_self_supervised_training:
            total_loss = self._add_self_supervised_losses(total_loss, logs, h_formula, h_structure, x_formula, x_struct)

        # Log metrics
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def _add_self_supervised_losses(
        self,
        total_loss: torch.Tensor,
        logs: dict,
        h_formula: torch.Tensor,
        h_structure: torch.Tensor | None,
        x_formula: torch.Tensor,
        x_structure: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add self-supervised losses to the total loss."""
        # Only continue if self-supervised training is enabled and modules are initialized
        if not self.enable_self_supervised_training or not hasattr(self, "ssl_module"):
            return total_loss

        # 1. Masked Feature Modeling loss
        if self.w.get("mfm", 0) > 0 and x_formula is not None:
            # Create encoder function for masked feature modeling
            encoder_fn = self.encoder.encode_masked

            # Compute MFM loss using the self-supervised module
            mfm_loss, mfm_logs = self.ssl_module.compute_masked_feature_loss(encoder_fn, x_formula, x_structure)

            # Add weighted loss
            total_loss += self.w["mfm"] * mfm_loss
            logs.update({f"train_{k}": v for k, v in mfm_logs.items()})

        # 2. Contrastive loss (only for multi-modal)
        if self.with_structure and h_structure is not None and self.w.get("contrastive", 0) > 0:
            contrastive_loss = self.ssl_module.compute_contrastive_loss(h_formula, h_structure)
            total_loss += self.w["contrastive"] * contrastive_loss
            logs["train_contrastive"] = contrastive_loss.detach()

        # 3. Cross-reconstruction loss (only for multi-modal)
        if (
            self.with_structure
            and h_structure is not None
            and x_structure is not None
            and self.w.get("cross_recon", 0) > 0
        ):
            cross_recon_loss = self.ssl_module.compute_cross_reconstruction_loss(
                h_formula, h_structure, x_formula, x_structure
            )
            total_loss += self.w["cross_recon"] * cross_recon_loss
            logs["train_cross_recon"] = cross_recon_loss.detach()

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step implementation.

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict, task_masks, temps)
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Total validation loss
        """
        # Unpack batch
        x, y_dict, task_masks, temps = batch

        # Forward pass
        preds = self(x, temps)

        # Calculate task losses
        total_loss = 0.0
        logs = {}

        # Get task mask indices mapping
        task_mask_indices = {name: i for i, name in enumerate(self.task_heads.keys())}

        # Process each prediction
        for name, pred in preds.items():
            if name not in y_dict:
                continue  # Skip tasks without targets

            head = self.task_heads[name]
            target = y_dict[name]

            # Get mask for this task
            mask_idx = task_mask_indices.get(name)
            if mask_idx is not None and mask_idx < task_masks.shape[1]:
                # Create sample mask tensor matching target shape
                sample_mask = task_masks[:, mask_idx].view(-1, 1)

                # Handle sequence task masks
                if isinstance(head, SequenceBaseHead) and len(target.shape) > 2:
                    sample_mask = sample_mask.unsqueeze(1).expand(-1, target.shape[1], -1)
            else:
                # Default: all samples valid
                sample_mask = torch.ones_like(target)

            # Compute loss
            loss, per_dim_loss = head.compute_loss(pred, target, sample_mask)

            # Accumulate loss (unweighted for validation)
            total_loss += loss

            # Log task losses
            logs[f"val_{name}_loss"] = loss.detach()

        # Log metrics
        self.log_dict(logs, prog_bar=True, on_epoch=True)

        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step."""
        x, _, _, temps = batch
        return self(x, temps)

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
