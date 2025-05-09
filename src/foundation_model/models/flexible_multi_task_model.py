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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from .components.gated_fusion import GatedFusion
from .components.structure_encoder import StructureEncoder
from .fc_layers import LinearBlock
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

    5. Pre-training Mechanisms (optional):
       When pretrain=True, enables self-supervised learning objectives:
       - Masked Feature Modeling (MFM): Similar to BERT's masked language modeling
       - Contrastive Learning: Aligns representations from different modalities
       - Cross-reconstruction: Reconstructs one modality from another

    Training Process:
    - Each batch's loss includes task-specific losses and optional pre-training losses
    - Uses manual optimization to support complex optimizer configurations
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
        Probability of modality dropout. During pre-training, there's this probability of
        randomly dropping the structure modality, forcing the model to learn to handle
        single-modality cases. Only relevant when with_structure=True and pretrain=True.
    pretrain : bool
        Whether to use pre-training objectives (masked feature modeling, contrastive learning,
        and cross-reconstruction). When True, these additional losses are added to task-specific losses.
    loss_weights : dict[str, float] | None
        Weight coefficients dictionary for balancing different loss components in the total loss.
        Includes the following keys:
        - "attr": Weight for attribute tasks (regression/classification) loss, default 1.0
        - "seq": Weight for sequence prediction task loss, default 1.0
        - "con": Weight for contrastive learning loss (only used when pretrain=True and with_structure=True), default 1.0
        - "cross": Weight for cross-reconstruction loss (only used when pretrain=True and with_structure=True), default 1.0
        - "mask": Weight for masked feature reconstruction loss (only used when pretrain=True), default 1.0
        Example: {"attr": 1.0, "seq": 0.5, "con": 0.1, "cross": 0.1, "mask": 0.2}
    mask_ratio : float
        Ratio of features to be randomly masked in masked feature modeling.
        Typical value is 0.15. Only used when pretrain=True.
    temperature : float
        Temperature coefficient in contrastive learning. Controls the smoothness of
        the similarity distribution, with smaller values increasing contrast.
        Typical value is 0.07. Only used when pretrain=True and with_structure=True.
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
        # Pre-training options
        pretrain: bool = False,
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

        # Store all parameters as instance variables
        self.shared_block_dims = shared_block_dims
        self.task_configs = task_configs

        # Normalization/residual options
        self.norm_shared = norm_shared
        self.residual_shared = residual_shared

        # Structure fusion parameters
        self.with_structure = with_structure
        self.struct_block_dims = struct_block_dims
        self.mod_dropout_p = modality_dropout_p

        # Pre-training parameters
        self.pretrain = pretrain
        self.mask_ratio = mask_ratio
        self.tau = temperature

        # Optimizer configurations
        self.shared_block_optimizer = shared_block_optimizer or OptimizerConfig(weight_decay=1e-2)

        # Store task configurations by name for easy reference
        self.task_configs_map = {cfg.name: cfg for cfg in self.task_configs}

        # Initialize task loss weights
        # 1. First set default weight 1.0 for each enabled task
        self.w = {cfg.name: 1.0 for cfg in self.task_configs if cfg.enabled}
        # 2. Add default weights for pre-training losses
        self.w.update({"con": 1.0, "cross": 1.0, "mask": 1.0})
        # 3. Apply user-provided weights to override defaults
        if loss_weights:
            self.w.update(loss_weights)

        # Initialize model components
        self._init_shared_layers()
        self._init_task_heads()

        # Task type tracking
        self.has_regression = any(tc.type == TaskType.REGRESSION for tc in task_configs if tc.enabled)
        self.has_classification = any(tc.type == TaskType.CLASSIFICATION for tc in task_configs if tc.enabled)
        self.has_sequence = any(tc.type == TaskType.SEQUENCE for tc in task_configs if tc.enabled)

        # Enable automatic optimization
        self.automatic_optimization = True

        # Initialize weights
        self._init_weights()

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

    def _init_shared_layers(self):
        """Initialize shared foundation encoder layers."""
        # Shared encoder (foundation model component)
        self.shared = LinearBlock(
            self.shared_block_dims,
            normalization=self.norm_shared,
            residual=self.residual_shared,
        )

        # Use task_configs[0].dims[0] as the deposit dimension
        if len(self.task_configs) > 0 and hasattr(self.task_configs[0], "dims"):
            deposit_dim = self.task_configs[0].dims[0]
        else:
            deposit_dim = self.shared_block_dims[-1]  # Same dimension as latent by default

        # Deposit layer serves two purposes:
        # 1. Acts as a buffer between shared encoder and task heads
        # 2. Provides an extension point for future continue learning capabilities
        #    where shared layers can be frozen while deposit layers are extended/trained
        self.deposit = nn.Sequential(
            nn.Linear(self.shared_block_dims[-1], deposit_dim),
            nn.Tanh(),
        )

        # Structure encoder and fusion
        if self.with_structure:
            if not self.struct_block_dims or self.struct_block_dims[-1] != self.shared_block_dims[-1]:
                raise ValueError("struct_block_dims must be provided and last dim equal to formula latent dim")
            self.struct_enc = StructureEncoder(
                self.struct_block_dims[0],
                self.struct_block_dims[1:],
                norm=self.norm_shared,
                residual=self.residual_shared,
            )
            self.fusion = GatedFusion(self.shared_block_dims[-1])

        # Pre-training decoders
        if self.pretrain:
            latent_dim = self.shared_block_dims[-1]
            self.dec_formula = nn.Linear(latent_dim, self.shared_block_dims[0], bias=False)
            if self.with_structure:
                self.dec_struct = nn.Linear(latent_dim, self.struct_block_dims[0], bias=False)

    def _init_task_heads(self):
        """Initialize task heads based on configuration."""
        deposit_dim = next(
            (c.dims[0] for c in self.task_configs if hasattr(c, "dims")),
            self.shared_block_dims[-1],  # Same as latent_dim by default
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

    def _encode(
        self, x_formula: torch.Tensor, x_struct: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode inputs through shared layers and structure fusion if enabled.

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
            Currently retained for potential future extensions, but not used directly.
        h_task : torch.Tensor
            Task input representation (B, deposit_dim) after deposit layer.
            Used as input for all task heads.
        """
        # Encode formula input
        h_f = self.shared(x_formula)

        # Apply structure fusion if enabled
        if self.with_structure:
            if x_struct is None:
                h_s = torch.zeros_like(h_f)
                has_s = torch.zeros(h_f.size(0), 1, device=h_f.device)
            else:
                h_s = self.struct_enc(x_struct)
                has_s = torch.ones(h_f.size(0), 1, device=h_f.device)
            # Gated fusion
            h_latent = self.fusion(h_f, h_s, has_s)
        else:
            h_latent = h_f

        # Apply deposit layer
        h_task = self.deposit(h_latent)

        return h_latent, h_task

    @staticmethod
    def _masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute masked MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values.
        tgt : torch.Tensor
            Target values.
        mask : torch.Tensor
            Binary mask for valid values.

        Returns
        -------
        total_loss : torch.Tensor
            Total MSE loss (scalar).
        per_attr : torch.Tensor
            Per-attribute losses.
        """
        losses = F.mse_loss(pred, tgt, reduction="none") * mask
        per_attr = torch.nan_to_num(losses.sum(0) / mask.sum(0), nan=0.0, posinf=0.0, neginf=0.0)
        return losses.sum() / mask.sum(), per_attr

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
        h_latent, h_task = self._encode(x_formula, x_struct)

        # Apply task heads - all task heads use h_task (deposit layer output)
        outputs = {}
        for name, head in self.task_heads.items():
            if isinstance(head, SequenceBaseHead):
                if temps is not None:
                    outputs[name] = head(h_task, temps)
            else:
                outputs[name] = head(h_task)

        return outputs

    # ----------- Pre-training helpers ----------- #
    def _mask_feat(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implement Masked Feature Modeling (MFM).

        Randomly masks a portion of the input features (determined by self.mask_ratio),
        sets the masked positions to 0, and returns both the masked features and the mask
        positions. During pre-training, the model attempts to reconstruct these masked
        feature values, similar to the Masked Language Modeling technique in BERT.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor

        Returns
        -------
        x_masked : torch.Tensor
            Masked feature tensor with the same shape as x
        mask : torch.Tensor
            Binary mask tensor indicating which positions were masked (1=masked)
        """
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def _contrastive(self, h_f: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss between formula and structure representations.

        The goal of contrastive learning is to bring different modal representations
        (formula and structure) of the same sample closer in the feature space,
        while pushing representations of different samples apart. Process steps:

        1. Apply L2 normalization to representations, placing them on a unit hypersphere
        2. Calculate cosine similarity matrix of normalized representations, divided by temperature tau
        3. Compute bidirectional cross-entropy loss (formula→structure and structure→formula), with diagonal target
        4. Return the average of both directional losses

        This contrastive learning method encourages the model to learn aligned
        representations across modalities, improving the quality of multi-modal data representation.

        Parameters
        ----------
        h_f : torch.Tensor
            Formula representation, shape (B,D)
        h_s : torch.Tensor
            Structure representation, shape (B,D)

        Returns
        -------
        torch.Tensor
            Scalar contrastive loss
        """
        h_f = F.normalize(h_f, dim=-1)
        h_s = F.normalize(h_s, dim=-1)
        logits = (h_f @ h_s.T) / self.tau
        tgt = torch.arange(h_f.size(0), device=h_f.device)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))

    # ----------- Lightning methods ----------- #
    def training_step(self, batch, batch_idx):
        """
        Training step implementation with multi-component loss calculation.

        This method implements the full training step, including:
        1. Multi-task loss calculation (regression, classification, sequence)
        2. Pre-training losses when pretrain=True (contrastive, cross-reconstruction, MFM)

        The total loss is a weighted sum of all enabled loss components, with weights
        specified by the loss_weights parameter during initialization. The following
        weights are applied to different loss components:
        - Task-specific weights: Each task uses its name as the key (e.g., self.w["band_gap"])
        - Pre-training loss weights:
          - self.w["con"]: Weight for contrastive learning loss
          - self.w["cross"]: Weight for cross-reconstruction loss
          - self.w["mask"]: Weight for masked feature modeling loss

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict, task_masks, temps)
            - x: Input features or tuple of (formula, structure) features
            - y_dict: Dictionary mapping task names to target values
            - task_masks: Binary mask indicating valid tasks for each sample
            - temps: Temperature/sequence points for sequence tasks
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Total weighted loss value
        """
        # Unpack batch
        x, y_dict, task_masks, temps = batch

        # Unpack inputs for structure fusion
        if self.with_structure and isinstance(x, (list, tuple)):
            x_formula, x_struct = x
        else:
            x_formula, x_struct = x, None

        # Get raw embeddings for pre-training tasks
        if self.pretrain:
            h_f = self.shared(x_formula)
            h_s = self.struct_enc(x_struct) if (self.with_structure and x_struct is not None) else None

            # Modality dropout during pre-training
            if h_s is not None and torch.rand(1).item() < self.mod_dropout_p:
                h_s, x_struct = None, None
        else:
            h_f, h_s = None, None

        # Forward pass
        preds = self(x, temps)

        # Calculate task losses
        total_loss = 0.0
        logs = {}

        # Get task mask indices mapping (map from task name to index in task_masks)
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

                # Handle sequence task masks (expand mask for sequence dimensions)
                if isinstance(head, SequenceBaseHead) and len(target.shape) > 2:
                    # Expand mask to match sequence dimensions (B, L, D) -> sample_mask is (B, 1)
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

            # Log each task's loss
            logs[f"train_{name}_loss"] = loss.detach()

        # Pre-training losses
        if self.pretrain and h_f is not None:
            # 1) Contrastive loss
            if h_s is not None and self.w.get("con", 0) > 0:
                l_con = self._contrastive(h_f, h_s)
                total_loss += self.w["con"] * l_con
                logs["train_con"] = l_con.detach()

            # 2) Cross-reconstruction loss
            if h_s is not None and self.w.get("cross", 0) > 0:
                l_cross = 0.5 * (
                    F.mse_loss(self.dec_struct(h_f), x_struct) + F.mse_loss(self.dec_formula(h_s), x_formula)
                )
                total_loss += self.w["cross"] * l_cross
                logs["train_cross"] = l_cross.detach()

            # 3) Masked feature modeling
            if self.w.get("mask", 0) > 0:
                xf_mask, mf = self._mask_feat(x_formula)
                l_mask = F.mse_loss(self.dec_formula(self.shared(xf_mask))[mf], x_formula[mf])
                if x_struct is not None:
                    xs_mask, ms = self._mask_feat(x_struct)
                    l_mask_s = F.mse_loss(self.dec_struct(self.struct_enc(xs_mask))[ms], x_struct[ms])
                    l_mask = 0.5 * (l_mask + l_mask_s)
                total_loss += self.w["mask"] * l_mask
                logs["train_mask"] = l_mask.detach()

        # Log metrics
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step implementation.

        This method evaluates the model's performance on validation data, calculating
        losses for all enabled task heads without applying optimization or pre-training losses.

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict, task_masks, temps)
            - x: Input features or tuple of (formula, structure) features
            - y_dict: Dictionary mapping task names to target values
            - task_masks: Binary mask indicating valid tasks for each sample
            - temps: Temperature/sequence points for sequence tasks
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Total validation loss value (unweighted sum of all task losses)
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

            # Log each task's loss
            logs[f"val_{name}_loss"] = loss.detach()

        # Log metrics
        self.log_dict(logs, prog_bar=True, on_epoch=True)

        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step.

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict, task_masks, temps)
        batch_idx : int
            Index of the current batch
        dataloader_idx : int
            Index of the dataloader

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of task predictions
        """
        x, _, _, temps = batch
        return self(x, temps)

    def _create_optimizer(self, params: list[torch.nn.Parameter], config: OptimizerConfig) -> torch.optim.Optimizer:
        """
        Create an optimizer based on the configuration.

        Parameters
        ----------
        params : list[torch.nn.Parameter]
            Parameters to optimize
        config : OptimizerConfig
            Optimizer configuration

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer instance
        """
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
        """
        Create a learning rate scheduler based on the configuration.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to schedule
        config : OptimizerConfig
            Scheduler configuration

        Returns
        -------
        _LRScheduler | None
            Configured scheduler instance or None if scheduler_type is "None"
        """
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

        # 1. Shared parameters (shared encoder + deposit)
        # Note: In future iterations, deposit layer might have its own optimizer
        # to facilitate continue learning scenarios where shared encoder is frozen
        # while deposit layers can be expanded and fine-tuned
        shared_params = list(self.shared.parameters()) + list(self.deposit.parameters())
        shared_opt = self._create_optimizer(shared_params, self.shared_block_optimizer)
        shared_sched = self._create_scheduler(shared_opt, self.shared_block_optimizer)

        if shared_sched:
            optimizers_and_schedulers.append(
                {
                    "optimizer": shared_opt,
                    "lr_scheduler": {
                        "scheduler": shared_sched,
                        "monitor": self.shared_block_optimizer.monitor,
                        "interval": self.shared_block_optimizer.interval,
                        "frequency": self.shared_block_optimizer.frequency,
                    },
                }
            )
        else:
            optimizers_and_schedulers.append(shared_opt)

        # 2. Structure encoder parameters if present
        if self.with_structure:
            # Use shared optimizer config for structure encoder
            struct_opt = self._create_optimizer(self.struct_enc.parameters(), self.shared_block_optimizer)
            struct_sched = self._create_scheduler(struct_opt, self.shared_block_optimizer)

            if struct_sched:
                optimizers_and_schedulers.append(
                    {
                        "optimizer": struct_opt,
                        "lr_scheduler": {
                            "scheduler": struct_sched,
                            "monitor": self.shared_block_optimizer.monitor,
                            "interval": self.shared_block_optimizer.interval,
                            "frequency": self.shared_block_optimizer.frequency,
                        },
                    }
                )
            else:
                optimizers_and_schedulers.append(struct_opt)

        # 3. Task head parameters
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

        return optimizers_and_schedulers
