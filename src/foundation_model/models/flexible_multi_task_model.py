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

    This model provides a shared representation layer (foundation encoder) and
    supports multiple task heads for different prediction tasks, including
    regression, classification, and sequence prediction.

    Parameters
    ----------
    shared_block_dims : list[int]
        Widths of shared MLP layers (foundation encoder output → deposit).
    task_configs : list[RegressionTaskConfig | ClassificationTaskConfig | SequenceTaskConfig]
        List of task configurations.
    norm_shared : bool
        Whether to apply normalization in shared layers.
    residual_shared : bool
        Whether to use residual connections in shared layers.
    shared_block_optimizer : OptimizerConfig | None
        Optimizer configuration for shared foundation encoder and deposit layer.
    with_structure : bool
        Whether to enable structure fusion.
    struct_block_dims : list[int] | None
        Dimensions for structure encoder, if with_structure is True.
    modality_dropout_p : float
        Dropout probability for modality fusion.
    pretrain : bool
        Whether to use pre-training objectives.
    loss_weights : dict[str, float] | None
        Weights for different loss components.
    mask_ratio : float
        Mask ratio for masked feature modeling.
    temperature : float
        Temperature for contrastive learning.
    freeze_encoder : bool
        Whether to freeze the encoder during fine-tuning.
    lora_rank : int
        Rank for LoRA adaptation (0 = off).
    lora_alpha : float
        Alpha scaling factor for LoRA.
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
        # LoRA options
        freeze_encoder: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
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

        # LoRA parameters
        self.freeze_encoder = freeze_encoder
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Optimizer configurations
        self.shared_block_optimizer = shared_block_optimizer or OptimizerConfig(weight_decay=1e-2)

        # Store task configurations by name for easy reference
        self.task_configs_map = {cfg.name: cfg for cfg in self.task_configs}

        # Loss weights (keys: con, cross, mask, attr, seq)
        self.w = {"con": 1, "cross": 1, "mask": 1, "attr": 1, "seq": 1}
        if loss_weights:
            self.w.update(loss_weights)

        # Initialize model components
        self._init_shared_layers()
        self._init_task_heads()

        # Task type tracking
        self.has_regression = any(tc.type == TaskType.REGRESSION for tc in task_configs if tc.enabled)
        self.has_classification = any(tc.type == TaskType.CLASSIFICATION for tc in task_configs if tc.enabled)
        self.has_sequence = any(tc.type == TaskType.SEQUENCE for tc in task_configs if tc.enabled)

        # Manual optimization
        self.automatic_optimization = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Freeze encoder if requested
        if self.freeze_encoder:
            for p in self.shared.parameters():
                p.requires_grad_(False)
            if self.with_structure:
                for p in self.struct_enc.parameters():
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
            deposit_dim = self.shared_block_dims[-1] // 2  # fallback

        # Deposit layer maps from shared latent space to task-specific space
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
            self.shared_block_dims[-1] // 2,
        )
        latent_dim = self.shared_block_dims[-1]

        self.task_heads = create_task_heads(
            task_configs=self.task_configs,
            deposit_dim=deposit_dim,
            latent_dim=latent_dim,
            lora_rank=self.lora_rank if self.lora_rank > 0 else None,
            lora_alpha=self.lora_alpha,
        )

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
        h_task : torch.Tensor
            Task input representation (B, deposit_dim).
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

        # Apply task heads
        outputs = {}
        for name, head in self.task_heads.items():
            if isinstance(head, SequenceBaseHead):
                if temps is not None:
                    outputs[name] = head(h_latent, temps)
            else:
                outputs[name] = head(h_task)

        return outputs

    # ----------- Pre-training helpers ----------- #
    def _mask_feat(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create masked version of features for masked feature modeling."""
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def _contrastive(self, h_f: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between formula and structure representations."""
        h_f = F.normalize(h_f, dim=-1)
        h_s = F.normalize(h_s, dim=-1)
        logits = (h_f @ h_s.T) / self.tau
        tgt = torch.arange(h_f.size(0), device=h_f.device)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))

    # ----------- Lightning methods ----------- #
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Unpack batch
        x, y_attr, mask_attr, temps, y_seq, mask_seq = batch

        # Get optimizers
        optimizers = self.optimizers()
        opt_shared, opt_task = optimizers[0], optimizers[1]
        opt_seq = optimizers[2] if len(optimizers) > 2 else None

        # Zero gradients
        opt_shared.zero_grad()
        opt_task.zero_grad()
        if opt_seq is not None:
            opt_seq.zero_grad()

        # Unpack inputs for structure fusion
        if self.with_structure and isinstance(x, (list, tuple)):
            x_formula, x_struct = x
        else:
            x_formula, x_struct = x, None

        # Get raw embeddings for pre-training tasks
        h_f = self.shared(x_formula)
        h_s = self.struct_enc(x_struct) if (self.with_structure and x_struct is not None) else None

        # Modality dropout during pre-training
        if self.pretrain and h_s is not None and torch.rand(1).item() < self.mod_dropout_p:
            h_s, x_struct = None, None

        # Forward pass
        preds = self(x, temps)

        # Calculate task losses
        total_loss = 0.0
        logs = {}

        # Regression/classification losses
        attr_losses = []
        for name, pred in preds.items():
            head = self.task_heads[name]
            if not isinstance(head, SequenceBaseHead):
                attr_loss, _ = head.compute_loss(pred, y_attr, mask_attr)
                attr_losses.append(attr_loss)
                logs[f"train_{name}_loss"] = attr_loss

        if attr_losses:
            attr_loss = torch.stack(attr_losses).mean()
            total_loss = self.w["attr"] * attr_loss
            logs["train_attr_loss"] = attr_loss

        # Sequence losses
        seq_losses = []
        for name, pred in preds.items():
            head = self.task_heads[name]
            if isinstance(head, SequenceBaseHead) and y_seq is not None:
                seq_loss, _ = head.compute_loss(pred, y_seq, mask_seq)
                seq_losses.append(seq_loss)
                logs[f"train_{name}_loss"] = seq_loss

        if seq_losses:
            seq_loss = torch.stack(seq_losses).mean()
            total_loss = total_loss + self.w["seq"] * seq_loss
            logs["train_seq_loss"] = seq_loss

        # Pre-training losses
        if self.pretrain:
            # 1) Contrastive loss
            if h_s is not None and self.w["con"] > 0:
                l_con = self._contrastive(h_f, h_s)
                total_loss += self.w["con"] * l_con
                logs["train_con"] = l_con

            # 2) Cross-reconstruction loss
            if h_s is not None and self.w["cross"] > 0:
                l_cross = 0.5 * (
                    F.mse_loss(self.dec_struct(h_f), x_struct) + F.mse_loss(self.dec_formula(h_s), x_formula)
                )
                total_loss += self.w["cross"] * l_cross
                logs["train_cross"] = l_cross

            # 3) Masked feature modeling
            if self.w["mask"] > 0:
                xf_mask, mf = self._mask_feat(x_formula)
                l_mask = F.mse_loss(self.dec_formula(self.shared(xf_mask))[mf], x_formula[mf])
                if x_struct is not None:
                    xs_mask, ms = self._mask_feat(x_struct)
                    l_mask_s = F.mse_loss(self.dec_struct(self.struct_enc(xs_mask))[ms], x_struct[ms])
                    l_mask = 0.5 * (l_mask + l_mask_s)
                total_loss += self.w["mask"] * l_mask
                logs["train_mask"] = l_mask

        # Backward pass and optimization
        self.manual_backward(total_loss)
        opt_shared.step()
        opt_task.step()
        if opt_seq is not None:
            opt_seq.step()

        # Log metrics
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Unpack batch
        x, y_attr, mask_attr, temps, y_seq, mask_seq = batch

        # Forward pass
        preds = self(x, temps)

        # Calculate task losses
        total_loss = 0.0
        logs = {}

        # Regression/classification losses
        attr_losses = []
        for name, pred in preds.items():
            head = self.task_heads[name]
            if not isinstance(head, SequenceBaseHead):
                attr_loss, _ = head.compute_loss(pred, y_attr, mask_attr)
                attr_losses.append(attr_loss)
                logs[f"val_{name}_loss"] = attr_loss

        if attr_losses:
            attr_loss = torch.stack(attr_losses).mean()
            total_loss = attr_loss
            logs["val_attr_loss"] = attr_loss

        # Sequence losses
        seq_losses = []
        for name, pred in preds.items():
            head = self.task_heads[name]
            if isinstance(head, SequenceBaseHead) and y_seq is not None:
                seq_loss, _ = head.compute_loss(pred, y_seq, mask_seq)
                seq_losses.append(seq_loss)
                logs[f"val_{name}_loss"] = seq_loss

        if seq_losses:
            seq_loss = torch.stack(seq_losses).mean()
            total_loss = total_loss + seq_loss
            logs["val_seq_loss"] = seq_loss

        # Log metrics
        self.log_dict(logs, prog_bar=True, on_epoch=True)

        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step."""
        x, _, _, temps, _, _ = batch
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
