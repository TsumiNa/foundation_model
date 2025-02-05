import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F

from .layers import LinearBlock, LinearLayer


class MultiTaskAttributePredictor(L.LightningModule):
    def __init__(
        self,
        shared_block_dims: list[int],
        task_block_dims: list[int],
        n_tasks: int,
        *,
        shared_block_lr: float = 0.005,
        task_block_lr: float = 0.005,
        norm_shared: bool = True,
        residual_shared: bool = False,
        norm_tasks: bool = True,
        residual_tasks: bool = False,
    ):
        super(MultiTaskAttributePredictor, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()
        self._model = None

        self._shared_block_lr = shared_block_lr
        self._task_block_lr = task_block_lr

        self._norm_shared = norm_shared
        self._residual_shared = residual_shared
        self._norm_tasks = norm_tasks
        self._residual_tasks = residual_tasks

        # Create shared block
        self.shard_block = LinearBlock(
            shared_block_dims,
            normalization=self._norm_shared,
            residual=self._residual_shared,
        )

        # Create intermediate layers
        self.deposit_layer = nn.Sequential(
            nn.Linear(shared_block_dims[-1], task_block_dims[0]),
            nn.Tanh(),
        )

        # Create task blocks
        self.task_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    LinearBlock(
                        task_block_dims[:-1],
                        normalization=self._norm_tasks,
                        residual=self._residual_tasks,
                    ),
                    LinearLayer(
                        task_block_dims[-2],
                        task_block_dims[-1],
                        activation=None,
                        normalization=False,
                    ),
                )
                for _ in range(n_tasks)
            ]
        )

        # Initialize parameters
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.shard_block(x)  # shared block
        x = self.deposit_layer(x)  # intermediate layer
        x = torch.cat([layer(x) for layer in self.task_blocks], dim=1)  # task blocks

        return x

    @staticmethod
    def masked_mse_loss(preds, targets, masks):
        losses = F.mse_loss(preds, targets, reduction="none") * masks
        per_attr_losses = torch.nan_to_num(  # [n_tasks]
            losses.sum(dim=0) / masks.sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0
        )
        loss = losses.sum() / masks.sum()
        return loss, per_attr_losses

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        lr1, lr2 = self.lr_schedulers()
        opt1.zero_grad()
        opt2.zero_grad()
        x, y, mask, attrs = batch
        y_pred = self(x)

        # Calculate per-attribute losses
        loss, per_attr_losses = self.masked_mse_loss(y_pred, y, mask)

        # Log per-attribute losses
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        attrs = np.array(attrs)
        self.log_dict(
            {
                f"{attr} (train_loss)": loss
                for attr, loss in zip(attrs[:, 0], per_attr_losses)
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.manual_backward(loss)
        opt1.step()
        opt2.step()
        lr1.step(loss)
        lr2.step(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask, attrs = batch
        y_pred = self(x)

        # Calculate per-attribute losses
        loss, per_attr_losses = self.masked_mse_loss(y_pred, y, mask)

        # Log per-attribute losses
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        attrs = np.array(attrs)
        self.log_dict(
            {
                f"{attr} (val_loss)": loss
                for attr, loss in zip(attrs[:, 0], per_attr_losses)
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, y, mask, attrs = batch
        y_pred = self(x)

        # Calculate per-attribute losses
        loss, per_attr_losses = self.masked_mse_loss(y_pred, y, mask)

        # Log per-attribute losses
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        attrs = np.array(attrs)
        self.log_dict(
            {
                f"{attr} (test_loss)": loss
                for attr, loss in zip(attrs[:, 0], per_attr_losses)
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, mask, attrs = batch
        y_pred = self(x)
        return {
            "preds": y_pred,
            "targets": y,
            "masks": mask,
            "attributes": attrs[0],
        }

    def configure_optimizers(self):
        # Optimizer for shared layers (shard_block and deposit_layer)
        shared_params = list(self.shard_block.parameters()) + list(
            self.deposit_layer.parameters()
        )
        shared_optimizer = optim.Adam(shared_params, lr=self._shared_block_lr)

        # Optimizer for task-specific layers (task_blocks)
        task_optimizer = optim.Adam(
            self.task_blocks.parameters(), lr=self._task_block_lr
        )

        # Learning rate schedulers
        shared_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            shared_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-4,
        )
        task_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            task_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-4,
        )

        return (
            {
                "optimizer": shared_optimizer,
                "lr_scheduler": {
                    "scheduler": shared_scheduler,
                    "monitor": "train_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
            {
                "optimizer": task_optimizer,
                "lr_scheduler": {
                    "scheduler": task_scheduler,
                    "monitor": "train_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
        )
