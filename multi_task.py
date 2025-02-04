from pathlib import Path
from typing import Callable, Literal, Sequence

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import (
    BasePredictionWriter,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.functional import F
from torch.utils.data import DataLoader, Dataset

from multi_task_splitter import MultiTaskSplitter
from plot_utils import plot_scatter_comparison


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CompoundDataModule(L.LightningDataModule):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        property_data: pd.DataFrame,
        splitter: Callable,
        property_fractions: dict[str, float],
        batch_size=32,
        num_workers=0,
    ):
        """
        Initialize the data module.

        Parameters
        ----------
        descriptor : pd.DataFrame
            Input features for the compounds
        property_data : pd.DataFrame
            Target properties for the compounds
        splitter : Callable
            Function to split data into train/val/test sets
        property_fractions : dict[str, float]
            Dictionary specifying what fraction of data to use for each property
            e.g., {"property_name": 0.8} means use 80% of available data for that property
        batch_size : int, optional
            Batch size for dataloaders, by default 32
        num_workers : int, optional
            Number of workers for dataloaders, by default 0
        """
        super().__init__()
        self.descriptor = descriptor
        self.property_data = property_data
        self.property_fractions = property_fractions
        self.batch_size = batch_size
        self.num_workers = num_workers
        if isinstance(splitter, MultiTaskSplitter):
            self.splitter = splitter
        else:
            # Default to MultiTaskSplitter if a custom splitter is not provided
            self.splitter = MultiTaskSplitter(train_ratio=0.9, val_ratio=0.1)

    def setup(self, stage: str = None):
        # Split indices using MultiTaskSplitter
        indices = self.splitter.split(self.property_data)
        if len(indices) < 2:
            raise ValueError("Splitter must return at least two sets of indices")
        if len(indices) == 2:
            train_indices, test_indices = indices
            val_indices = []
        else:
            train_indices, val_indices, test_indices = indices
        self.train_dataset = CompoundDataset(
            self.descriptor.iloc[train_indices],
            self.property_data.iloc[train_indices],
            **self.property_fractions,
        )
        # Create validation dataset if validation indices are provided
        if len(val_indices) > 0:
            self.val_dataset = CompoundDataset(
                self.descriptor.iloc[val_indices],
                self.property_data.iloc[val_indices],
            )

        # Create test dataset if test indices are provided
        if len(test_indices) > 0:
            self.test_dataset = CompoundDataset(
                self.descriptor.iloc[test_indices],
                self.property_data.iloc[test_indices],
            )
        else:
            self.test_dataset = CompoundDataset(
                self.descriptor.iloc[val_indices],
                self.property_data.iloc[val_indices],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if hasattr(self, "val_dataset"):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None

    def test_dataloader(self):
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None

    def predict_dataloader(self):
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return None


class CompoundDataset(Dataset):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        property: pd.DataFrame,
        **property_fractions,
    ):
        """
        Custom dataset for compounds.

        Parameters
        ----------
        descriptor : pd.DataFrame
            Input features for the compounds
        property : pd.DataFrame
            Target properties for the compounds
        property_fractions : dict
            Dictionary specifying what fraction of data to use for each property
            e.g., {"property_name": 0.8} means use 80% of available data for that property
        """
        # Ensure descriptor and property have matching indices
        if not descriptor.index.equals(property.index):
            raise ValueError("descriptor and property must have matching indices")

        # Get attributes from property columns
        self.attributes = list(property.columns)
        if not self.attributes:
            raise ValueError("property DataFrame must have at least one column")

        # Input features
        self.x = descriptor.values

        # Output attributes - select all columns from property
        self.y = property.values.astype(np.float32)

        # Create initial masks based on non-nan values
        self.mask = (~np.isnan(self.y)).astype(int)

        # Initialize property fractions with default values (use all available data)
        self._property_fractions = {attr: 1.0 for attr in self.attributes}

        # Validate and update property fractions if provided
        if property_fractions:
            # Validate attributes
            invalid_attrs = set(property_fractions.keys()) - set(self.attributes)
            if invalid_attrs:
                raise ValueError(
                    f"Invalid attributes in property_fractions: {invalid_attrs}. "
                    f"Valid attributes are: {self.attributes}"
                )

            # Validate percentages
            invalid_percents = [
                (attr, percent)
                for attr, percent in property_fractions.items()
                if not 0 <= percent <= 1
            ]
            if invalid_percents:
                raise ValueError(
                    "Percentages must be between 0 and 1. Invalid values: "
                    f"{invalid_percents}"
                )

            # Update with provided values
            self._property_fractions.update(property_fractions)

            # Apply fractions only to non-nan values
            for attr_name, fraction in self._property_fractions.items():
                attr_idx = self.attributes.index(attr_name)
                # Get indices where values are not nan
                valid_indices = np.where(~np.isnan(self.y[:, attr_idx]))[0]
                if len(valid_indices) > 0:  # Only proceed if there are valid values
                    # Calculate number of samples to use based on fraction
                    num_to_use = int(len(valid_indices) * fraction)
                    # Randomly select indices to keep
                    keep_indices = np.random.choice(
                        valid_indices, num_to_use, replace=False
                    )
                    # Mask all valid indices except those we keep
                    mask_indices = np.setdiff1d(valid_indices, keep_indices)
                    self.mask[mask_indices, attr_idx] = 0

        # Fill all nan to 0 after masking
        self.y = np.nan_to_num(self.y)

        # Convert to tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]

    @property
    def fractions(self) -> dict[str, float]:
        """
        Get the fraction of data used for each property.

        Returns
        -------
        dict[str, float]
            Dictionary containing the fraction of data used for each property,
            where 1.0 means using all available data and 0.5 means using half.
        """
        return self._property_fractions.copy()


class LinearLayer(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        normalization=True,
        activation: None | nn.Module = nn.LeakyReLU(0.1),
    ):
        """
        Parameters
        ----------
        n_in: int
            Size of each input sample.
        n_out: int
            Size of each output sample
        """
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.normal = nn.BatchNorm1d(n_out) if normalization else None
        self.activation = nn.LeakyReLU(0.1)
        self.activation = activation

    def forward(self, x):
        _out = self.layer(x)
        if self.normal:
            _out = self.normal(_out)
        if self.activation:
            _out = self.activation(_out)

        return _out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_features,
        normalization=True,
        n_layers=2,
        layer_activation: None | nn.Module = nn.LeakyReLU(0.1),
        output_active: None | nn.Module = nn.LeakyReLU(0.1),
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                LinearLayer(
                    n_features,
                    n_features,
                    normalization=normalization,
                    activation=layer_activation,
                )
                if i != n_layers - 1
                else LinearLayer(
                    n_features, n_features, normalization=normalization, activation=None
                )
                for i in range(n_layers)
            ]
        )
        self.output_active = output_active

    def forward(self, x):
        y = self.layers(x)
        y += x
        if self.output_active:
            return self.output_active(y)
        return y


class LinearBlock(nn.Module):
    def __init__(
        self,
        shared_layer_dims: Sequence[int],
        normalization=True,
        residual=False,
        layer_activation: None | nn.Module = nn.LeakyReLU(0.1),
        output_active: None | nn.Module = nn.LeakyReLU(0.1),
    ):
        super().__init__()
        counter = len(shared_layer_dims) - 1
        if counter < 1:
            raise ValueError("shared_layer_dims must have at least 2 elements")

        if residual:
            self.layers = nn.Sequential(
                *[
                    # Add residual block after each layer
                    nn.Sequential(
                        LinearLayer(
                            shared_layer_dims[i],
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            activation=layer_activation,
                        ),
                        ResidualBlock(
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            layer_activation=layer_activation,
                            output_active=None,
                        ),
                    )
                    if i == counter - 1 and output_active is None
                    else nn.Sequential(
                        LinearLayer(
                            shared_layer_dims[i],
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            activation=layer_activation,
                        ),
                        ResidualBlock(
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            layer_activation=layer_activation,
                            output_active=output_active,
                        ),
                    )
                    for i in range(counter)
                ]
            )
        else:
            self.layers = nn.Sequential(
                *[
                    LinearLayer(
                        shared_layer_dims[i],
                        shared_layer_dims[i + 1],
                        normalization=normalization,
                        activation=None,
                    )
                    if i == counter - 1 and output_active is None
                    else LinearLayer(
                        shared_layer_dims[i],
                        shared_layer_dims[i + 1],
                        normalization=normalization,
                        activation=layer_activation,
                    )
                    for i in range(counter)
                ]
            )

    def forward(self, x):
        return self.layers(x)


class MultiTaskPropertyPredictor(L.LightningModule):
    def __init__(
        self,
        shared_block_dims: Sequence[int],
        task_block_dims: Sequence[int],
        n_tasks: int,
        *,
        shared_block_lr: float = 0.005,
        task_block_lr: float = 0.005,
        norm_shared: bool = True,
        residual_shared: bool = False,
        norm_tasks: bool = True,
        residual_tasks: bool = False,
    ):
        super(MultiTaskPropertyPredictor, self).__init__()
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
        loss = F.mse_loss(preds, targets, reduction="none") * masks
        return loss.sum() / masks.sum()

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        lr1, lr2 = self.lr_schedulers()
        opt1.zero_grad()
        opt2.zero_grad()
        x, y, mask = batch
        y_pred = self(x)
        # loss = self.masked_mse_loss(y_pred, y, mask)

        # Calculate per-attribute losses
        losses = F.mse_loss(y_pred, y, reduction="none") * mask  # [batch_size, n_tasks]
        per_attr_losses = losses.sum(dim=0) / mask.sum(dim=0)  # [n_tasks]
        loss = losses.sum() / mask.sum()

        # Log per-attribute losses
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            {f"train_loss_attr_{i}": loss for i, loss in enumerate(per_attr_losses)},
            on_step=True,
            on_epoch=False,
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
        x, y, mask = batch
        y_pred = self(x)

        # Calculate per-attribute losses
        losses = F.mse_loss(y_pred, y, reduction="none") * mask  # [batch_size, n_tasks]
        per_attr_losses = losses.sum(dim=0) / mask.sum(dim=0)  # [n_tasks]
        loss = losses.sum() / mask.sum()

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
        self.log_dict(
            {f"val_loss_attr_{i}": loss for i, loss in enumerate(per_attr_losses)},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_pred = self(x)

        # Calculate per-attribute losses
        losses = F.mse_loss(y_pred, y, reduction="none") * mask  # [batch_size, n_tasks]
        per_attr_losses = losses.sum(dim=0) / mask.sum(dim=0)  # [n_tasks]
        loss = losses.sum() / mask.sum()

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
        self.log_dict(
            {f"test_loss_attr_{i}": loss for i, loss in enumerate(per_attr_losses)},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, mask = batch
        y_pred = self(x)
        return {
            "preds": y_pred,
            "targets": y,
            "masks": mask,
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
            # verbose=True,
        )
        task_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            task_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-4,
            # verbose=True,
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


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str | None = None,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
    ):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir) if output_dir else None

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        path = self.output_dir if self.output_dir else Path(trainer.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            predictions,
            path / f"rank_{trainer.global_rank}.pt",
        )


def train_and_evaluate(
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: int | list | str = "auto",
    profiler: str | None = None,
    default_root_dir: str | None = None,
    strategy: str = "auto",
    fast_dev_run: bool = False,
    log_name: str = "my_exp_name",
    verbose: bool = True,
    raw_predictions_dir: str | None = None,
):
    """
    Train and evaluate a PyTorch Lightning model using the Lightning Trainer.

    Parameters
    ----------
    model : pl.LightningModule
        The PyTorch Lightning model to train
    datamodule : pl.LightningDataModule
        The data module containing train/val/test data
    max_epochs : int, optional
        Maximum number of epochs to train for, by default 100
    accelerator : str, optional
        The accelerator to use ("cpu", "gpu", "tpu", "auto"), by default "auto"
    devices : int | list | str, optional
        The devices to use for training, by default "auto"
    default_root_dir : str, optional
        Root directory for logs and checkpoints, by default None

    Returns
    -------
    tuple
        A tuple containing (avg_test_losses, all_preds, all_targets, all_masks)
    """

    # Set default_root_dir to "lightning_logs" if None
    if default_root_dir is None:
        default_root_dir = "lightning_logs"

    # Configure callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        min_delta=1e-4,
        verbose=verbose,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
        save_last=True,
        verbose=verbose,
    )

    # Configure loggers
    csv_logger = CSVLogger(f"{default_root_dir}/common_logs", name=log_name)
    tb_logger = TensorBoardLogger(f"{default_root_dir}/tb_logs", name=log_name)

    # # Set default raw_predictions_dir if None
    # if raw_predictions_dir is None:
    #     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #     raw_predictions_dir = str(Path("raw_predictions_dir") / timestamp)

    # pred_writer = PredictionWriter(
    #     output_dir=raw_predictions_dir, write_interval="epoch"
    # )

    # Configure trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=[csv_logger, tb_logger],
        callbacks=[early_stopping, checkpoint_callback],
        default_root_dir=default_root_dir,
        profiler=profiler,
        strategy=strategy,
        fast_dev_run=fast_dev_run,
    )

    # Train and test the model
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)
    return

    # Get predictions for plotting
    trainer.predict(model, datamodule=datamodule)

    # Load predictions from disk if not available
    all_preds = []
    all_targets = []
    all_masks = []

    attributes = list(datamodule.property_data.columns)
    for p in Path(raw_predictions_dir).glob("*.pt"):
        prediction = torch.load(p, map_location="cpu")
        preds = np.concatenate([batch["preds"] for batch in prediction], axis=0)
        targets = np.concatenate([batch["targets"] for batch in prediction], axis=0)
        masks = np.concatenate([batch["masks"] for batch in prediction], axis=0)

        all_preds.append(preds)
        all_targets.append(targets)
        all_masks.append(masks)

    if (
        not (len(all_preds) == len(all_targets) == len(all_masks))
        or len(all_preds) == 0
    ):
        raise ValueError("Predictions, targets, and masks must have the same length")

    # Concatenate predictions for plotting
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    if (
        not (all_preds.shape == all_targets.shape == all_masks.shape)
        or all_preds.shape[0] == 0
    ):
        raise ValueError("The shapes of predictions, targets, and masks do not match")

    # Calculate per-attribute losses
    avg_test_losses = []

    for attr_idx in range(all_preds.shape[1]):
        mask = all_masks[:, attr_idx] == 1
        if mask.sum() > 0:
            mse = (
                (all_preds[mask, attr_idx] - all_targets[mask, attr_idx]) ** 2
            ).mean()
            avg_test_losses.append(mse)
        else:
            avg_test_losses.append(float("nan"))
    avg_test_losses = np.array(avg_test_losses)

    # Print per-attribute losses if verbose is True
    if verbose:
        print("\nTest Losses by Attribute:")
        for attr_idx, attr_name in enumerate(attributes):
            print(
                f"{attr_name}: {avg_test_losses[attr_idx]:.4f} -- size: {int(all_masks[:, attr_idx].sum()):,}"
            )

    return avg_test_losses, all_preds, all_targets, all_masks, attributes


def plot_predictions(
    all_preds,
    all_targets,
    all_masks,
    attributes: list[str],
    *,
    savefig=None,
    suffix=None,
    no_show=False,
    return_stat=False,
):
    """
    Generate scatter plots comparing predicted and target values for multiple attributes.

    This function concatenates the provided predictions, targets, and masks, and then for each attribute
    creates a scatter plot comparing the predictions to the targets, using the provided mask to filter valid
    entries. Optionally, the generated figures can be saved to disk, and statistics of the comparisons are
    collected and returned as a DataFrame if requested.

    Parameters:
        all_preds (list of numpy.array): A list of arrays containing the prediction values. These arrays are concatenated along the axis 0.
        all_targets (list of numpy.array): A list of arrays containing the target values, concatenated similarly to all_preds.
        all_masks (list of numpy.array): A list of arrays containing the mask indicators (1 for valid entries) for each attribute.
        attributes (list of str): A list of attribute names corresponding to each prediction/target column.
        savefig (str, optional): Directory path where the generated plots will be saved. If None, figures are not saved.
        suffix (str, optional): A string suffix that will be appended to the save directory if provided.
        no_show (bool, optional): If True, the figures are cleared (not shown) after creation.
        return_stat (bool, optional): If True, returns a pandas DataFrame containing statistics of the scatter comparisons.

    Returns:
        pandas.DataFrame or None: A DataFrame containing statistical information for each attribute if return_stat is True; otherwise, None.
    """
    all_stat = []
    for m in range(len(attributes)):
        mask_m = all_masks[:, m] == 1
        preds_m = all_preds[mask_m, m]
        targets_m = all_targets[mask_m, m]

        # Create DataFrame for plotting
        fig, _, stat = plot_scatter_comparison(
            targets_m, preds_m, title=attributes[m], return_stat=True
        )
        if savefig and isinstance(savefig, str):
            savefig_ = f"{savefig}/{suffix if suffix else ''}"
            _ = Path(savefig_).mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{savefig_}/{attributes[m]}.png", bbox_inches="tight")
        stat["property"] = attributes[m]
        all_stat.append(stat)

        if no_show:
            plt.cla()
            plt.clf()
            plt.close()

    if return_stat:
        return pd.DataFrame(all_stat)
