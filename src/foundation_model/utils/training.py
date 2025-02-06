import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger


def training(
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
    early_stopping_config: dict | None = None,
    checkpoint_config: dict | None = None,
):
    """
    Train and evaluate a PyTorch Lightning model using the Lightning Trainer.

    Parameters
    ----------
    model : L.LightningModule
        The PyTorch Lightning model to train
    datamodule : L.LightningDataModule
        The data module containing train/val/test data
    max_epochs : int, optional
        Maximum number of epochs to train for, by default 100
    accelerator : str, optional
        The accelerator to use ("cpu", "gpu", "tpu", "auto"), by default "auto"
    devices : int | list | str, optional
        The devices to use for training, by default "auto"
    profiler : str | None, optional
        The profiler to use for performance analysis
    default_root_dir : str | None, optional
        Root directory for logs and checkpoints, by default None
    strategy : str, optional
        Training strategy to use, by default "auto"
    fast_dev_run : bool, optional
        Whether to run a fast development run, by default False
    log_name : str, optional
        Name for the experiment logs, by default "my_exp_name"
    verbose : bool, optional
        Whether to print verbose output, by default True
    early_stopping_config : dict | None, optional
        Configuration for early stopping callback
    checkpoint_config : dict | None, optional
        Configuration for model checkpoint callback
    """
    # Set default_root_dir to "lightning_logs" if None
    if default_root_dir is None:
        default_root_dir = "lightning_logs"

    # Configure callbacks
    early_stopping = (
        EarlyStopping(
            monitor=early_stopping_config.get("monitor", "val_loss"),
            patience=early_stopping_config.get("patience", 20),
            mode=early_stopping_config.get("mode", "min"),
            min_delta=early_stopping_config.get("min_delta", 1e-4),
            verbose=verbose,
        )
        if early_stopping_config is not None
        else EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            min_delta=1e-4,
            verbose=verbose,
        )
    )

    checkpoint_callback = (
        ModelCheckpoint(
            monitor=checkpoint_config.get("monitor", "val_loss"),
            filename="model-{epoch:02d}-{val_loss:.4f}",
            save_top_k=checkpoint_config.get("save_top_k", 1),
            mode=checkpoint_config.get("mode", "min"),
            save_last=checkpoint_config.get("save_last", True),
            verbose=verbose,
        )
        if checkpoint_config is not None
        else ModelCheckpoint(
            monitor="val_loss",
            filename="model-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
            save_last=True,
            verbose=verbose,
        )
    )

    # Configure loggers with experiment name
    csv_logger = CSVLogger(
        save_dir=f"{default_root_dir}/common_logs",
        name=log_name,
        version=None,  # Auto-increment version
    )
    tb_logger = TensorBoardLogger(
        save_dir=f"{default_root_dir}/tb_logs",
        name=log_name,
        version=None,  # Auto-increment version
    )

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

    return trainer
