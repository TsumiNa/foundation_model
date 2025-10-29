import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from foundation_model.configs.callback_config import (
    EarlyStoppingConfig,
    ModelCheckpointConfig,
)


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
    log_name: str = "my_exp_name",  # Used for default logger/checkpoint paths
    verbose: bool = True,  # General verbosity for callbacks
    early_stopping_config: EarlyStoppingConfig | bool | None = None,
    checkpoint_config: ModelCheckpointConfig | bool | None = None,
    csv_logger_config: dict | bool | None = None,
    tb_logger_config: dict | bool | None = None,
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
    early_stopping_config : EarlyStoppingConfig | bool | None, optional
        Configuration for early stopping. Can be an EarlyStoppingConfig instance,
        True to use defaults, False to disable, or None to use defaults.
    checkpoint_config : ModelCheckpointConfig | bool | None, optional
        Configuration for model checkpointing. Can be a ModelCheckpointConfig instance,
        True to use defaults, False to disable, or None to use defaults.
    csv_logger_config : dict | bool | None, optional
        Configuration for CSV logger. Can be a dict of parameters,
        True to use defaults, False to disable, or None to use defaults.
    tb_logger_config : dict | bool | None, optional
        Configuration for TensorBoard logger. Can be a dict of parameters,
        True to use defaults, False to disable, or None to use defaults.
    """
    # Set default_root_dir to "lightning_logs" if None
    if default_root_dir is None:
        default_root_dir = "lightning_logs"

    callbacks = []
    loggers = []

    # Configure EarlyStopping
    if early_stopping_config is False:
        pass  # Disabled
    elif isinstance(early_stopping_config, EarlyStoppingConfig):
        if early_stopping_config.enabled:
            callbacks.append(EarlyStopping(**early_stopping_config.model_dump(exclude={"enabled"}, exclude_none=True)))
    else:  # None or True (use defaults)
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10, mode="min", min_delta=1e-4, verbose=verbose))

    # Configure ModelCheckpoint
    if checkpoint_config is False:
        pass  # Disabled
    elif isinstance(checkpoint_config, ModelCheckpointConfig):
        if checkpoint_config.enabled:
            cfg_dict = checkpoint_config.model_dump(exclude={"enabled"}, exclude_none=True)
            if cfg_dict.get("dirpath") is None:
                cfg_dict["dirpath"] = f"{default_root_dir}/checkpoints/{log_name}"

            # Ensure filename is sensible if not provided or if monitor is None
            if cfg_dict.get("filename") is None:
                monitored_metric = cfg_dict.get("monitor")
                if monitored_metric:
                    # Ensure the metric name is valid for a filename
                    safe_metric_name = monitored_metric.replace("/", "_")
                    cfg_dict["filename"] = f"{{epoch:02d}}-{{{safe_metric_name}:.4f}}"
                else:
                    # If no monitor, save by epoch and step
                    cfg_dict["filename"] = "{epoch:02d}-{step:06d}"

            callbacks.append(ModelCheckpoint(**cfg_dict))
    else:  # None or True (use defaults)
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"{default_root_dir}/checkpoints/{log_name}",
                filename="{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                save_last=True,
                verbose=verbose,
            )
        )

    # Configure CSVLogger
    if csv_logger_config is False:
        pass  # Disabled
    elif isinstance(csv_logger_config, dict):
        loggers.append(CSVLogger(**csv_logger_config))
    else:  # None or True (use defaults)
        loggers.append(
            CSVLogger(
                save_dir=f"{default_root_dir}/csv_logs",
                name=log_name,
                version=None,  # Auto-increment version
            )
        )

    # Configure TensorBoardLogger
    if tb_logger_config is False:
        pass  # Disabled
    elif isinstance(tb_logger_config, dict):
        loggers.append(TensorBoardLogger(**tb_logger_config))
    else:  # None or True (use defaults)
        loggers.append(
            TensorBoardLogger(
                save_dir=f"{default_root_dir}/tb_logs",
                name=log_name,
                version=None,  # Auto-increment version
            )
        )

    # Ensure loggers list is not empty for Trainer, or pass True for default logger
    trainer_logger = loggers if loggers else True

    # Configure trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=trainer_logger,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        profiler=profiler,
        strategy=strategy,
        fast_dev_run=fast_dev_run,
    )

    # Train and test the model
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)

    return trainer
