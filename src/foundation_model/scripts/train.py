#!/usr/bin/env python
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})


def cli_main():
    """
    Main function to run the Lightning CLI.
    Initializes LightningCLI with the model and datamodule.
    The CLI handles parsing arguments, loading configurations, and running
    the appropriate trainer actions (fit, validate, test, predict).
    """
    cli = LightningCLI(
        FlexibleMultiTaskModel,
        CompoundDataModule,
        auto_configure_optimizers=False,
        save_config_callback=LoggerSaveConfigCallback,
        # subclass_mode_model=True, # Uncomment if multiple model classes are possible
        # subclass_mode_data=True,  # Uncomment if multiple data module classes are possible
    )
    # No explicit cli.trainer.fit() or other calls are needed here.
    # LightningCLI manages the execution flow based on command-line subcommands.


if __name__ == "__main__":
    cli_main()
