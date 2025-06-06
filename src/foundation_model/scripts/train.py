#!/usr/bin/env python
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger

from foundation_model.data.datamodule import CompoundDataModule  # noqa: F401
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel  # noqa: F401


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})


class StrictFalseLightningCLI(LightningCLI):
    def instantiate_classes(self):
        model, data, trainer, callbacks, loggers = super().instantiate_classes()
        ckpt_path = self.config_fit.get("ckpt_path", None)
        if ckpt_path and hasattr(model, "load_from_checkpoint"):
            model = model.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)
        self.model = model
        return model, data, trainer, callbacks, loggers


def cli_main():
    """
    Main function to run the Lightning CLI.
    Initializes LightningCLI with the model and datamodule.
    The CLI handles parsing arguments, loading configurations, and running
    the appropriate trainer actions (fit, validate, test, predict).
    """
    LightningCLI(
        # subclass_mode_model=True,
        # subclass_mode_data=True,
        auto_configure_optimizers=False,
        save_config_callback=LoggerSaveConfigCallback,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    # No explicit cli.trainer.fit() or other calls are needed here.
    # LightningCLI manages the execution flow based on command-line subcommands.


if __name__ == "__main__":
    cli_main()
