#!/usr/bin/env python
import argparse

import pandas as pd
import torch
from xenonpy.descriptor import Compositions

from foundation_model.configs.model_config import ExperimentConfig, ModelConfig
from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.data.splitter import MultiTaskSplitter
from foundation_model.models.multi_task import MultiTaskAttributePredictor
from foundation_model.utils.training import training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-task attribute prediction training script"
    )
    # Attribute configuration
    parser.add_argument(
        "--mp_attrs_rate",
        type=float,
        default=1.0,
        help="Sampling rate for Materials Project attributes (default: 1.0)",
    )
    parser.add_argument(
        "--filter_attributes",
        action="store_true",
        help="If set, only keeps attributes specified in attribute_rates",
    )
    # Training configuration
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers for data loading",
    )

    # Dataset configuration
    parser.add_argument(
        "--train_ratio",
        type=float,
        help="Ratio of training data",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        help="Ratio of validation data",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        help="Ratio of test data",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for data splitting",
    )

    # Trainer configuration
    parser.add_argument(
        "--accelerator",
        type=str,
        help="Accelerator type for training",
    )
    parser.add_argument(
        "--devices",
        type=int,
        help="Number of devices to use",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Training strategy",
    )

    # Experiment configuration
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Name of the experiment for logging",
    )

    # Path configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing raw data",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory for results",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Directory for logs",
    )

    # Learning rate configuration
    parser.add_argument(
        "--shared_block_lr",
        type=float,
        help="Learning rate for shared blocks (default: 0.01)",
    )
    parser.add_argument(
        "--task_block_lr",
        type=float,
        help="Learning rate for task-specific blocks (default: 0.05)",
    )
    return parser.parse_args()


def update_config_from_args(
    config: ExperimentConfig, model_config: ModelConfig, args: argparse.Namespace
) -> None:
    """Update configuration with command line arguments."""
    for key, value in vars(args).items():
        if value is not None:  # Only override if argument was provided
            if key == "shared_block_lr":
                model_config.shared_block_lr = value
            elif key == "task_block_lr":
                model_config.task_block_lr = value
            elif hasattr(config, key):
                setattr(config, key, value)
            elif key in config.trainer_config:
                config.trainer_config[key] = value


def main():
    # Parse command line arguments
    args = parse_args()

    # Load configurations
    model_config = ModelConfig()
    exp_config = ExperimentConfig()

    # Update experiment and model configs with command line arguments
    update_config_from_args(exp_config, model_config, args)

    # Set float32 matmul precision
    # see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("high")

    # Load dataset
    qc_ac_te_mp_dataset = pd.read_pickle(
        f"{exp_config.data_dir}/qc_ac_te_mp_rebuild_T=290K_20250202.pd.xz"
    )

    # Update attribute rates based on groups and rate
    attribute_rates = {}

    # Set ac_qc_starry_attrs to fixed rate 1.0
    for attr in ExperimentConfig.ac_qc_starry_attrs:
        attribute_rates[attr] = 1.0

    # Set mp_attrs to specified rate
    for attr in ExperimentConfig.mp_attrs:
        attribute_rates[attr] = args.mp_attrs_rate

    # Update ExperimentConfig's attribute_rates
    exp_config.attribute_rates = attribute_rates

    # Use attribute lists from ExperimentConfig
    all_attrs = ExperimentConfig.ac_qc_starry_attrs + ExperimentConfig.mp_attrs
    qc_ac_te_mp_attrs = qc_ac_te_mp_dataset[all_attrs]

    # Preprocess data
    from sklearn.preprocessing import QuantileTransformer

    qt = QuantileTransformer(output_distribution="normal").set_output(
        transform="pandas"
    )
    qc_ac_te_mp_attrs = qt.fit_transform(qc_ac_te_mp_attrs)

    # Calculate descriptors
    comp_calc = Compositions(featurizers=exp_config.xenonpy_featurizers, n_jobs=20)
    all_comp_desc = comp_calc.fit_transform(qc_ac_te_mp_dataset).dropna()
    qc_ac_te_mp_attrs = qc_ac_te_mp_attrs.loc[all_comp_desc.index]

    used_attrs = qc_ac_te_mp_attrs[all_attrs].dropna(how="all")
    used_desc = all_comp_desc.loc[used_attrs.index]

    # Prepare data module
    splitter = MultiTaskSplitter(
        train_ratio=exp_config.train_ratio,
        val_ratio=exp_config.val_ratio,
        test_ratio=exp_config.test_ratio,
        random_state=exp_config.random_seed,
    )

    datamodule = CompoundDataModule(
        descriptor=used_desc,
        attributes=used_attrs,
        splitter=splitter,
        attribute_rates=exp_config.attribute_rates,
        filter_attributes=args.filter_attributes,
        batch_size=exp_config.batch_size,
        num_workers=exp_config.num_workers,
    )

    # Update input dimension in model config
    model_config.shared_block_dims[0] = used_desc.shape[1]

    # Setup datamodule to get actual number of tasks
    datamodule.setup()
    n_tasks = len(datamodule.train_dataset.attribute_names)

    # Initialize model with correct number of tasks
    model = MultiTaskAttributePredictor(
        shared_block_dims=model_config.shared_block_dims,
        task_block_dims=model_config.task_block_dims,
        n_tasks=n_tasks,  # Use actual number of tasks after filtering
        norm_shared=model_config.norm_shared,
        norm_tasks=model_config.norm_tasks,
        residual_shared=model_config.residual_shared,
        residual_tasks=model_config.residual_tasks,
        shared_block_lr=model_config.shared_block_lr,
        task_block_lr=model_config.task_block_lr,
    )
    # # Compile the model
    # model = torch.compile(model)

    # Train model
    training(
        model=model,
        datamodule=datamodule,
        max_epochs=exp_config.max_epochs,
        accelerator=exp_config.trainer_config["accelerator"],
        devices=exp_config.trainer_config["devices"],
        strategy=exp_config.trainer_config["strategy"],
        default_root_dir=exp_config.log_dir,
        early_stopping_config=exp_config.early_stopping_config,
        checkpoint_config=exp_config.checkpoint_config,
        log_name=exp_config.exp_name,
    )


if __name__ == "__main__":
    main()
