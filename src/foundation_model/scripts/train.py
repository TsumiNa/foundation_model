#!/usr/bin/env python
import argparse

import pandas as pd
import torch
from xenonpy.descriptor import Compositions

from foundation_model.configs.model_config import ExperimentConfig, ModelConfig
from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.data.splitter import MultiTaskSplitter
from foundation_model.models.multi_task import MultiTaskPropertyPredictor
from foundation_model.utils.training import training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-task property prediction training script"
    )
    # Property rate configuration
    parser.add_argument(
        "--mp_props_rate",
        type=float,
        default=1.0,
        help="Sampling rate for Materials Project properties (default: 1.0)",
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

    # Update property fractions based on groups and rate
    property_fractions = {}

    # Set ac_qc_starry_props to fixed rate 1.0
    for prop in ExperimentConfig.ac_qc_starry_props:
        property_fractions[prop] = 1.0

    # Set mp_props to specified rate
    for prop in ExperimentConfig.mp_props:
        property_fractions[prop] = args.mp_props_rate

    # Update ExperimentConfig's property_fractions
    exp_config.property_fractions = property_fractions

    # Use property lists from ExperimentConfig
    all_props = ExperimentConfig.ac_qc_starry_props + ExperimentConfig.mp_props
    qc_ac_te_mp_props = qc_ac_te_mp_dataset[all_props]

    # Preprocess data
    from sklearn.preprocessing import QuantileTransformer

    qt = QuantileTransformer(output_distribution="normal").set_output(
        transform="pandas"
    )
    qc_ac_te_mp_props = qt.fit_transform(qc_ac_te_mp_props)

    # Calculate descriptors
    comp_calc = Compositions(featurizers=exp_config.xenonpy_featurizers, n_jobs=20)
    all_comp_desc = comp_calc.fit_transform(qc_ac_te_mp_dataset).dropna()
    qc_ac_te_mp_props = qc_ac_te_mp_props.loc[all_comp_desc.index]

    used_props = qc_ac_te_mp_props[all_props].dropna(how="all")
    used_desc = all_comp_desc.loc[used_props.index]

    # Prepare data module
    splitter = MultiTaskSplitter(
        train_ratio=exp_config.train_ratio,
        val_ratio=exp_config.val_ratio,
        test_ratio=exp_config.test_ratio,
        random_state=exp_config.random_seed,
    )

    datamodule = CompoundDataModule(
        descriptor=used_desc,
        property_data=used_props,
        splitter=splitter,
        property_fractions=exp_config.property_fractions,
        batch_size=exp_config.batch_size,
        num_workers=exp_config.num_workers,
    )

    # Update input dimension in model config
    model_config.shared_block_dims[0] = used_desc.shape[1]

    # Initialize model
    model = MultiTaskPropertyPredictor(
        shared_block_dims=model_config.shared_block_dims,
        task_block_dims=model_config.task_block_dims,
        n_tasks=len(exp_config.property_fractions),
        norm_shared=model_config.norm_shared,
        norm_tasks=model_config.norm_tasks,
        residual_shared=model_config.residual_shared,
        residual_tasks=model_config.residual_tasks,
        shared_block_lr=model_config.shared_block_lr,
        task_block_lr=model_config.task_block_lr,
    )

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
