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
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="Accelerator type for training"
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices to use"
    )
    parser.add_argument(
        "--strategy", type=str, default="auto", help="Training strategy"
    )
    parser.add_argument(
        "--default_root_dir",
        type=str,
        default=None,
        help="Default root directory for logs and checkpoints",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading"
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Load configurations
    model_config = ModelConfig()
    exp_config = ExperimentConfig()

    # Set float32 matmul precision
    # see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("high")

    # Load dataset
    qc_ac_te_mp_dataset = pd.read_pickle(
        "data/raw/qc_ac_te_mp_rebuild_T=290K_20250202.pd.xz"
    )

    # Define properties
    ac_qc_starry_props = [
        "Seebeck coefficient",
        "Thermal conductivity",
        "Electrical resistivity",
        "Magnetic susceptibility",
        "Specific heat capacity",
        "Hall coefficient",
        "ZT",
        "Power factor",
        "Carrier concentration",
        "Electrical conductivity",
        "Thermopower",
        "Lattice thermal conductivity",
        "Hall mobility",
        "Electronic contribution",
        "Electronic thermal conductivity",
    ]

    mp_props = [
        "Band gap",
        "Density",
        "Efermi",
        "Final energy per atom",
        "Formation energy per atom",
        "Total magnetization",
        "Volume",
    ]

    all_props = ac_qc_starry_props + mp_props
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
        train_ratio=model_config.train_ratio,
        val_ratio=model_config.val_ratio,
        test_ratio=model_config.test_ratio,
        random_state=model_config.random_seed,
    )

    datamodule = CompoundDataModule(
        descriptor=used_desc,
        property_data=used_props,
        splitter=splitter,
        property_fractions=model_config.property_fractions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Update input dimension in model config
    model_config.shared_block_dims[0] = used_desc.shape[1]

    # Initialize model
    model = MultiTaskPropertyPredictor(
        shared_block_dims=model_config.shared_block_dims,
        task_block_dims=model_config.task_block_dims,
        n_tasks=len(model_config.property_fractions),
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
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        default_root_dir=args.default_root_dir,
    )


if __name__ == "__main__":
    main()
