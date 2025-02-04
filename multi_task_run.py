#!/usr/bin/env python
# coding: utf-8

import argparse

import joblib
import pandas as pd

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
import torch

# user-friendly print
from multi_task import (
    CompoundDataModule,
    MultiTaskPropertyPredictor,
    plot_predictions,
    train_and_evaluate,
)
from multi_task_splitter import MultiTaskSplitter

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-task property prediction training script"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="Accelerator type for training"
    )
    parser.add_argument(
        "--devices", type=int, default=4, help="Number of devices to use"
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
    args = parse_args()
    qc_ac_te_mp_dataset = pd.read_pickle(
        "common_data/qc_ac_te_mp_rebuild_T=290K_20250202.pd.xz"
    )

    qc_ac_te_mp_dataset.head(3)
    qc_ac_te_mp_dataset.shape

    starry_props = [
        "Thermal conductivity",
        "Carrier concentration",
        "Electrical conductivity",
        "Thermopower",
        "Electrical resistivity",
        "Power factor",
        "Seebeck coefficient",
        "Lattice thermal conductivity",
        "ZT",
        "Hall mobility",
        "Electronic contribution",
        "Electronic thermal conductivity",
    ]

    ac_qc_props = [
        "Seebeck coefficient",
        "Thermal conductivity",
        "Electrical resistivity",
        "Magnetic susceptibility",
        "Specific heat capacity",
        "Hall coefficient",
        "ZT",
        "Power factor",
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

    ac_qc_starry_shared_props = [
        "Seebeck coefficient",
        "Thermal conductivity",
        "Electrical resistivity",
        "Power factor",
        "ZT",
    ]

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

    all_props = ac_qc_starry_props + mp_props

    qc_ac_te_mp_props = qc_ac_te_mp_dataset[all_props]

    # In[4]:

    from sklearn.preprocessing import QuantileTransformer

    qt = QuantileTransformer(output_distribution="normal").set_output(
        transform="pandas"
    )

    qc_ac_te_mp_props = qt.fit_transform(qc_ac_te_mp_props)

    qc_ac_te_mp_props.shape
    qc_ac_te_mp_props.head(3)

    # ----

    # In[5]:

    from xenonpy.descriptor import Compositions

    featurizers = ["WeightedAverage", "WeightedVariance", "MaxPooling", "MinPooling"]
    comp_calc = Compositions(
        featurizers=featurizers, n_jobs=20
    )  # use specific featurizers

    # In[6]:

    all_comp_desc = comp_calc.fit_transform(qc_ac_te_mp_dataset).dropna()
    qc_ac_te_mp_props = qc_ac_te_mp_props.loc[all_comp_desc.index]

    used_props = qc_ac_te_mp_props[all_props].dropna(how="all")
    used_desc = all_comp_desc.loc[used_props.index]

    # Prepare datasets
    property_fractions = {
        "Seebeck coefficient": 1.0,
        "Thermal conductivity": 1.0,
        "Electrical resistivity": 1.0,
        "Magnetic susceptibility": 1.0,
        "Specific heat capacity": 1.0,
        "Electrical conductivity": 1.0,
        "ZT": 1.0,
        "Hall coefficient": 1.0,
        "Power factor": 1.0,
        "Carrier concentration": 1.0,
        "Thermopower": 1.0,
        "Lattice thermal conductivity": 1.0,
        "Hall mobility": 1.0,
        "Electronic contribution": 1.0,
        "Electronic thermal conductivity": 1.0,
        "Band gap": 1.0,
        "Density": 1.0,
        "Efermi": 1.0,
        "Final energy per atom": 1.0,
        "Formation energy per atom": 1.0,
        "Total magnetization": 1.0,
        "Volume": 1.0,
    }

    splitter = MultiTaskSplitter(
        train_ratio=0.8, val_ratio=0.2, test_ratio=0.0, random_state=42
    )

    datamodule = CompoundDataModule(
        descriptor=used_desc,
        property_data=used_props,
        splitter=splitter,
        property_fractions=property_fractions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize model
    shared_block_dims = [used_desc.shape[1], 128, 64, 32]
    task_block_dims = [32, 16, 1]
    model = MultiTaskPropertyPredictor(
        shared_block_dims=shared_block_dims,
        task_block_dims=task_block_dims,
        n_tasks=len(property_fractions),
        norm_shared=True,
        norm_tasks=True,
        residual_shared=True,
        residual_tasks=True,
        shared_block_lr=0.005,
        task_block_lr=0.01,
    )
    # model = torch.compile(model)

    return train_and_evaluate(
        model=model,
        datamodule=datamodule,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        default_root_dir=args.default_root_dir,
    )


if __name__ == "__main__":
    import joblib

    # Check for CUDA availability
    results = main()
    print("==================all done=======================================")

    with open("scaling_results.pkl.z", "wb") as fo:
        joblib.dump(
            results,
            fo,
        )
