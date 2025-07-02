#!/usr/bin/env python3
"""
Test script to verify that the fixed PredictionDataFrameWriter correctly handles
multi-GPU distributed prediction and saves all 48,998 samples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(".") / "src"))

import lightning as L

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import ExtendRegressionTaskConfig, TaskType
from foundation_model.scripts.callbacks.prediction_writer import PredictionDataFrameWriter


def test_multi_gpu_prediction():
    """Test multi-GPU prediction with the fixed PredictionDataFrameWriter."""

    print("=== Testing Multi-GPU Prediction with Fixed PredictionDataFrameWriter ===")

    # Create task configuration
    task_configs = [
        ExtendRegressionTaskConfig(
            name="dos",
            type=TaskType.ExtendRegression,
            data_column="DOS density (normalized)",
            t_column="DOS energy",
            x_dim=[128, 32, 16],
            t_dim=[32, 16],
            t_encoding_method="fc",
            norm=True,
            residual=False,
            weight=1.0,
            enabled=True,
        )
    ]

    # Create DataModule
    print("Creating DataModule...")
    datamodule = CompoundDataModule(
        formula_desc_source="/data/foundation_model/data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet",
        attributes_source="/data/foundation_model/data/qc_ac_te_mp_dos_reformat_20250615.pd.parquet",
        task_configs=task_configs,
        batch_size=256,
        num_workers=0,
        predict_idx="all",
        val_split=0.1,
        test_split=0.1,
        train_random_seed=42,
        test_random_seed=24,
    )

    # Load model
    print("Loading model...")
    model = FlexibleMultiTaskModel.load_from_checkpoint(
        "samples/example_logs/basic_run/basic_experiment_20250702_003437/fit/checkpoints/last.ckpt", strict=True
    )

    # Create prediction writer with fixed distributed handling
    print("Creating PredictionDataFrameWriter with distributed support...")
    prediction_writer = PredictionDataFrameWriter(output_path="test_multi_gpu_predictions", write_interval="epoch")

    # Test with 2 GPUs (the problematic case)
    print("Creating trainer with 2 GPUs...")
    trainer = L.Trainer(
        accelerator="auto",
        devices=2,  # This was causing the 50% data loss
        logger=False,
        callbacks=[prediction_writer],
    )

    # Run prediction
    print("Starting multi-GPU prediction test...")
    print("Expected: 48,998 samples")
    print("Previous result with 2 GPUs: ~24,499 samples (50% loss)")
    print("Expected result after fix: 48,998 samples (100%)")

    trainer.predict(model, datamodule=datamodule)

    print("Multi-GPU prediction completed!")

    # Verify results
    print("\n=== Verification ===")
    try:
        import pandas as pd

        df = pd.read_csv("test_multi_gpu_predictions/predictions.csv")
        print(f"Result: {len(df)} samples saved")
        print("Expected: 48,998 samples")
        print(f"Success: {len(df) == 48998}")

        if len(df) == 48998:
            print("✅ FIXED: Multi-GPU prediction now saves all samples correctly!")
        else:
            print(f"❌ ISSUE: Still missing {48998 - len(df)} samples")

        print(f"Columns: {list(df.columns)}")
        print("First few rows:")
        print(df.head())

    except Exception as e:
        print(f"Error reading results: {e}")


if __name__ == "__main__":
    test_multi_gpu_prediction()
