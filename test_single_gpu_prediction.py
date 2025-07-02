#!/usr/bin/env python3
"""
Test script to verify single GPU prediction compatibility after multi-GPU fix.
This ensures the distributed prediction fix doesn't break single GPU functionality.
"""

import os
import shutil
from pathlib import Path

import lightning as L
import pandas as pd
import torch

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.scripts.callbacks.prediction_writer import PredictionDataFrameWriter


def test_single_gpu_prediction():
    """Test single GPU prediction to ensure compatibility after multi-GPU fix."""
    print("=== Testing Single GPU Prediction with Fixed PredictionDataFrameWriter ===")

    # Create temporary output directory
    output_dir = Path("test_single_gpu_predictions")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading model first...")
    # Load the trained model (use the latest checkpoint) without overriding task_configs
    checkpoint_path = "samples/example_logs/basic_run/basic_experiment_20250702_003437/fit/checkpoints/last.ckpt"
    model = FlexibleMultiTaskModel.load_from_checkpoint(
        checkpoint_path,
        strict=True,
    )

    print("Creating DataModule...")
    # Use the same task configs as the loaded model
    task_configs = model.task_configs

    # Create DataModule
    datamodule = CompoundDataModule(
        formula_desc_source="/data/foundation_model/data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet",
        attributes_source="/data/foundation_model/data/qc_ac_te_mp_dos_reformat_20250615.pd.parquet",
        task_configs=task_configs,
        batch_size=512,  # Larger batch size for single GPU
        num_workers=0,
        predict_idx="all",  # Test with all data
        val_split=0.1,
        test_split=0.1,
        train_random_seed=42,
        test_random_seed=24,
    )

    print("Creating PredictionDataFrameWriter with single GPU support...")
    # Create prediction writer
    prediction_writer = PredictionDataFrameWriter(output_path=str(output_dir), write_interval="epoch")

    print("Creating trainer with 1 GPU...")
    # Create trainer with single GPU
    trainer = L.Trainer(
        devices=1,  # Single GPU
        accelerator="auto",
        logger=False,
        enable_progress_bar=True,
        callbacks=[prediction_writer],
    )

    print("Starting single GPU prediction test...")
    print("Expected: 48,998 samples")
    print("Testing single GPU compatibility after multi-GPU fix")

    # Run prediction
    trainer.predict(model, datamodule=datamodule)

    print("Single GPU prediction completed!")

    # Verify results
    print("\n=== Verification ===")

    # Check if prediction files exist
    csv_file = output_dir / "predictions.csv"
    parquet_file = output_dir / "predictions.pd.parquet"
    pickle_file = output_dir / "predictions.pd.xz"

    files_exist = all([csv_file.exists(), parquet_file.exists(), pickle_file.exists()])

    if not files_exist:
        print("❌ ERROR: Prediction files not found!")
        missing_files = [f for f in [csv_file, parquet_file, pickle_file] if not f.exists()]
        print(f"Missing files: {missing_files}")
        return False

    # Load and verify the results
    try:
        df = pd.read_csv(csv_file, index_col=0)
        result_count = len(df)
        expected_count = 48998

        print(f"Result: {result_count} samples saved")
        print(f"Expected: {expected_count} samples")

        success = result_count == expected_count
        print(f"Success: {success}")

        if success:
            print("✅ SUCCESS: Single GPU prediction works correctly!")
            print("✅ COMPATIBILITY: Multi-GPU fix doesn't break single GPU functionality!")
        else:
            print(f"❌ ISSUE: Expected {expected_count} samples, got {result_count}")
            return False

        # Show sample results
        print(f"Columns: {df.columns.tolist()}")
        print("First few rows:")
        print(df.head())

        # Performance check - single GPU should be reasonably fast
        print("\nFile sizes:")
        print(f"CSV: {csv_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"Parquet: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"Pickle: {pickle_file.stat().st_size / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to load or verify results: {e}")
        return False


def compare_with_multi_gpu_results():
    """Compare single GPU results with multi-GPU results if available."""
    print("\n=== Comparing with Multi-GPU Results ===")

    single_gpu_file = Path("test_single_gpu_predictions/predictions.csv")
    multi_gpu_file = Path("test_multi_gpu_predictions/predictions.csv")

    if not single_gpu_file.exists():
        print("❌ Single GPU results not found")
        return False

    if not multi_gpu_file.exists():
        print("⚠️  Multi-GPU results not found - skipping comparison")
        return True

    try:
        single_df = pd.read_csv(single_gpu_file, index_col=0)
        multi_df = pd.read_csv(multi_gpu_file, index_col=0)

        print(f"Single GPU samples: {len(single_df)}")
        print(f"Multi GPU samples: {len(multi_df)}")

        if len(single_df) != len(multi_df):
            print("❌ Sample count mismatch between single and multi GPU")
            return False

        # Check if the data structure is the same
        if not single_df.columns.equals(multi_df.columns):
            print("❌ Column structure mismatch")
            print(f"Single GPU columns: {single_df.columns.tolist()}")
            print(f"Multi GPU columns: {multi_df.columns.tolist()}")
            return False

        print("✅ Data structure matches between single and multi GPU")

        # Check if indices are the same (they should be for same predict_idx="all")
        if not single_df.index.equals(multi_df.index):
            print("⚠️  Index order differs (this is expected and OK)")
        else:
            print("✅ Index order matches")

        return True

    except Exception as e:
        print(f"❌ ERROR during comparison: {e}")
        return False


if __name__ == "__main__":
    # Set environment for single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU

    print("🔧 Environment setup:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

    # Run single GPU test
    success = test_single_gpu_prediction()

    # Compare with multi-GPU if available
    if success:
        compare_with_multi_gpu_results()

    print(f"\n{'=' * 60}")
    if success:
        print("🎉 SINGLE GPU TEST PASSED!")
        print("✅ Multi-GPU fix maintains single GPU compatibility")
    else:
        print("❌ SINGLE GPU TEST FAILED!")
        print("⚠️  Multi-GPU fix may have broken single GPU functionality")
    print(f"{'=' * 60}")
