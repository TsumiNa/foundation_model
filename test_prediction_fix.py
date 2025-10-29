#!/usr/bin/env python3

"""
Test script to verify the prediction fix for KernelRegression tasks.
This script tests the complete prediction pipeline to ensure all samples are processed correctly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import KernelRegressionTaskConfig, TaskType


def test_prediction_pipeline():
    """Test the complete prediction pipeline with diagnostic logging."""

    print("=" * 80)
    print("TESTING PREDICTION PIPELINE FIX")
    print("=" * 80)

    # Load data to check initial counts
    print("\n1. LOADING RAW DATA...")
    try:
        raw_data = pd.read_parquet("data/qc_ac_te_mp_dos_reformat_20250615.pd.parquet")
        print(f"   Raw data loaded: {len(raw_data)} samples")
    except Exception as e:
        print(f"   Error loading raw data: {e}")
        return False

    # Create task configuration
    print("\n2. CREATING TASK CONFIGURATION...")
    task_configs = [
        KernelRegressionTaskConfig(
            name="dos",
            type=TaskType.KERNEL_REGRESSION,
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
    print("\n3. CREATING DATAMODULE...")
    try:
        datamodule = CompoundDataModule(
            formula_desc_source="data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet",
            attributes_source="data/qc_ac_te_mp_dos_reformat_20250615.pd.parquet",
            task_configs=task_configs,
            batch_size=256,
            num_workers=0,
            predict_idx="all",
            val_split=0.1,
            test_split=0.1,
            train_random_seed=42,
            test_random_seed=24,
        )

        # Setup for prediction
        datamodule.setup(stage="predict")
        print("   DataModule setup complete")
        print(f"   Predict dataset size: {len(datamodule.predict_dataset) if datamodule.predict_dataset else 0}")

        # Create predict dataloader
        predict_dataloader = datamodule.predict_dataloader()
        if predict_dataloader is None:
            print("   ERROR: predict_dataloader is None")
            return False

        total_batches = len(predict_dataloader)
        print(f"   Predict dataloader created: {total_batches} batches")

    except Exception as e:
        print(f"   Error creating DataModule: {e}")
        return False

    # Create a simple model for testing
    print("\n4. CREATING MODEL...")
    try:
        model = FlexibleMultiTaskModel(
            shared_block_dims=[290, 128],
            task_configs=task_configs,
        )
        model.eval()
        print("   Model created successfully")

    except Exception as e:
        print(f"   Error creating model: {e}")
        return False

    # Test prediction on a few batches
    print("\n5. TESTING PREDICTION...")
    total_input_samples = 0
    total_output_samples = 0
    batch_count = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(predict_dataloader):
                if batch_idx >= 3:  # Test only first 3 batches
                    break

                batch_count += 1

                # Count input samples
                x_formula = batch[0]
                batch_size = x_formula.shape[0]
                total_input_samples += batch_size

                # Get sequence data
                task_sequence_data_batch = batch[3] if len(batch) > 3 else {}
                dos_sequences = task_sequence_data_batch.get("dos", [])

                if isinstance(dos_sequences, list):
                    sequence_lengths = [len(seq) for seq in dos_sequences]
                    total_sequence_points = sum(sequence_lengths)
                    print(f"   Batch {batch_idx + 1}: {batch_size} samples, {total_sequence_points} sequence points")
                    print(
                        f"     Sequence lengths: min={min(sequence_lengths)}, max={max(sequence_lengths)}, avg={np.mean(sequence_lengths):.1f}"
                    )

                # Run prediction
                try:
                    predictions = model.predict_step(batch, batch_idx)

                    # Count output samples
                    for key, value in predictions.items():
                        if key.startswith("dos_"):
                            if isinstance(value, list):
                                output_count = len(value)
                                total_output_samples += output_count
                                print(f"     Output '{key}': {output_count} samples")
                            else:
                                print(f"     Output '{key}': {type(value)} (unexpected format)")
                            break

                except Exception as e:
                    print(f"   ERROR in prediction for batch {batch_idx}: {e}")
                    return False

        print("\n6. PREDICTION SUMMARY:")
        print(f"   Batches processed: {batch_count}")
        print(f"   Total input samples: {total_input_samples}")
        print(f"   Total output samples: {total_output_samples}")
        print(f"   Sample preservation ratio: {total_output_samples / total_input_samples:.3f}")

        # Check if we have 1:1 mapping
        if total_output_samples == total_input_samples:
            print("   ✅ SUCCESS: All input samples have corresponding outputs!")
            return True
        else:
            print("   ❌ FAILURE: Sample count mismatch!")
            return False

    except Exception as e:
        print(f"   Error during prediction testing: {e}")
        return False


if __name__ == "__main__":
    success = test_prediction_pipeline()
    if success:
        print("\n🎉 PREDICTION PIPELINE FIX VERIFIED!")
        sys.exit(0)
    else:
        print("\n💥 PREDICTION PIPELINE STILL HAS ISSUES!")
        sys.exit(1)
