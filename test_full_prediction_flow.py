#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import ExtendRegressionTaskConfig, RegressionTaskConfig, TaskType


def test_full_prediction_flow():
    """Test the full prediction flow to find where samples are lost."""

    print("=== TESTING FULL PREDICTION FLOW ===")

    # Create task configs
    task_configs = [
        RegressionTaskConfig(
            name="density",
            type=TaskType.REGRESSION,
            data_column="Density (normalized)",
            dims=[128, 64, 32, 1],
            enabled=True,
        ),
        ExtendRegressionTaskConfig(
            name="dos",
            type=TaskType.ExtendRegression,
            data_column="DOS density (normalized)",
            t_column="DOS energy",
            x_dim=[128, 32, 16],
            t_dim=[32, 16],
            t_encoding_method="fc",
            enabled=True,
        ),
    ]

    # Initialize datamodule
    datamodule = CompoundDataModule(
        formula_desc_source="/data/foundation_model/data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet",
        attributes_source="/data/foundation_model/data/qc_ac_te_mp_dos_reformat_20250615.pd.parquet",
        task_configs=task_configs,
        batch_size=8,
        num_workers=0,
        predict_idx="all",
    )

    # Setup for prediction
    datamodule.setup(stage="predict")

    if datamodule.predict_dataset is None:
        print("ERROR: predict_dataset is None")
        return

    print(f"Dataset length: {len(datamodule.predict_dataset)}")

    # Initialize model
    model = FlexibleMultiTaskModel(
        shared_block_dims=[290, 128],
        task_configs=task_configs,
        with_structure=False,
    )

    # Get predict dataloader
    predict_dataloader = datamodule.predict_dataloader()

    if predict_dataloader is None:
        print("ERROR: predict_dataloader is None")
        return

    print("\n=== TESTING MODEL PREDICTION ===")

    # Test a few batches
    total_input_samples = 0
    total_output_samples = 0

    for batch_idx, batch in enumerate(predict_dataloader):
        if batch_idx >= 5:  # Test first 5 batches
            break

        print(f"\nBatch {batch_idx}:")

        # Get batch data
        model_input_x, sample_y_dict, sample_task_masks_dict, sample_t_sequences_dict = batch

        input_batch_size = len(sample_y_dict["dos"])
        total_input_samples += input_batch_size

        print(f"  Input batch size: {input_batch_size}")

        # Check DOS t_sequence data
        dos_t_sequence = sample_t_sequences_dict["dos"]
        print(f"  DOS t_sequence type: {type(dos_t_sequence)}")
        print(f"  DOS t_sequence length: {len(dos_t_sequence)}")

        # Count valid vs placeholder samples
        valid_samples = 0
        placeholder_samples = 0

        for i in range(input_batch_size):
            t_seq = dos_t_sequence[i]
            if len(t_seq) == 1 and t_seq[0].item() == 0.0:
                placeholder_samples += 1
            else:
                valid_samples += 1

        print(f"  Valid samples: {valid_samples}, Placeholder samples: {placeholder_samples}")

        # Test model prediction
        try:
            with torch.no_grad():
                predictions = model.predict_step(batch, batch_idx)

            print(f"  Prediction keys: {list(predictions.keys())}")

            # Check DOS predictions
            if "dos_value" in predictions:
                dos_predictions = predictions["dos_value"]
                print(f"  DOS predictions type: {type(dos_predictions)}")

                if isinstance(dos_predictions, list):
                    output_batch_size = len(dos_predictions)
                    print(f"  Output batch size: {output_batch_size}")
                    total_output_samples += output_batch_size

                    # Check individual predictions
                    for i in range(min(3, output_batch_size)):  # Show first 3
                        pred = dos_predictions[i]
                        print(
                            f"    Sample {i}: prediction shape={pred.shape if hasattr(pred, 'shape') else 'N/A'}, type={type(pred)}"
                        )

                elif isinstance(dos_predictions, np.ndarray):
                    output_batch_size = dos_predictions.shape[0] if dos_predictions.ndim > 0 else 1
                    print(f"  Output batch size: {output_batch_size}")
                    total_output_samples += output_batch_size
                    print(f"  DOS predictions shape: {dos_predictions.shape}")
                else:
                    print(f"  Unexpected DOS predictions type: {type(dos_predictions)}")
            else:
                print("  No 'dos_value' in predictions!")

        except Exception as e:
            print(f"  ERROR during prediction: {e}")
            import traceback

            traceback.print_exc()

    print("\n=== SUMMARY ===")
    print(f"Total input samples: {total_input_samples}")
    print(f"Total output samples: {total_output_samples}")
    print(f"Sample loss: {total_input_samples - total_output_samples}")
    print(f"Sample retention rate: {total_output_samples / total_input_samples * 100:.1f}%")


if __name__ == "__main__":
    test_full_prediction_flow()
