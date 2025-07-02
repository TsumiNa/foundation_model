#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.model_config import ExtendRegressionTaskConfig, RegressionTaskConfig, TaskType


def test_extend_regression_issue():
    """Test the ExtendRegression issue with NaN handling."""

    print("=== TESTING EXTEND REGRESSION NaN HANDLING ===")

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
        batch_size=4,
        num_workers=0,
        predict_idx="all",
    )

    # Setup for prediction
    datamodule.setup(stage="predict")

    if datamodule.predict_dataset is None:
        print("ERROR: predict_dataset is None")
        return

    print(f"Dataset length: {len(datamodule.predict_dataset)}")

    # Get predict dataloader
    predict_dataloader = datamodule.predict_dataloader()

    if predict_dataloader is None:
        print("ERROR: predict_dataloader is None")
        return

    # Analyze first few batches
    print("\n=== ANALYZING BATCHES ===")

    total_samples_processed = 0
    dos_samples_with_valid_data = 0
    dos_samples_with_placeholder_data = 0

    for batch_idx, batch in enumerate(predict_dataloader):
        if batch_idx >= 10:  # Only check first 10 batches
            break

        model_input_x, sample_y_dict, sample_task_masks_dict, sample_t_sequences_dict = batch

        batch_size = len(sample_y_dict["dos"])
        total_samples_processed += batch_size

        print(f"\nBatch {batch_idx}:")
        print(f"  Batch size: {batch_size}")

        # Check DOS data
        dos_y_data = sample_y_dict["dos"]
        dos_t_data = sample_t_sequences_dict["dos"]

        print(f"  DOS y_data type: {type(dos_y_data)}")
        print(f"  DOS y_data length: {len(dos_y_data)}")

        # Analyze each sample in the batch
        for i in range(batch_size):
            y_seq = dos_y_data[i]
            t_seq = dos_t_data[i]

            # Check if this is a placeholder (all zeros or length 1)
            if len(y_seq) == 1 and y_seq[0].item() == 0.0:
                dos_samples_with_placeholder_data += 1
                if batch_idx == 0 and i < 3:  # Show details for first few
                    print(f"    Sample {i}: PLACEHOLDER - y_seq length={len(y_seq)}, t_seq length={len(t_seq)}")
            else:
                dos_samples_with_valid_data += 1
                if batch_idx == 0 and i < 3:  # Show details for first few
                    print(f"    Sample {i}: VALID - y_seq length={len(y_seq)}, t_seq length={len(t_seq)}")

    print("\n=== SUMMARY ===")
    print(f"Total samples processed: {total_samples_processed}")
    print(f"DOS samples with valid data: {dos_samples_with_valid_data}")
    print(f"DOS samples with placeholder data: {dos_samples_with_placeholder_data}")
    print(f"Percentage with placeholder data: {dos_samples_with_placeholder_data / total_samples_processed * 100:.1f}%")

    # Now test the model expansion logic
    print("\n=== TESTING MODEL EXPANSION LOGIC ===")

    # Get one batch
    batch = next(iter(predict_dataloader))
    model_input_x, sample_y_dict, sample_task_masks_dict, sample_t_sequences_dict = batch

    # Simulate h_task (deposit layer output)
    batch_size = len(sample_y_dict["dos"])
    h_task = torch.randn(batch_size, 128)  # Simulate deposit layer output

    # Get DOS t_sequence
    dos_t_sequence = sample_t_sequences_dict["dos"]

    print(f"Input batch size: {batch_size}")
    print(f"DOS t_sequence type: {type(dos_t_sequence)}")
    print(f"DOS t_sequence length: {len(dos_t_sequence)}")

    # Test the expansion logic manually
    expanded_h_list = []
    expanded_t_list = []

    for batch_idx in range(batch_size):
        t_sample = dos_t_sequence[batch_idx]
        h_sample = h_task[batch_idx]

        print(
            f"Sample {batch_idx}: t_sample shape={t_sample.shape}, values={t_sample[:5].tolist() if len(t_sample) > 5 else t_sample.tolist()}"
        )

        # This is the problematic logic from the model
        if t_sample.numel() > 0:
            valid_mask = t_sample != 0.0
            valid_t = t_sample[valid_mask] if valid_mask.any() else t_sample

            print(f"  valid_mask.any(): {valid_mask.any()}")
            print(f"  len(valid_t): {len(valid_t)}")

            if len(valid_t) > 0:
                h_replicated = h_sample.unsqueeze(0).repeat(len(valid_t), 1)
                expanded_h_list.append(h_replicated)
                expanded_t_list.append(valid_t)
                print("  -> INCLUDED in expansion")
            else:
                print("  -> EXCLUDED from expansion (no valid t values)")
        else:
            print("  -> EXCLUDED from expansion (empty t_sample)")

    if expanded_h_list:
        expanded_h_task = torch.cat(expanded_h_list, dim=0)
        expanded_t = torch.cat(expanded_t_list, dim=0)
        print("\nExpansion result:")
        print(f"  Original batch size: {batch_size}")
        print(f"  Expanded size: {len(expanded_h_task)}")
        print(f"  Samples included: {len(expanded_h_list)}")
        print(f"  Samples excluded: {batch_size - len(expanded_h_list)}")
    else:
        print("\nExpansion result: NO SAMPLES INCLUDED!")


if __name__ == "__main__":
    test_extend_regression_issue()
