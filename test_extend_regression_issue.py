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

    # Now test sample_task_masks_dict
    print("\n=== TESTING SAMPLE_TASK_MASKS_DICT ===")

    total_mask_samples_processed = 0
    dos_mask_valid_count = 0
    dos_mask_invalid_count = 0
    density_mask_valid_count = 0
    density_mask_invalid_count = 0

    for batch_idx, batch in enumerate(predict_dataloader):
        if batch_idx >= 10:  # Only check first 10 batches
            break

        model_input_x, sample_y_dict, sample_task_masks_dict, sample_t_sequences_dict = batch

        batch_size = len(sample_y_dict["dos"])
        total_mask_samples_processed += batch_size

        print(f"\nBatch {batch_idx} - Mask Analysis:")
        print(f"  Batch size: {batch_size}")

        # Test mask structure
        print(f"  Available mask keys: {list(sample_task_masks_dict.keys())}")

        # Test DOS masks (ExtendRegression - should be List[Tensor])
        if "dos" in sample_task_masks_dict:
            dos_masks = sample_task_masks_dict["dos"]
            dos_y_data = sample_y_dict["dos"]

            print(f"  DOS mask type: {type(dos_masks)}")
            print(f"  DOS mask length: {len(dos_masks)}")

            # Verify it's List[Tensor] format
            if isinstance(dos_masks, list):
                print("  ✓ DOS masks correctly use List[Tensor] format")

                # Check each sample's mask
                for i in range(min(batch_size, 3)):  # Check first 3 samples
                    mask_tensor = dos_masks[i]
                    y_tensor = dos_y_data[i]

                    print(f"    Sample {i}:")
                    print(f"      Mask shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}")
                    print(f"      Y data shape: {y_tensor.shape}")

                    # Verify shapes match
                    if mask_tensor.shape == y_tensor.shape:
                        print("      ✓ Mask and y_data shapes match")
                    else:
                        print("      ✗ Mask and y_data shapes don't match!")

                    # Verify mask dtype is bool
                    if mask_tensor.dtype == torch.bool:
                        print("      ✓ Mask dtype is bool")
                    else:
                        print(f"      ✗ Mask dtype is {mask_tensor.dtype}, expected bool")

                    # Check mask values logic
                    mask_all_true = torch.all(mask_tensor).item()
                    mask_all_false = torch.all(~mask_tensor).item()
                    mask_valid_count = torch.sum(mask_tensor).item()

                    # Check if this sample has placeholder data
                    is_placeholder = len(y_tensor) == 1 and y_tensor[0].item() == 0.0

                    print(
                        f"      Y data length: {len(y_tensor)}, first value: {y_tensor[0].item() if len(y_tensor) > 0 else 'N/A'}"
                    )
                    print(f"      Is placeholder: {is_placeholder}")
                    print(f"      Mask valid count: {mask_valid_count}/{len(mask_tensor)}")

                    if is_placeholder:
                        if mask_all_false:
                            print("      ✓ Placeholder sample correctly has all-False mask")
                            dos_mask_invalid_count += 1
                        else:
                            print("      ✗ Placeholder sample should have all-False mask but doesn't")
                    else:
                        if mask_all_true:
                            print("      ✓ Valid sample correctly has all-True mask")
                            dos_mask_valid_count += 1
                        else:
                            print(f"      ? Valid sample has partial mask: {mask_valid_count}/{len(mask_tensor)} True")
                            # This might be due to random masking or other valid reasons
                            dos_mask_valid_count += 1

                # Count valid/invalid masks for this batch
                for i in range(batch_size):
                    mask_tensor = dos_masks[i]
                    y_tensor = dos_y_data[i]
                    is_placeholder = len(y_tensor) == 1 and y_tensor[0].item() == 0.0
                    mask_has_valid = torch.any(mask_tensor).item()

                    if not is_placeholder and mask_has_valid:
                        if i >= 3:  # Only count those not already counted above
                            dos_mask_valid_count += 1
                    elif is_placeholder and not mask_has_valid:
                        if i >= 3:  # Only count those not already counted above
                            dos_mask_invalid_count += 1
            else:
                print(f"  ✗ DOS masks should be List[Tensor] but got {type(dos_masks)}")

        # Test Density masks (Regression - should be Tensor)
        if "density" in sample_task_masks_dict:
            density_masks = sample_task_masks_dict["density"]
            density_y_data = sample_y_dict["density"]

            print(f"  Density mask type: {type(density_masks)}")
            print(f"  Density mask shape: {density_masks.shape}")

            # Verify it's Tensor format
            if isinstance(density_masks, torch.Tensor):
                print("  ✓ Density masks correctly use Tensor format")

                # Verify shapes match
                if density_masks.shape[0] == density_y_data.shape[0]:
                    print("  ✓ Density mask and y_data batch sizes match")
                else:
                    print("  ✗ Density mask and y_data batch sizes don't match!")

                # Verify mask dtype is bool
                if density_masks.dtype == torch.bool:
                    print("  ✓ Density mask dtype is bool")
                else:
                    print(f"  ✗ Density mask dtype is {density_masks.dtype}, expected bool")

                # Count valid masks
                valid_count = torch.sum(density_masks).item()
                invalid_count = density_masks.shape[0] - valid_count

                density_mask_valid_count += valid_count
                density_mask_invalid_count += invalid_count

                print(f"  Density mask stats: {valid_count} valid, {invalid_count} invalid")

                # Check first few samples
                for i in range(min(batch_size, 3)):
                    mask_val = density_masks[i].item() if density_masks.dim() > 1 else density_masks[i].item()
                    y_val = density_y_data[i].item() if hasattr(density_y_data[i], "item") else density_y_data[i]
                    print(f"    Sample {i}: mask={mask_val}, y_data={y_val}")
            else:
                print(f"  ✗ Density masks should be Tensor but got {type(density_masks)}")

    print("\n=== MASK TESTING SUMMARY ===")
    print(f"Total samples processed: {total_mask_samples_processed}")
    print("DOS mask summary:")
    print(f"  Valid (non-placeholder): {dos_mask_valid_count}")
    print(f"  Invalid (placeholder): {dos_mask_invalid_count}")
    print(f"  Percentage with valid DOS masks: {dos_mask_valid_count / total_mask_samples_processed * 100:.1f}%")
    print("Density mask summary:")
    print(f"  Valid: {density_mask_valid_count}")
    print(f"  Invalid: {density_mask_invalid_count}")
    print(
        f"  Percentage with valid Density masks: {density_mask_valid_count / total_mask_samples_processed * 100:.1f}%"
    )

    # Test mask consistency with data
    print("\n=== MASK-DATA CONSISTENCY CHECK ===")
    consistency_issues = 0

    for batch_idx, batch in enumerate(predict_dataloader):
        if batch_idx >= 5:  # Check first 5 batches for consistency
            break

        model_input_x, sample_y_dict, sample_task_masks_dict, sample_t_sequences_dict = batch

        # Check DOS consistency
        if "dos" in sample_task_masks_dict and "dos" in sample_y_dict:
            dos_masks = sample_task_masks_dict["dos"]
            dos_y_data = sample_y_dict["dos"]

            for i, (mask, y_data) in enumerate(zip(dos_masks, dos_y_data)):
                is_placeholder = len(y_data) == 1 and y_data[0].item() == 0.0
                mask_indicates_invalid = torch.all(~mask).item()

                # Consistency check: placeholder data should have all-False mask
                if is_placeholder and not mask_indicates_invalid:
                    print(
                        f"  ✗ Consistency issue in batch {batch_idx}, sample {i}: placeholder data but mask not all-False"
                    )
                    consistency_issues += 1
                elif not is_placeholder and mask_indicates_invalid:
                    print(f"  ✗ Consistency issue in batch {batch_idx}, sample {i}: valid data but mask all-False")
                    consistency_issues += 1

    if consistency_issues == 0:
        print("  ✓ No mask-data consistency issues found")
    else:
        print(f"  ✗ Found {consistency_issues} mask-data consistency issues")

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
