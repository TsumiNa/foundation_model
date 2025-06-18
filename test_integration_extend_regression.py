#!/usr/bin/env python3
"""
Integration test for ExtendRegressionHead with FlexibleMultiTaskModel.
Tests the DOSDataset-style data expansion logic.
"""

import torch

from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import ExtendRegressionTaskConfig


def test_flexible_model_with_extend_regression():
    """Test FlexibleMultiTaskModel with ExtendRegressionHead integration."""
    print("Testing FlexibleMultiTaskModel with ExtendRegressionHead integration...")

    # Create ExtendRegressionTaskConfig
    extend_config = ExtendRegressionTaskConfig(
        name="test_dos_prediction",
        x_dim=[128, 64, 32],  # Input from shared encoder: 128, hidden layers: 64, output: 32
        t_dim=[16, 8],  # t_embedding_dim=16, hidden=8, output=1 (will be added automatically)
        interaction_dim=8,
        t_encoding_method="fourier",
    )

    # Create FlexibleMultiTaskModel
    model = FlexibleMultiTaskModel(
        shared_block_dims=[64, 128],  # Input: 64, Output (deposit_dim): 128
        task_configs=[extend_config],
        norm_shared=True,
        residual_shared=False,
    )

    print(f"✓ Model created with task: {extend_config.name}")
    print(f"  - Deposit dim: {model.deposit_dim}")
    print(f"  - Has extend regression: {model.has_extend_regression}")

    # Test data expansion logic
    batch_size = 3
    seq_len = 4

    # Create test inputs
    x_formula = torch.randn(batch_size, 64)  # Feature input
    t_sequence = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.0],  # Sample 1: 3 valid t values
            [0.5, 1.5, 0.0, 0.0],  # Sample 2: 2 valid t values
            [0.1, 0.2, 0.3, 0.4],  # Sample 3: 4 valid t values
        ]
    )  # Shape: (3, 4)

    task_sequence_data_batch = {"test_dos_prediction": t_sequence}

    print("✓ Test data created:")
    print(f"  - x_formula shape: {x_formula.shape}")
    print(f"  - t_sequence shape: {t_sequence.shape}")
    print("  - Valid t counts: [3, 2, 4] = 9 total")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x_formula, task_sequence_data_batch)

    pred_tensor = outputs["test_dos_prediction"]
    print("✓ Forward pass successful:")
    print(f"  - Output shape: {pred_tensor.shape}")
    print(f"  - Expected total points: 9, Got: {pred_tensor.shape[0]}")

    # Verify the expansion worked correctly
    expected_total_points = 3 + 2 + 4  # 9 total valid t values
    assert pred_tensor.shape[0] == expected_total_points, (
        f"Expected {expected_total_points} points, got {pred_tensor.shape[0]}"
    )
    assert pred_tensor.shape[1] == 1, f"Expected output dim 1, got {pred_tensor.shape[1]}"

    print("✓ Data expansion verification passed")

    # Test with different t_encoding_method
    extend_config_fc = ExtendRegressionTaskConfig(
        name="test_dos_fc",
        x_dim=[128, 64, 32],
        t_dim=[20, 10],  # t_embedding_dim=20
        interaction_dim=8,
        t_encoding_method="fc",
    )

    model_fc = FlexibleMultiTaskModel(
        shared_block_dims=[64, 128], task_configs=[extend_config_fc], norm_shared=True, residual_shared=False
    )

    task_sequence_data_batch_fc = {"test_dos_fc": t_sequence}

    model_fc.eval()
    with torch.no_grad():
        outputs_fc = model_fc(x_formula, task_sequence_data_batch_fc)

    pred_tensor_fc = outputs_fc["test_dos_fc"]
    print("✓ FC encoding test successful:")
    print(f"  - Output shape: {pred_tensor_fc.shape}")

    # Test edge case: all zeros (no valid t values)
    t_sequence_zeros = torch.zeros(batch_size, seq_len)
    task_sequence_data_batch_zeros = {"test_dos_prediction": t_sequence_zeros}

    with torch.no_grad():
        outputs_zeros = model(x_formula, task_sequence_data_batch_zeros)

    pred_tensor_zeros = outputs_zeros["test_dos_prediction"]
    print("✓ Edge case (all zeros) test:")
    print(f"  - Output shape: {pred_tensor_zeros.shape}")
    print(f"  - Expected 0 points, Got: {pred_tensor_zeros.shape[0]}")

    assert pred_tensor_zeros.shape[0] == 0, f"Expected 0 points for all-zero input, got {pred_tensor_zeros.shape[0]}"

    print("\n🎉 All integration tests passed!")


if __name__ == "__main__":
    test_flexible_model_with_extend_regression()
