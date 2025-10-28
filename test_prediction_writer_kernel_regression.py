#!/usr/bin/env python3

"""
Test PredictionDataFrameWriter compatibility with the KernelRegressionHead.

This test verifies that the Lightning callback can correctly process
predictions from KernelRegressionHead and save them to DataFrame format.
"""

import tempfile
from pathlib import Path

import pandas as pd
import torch

from foundation_model.models.model_config import KernelRegressionTaskConfig
from foundation_model.models.task_head.kernel_regression import KernelRegressionHead
from foundation_model.scripts.callbacks.prediction_writer import PredictionDataFrameWriter


def test_prediction_writer_compatibility():
    """Test that PredictionDataFrameWriter works with KernelRegressionHead output format."""

    print("Testing PredictionDataFrameWriter compatibility with KernelRegressionHead...")

    # 1. Create KernelRegressionHead
    config = KernelRegressionTaskConfig(
        name="test_dos_prediction",
        x_dim=[64, 32, 16],
        t_dim=[32, 16, 8],
        kernel_num_centers=5,
        t_encoding_method="fourier",
    )

    head = KernelRegressionHead(config)
    print(f"✓ Created KernelRegressionHead with task: {config.name}")

    # 2. Create test data - simulate expanded format from FlexibleMultiTaskModel
    batch_size = 3
    total_points = 10  # Total expanded points across all samples

    # Simulate model forward output (this would come from FlexibleMultiTaskModel forward pass)
    raw_output = torch.randn(total_points, 1)  # Shape: (total_points, 1)

    # 3. Get predictions using head.predict (this is what FlexibleMultiTaskModel.predict_step calls)
    predictions = head.predict(raw_output)
    print("✓ Head predictions generated:")
    for key, value in predictions.items():
        print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")

    # 4. Simulate multiple batches (as would happen in Lightning)
    batch_predictions = []

    # Batch 1: 4 points
    batch1_output = torch.randn(4, 1)
    batch1_pred = head.predict(batch1_output)
    batch_predictions.append(batch1_pred)

    # Batch 2: 3 points
    batch2_output = torch.randn(3, 1)
    batch2_pred = head.predict(batch2_output)
    batch_predictions.append(batch2_pred)

    # Batch 3: 3 points
    batch3_output = torch.randn(3, 1)
    batch3_pred = head.predict(batch3_output)
    batch_predictions.append(batch3_pred)

    print(f"✓ Created {len(batch_predictions)} batch predictions")

    # 5. Test PredictionDataFrameWriter
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = PredictionDataFrameWriter(output_path=tmp_dir, write_interval="epoch")

        # Process predictions (this is what Lightning calls)
        df = writer._process_predictions(batch_predictions)

        print("✓ DataFrame created successfully:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Data types:\n{df.dtypes}")

        # Verify data
        expected_total_points = 4 + 3 + 3
        assert len(df) == expected_total_points, f"Expected {expected_total_points} rows, got {len(df)}"

        # Check column name format
        expected_col = "test_dos_prediction_value"
        assert expected_col in df.columns, f"Expected column '{expected_col}' not found"

        # Check data types and values
        assert df[expected_col].dtype == float, f"Expected float dtype, got {df[expected_col].dtype}"
        assert not df[expected_col].isna().any(), "Found NaN values in predictions"

        print("✓ Data validation passed:")
        print(f"  - Total points: {len(df)}")
        print(f"  - Column name: {expected_col}")
        print(f"  - Data range: [{df[expected_col].min():.3f}, {df[expected_col].max():.3f}]")

        # 6. Test file saving
        try:
            # Simulate write_on_epoch_end
            csv_path = Path(tmp_dir) / "predictions.csv"
            pickle_path = Path(tmp_dir) / "predictions.pd.xz"

            df.to_csv(csv_path, index=True)
            df.to_pickle(pickle_path, compression="xz")

            # Verify files exist and can be read
            assert csv_path.exists(), "CSV file was not created"
            assert pickle_path.exists(), "Pickle file was not created"

            # Test reading back
            df_csv = pd.read_csv(csv_path, index_col=0)
            df_pickle = pd.read_pickle(pickle_path)

            assert df_csv.shape == df.shape, "CSV readback shape mismatch"
            assert df_pickle.shape == df.shape, "Pickle readback shape mismatch"

            print("✓ File I/O test passed:")
            print(f"  - CSV saved/loaded: {csv_path}")
            print(f"  - Pickle saved/loaded: {pickle_path}")

        except Exception as e:
            print(f"✗ File I/O test failed: {e}")
            raise

    # 7. Test edge cases
    print("\n--- Testing edge cases ---")

    # Empty predictions
    empty_df = writer._process_predictions([])
    assert empty_df.empty, "Expected empty DataFrame for empty predictions"
    print("✓ Empty predictions handled correctly")

    # Single point prediction
    single_output = torch.randn(1, 1)
    single_pred = head.predict(single_output)
    single_df = writer._process_predictions([single_pred])
    assert len(single_df) == 1, f"Expected 1 row for single prediction, got {len(single_df)}"
    print("✓ Single point prediction handled correctly")

    print("\n🎉 All PredictionDataFrameWriter compatibility tests passed!")


def test_multiple_tasks_compatibility():
    """Test PredictionDataFrameWriter with multiple kernel regression tasks."""

    print("\n--- Testing multiple KernelRegression tasks ---")

    # Create two KernelRegression tasks
    config1 = KernelRegressionTaskConfig(name="dos_prediction", x_dim=[64, 32], t_dim=[16, 8], kernel_num_centers=4)

    config2 = KernelRegressionTaskConfig(
        name="bandStructure",  # Test different naming format
        x_dim=[64, 16],
        t_dim=[8, 4],
        kernel_num_centers=3,
    )

    head1 = KernelRegressionHead(config1)
    head2 = KernelRegressionHead(config2)

    # Simulate batch predictions with both tasks
    batch_output = torch.randn(5, 1)

    pred1 = head1.predict(batch_output)
    pred2 = head2.predict(batch_output)

    # Combine predictions (as FlexibleMultiTaskModel.predict_step would)
    combined_pred = {**pred1, **pred2}

    batch_predictions = [combined_pred]

    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = PredictionDataFrameWriter(output_path=tmp_dir)
        df = writer._process_predictions(batch_predictions)

        print("✓ Multi-task DataFrame created:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")

        # Verify both tasks' columns are present
        expected_cols = ["dos_prediction_value", "band_structure_value"]
        for col in expected_cols:
            assert col in df.columns, f"Expected column '{col}' not found"

        assert len(df) == 5, f"Expected 5 rows, got {len(df)}"

        print("✓ Multi-task compatibility verified")


if __name__ == "__main__":
    test_prediction_writer_compatibility()
    test_multiple_tasks_compatibility()
    print("\n🎉 All tests completed successfully!")
