import numpy as np
import pandas as pd

from multi_task_splitter import MultiTaskSplitter


def create_test_data():
    """Create synthetic data with NaN values for testing."""
    np.random.seed(42)
    n_samples = 100

    # Create three tasks with different amounts of available data
    data = pd.DataFrame(
        {
            "task1": np.random.randn(n_samples),  # All samples available
            "task2": np.random.randn(n_samples),  # Will make 30% NaN
            "task3": np.random.randn(n_samples),  # Will make 60% NaN
        }
    )

    # Introduce NaN values
    mask2 = np.random.choice([True, False], size=n_samples, p=[0.7, 0.3])
    mask3 = np.random.choice([True, False], size=n_samples, p=[0.4, 0.6])

    data.loc[~mask2, "task2"] = np.nan
    data.loc[~mask3, "task3"] = np.nan

    return data


def test_splitter():
    """Test the MultiTaskSplitter with synthetic data."""
    # Create test data
    data = create_test_data()

    print("Data availability per task:")
    print(data.notna().sum())
    print("\nPercentage of available data per task:")
    print(data.notna().mean() * 100)

    # Test train/val split (90/10)
    splitter = MultiTaskSplitter(
        train_ratio=0.9, val_ratio=0.1, test_ratio=0.0, random_state=42
    )

    # Split data
    train_idx, val_idx, test_idx = splitter.split(data)

    print("\nSplit sizes:")
    print(f"Train: {len(train_idx)}")
    print(f"Val: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")

    # Check task representation in each split
    print("\nTask representation in splits:")
    splits = {"Train": train_idx, "Val": val_idx}
    for split_name, indices in splits.items():
        print(f"\n{split_name} split:")
        split_data = data.iloc[indices]
        print(split_data.notna().sum())
        print(f"Percentage of total data: {len(indices) / len(data) * 100:.1f}%")


if __name__ == "__main__":
    test_splitter()
