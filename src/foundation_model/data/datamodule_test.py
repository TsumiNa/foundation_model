import logging  # Import logging
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from .datamodule import CompoundDataModule


# --- Mock Objects (can be shared or re-defined if specific variations needed) ---
class MockTaskType:  # Re-using from dataset_test or define locally
    REGRESSION = SimpleNamespace(name="REGRESSION")
    SEQUENCE = SimpleNamespace(name="SEQUENCE")


class MockTaskConfig:  # Re-using from dataset_test or define locally
    def __init__(
        self, name, task_type, enabled=True, dims=None, optimizer=None
    ):  # Added optimizer for datamodule tests
        self.name = name
        self.type = task_type
        self.enabled = enabled
        self.dims = dims
        self.optimizer = optimizer


# --- Fixtures ---
@pytest.fixture
def base_formula_df():
    return pd.DataFrame({"f1": np.random.rand(20), "f2": np.random.rand(20)}, index=[f"s{i}" for i in range(20)])


@pytest.fixture
def base_attributes_df():
    data = {
        "task1_regression_value": np.random.rand(20),
        "task2_sequence_series": [[np.random.rand() for _ in range(5)] for _ in range(20)],
        "task2_temps": [[np.random.rand() for _ in range(5)] for _ in range(20)],
        "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,  # Predefined split
    }
    return pd.DataFrame(data, index=[f"s{i}" for i in range(20)])


@pytest.fixture
def base_structure_df():
    return pd.DataFrame({"s1": np.random.rand(20), "s2": np.random.rand(20)}, index=[f"s{i}" for i in range(20)])


@pytest.fixture
def sample_task_configs_dm():  # Slightly different from dataset_test if needed
    return [
        MockTaskConfig(name="task1", task_type=MockTaskType.REGRESSION, dims=[None, 1]),
        MockTaskConfig(name="task2", task_type=MockTaskType.SEQUENCE, dims=[None, 5]),
    ]


@pytest.fixture
def temp_files_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# --- Test Cases ---


def test_datamodule_init_with_dfs(base_formula_df, base_attributes_df, sample_task_configs_dm):
    """Test initialization with pre-loaded DataFrames."""
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=base_attributes_df,
        task_configs=sample_task_configs_dm,
        batch_size=4,
    )
    assert dm.formula_df is not None
    assert dm.attributes_df is not None
    assert dm.structure_df is None
    assert not dm.actual_with_structure


def test_datamodule_init_with_structure_dfs(
    base_formula_df, base_attributes_df, base_structure_df, sample_task_configs_dm
):
    """Test initialization with structure DataFrame."""
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=base_attributes_df,
        structure_desc_source=base_structure_df,
        task_configs=sample_task_configs_dm,
        with_structure=True,
        batch_size=4,
    )
    assert dm.structure_df is not None
    assert dm.actual_with_structure
    assert len(dm.formula_df) == len(dm.structure_df)  # Check alignment


def test_datamodule_load_data_from_paths(temp_files_dir, base_formula_df, base_attributes_df, sample_task_configs_dm):
    """Test _load_data method with file paths."""
    formula_pkl_path = os.path.join(temp_files_dir, "formula.pkl")
    attributes_csv_path = os.path.join(temp_files_dir, "attributes.csv")

    base_formula_df.to_pickle(formula_pkl_path)
    base_attributes_df.to_csv(attributes_csv_path)  # index=True is default for to_csv

    dm = CompoundDataModule(
        formula_desc_source=formula_pkl_path,
        attributes_source=attributes_csv_path,
        task_configs=sample_task_configs_dm,
        batch_size=4,
    )
    assert dm.formula_df is not None
    assert dm.attributes_df is not None
    pd.testing.assert_frame_equal(dm.formula_df, base_formula_df, check_dtype=False)
    # CSV loading might change dtypes for object columns (like list of lists for series)
    # So we might only compare shapes or specific columns if dtypes are an issue after CSV load.
    # For this test, assuming attributes_df structure is simple enough or CompoundDataset handles it.
    assert dm.attributes_df.shape == base_attributes_df.shape


def test_datamodule_alignment(base_formula_df, base_attributes_df, sample_task_configs_dm):
    """Test DataFrame alignment logic."""
    # Create misaligned data
    formula_misaligned = base_formula_df.sample(frac=0.8, random_state=1)  # 80% of samples
    attributes_misaligned = base_attributes_df.sample(frac=0.7, random_state=2)  # 70% of samples

    dm = CompoundDataModule(
        formula_desc_source=formula_misaligned,
        attributes_source=attributes_misaligned,
        task_configs=sample_task_configs_dm,
    )
    # Check that lengths are now aligned to the intersection of indices
    common_idx_len = len(attributes_misaligned.index.intersection(formula_misaligned.index))
    assert len(dm.formula_df) == common_idx_len
    assert len(dm.attributes_df) == common_idx_len


def test_datamodule_setup_split_column(base_formula_df, base_attributes_df, sample_task_configs_dm):
    """Test setup method using the 'split' column."""
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=base_attributes_df,  # base_attributes_df has 'split' column
        task_configs=sample_task_configs_dm,
        batch_size=4,
    )
    dm.setup(stage="fit")
    assert len(dm.train_idx) == 10
    assert len(dm.val_idx) == 5
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None

    dm.setup(stage="test")
    assert len(dm.test_idx) == 5
    assert dm.test_dataset is not None


def test_datamodule_setup_random_split(base_formula_df, base_attributes_df, sample_task_configs_dm):
    """Test setup method with random train/val/test splits."""
    # Remove 'split' column to force random split
    attributes_no_split = base_attributes_df.drop(columns=["split"])
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_no_split,
        task_configs=sample_task_configs_dm,
        val_split=0.2,  # 20% of (100%-10%) = 20% of 90% = 18% of total for val
        test_split=0.1,  # 10% of total for test
        batch_size=4,
    )
    dm.setup(stage="fit")
    # Total 20 samples. Test = 0.1*20 = 2. Train_val = 18. Val = 0.2 * 18 ~ 3-4. Train ~ 14-15
    # Effective val_split = 0.2 / (1-0.1) = 0.2 / 0.9 = 0.222...
    # Val samples = round(18 * 0.222...) = round(4) = 4
    # Train samples = 18 - 4 = 14
    assert len(dm.test_idx) == 2  # 0.1 * 20
    assert len(dm.val_idx) == 4  # (20-2)*0.2 / (1-0.1) -> 18 * (0.2/0.9) = 18 * 0.222... = 4
    assert len(dm.train_idx) == 14  # 20 - 2 - 4
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None


def test_datamodule_setup_test_all(base_formula_df, base_attributes_df, sample_task_configs_dm):
    """Test setup with test_all=True."""
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=base_attributes_df,
        task_configs=sample_task_configs_dm,
        test_all=True,
        batch_size=4,
    )
    dm.setup(stage="test")
    assert len(dm.test_idx) == 20
    assert dm.train_idx.empty
    assert dm.val_idx.empty
    assert dm.test_dataset is not None


def test_datamodule_dataloaders(base_formula_df, base_attributes_df, sample_task_configs_dm):
    """Test that dataloaders are created."""
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=base_attributes_df,
        task_configs=sample_task_configs_dm,
        batch_size=4,
    )
    dm.setup(stage="fit")
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
    dm.setup(stage="test")
    assert dm.test_dataloader() is not None
    dm.setup(stage="predict")
    assert dm.predict_dataloader() is not None


def test_datamodule_structure_usage_in_dataset(
    base_formula_df, base_attributes_df, base_structure_df, sample_task_configs_dm
):
    """Test that use_structure_for_this_dataset is correctly passed to CompoundDataset."""
    # Case 1: with_structure=True, structure_df is valid
    dm_with_struct = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=base_attributes_df,
        structure_desc_source=base_structure_df,
        task_configs=sample_task_configs_dm,
        with_structure=True,
    )
    dm_with_struct.setup("fit")
    assert dm_with_struct.train_dataset.use_structure_for_this_dataset is True
    assert dm_with_struct.train_dataset.x_struct is not None

    # Case 2: with_structure=True, but structure_df is None (e.g. load failed or not provided)
    dm_struct_missing = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=base_attributes_df,
        structure_desc_source=None,  # Structure not provided
        task_configs=sample_task_configs_dm,
        with_structure=True,
    )
    dm_struct_missing.setup("fit")
    assert dm_struct_missing.train_dataset.use_structure_for_this_dataset is False
    assert dm_struct_missing.train_dataset.x_struct is None

    # Case 3: with_structure=False
    dm_no_struct_flag = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=base_attributes_df,
        structure_desc_source=base_structure_df,  # Provide it, but flag is false
        task_configs=sample_task_configs_dm,
        with_structure=False,
    )
    dm_no_struct_flag.setup("fit")
    assert dm_no_struct_flag.train_dataset.use_structure_for_this_dataset is False
    assert dm_no_struct_flag.train_dataset.x_struct is None


def test_datamodule_init_with_empty_sources_raises_value_error(sample_task_configs_dm):
    """Test CompoundDataModule raises ValueError if initialized with empty DataFrames."""
    empty_df = pd.DataFrame()
    non_empty_df = pd.DataFrame({"col1": [1]})  # Needs to be non-empty to pass first check

    with pytest.raises(ValueError, match="Formula and attributes DataFrames cannot be empty after loading."):
        CompoundDataModule(
            formula_desc_source=empty_df,  # This one is empty
            attributes_source=non_empty_df,  # This one is not
            task_configs=sample_task_configs_dm,
            batch_size=4,
        )

    with pytest.raises(ValueError, match="Formula and attributes DataFrames cannot be empty after loading."):
        CompoundDataModule(
            formula_desc_source=non_empty_df,  # This one is not
            attributes_source=empty_df,  # This one is empty
            task_configs=sample_task_configs_dm,
            batch_size=4,
        )

    with pytest.raises(ValueError, match="Formula and attributes DataFrames cannot be empty after loading."):
        CompoundDataModule(
            formula_desc_source=empty_df,  # Both empty
            attributes_source=empty_df,
            task_configs=sample_task_configs_dm,
            batch_size=4,
        )

    # Also test the None case (file not found)
    with pytest.raises(ValueError, match="formula_desc_source and attributes_source must be successfully loaded"):
        CompoundDataModule(
            formula_desc_source="non_existent_file.pkl",  # This will return None from _load_data
            attributes_source=non_empty_df,
            task_configs=sample_task_configs_dm,
            batch_size=4,
        )


def test_datamodule_empty_dataset_returns_none_dataloader(sample_task_configs_dm, caplog):
    """Test that an empty dataset results in a None dataloader and logs a warning/info."""
    # Create a DM instance more carefully for testing dataloader logic
    # We need non-empty formula_df and attributes_df to pass __init__
    # then we'll manually set indices and datasets to test the dataloader part.
    dummy_formula = pd.DataFrame({"f1": [1, 2]}, index=["s0", "s1"])
    dummy_attrs = pd.DataFrame({"task1_regression_value": [1, 2]}, index=["s0", "s1"])

    dm = CompoundDataModule(
        formula_desc_source=dummy_formula,
        attributes_source=dummy_attrs,
        task_configs=sample_task_configs_dm,  # Use the provided fixture
        batch_size=4,
    )

    # Force empty datasets for testing dataloader methods
    dm.train_idx = pd.Index([])
    dm.val_idx = pd.Index([])
    dm.test_idx = pd.Index([])
    dm.predict_idx = pd.Index([])  # predict_idx is also set in setup

    # Manually set datasets to None to trigger the specific log messages
    dm.train_dataset = None
    dm.val_dataset = None
    dm.test_dataset = None
    dm.predict_dataset = None  # This is set in setup(stage='predict')

    # Test train_dataloader
    caplog.clear()  # Clear previous logs
    assert dm.train_dataloader() is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "train_dataloader: Train dataset is empty or not initialized. Returning None." in caplog.records[0].message

    # Test val_dataloader
    caplog.clear()
    dm_logger = logging.getLogger("foundation_model.data.datamodule")
    original_level = dm_logger.getEffectiveLevel()
    dm_logger.setLevel(logging.INFO)
    try:
        assert dm.val_dataloader() is None
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert (
            "Validation dataset is empty or not initialized. Returning None for val_dataloader."
            in caplog.records[0].message
        )
    finally:
        dm_logger.setLevel(original_level)

    # Test test_dataloader
    caplog.clear()
    dm_logger.setLevel(logging.INFO)
    try:
        assert dm.test_dataloader() is None
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert (
            "Test dataset is empty or not initialized. Returning None for test_dataloader." in caplog.records[0].message
        )
    finally:
        dm_logger.setLevel(original_level)

    # Test predict_dataloader
    caplog.clear()
    dm_logger.setLevel(logging.INFO)
    try:
        assert dm.predict_dataloader() is None
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert (
            "Predict dataset is empty or not initialized. Returning None for predict_dataloader."
            in caplog.records[0].message
        )
    finally:
        dm_logger.setLevel(original_level)


# TODO:
# - Test _load_data with np.ndarray source.
# - Test _load_data with unsupported file type.
# - Test _load_data with FileNotFoundError.
# - Test alignment when structure_df index doesn't match and it gets set to None.
# - Test setup() when val_split is 0 or test_split is 0.
# - Test setup() when train_idx becomes empty after splits.
# - Test predict_idx logic in setup(stage="predict") more thoroughly.
# - Test task_masking_ratios being passed to train_dataset but not others.
