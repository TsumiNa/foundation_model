from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from foundation_model.data.dataset import CompoundDataset


# --- Mock Objects ---
class MockTaskType:
    REGRESSION = SimpleNamespace(name="REGRESSION")
    CLASSIFICATION = SimpleNamespace(name="CLASSIFICATION")
    SEQUENCE = SimpleNamespace(name="SEQUENCE")


class MockTaskConfig:
    def __init__(self, name, task_type, enabled=True, dims=None, num_classes=None):
        self.name = name
        self.type = task_type
        self.enabled = enabled
        self.dims = dims
        self.num_classes = num_classes
        # Add other attributes if CompoundDataset specifically checks for them,
        # e.g., optimizer, but for basic data processing, these might be enough.


# --- Fixtures ---


@pytest.fixture
def sample_formula_desc_df():
    """Returns a sample formula descriptor DataFrame."""
    return pd.DataFrame(
        {"feat_0": [0.1, 0.2, 0.3, 0.4, 0.5], "feat_1": [1.1, 1.2, 1.3, 1.4, 1.5]}, index=[f"id_{i}" for i in range(5)]
    )


@pytest.fixture
def sample_structure_desc_df():
    """Returns a sample structure descriptor DataFrame."""
    return pd.DataFrame(
        {"struct_feat_0": [10.1, 10.2, 10.3, 10.4, 10.5], "struct_feat_1": [11.1, 11.2, 11.3, 11.4, 11.5]},
        index=[f"id_{i}" for i in range(5)],
    )


@pytest.fixture
def sample_attributes_df():
    """Returns a sample attributes DataFrame with various data types and NaNs."""
    data = {
        # Regression task
        "task_reg_regression_value": [1.0, 2.0, np.nan, 4.0, 5.0],
        # Sequence task
        "task_seq_sequence_series": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [np.nan, np.nan, np.nan],  # All NaN sequence
            [0.7, np.nan, 0.9],  # Partial NaN sequence
            [1.0, 1.1, 1.2],
        ],
        "task_seq_temps": [[10, 20, 30], [10, 20, 30], [10, 20, 30], [10, 20, 30], [10, 20, 30]],
        # Another regression task for variety
        "task_another_reg_regression_value": [10.0, 20.0, 30.0, 40.0, np.nan],
        # A task that might be disabled or have missing columns
        "task_missing_regression_value": [100, 200, 300, 400, 500],
    }
    return pd.DataFrame(data, index=[f"id_{i}" for i in range(5)])


@pytest.fixture
def sample_task_configs():
    """Returns a list of sample MockTaskConfig objects."""
    return [
        MockTaskConfig(name="task_reg", task_type=MockTaskType.REGRESSION, dims=[None, 1]),
        MockTaskConfig(name="task_seq", task_type=MockTaskType.SEQUENCE, dims=[None, 3]),  # Assuming seq_len is 3
        MockTaskConfig(name="task_another_reg", task_type=MockTaskType.REGRESSION, dims=[None, 1]),
        MockTaskConfig(name="task_disabled", task_type=MockTaskType.REGRESSION, enabled=False),
        MockTaskConfig(
            name="task_fully_missing_data_col", task_type=MockTaskType.REGRESSION, dims=[None, 1]
        ),  # This task's data column won't exist in attributes_df
    ]


# --- Test Cases ---


def test_dataset_initialization_basic(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test basic initialization with formula and attributes."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        dataset_name="test_basic",
    )
    assert len(dataset) == 5
    assert dataset.x_formula.shape == (5, 2)
    assert dataset.x_struct is None
    assert "task_reg" in dataset.y_dict
    assert "task_seq" in dataset.y_dict
    assert "task_seq" in dataset.temps_dict
    assert "task_another_reg" in dataset.y_dict
    assert "task_disabled" not in dataset.enabled_task_names  # Check if disabled task is skipped
    assert "task_fully_missing_data_col" in dataset.y_dict  # Should have placeholder


def test_dataset_initialization_with_structure(
    sample_formula_desc_df, sample_attributes_df, sample_structure_desc_df, sample_task_configs
):
    """Test initialization with structure data."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        structure_desc=sample_structure_desc_df,
        use_structure_for_this_dataset=True,
        dataset_name="test_with_structure",
    )
    assert len(dataset) == 5
    assert dataset.x_formula.shape == (5, 2)
    assert dataset.x_struct is not None
    assert dataset.x_struct.shape == (5, 2)


def test_dataset_initialization_structure_mismatch(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test ValueError if structure length mismatches formula length."""
    short_structure_df = pd.DataFrame({"struct_feat_0": [10.1, 10.2]}, index=["id_0", "id_1"])
    with pytest.raises(ValueError, match="formula_desc and structure_desc must have the same number of samples."):
        CompoundDataset(
            formula_desc=sample_formula_desc_df,
            attributes=sample_attributes_df,
            task_configs=sample_task_configs,
            structure_desc=short_structure_df,
            use_structure_for_this_dataset=True,
            dataset_name="test_struct_mismatch",
        )


def test_task_data_processing_regression(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Verify y_dict and masks for a regression task."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        dataset_name="test_reg_processing",
    )
    task_name = "task_reg"
    assert task_name in dataset.y_dict
    assert dataset.y_dict[task_name].shape == (5, 1)
    # Expected values (NaNs become 0)
    expected_y = torch.tensor([[1.0], [2.0], [0.0], [4.0], [5.0]], dtype=torch.float32)
    assert torch.allclose(dataset.y_dict[task_name], expected_y)

    assert task_name in dataset.task_masks_dict
    assert dataset.task_masks_dict[task_name].dtype == torch.bool
    # Expected mask (True where not NaN)
    expected_mask = torch.tensor([[True], [True], [False], [True], [True]], dtype=torch.bool)
    assert torch.equal(dataset.task_masks_dict[task_name], expected_mask)


def test_task_data_processing_sequence(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Verify y_dict, temps_dict, and masks for a sequence task."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        dataset_name="test_seq_processing",
    )
    task_name = "task_seq"
    assert task_name in dataset.y_dict
    assert dataset.y_dict[task_name].shape == (5, 3)  # 5 samples, 3 sequence points
    # Expected y (NaNs become 0)
    expected_y_seq_vals = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0], [0.7, 0.0, 0.9], [1.0, 1.1, 1.2]]
    assert torch.allclose(dataset.y_dict[task_name], torch.tensor(expected_y_seq_vals, dtype=torch.float32))

    assert task_name in dataset.temps_dict
    assert dataset.temps_dict[task_name].shape == (5, 3, 1)  # 5 samples, 3 seq points, 1 channel
    expected_temps = torch.tensor([[[10], [20], [30]]] * 5, dtype=torch.float32)  # Repeated for each sample
    assert torch.allclose(dataset.temps_dict[task_name], expected_temps)

    assert task_name in dataset.task_masks_dict
    assert dataset.task_masks_dict[task_name].dtype == torch.bool
    # Mask for sequence is True if NOT ALL points are NaN
    expected_mask_seq = torch.tensor([[True], [True], [False], [True], [True]], dtype=torch.bool)
    assert torch.equal(dataset.task_masks_dict[task_name], expected_mask_seq)


def test_task_data_processing_missing_columns(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test behavior when expected attribute columns for a task are missing."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,  # sample_task_configs includes 'task_fully_missing_data_col'
        dataset_name="test_missing_cols",
    )
    task_name = "task_fully_missing_data_col"
    assert task_name in dataset.y_dict
    assert dataset.y_dict[task_name].shape == (5, 1)  # Placeholder shape
    assert torch.all(dataset.y_dict[task_name] == 0)  # Placeholder is zeros

    assert task_name in dataset.task_masks_dict
    assert dataset.task_masks_dict[task_name].shape == (5, 1)
    assert torch.all(~dataset.task_masks_dict[task_name])  # All False mask


def test_nan_masking(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test that NaNs in attributes correctly generate masks (already covered in specific task tests, but good to have a focused one)."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        dataset_name="test_nan_masking",
    )
    # Regression task 'task_reg'
    expected_mask_reg = torch.tensor([[True], [True], [False], [True], [True]], dtype=torch.bool)
    assert torch.equal(dataset.task_masks_dict["task_reg"], expected_mask_reg)

    # Sequence task 'task_seq' (mask based on all-NaN sequences)
    expected_mask_seq = torch.tensor([[True], [True], [False], [True], [True]], dtype=torch.bool)
    assert torch.equal(dataset.task_masks_dict["task_seq"], expected_mask_seq)


def test_ratio_masking_train(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test task_masking_ratios for a training dataset."""
    task_name_to_mask = "task_reg"  # This task has 4 valid samples initially
    masking_ratios = {task_name_to_mask: 0.5}  # Keep 50% of valid samples

    # Seed for reproducibility of np.random.choice
    np.random.seed(42)
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        task_masking_ratios=masking_ratios,
        is_predict_set=False,  # Training mode
        dataset_name="test_ratio_mask_train",
    )

    original_valid_mask = torch.tensor([True, True, False, True, True], dtype=torch.bool)
    num_originally_valid = torch.sum(original_valid_mask).item()  # 4

    final_mask = dataset.task_masks_dict[task_name_to_mask].squeeze()
    num_finally_valid = torch.sum(final_mask).item()

    # Expected number to keep is 50% of 4 = 2
    assert num_finally_valid == int(np.round(num_originally_valid * 0.5))  # Should be 2
    # Ensure that the False from NaN is still False
    assert not final_mask[2].item()
    # Ensure that only originally True items could have been set to False
    assert torch.all(final_mask <= original_valid_mask)


def test_ratio_masking_predict(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test task_masking_ratios is ignored for a predict dataset."""
    task_name_to_mask = "task_reg"
    masking_ratios = {task_name_to_mask: 0.5}

    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        task_masking_ratios=masking_ratios,
        is_predict_set=True,  # Predict mode
        dataset_name="test_ratio_mask_predict",
    )
    # Mask should only be based on NaNs, not the ratio
    expected_mask = torch.tensor([[True], [True], [False], [True], [True]], dtype=torch.bool)
    assert torch.equal(dataset.task_masks_dict[task_name_to_mask], expected_mask)


def test_getitem_predict_mode(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test __getitem__ when is_predict_set=True."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        is_predict_set=True,
        dataset_name="test_getitem_predict",
    )
    model_input_x, y_dict, masks_dict, temps_dict = dataset[0]

    assert isinstance(model_input_x, torch.Tensor)  # Should be just x_formula as no structure is used here
    assert torch.allclose(model_input_x, dataset.x_formula[0])
    assert "task_reg" in y_dict
    assert "task_reg" in masks_dict


def test_getitem_predict_mode_with_structure(
    sample_formula_desc_df, sample_attributes_df, sample_structure_desc_df, sample_task_configs
):
    """Test __getitem__ when is_predict_set=True and structure is used."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        structure_desc=sample_structure_desc_df,
        use_structure_for_this_dataset=True,
        is_predict_set=True,  # Predict mode
        dataset_name="test_getitem_predict_with_struct",
    )
    model_input_x, y_dict, masks_dict, temps_dict = dataset[0]

    assert isinstance(model_input_x, tuple)  # Should be (x_formula, x_struct)
    assert len(model_input_x) == 2
    assert torch.allclose(model_input_x[0], dataset.x_formula[0])
    assert torch.allclose(model_input_x[1], dataset.x_struct[0])
    assert "task_reg" in y_dict
    assert "task_reg" in masks_dict


def test_getitem_train_mode_no_structure(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test __getitem__ for training without structure."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        is_predict_set=False,
        use_structure_for_this_dataset=False,
        dataset_name="test_getitem_train_no_struct",
    )
    model_input_x, _, _, _ = dataset[0]
    assert isinstance(model_input_x, torch.Tensor)
    assert torch.allclose(model_input_x, dataset.x_formula[0])


def test_getitem_train_mode_with_structure(
    sample_formula_desc_df, sample_attributes_df, sample_structure_desc_df, sample_task_configs
):
    """Test __getitem__ for training with structure."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        structure_desc=sample_structure_desc_df,
        is_predict_set=False,
        use_structure_for_this_dataset=True,
        dataset_name="test_getitem_train_with_struct",
    )
    model_input_x, _, _, _ = dataset[0]
    assert isinstance(model_input_x, tuple)
    assert len(model_input_x) == 2
    assert torch.allclose(model_input_x[0], dataset.x_formula[0])
    assert torch.allclose(model_input_x[1], dataset.x_struct[0])


def test_attribute_names_property(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test the attribute_names property."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        dataset_name="test_attr_names",
    )
    expected_names = ["task_reg", "task_seq", "task_another_reg", "task_fully_missing_data_col"]
    assert sorted(dataset.attribute_names) == sorted(expected_names)


def test_input_dtypes_conversion(sample_attributes_df, sample_task_configs):
    """Test that input DataFrames with different dtypes are converted to float32 for tensors."""
    # Create formula_desc with int dtype
    formula_int_df = pd.DataFrame(
        {"feat_0": [1, 2, 3, 4, 5], "feat_1": [11, 12, 13, 14, 15]},
        index=[f"id_{i}" for i in range(5)],
        dtype=np.int32,
    )
    # Create structure_desc with a mix of int and float
    structure_mixed_df = pd.DataFrame(
        {"struct_feat_0": [10, 10, 10, 10, 10], "struct_feat_1": [11.1, 11.2, 11.3, 11.4, 11.5]},
        index=[f"id_{i}" for i in range(5)],
    )
    structure_mixed_df["struct_feat_0"] = structure_mixed_df["struct_feat_0"].astype(np.int64)

    dataset = CompoundDataset(
        formula_desc=formula_int_df,
        attributes=sample_attributes_df,  # sample_attributes_df already has floats and objects (lists)
        task_configs=sample_task_configs,
        structure_desc=structure_mixed_df,
        use_structure_for_this_dataset=True,
        dataset_name="test_dtypes",
    )

    assert dataset.x_formula.dtype == torch.float32
    assert dataset.x_struct.dtype == torch.float32

    # Check y_dict dtypes (regression and sequence should be float32)
    for task_name, y_tensor in dataset.y_dict.items():
        task_cfg = next(tc for tc in sample_task_configs if tc.name == task_name)
        if task_cfg.type == MockTaskType.REGRESSION or task_cfg.type == MockTaskType.SEQUENCE:
            assert y_tensor.dtype == torch.float32, f"Task {task_name} y_dict dtype mismatch"
        # Classification tasks (if added) would be long

    # Check temps_dict dtype (should be float32)
    for task_name, temps_tensor in dataset.temps_dict.items():
        assert temps_tensor.dtype == torch.float32, f"Task {task_name} temps_dict dtype mismatch"


# TODO: Add more tests:
# - Edge case: empty formula_desc or attributes (should raise error or handle gracefully if allowed by design)
# - Edge case: task_configs list is empty (should raise error)
# - Test sequence data where individual items in the list/array are not all numbers (e.g. strings, though unlikely)
# - Test when `temps` column is missing for a sequence task.


def test_empty_inputs_raise_error(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test that empty DataFrames or task_configs list raise ValueError."""
    empty_df = pd.DataFrame()
    non_empty_formula_df = sample_formula_desc_df
    non_empty_attributes_df = sample_attributes_df
    non_empty_task_configs = sample_task_configs

    with pytest.raises(ValueError, match="formula_desc DataFrame cannot be empty."):
        CompoundDataset(
            formula_desc=empty_df,
            attributes=non_empty_attributes_df,
            task_configs=non_empty_task_configs,
            dataset_name="test_empty_formula",
        )

    with pytest.raises(ValueError, match="attributes DataFrame cannot be empty."):
        CompoundDataset(
            formula_desc=non_empty_formula_df,
            attributes=empty_df,
            task_configs=non_empty_task_configs,
            dataset_name="test_empty_attributes",
        )

    with pytest.raises(ValueError, match="task_configs list cannot be empty."):
        CompoundDataset(
            formula_desc=non_empty_formula_df,
            attributes=non_empty_attributes_df,
            task_configs=[],  # Empty task_configs
            dataset_name="test_empty_task_configs",
        )

    # Test case where formula_desc or attributes_df might be None (though type hints suggest DataFrame)
    # This is more about robust handling if None is somehow passed.
    # The class currently expects DataFrames, so this might be redundant if type checking is strict.
    with pytest.raises(TypeError, match="formula_desc must be pd.DataFrame or np.ndarray"):
        CompoundDataset(
            formula_desc=None,
            attributes=non_empty_attributes_df,
            task_configs=non_empty_task_configs,
            dataset_name="test_none_formula",
        )


def test_sequence_data_with_non_numeric(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test that non-numeric data in sequence series raises an error."""
    attributes_bad_seq = sample_attributes_df.copy()
    # Ensure the column can hold arbitrary objects before assigning a mixed-type list
    attributes_bad_seq["task_seq_sequence_series"] = attributes_bad_seq["task_seq_sequence_series"].astype("object")
    # Introduce a string into a sequence using .at for scalar assignment
    attributes_bad_seq.at["id_0", "task_seq_sequence_series"] = [0.1, "not_a_number", 0.3]

    with pytest.raises(TypeError):  # Expect TypeError when converting list with string to tensor
        CompoundDataset(
            formula_desc=sample_formula_desc_df,
            attributes=attributes_bad_seq,
            task_configs=sample_task_configs,
            dataset_name="test_bad_seq_data",
        )

    attributes_bad_temps = sample_attributes_df.copy()
    # Ensure the column can hold arbitrary objects
    attributes_bad_temps["task_seq_temps"] = attributes_bad_temps["task_seq_temps"].astype("object")
    # Introduce a string into temps using .at for scalar assignment
    attributes_bad_temps.at["id_0", "task_seq_temps"] = [10, "bad_temp", 30]
    with pytest.raises(TypeError):  # Expect TypeError for temps as well
        CompoundDataset(
            formula_desc=sample_formula_desc_df,
            attributes=attributes_bad_temps,
            task_configs=sample_task_configs,
            dataset_name="test_bad_temps_data",
        )


def test_missing_temps_for_sequence_task(sample_formula_desc_df, sample_attributes_df, sample_task_configs, caplog):
    """Test behavior when 'temps' column is missing for a sequence task."""
    attributes_no_temps = sample_attributes_df.drop(columns=["task_seq_temps"])

    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=attributes_no_temps,
        task_configs=sample_task_configs,  # task_seq is a sequence task
        dataset_name="test_missing_temps",
    )

    task_name = "task_seq"
    assert task_name in dataset.temps_dict
    # Expect a placeholder (e.g., zeros) for temps
    # The shape should match y_dict for that task, but with an added channel dim (B, SeqLen, 1)
    expected_temps_shape = (len(sample_formula_desc_df), dataset.y_dict[task_name].shape[1], 1)
    assert dataset.temps_dict[task_name].shape == expected_temps_shape
    assert torch.all(dataset.temps_dict[task_name] == 0)  # Check if placeholder is zeros

    # Check for a warning log message
    temps_col_name = f"{task_name}_temps"  # Reconstruct for the log message
    expected_log_message = f"[{dataset.dataset_name}] Task '{task_name}': Temps column '{temps_col_name}' not found for sequence task. Using zero placeholder."
    assert any(expected_log_message in record.message and record.levelname == "WARNING" for record in caplog.records)
