import logging  # Add logging import

import numpy as np
import pandas as pd
import pytest
import torch

from foundation_model.data.dataset import CompoundDataset
from foundation_model.models.model_config import (
    KernelRegressionTaskConfig,
    RegressionTaskConfig,
    TaskType,
)

# --- Fixtures ---


@pytest.fixture
def sample_formula_desc_df():
    """Returns a sample formula descriptor DataFrame."""
    return pd.DataFrame(
        {"feat_0": [0.1, 0.2, 0.3, 0.4, 0.5], "feat_1": [1.1, 1.2, 1.3, 1.4, 1.5]}, index=[f"id_{i}" for i in range(5)]
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
    """Returns a list of sample TaskConfig objects."""
    return [
        RegressionTaskConfig(name="task_reg", type=TaskType.REGRESSION, data_column="task_reg_regression_value"),
        KernelRegressionTaskConfig(
            name="task_seq",
            type=TaskType.KERNEL_REGRESSION,
            data_column="task_seq_sequence_series",
            t_column="task_seq_temps",
        ),
        RegressionTaskConfig(
            name="task_another_reg", type=TaskType.REGRESSION, data_column="task_another_reg_regression_value"
        ),
        RegressionTaskConfig(
            name="task_disabled", type=TaskType.REGRESSION, data_column="task_disabled_regression_value", enabled=False
        ),
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
    assert "task_reg" in dataset.y_dict
    assert "task_seq" in dataset.y_dict
    assert "task_seq" in dataset.t_sequences_dict
    assert "task_another_reg" in dataset.y_dict
    assert "task_disabled" not in dataset.enabled_task_names  # Check disabled task


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


def test_task_data_processing_sequence(sample_formula_desc_df, sample_attributes_df):
    """Verify y_dict, temps_dict, and masks for a sequence task."""
    # Create a task config that only includes valid tasks for this test
    valid_task_configs = [
        RegressionTaskConfig(name="task_reg", type=TaskType.REGRESSION, data_column="task_reg_regression_value"),
        KernelRegressionTaskConfig(
            name="task_seq",
            type=TaskType.KERNEL_REGRESSION,
            data_column="task_seq_sequence_series",
            t_column="task_seq_temps",
        ),
    ]
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=valid_task_configs,
        dataset_name="test_seq_processing",
    )
    task_name = "task_seq"
    assert task_name in dataset.y_dict
    # For KernelRegression, y_dict stores List[Tensor]
    assert isinstance(dataset.y_dict[task_name], list)
    assert len(dataset.y_dict[task_name]) == 5  # 5 samples
    # Check first sample
    expected_y_first = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    assert torch.allclose(dataset.y_dict[task_name][0], expected_y_first)
    # Check third sample (all NaN becomes 0)
    expected_y_third = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    assert torch.allclose(dataset.y_dict[task_name][2], expected_y_third)

    assert task_name in dataset.t_sequences_dict
    # For KernelRegression, t_sequences_dict stores List[Tensor], so check the first sample
    assert len(dataset.t_sequences_dict[task_name]) == 5  # 5 samples
    expected_temps = torch.tensor([10, 20, 30], dtype=torch.float32)  # Single sample
    assert torch.allclose(dataset.t_sequences_dict[task_name][0], expected_temps)

    assert task_name in dataset.task_masks_dict
    # For KernelRegression, task_masks_dict also stores List[Tensor]
    assert isinstance(dataset.task_masks_dict[task_name], list)
    assert len(dataset.task_masks_dict[task_name]) == 5
    # Check that first sample mask is all True (valid data)
    assert torch.all(dataset.task_masks_dict[task_name][0])
    # Check that third sample mask is all False (all NaN data)
    assert torch.all(~dataset.task_masks_dict[task_name][2])


def test_task_data_processing_missing_columns(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test behavior when expected attribute columns for a task are missing."""
    configs_with_missing = sample_task_configs + [
        RegressionTaskConfig(
            name="task_fully_missing_data_col",
            type=TaskType.REGRESSION,
            data_column="non_existent_column",
        )
    ]
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=configs_with_missing,
        dataset_name="test_missing_cols",
    )
    placeholder = dataset.y_dict["task_fully_missing_data_col"]
    assert placeholder.shape[0] == len(sample_attributes_df)
    assert torch.allclose(placeholder, torch.zeros_like(placeholder))
    assert not dataset.task_masks_dict["task_fully_missing_data_col"].any()


def test_nan_masking(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test that NaNs in attributes correctly generate masks."""
    # (already covered in specific task tests, but good to have a focused one)
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
    expected_mask_seq = torch.tensor([True, True, False, True, True], dtype=torch.bool)
    seq_masks = dataset.task_masks_dict["task_seq"]
    assert isinstance(seq_masks, list)
    actual_seq_mask = torch.tensor([mask.any().item() for mask in seq_masks], dtype=torch.bool)
    assert torch.equal(actual_seq_mask, expected_mask_seq)


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


def test_task_masking_seed_controls_rng(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Ensure task_masking_seed makes masking deterministic."""
    task_name = "task_reg"
    masking_ratio = 0.5
    masking_ratios = {task_name: masking_ratio}
    masking_seed = 123

    dataset_seeded = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        task_masking_ratios=masking_ratios,
        task_masking_seed=masking_seed,
        is_predict_set=False,
        dataset_name="test_ratio_mask_seeded",
    )

    final_mask = dataset_seeded.task_masks_dict[task_name].squeeze().numpy()
    base_mask = np.array([True, True, False, True, True])
    expected_mask = base_mask.copy()
    valid_indices = np.where(base_mask)[0]
    num_to_keep = int(np.round(len(valid_indices) * masking_ratio))
    num_to_drop = len(valid_indices) - num_to_keep
    if num_to_drop > 0:
        rng = np.random.default_rng(masking_seed)
        indices_to_drop = rng.choice(valid_indices, size=num_to_drop, replace=False)
        expected_mask[indices_to_drop] = False
    assert np.array_equal(final_mask, expected_mask)

    # Changing the global NumPy RNG should not affect the seeded behavior
    np.random.seed(999)
    dataset_seeded_repeat = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        task_masking_ratios=masking_ratios,
        task_masking_seed=masking_seed,
        is_predict_set=False,
        dataset_name="test_ratio_mask_seeded_repeat",
    )
    repeated_mask = dataset_seeded_repeat.task_masks_dict[task_name].squeeze().numpy()
    assert np.array_equal(final_mask, repeated_mask)


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


def test_getitem_train_mode_no_structure(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test __getitem__ for training without structure."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        is_predict_set=False,
        dataset_name="test_getitem_train_no_struct",
    )
    model_input_x, _, _, _ = dataset[0]
    assert isinstance(model_input_x, torch.Tensor)
    assert torch.allclose(model_input_x, dataset.x_formula[0])


def test_attribute_names_property(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test the attribute_names property."""
    dataset = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        dataset_name="test_attr_names",
    )
    expected_names = ["task_reg", "task_seq", "task_another_reg"]
    assert sorted(dataset.attribute_names) == sorted(expected_names)


def test_input_dtypes_conversion(sample_attributes_df, sample_task_configs):
    """Test that input DataFrames with different dtypes are converted to float32."""
    formula_int_df = pd.DataFrame(
        {
            "feat_0": [1, 2, 3, 4, 5],
            "feat_1": [11, 12, 13, 14, 15],
        },
        index=[f"id_{i}" for i in range(5)],
        dtype=np.int32,
    )

    dataset = CompoundDataset(
        formula_desc=formula_int_df,
        attributes=sample_attributes_df,
        task_configs=sample_task_configs,
        dataset_name="test_dtypes",
    )

    assert dataset.x_formula.dtype == torch.float32

    for task_name, y_value in dataset.y_dict.items():
        task_cfg = next(tc for tc in sample_task_configs if tc.name == task_name)  # type: ignore
        if task_cfg.type in {TaskType.REGRESSION, TaskType.KERNEL_REGRESSION}:
            if isinstance(y_value, list):
                assert all(t.dtype == torch.float32 for t in y_value), f"Task {task_name} y_dict dtype mismatch"
            else:
                assert y_value.dtype == torch.float32, f"Task {task_name} y_dict dtype mismatch"
        elif task_cfg.type == TaskType.CLASSIFICATION:
            assert y_value.dtype == torch.long, f"Task {task_name} y_dict dtype mismatch for classification"

    for task_name, temps_list in dataset.t_sequences_dict.items():
        assert isinstance(temps_list, list)
        assert temps_list[0].dtype == torch.float32, f"Task {task_name} t_sequences_dict dtype mismatch"


# TODO: Add more tests:
# - Edge case: empty formula_desc or attributes (should raise error or handle gracefully if allowed by design)
# - Edge case: task_configs list is empty (should raise error)
# - Test sequence data where individual items in the list/array are not all numbers
#   (e.g. strings, though unlikely)
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
    # The class currently expects DataFrames, so this might be redundant
    # if type checking is strict.
    with pytest.raises(TypeError, match="formula_desc must be pd.DataFrame or np.ndarray"):
        CompoundDataset(
            formula_desc=None,  # type: ignore
            attributes=non_empty_attributes_df,
            task_configs=non_empty_task_configs,
            dataset_name="test_none_formula",
        )


def test_sequence_data_with_non_numeric(sample_formula_desc_df, sample_attributes_df, sample_task_configs, caplog):
    """Test that non-numeric data in sequence series is handled by creating NaNs, then converted by nan_to_num."""
    attributes_bad_seq = sample_attributes_df.copy()
    attributes_bad_seq["task_seq_sequence_series"] = attributes_bad_seq["task_seq_sequence_series"].astype("object")
    attributes_bad_seq.at["id_0", "task_seq_sequence_series"] = [0.1, "not_a_number", 0.3]

    dataset_bad_seq = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=attributes_bad_seq,
        task_configs=sample_task_configs,
        dataset_name="test_bad_seq_data",
    )
    # Check that the problematic entry became [0.1, 0.0, 0.3] after nan_to_num(nan=0.0)
    # The middle "not_a_number" should have been parsed as np.nan by _parse_structured_element, then 0.0
    assert torch.allclose(dataset_bad_seq.y_dict["task_seq"][0], torch.tensor([0.1, 0.0, 0.3], dtype=torch.float32))

    attributes_bad_temps = sample_attributes_df.copy()
    attributes_bad_temps["task_seq_temps"] = attributes_bad_temps["task_seq_temps"].astype("object")
    attributes_bad_temps.at["id_0", "task_seq_temps"] = [10, "bad_temp", 30]

    dataset_bad_temps = CompoundDataset(
        formula_desc=sample_formula_desc_df,
        attributes=attributes_bad_temps,
        task_configs=sample_task_configs,
        dataset_name="test_bad_temps_data",
    )
    # Check that the problematic entry became [10, 0, 30]
    assert torch.allclose(
        dataset_bad_temps.t_sequences_dict["task_seq"][0], torch.tensor([10.0, 0.0, 30.0], dtype=torch.float32)
    )


def test_missing_specified_t_column_raises_error(sample_formula_desc_df, sample_attributes_df, sample_task_configs):
    """Test ValueError if a specified t_column is missing for a KernelRegression task."""
    attributes_no_temps = sample_attributes_df.drop(columns=["task_seq_temps"])

    # sample_task_configs already has task_seq configured with t_column="task_seq_temps"

    with pytest.raises(
        ValueError, match="T-parameter column 'task_seq_temps' for task 'task_seq' not found in attributes data."
    ):
        CompoundDataset(
            formula_desc=sample_formula_desc_df,
            attributes=attributes_no_temps,
            task_configs=sample_task_configs,
            dataset_name="test_missing_specified_t_column",
        )


def test_kernel_regression_task_no_t_column_specified(sample_formula_desc_df, sample_attributes_df, caplog):
    """Test behavior when t_column is not specified for a KernelRegression task (uses placeholder)."""
    caplog.set_level(logging.INFO)  # Set caplog level to INFO

    task_configs_no_t_spec = [
        RegressionTaskConfig(name="task_reg", type=TaskType.REGRESSION, data_column="task_reg_regression_value"),
        KernelRegressionTaskConfig(
            name="task_seq", type=TaskType.KERNEL_REGRESSION, data_column="task_seq_sequence_series", t_column=""
        ),  # t_column is empty
    ]

    with pytest.raises(ValueError, match="t_column for KernelRegression task 'task_seq' must be specified."):
        CompoundDataset(
            formula_desc=sample_formula_desc_df,
            attributes=sample_attributes_df,  # attributes_df still has "task_seq_temps" but it won't be used
            task_configs=task_configs_no_t_spec,
            dataset_name="test_no_t_column_spec",
        )
