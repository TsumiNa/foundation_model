# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import logging  # Add logging import
import zlib

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
def sample_compositions():
    return [f"id_{i}" for i in range(5)]


@pytest.fixture
def sample_descriptors():
    """Descriptor features indexed by composition."""
    return pd.DataFrame(
        {"feat_0": [0.1, 0.2, 0.3, 0.4, 0.5], "feat_1": [1.1, 1.2, 1.3, 1.4, 1.5]},
        index=[f"id_{i}" for i in range(5)],
    )


@pytest.fixture
def sample_attributes_df():
    """A combined attributes frame (indexed by composition) shared across tasks in tests.

    The dataset reads only each task's own ``data_column`` / ``t_column`` from its frame, so a
    single combined frame can serve as every task's frame without cross-talk.
    """
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


def _task_frames(task_configs, attributes_df):
    """Build a per-task frame mapping that shares the combined attributes frame."""
    return {cfg.name: attributes_df for cfg in task_configs}


def _make_dataset(compositions, descriptors, task_configs, attributes_df, **kwargs):
    return CompoundDataset(
        compositions=compositions,
        descriptors=descriptors,
        task_frames=_task_frames(task_configs, attributes_df),
        task_configs=task_configs,
        **kwargs,
    )


# --- Test Cases ---


def test_dataset_initialization_basic(
    sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs
):
    """Test basic initialization with descriptors and per-task frames."""
    dataset = _make_dataset(
        sample_compositions, sample_descriptors, sample_task_configs, sample_attributes_df, dataset_name="test_basic"
    )
    assert len(dataset) == 5
    assert dataset.x_formula.shape == (5, 2)
    assert "task_reg" in dataset.y_dict
    assert "task_seq" in dataset.y_dict
    assert "task_seq" in dataset.t_sequences_dict
    assert "task_another_reg" in dataset.y_dict
    assert "task_disabled" not in dataset.enabled_task_names  # Check disabled task


def test_task_data_processing_regression(
    sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs
):
    """Verify y_dict and masks for a regression task."""
    dataset = _make_dataset(
        sample_compositions, sample_descriptors, sample_task_configs, sample_attributes_df, dataset_name="test_reg"
    )
    task_name = "task_reg"
    assert task_name in dataset.y_dict
    assert dataset.y_dict[task_name].shape == (5, 1)
    expected_y = torch.tensor([[1.0], [2.0], [0.0], [4.0], [5.0]], dtype=torch.float32)
    assert torch.allclose(dataset.y_dict[task_name], expected_y)

    assert task_name in dataset.task_masks_dict
    assert dataset.task_masks_dict[task_name].dtype == torch.bool
    expected_mask = torch.tensor([[True], [True], [False], [True], [True]], dtype=torch.bool)
    assert torch.equal(dataset.task_masks_dict[task_name], expected_mask)


def test_task_data_processing_sequence(sample_compositions, sample_descriptors, sample_attributes_df):
    """Verify y_dict, t_sequences_dict, and masks for a sequence task."""
    valid_task_configs = [
        RegressionTaskConfig(name="task_reg", type=TaskType.REGRESSION, data_column="task_reg_regression_value"),
        KernelRegressionTaskConfig(
            name="task_seq",
            type=TaskType.KERNEL_REGRESSION,
            data_column="task_seq_sequence_series",
            t_column="task_seq_temps",
        ),
    ]
    dataset = _make_dataset(
        sample_compositions, sample_descriptors, valid_task_configs, sample_attributes_df, dataset_name="test_seq"
    )
    task_name = "task_seq"
    assert task_name in dataset.y_dict
    assert isinstance(dataset.y_dict[task_name], list)
    assert len(dataset.y_dict[task_name]) == 5
    expected_y_first = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    assert torch.allclose(dataset.y_dict[task_name][0], expected_y_first)
    expected_y_third = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    assert torch.allclose(dataset.y_dict[task_name][2], expected_y_third)

    assert task_name in dataset.t_sequences_dict
    assert len(dataset.t_sequences_dict[task_name]) == 5
    expected_temps = torch.tensor([10, 20, 30], dtype=torch.float32)
    assert torch.allclose(dataset.t_sequences_dict[task_name][0], expected_temps)

    assert task_name in dataset.task_masks_dict
    assert isinstance(dataset.task_masks_dict[task_name], list)
    assert len(dataset.task_masks_dict[task_name]) == 5
    assert torch.all(dataset.task_masks_dict[task_name][0])
    assert torch.all(~dataset.task_masks_dict[task_name][2])


def test_task_data_processing_missing_columns(
    sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs
):
    """Test behavior when expected attribute columns for a task are missing."""
    extra = RegressionTaskConfig(
        name="task_fully_missing_data_col",
        type=TaskType.REGRESSION,
        data_column="non_existent_column",
    )
    configs_with_missing = sample_task_configs + [extra]
    dataset = _make_dataset(
        sample_compositions,
        sample_descriptors,
        configs_with_missing,
        sample_attributes_df,
        dataset_name="test_missing_cols",
    )
    placeholder = dataset.y_dict["task_fully_missing_data_col"]
    assert placeholder.shape[0] == len(sample_compositions)
    assert torch.allclose(placeholder, torch.zeros_like(placeholder))
    assert not dataset.task_masks_dict["task_fully_missing_data_col"].any()


def test_task_frame_missing_compositions_are_masked(sample_compositions, sample_descriptors, sample_attributes_df):
    """Compositions absent from a task's frame are reindexed to NaN and masked out."""
    configs = [RegressionTaskConfig(name="task_reg", data_column="task_reg_regression_value")]
    # Task frame only covers the first two compositions.
    partial_frame = sample_attributes_df.loc[["id_0", "id_1"]]
    dataset = CompoundDataset(
        compositions=sample_compositions,
        descriptors=sample_descriptors,
        task_frames={"task_reg": partial_frame},
        task_configs=configs,
        dataset_name="test_partial_frame",
    )
    mask = dataset.task_masks_dict["task_reg"].squeeze()
    # id_0, id_1 valid; id_2..id_4 absent from frame -> masked False.
    assert mask.tolist() == [True, True, False, False, False]
    # y for absent compositions is the 0.0 placeholder.
    assert dataset.y_dict["task_reg"][3].item() == 0.0


def test_nan_masking(sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs):
    """Test that NaNs in attributes correctly generate masks."""
    dataset = _make_dataset(
        sample_compositions, sample_descriptors, sample_task_configs, sample_attributes_df, dataset_name="test_nan"
    )
    expected_mask_reg = torch.tensor([[True], [True], [False], [True], [True]], dtype=torch.bool)
    assert torch.equal(dataset.task_masks_dict["task_reg"], expected_mask_reg)

    expected_mask_seq = torch.tensor([True, True, False, True, True], dtype=torch.bool)
    seq_masks = dataset.task_masks_dict["task_seq"]
    assert isinstance(seq_masks, list)
    actual_seq_mask = torch.tensor([mask.any().item() for mask in seq_masks], dtype=torch.bool)
    assert torch.equal(actual_seq_mask, expected_mask_seq)


def test_ratio_masking_train(sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs):
    """Test task_masking_ratios for a training dataset."""
    task_name_to_mask = "task_reg"  # 4 valid samples initially
    masking_ratios = {task_name_to_mask: 0.5}

    np.random.seed(42)
    dataset = _make_dataset(
        sample_compositions,
        sample_descriptors,
        sample_task_configs,
        sample_attributes_df,
        task_masking_ratios=masking_ratios,
        is_predict_set=False,
        dataset_name="test_ratio_mask_train",
    )

    original_valid_mask = torch.tensor([True, True, False, True, True], dtype=torch.bool)
    num_originally_valid = torch.sum(original_valid_mask).item()  # 4

    final_mask = dataset.task_masks_dict[task_name_to_mask].squeeze()
    num_finally_valid = torch.sum(final_mask).item()

    assert num_finally_valid == int(np.round(num_originally_valid * 0.5))  # 2
    assert not final_mask[2].item()
    assert torch.all(final_mask <= original_valid_mask)


def test_ratio_masking_predict(sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs):
    """Test task_masking_ratios is ignored for a predict dataset."""
    task_name_to_mask = "task_reg"
    masking_ratios = {task_name_to_mask: 0.5}

    dataset = _make_dataset(
        sample_compositions,
        sample_descriptors,
        sample_task_configs,
        sample_attributes_df,
        task_masking_ratios=masking_ratios,
        is_predict_set=True,
        dataset_name="test_ratio_mask_predict",
    )
    expected_mask = torch.tensor([[True], [True], [False], [True], [True]], dtype=torch.bool)
    assert torch.equal(dataset.task_masks_dict[task_name_to_mask], expected_mask)


def test_task_masking_seed_controls_rng(
    sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs
):
    """The keep mask is drawn from the documented per-task stream: rng([seed, crc32(task), epoch])."""
    task_name = "task_reg"
    masking_ratio = 0.5
    masking_ratios = {task_name: masking_ratio}
    masking_seed = 123

    dataset_seeded = _make_dataset(
        sample_compositions,
        sample_descriptors,
        sample_task_configs,
        sample_attributes_df,
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
        rng = np.random.default_rng([masking_seed, zlib.crc32(task_name.encode()), 0])
        indices_to_drop = rng.choice(valid_indices, size=num_to_drop, replace=False)
        expected_mask[indices_to_drop] = False
    assert np.array_equal(final_mask, expected_mask)

    np.random.seed(999)
    dataset_seeded_repeat = _make_dataset(
        sample_compositions,
        sample_descriptors,
        sample_task_configs,
        sample_attributes_df,
        task_masking_ratios=masking_ratios,
        task_masking_seed=masking_seed,
        is_predict_set=False,
        dataset_name="test_ratio_mask_seeded_repeat",
    )
    repeated_mask = dataset_seeded_repeat.task_masks_dict[task_name].squeeze().numpy()
    assert np.array_equal(final_mask, repeated_mask)


# --- keep-ratio masking: per-task streams & epoch resampling ---


def _wide_reg_dataset(n_rows=40, *, ratios=None, seed=123, extra_task=False, nan_rows=(), predict=False):
    """A one/two-regression-task dataset big enough that mask collisions are implausible."""
    comps = [f"m{i}" for i in range(n_rows)]
    desc = pd.DataFrame({"f0": np.arange(n_rows, dtype=float), "f1": np.ones(n_rows)}, index=comps)
    y_a = np.arange(n_rows, dtype=float)
    y_a[list(nan_rows)] = np.nan
    frames = {"task_a": pd.DataFrame({"y_a": y_a}, index=comps)}
    configs = [RegressionTaskConfig(name="task_a", type=TaskType.REGRESSION, data_column="y_a")]
    if extra_task:
        frames["task_b"] = pd.DataFrame({"y_b": np.arange(n_rows, dtype=float)}, index=comps)
        configs.append(RegressionTaskConfig(name="task_b", type=TaskType.REGRESSION, data_column="y_b"))
    return CompoundDataset(
        compositions=comps,
        descriptors=desc,
        task_frames=frames,
        task_configs=configs,
        task_masking_ratios=ratios,
        task_masking_seed=seed,
        is_predict_set=predict,
        dataset_name="wide",
    )


def test_mask_draw_independent_of_other_tasks():
    """A task's keep mask depends only on (seed, task, epoch) — not on which tasks share the dataset."""
    alone = _wide_reg_dataset(ratios={"task_a": 0.5})
    with_b = _wide_reg_dataset(ratios={"task_a": 0.5, "task_b": 0.3}, extra_task=True)
    assert torch.equal(alone.task_masks_dict["task_a"], with_b.task_masks_dict["task_a"])


def test_resample_epoch_zero_matches_construction():
    dataset = _wide_reg_dataset(ratios={"task_a": 0.5})
    construction = dataset.task_masks_dict["task_a"].clone()
    dataset.resample_task_masks(epoch=0)
    assert torch.equal(construction, dataset.task_masks_dict["task_a"])


def test_resample_changes_across_epochs_and_is_deterministic():
    dataset = _wide_reg_dataset(ratios={"task_a": 0.5})
    epoch0 = dataset.task_masks_dict["task_a"].clone()
    dataset.resample_task_masks(epoch=1)
    epoch1 = dataset.task_masks_dict["task_a"].clone()
    assert not torch.equal(epoch0, epoch1)
    assert epoch1.sum() == epoch0.sum()  # subset size is preserved
    dataset.resample_task_masks(epoch=1)
    assert torch.equal(epoch1, dataset.task_masks_dict["task_a"])  # same epoch -> same draw


def test_resample_never_unmasks_missing_labels():
    nan_rows = (0, 7, 13, 21)
    dataset = _wide_reg_dataset(ratios={"task_a": 0.5}, nan_rows=nan_rows)
    n_valid = 40 - len(nan_rows)
    expected_kept = int(np.round(n_valid * 0.5))
    for epoch in range(5):
        dataset.resample_task_masks(epoch=epoch)
        mask = dataset.task_masks_dict["task_a"].squeeze().numpy()
        assert int(mask.sum()) == expected_kept
        assert not mask[list(nan_rows)].any()


def test_resample_noop_for_predict_set():
    dataset = _wide_reg_dataset(ratios={"task_a": 0.5}, predict=True)
    full = dataset.task_masks_dict["task_a"].clone()
    assert int(full.sum()) == 40  # predict sets never subsample
    dataset.resample_task_masks(epoch=3)
    assert torch.equal(full, dataset.task_masks_dict["task_a"])


def test_resample_rebuilds_kernel_masks(
    sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs
):
    """KR keep masks stay List[Tensor], all-ones/all-zeros per row, and the all-NaN row stays off."""
    dataset = _make_dataset(
        sample_compositions,
        sample_descriptors,
        sample_task_configs,
        sample_attributes_df,
        task_masking_ratios={"task_seq": 0.5},
        task_masking_seed=7,
        dataset_name="test_kr_resample",
    )
    for epoch in (0, 1, 2):
        dataset.resample_task_masks(epoch=epoch)
        masks = dataset.task_masks_dict["task_seq"]
        assert isinstance(masks, list) and len(masks) == 5
        row_mask = [bool(m.any()) for m in masks]
        assert all(m.all() or not m.any() for m in masks)  # per-row all-on or all-off
        assert not row_mask[2]  # id_2 is the all-NaN sequence -> never selected
        assert sum(row_mask) == int(np.round(4 * 0.5))


def test_getitem_predict_mode(sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs):
    """Test __getitem__ when is_predict_set=True."""
    dataset = _make_dataset(
        sample_compositions,
        sample_descriptors,
        sample_task_configs,
        sample_attributes_df,
        is_predict_set=True,
        dataset_name="test_getitem_predict",
    )
    model_input_x, y_dict, masks_dict, temps_dict = dataset[0]

    assert isinstance(model_input_x, torch.Tensor)
    assert torch.allclose(model_input_x, dataset.x_formula[0])
    assert "task_reg" in y_dict
    assert "task_reg" in masks_dict


def test_getitem_train_mode_no_structure(
    sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs
):
    """Test __getitem__ for training without structure."""
    dataset = _make_dataset(
        sample_compositions,
        sample_descriptors,
        sample_task_configs,
        sample_attributes_df,
        is_predict_set=False,
        dataset_name="test_getitem_train",
    )
    model_input_x, _, _, _ = dataset[0]
    assert isinstance(model_input_x, torch.Tensor)
    assert torch.allclose(model_input_x, dataset.x_formula[0])


def test_attribute_names_property(sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs):
    """Test the attribute_names property."""
    dataset = _make_dataset(
        sample_compositions, sample_descriptors, sample_task_configs, sample_attributes_df, dataset_name="test_attr"
    )
    expected_names = ["task_reg", "task_seq", "task_another_reg"]
    assert sorted(dataset.attribute_names) == sorted(expected_names)


def test_input_dtypes_conversion(sample_compositions, sample_attributes_df, sample_task_configs):
    """Test that descriptor DataFrames with int dtypes are converted to float32."""
    descriptors_int = pd.DataFrame(
        {"feat_0": [1, 2, 3, 4, 5], "feat_1": [11, 12, 13, 14, 15]},
        index=[f"id_{i}" for i in range(5)],
        dtype=np.int32,
    )

    dataset = _make_dataset(
        sample_compositions, descriptors_int, sample_task_configs, sample_attributes_df, dataset_name="test_dtypes"
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


def test_empty_inputs_raise_error(sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs):
    """Test that empty/invalid inputs raise the expected errors."""
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="descriptors DataFrame cannot be empty."):
        CompoundDataset(
            compositions=sample_compositions,
            descriptors=empty_df,
            task_frames=_task_frames(sample_task_configs, sample_attributes_df),
            task_configs=sample_task_configs,
            dataset_name="test_empty_descriptors",
        )

    with pytest.raises(ValueError, match="compositions cannot be empty."):
        CompoundDataset(
            compositions=[],
            descriptors=sample_descriptors,
            task_frames=_task_frames(sample_task_configs, sample_attributes_df),
            task_configs=sample_task_configs,
            dataset_name="test_empty_compositions",
        )

    with pytest.raises(ValueError, match="task_configs list cannot be empty."):
        CompoundDataset(
            compositions=sample_compositions,
            descriptors=sample_descriptors,
            task_frames={},
            task_configs=[],
            dataset_name="test_empty_task_configs",
        )

    with pytest.raises(TypeError, match="descriptors must be a pd.DataFrame"):
        CompoundDataset(
            compositions=sample_compositions,
            descriptors=None,  # type: ignore
            task_frames=_task_frames(sample_task_configs, sample_attributes_df),
            task_configs=sample_task_configs,
            dataset_name="test_none_descriptors",
        )


def test_missing_descriptor_composition_raises(sample_descriptors, sample_attributes_df, sample_task_configs):
    """A composition without a descriptor row must raise a clear error."""
    compositions = [f"id_{i}" for i in range(5)] + ["id_missing"]
    with pytest.raises(ValueError, match="missing 1 composition"):
        CompoundDataset(
            compositions=compositions,
            descriptors=sample_descriptors,  # only id_0..id_4
            task_frames=_task_frames(sample_task_configs, sample_attributes_df),
            task_configs=sample_task_configs,
            dataset_name="test_missing_desc",
        )


def test_sequence_data_with_non_numeric(
    sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs
):
    """Non-numeric sequence data is parsed to NaN then nan_to_num'd to 0."""
    attributes_bad_seq = sample_attributes_df.copy()
    attributes_bad_seq["task_seq_sequence_series"] = attributes_bad_seq["task_seq_sequence_series"].astype("object")
    attributes_bad_seq.at["id_0", "task_seq_sequence_series"] = [0.1, "not_a_number", 0.3]

    dataset_bad_seq = _make_dataset(
        sample_compositions, sample_descriptors, sample_task_configs, attributes_bad_seq, dataset_name="test_bad_seq"
    )
    assert torch.allclose(dataset_bad_seq.y_dict["task_seq"][0], torch.tensor([0.1, 0.0, 0.3], dtype=torch.float32))

    attributes_bad_temps = sample_attributes_df.copy()
    attributes_bad_temps["task_seq_temps"] = attributes_bad_temps["task_seq_temps"].astype("object")
    attributes_bad_temps.at["id_0", "task_seq_temps"] = [10, "bad_temp", 30]

    dataset_bad_temps = _make_dataset(
        sample_compositions,
        sample_descriptors,
        sample_task_configs,
        attributes_bad_temps,
        dataset_name="test_bad_temps",
    )
    assert torch.allclose(
        dataset_bad_temps.t_sequences_dict["task_seq"][0], torch.tensor([10.0, 0.0, 30.0], dtype=torch.float32)
    )


def test_missing_specified_t_column_raises_error(
    sample_compositions, sample_descriptors, sample_attributes_df, sample_task_configs
):
    """Test ValueError if a specified t_column is missing for a KernelRegression task."""
    attributes_no_temps = sample_attributes_df.drop(columns=["task_seq_temps"])

    with pytest.raises(
        ValueError, match="T-parameter column 'task_seq_temps' for task 'task_seq' not found in task data."
    ):
        _make_dataset(
            sample_compositions,
            sample_descriptors,
            sample_task_configs,
            attributes_no_temps,
            dataset_name="test_missing_t",
        )


def test_kernel_regression_task_no_t_column_specified(
    sample_compositions, sample_descriptors, sample_attributes_df, caplog
):
    """Test behavior when t_column is not specified for a KernelRegression task."""
    caplog.set_level(logging.INFO)

    task_configs_no_t_spec = [
        RegressionTaskConfig(name="task_reg", type=TaskType.REGRESSION, data_column="task_reg_regression_value"),
        KernelRegressionTaskConfig(
            name="task_seq", type=TaskType.KERNEL_REGRESSION, data_column="task_seq_sequence_series", t_column=""
        ),
    ]

    with pytest.raises(ValueError, match="t_column for KernelRegression task 'task_seq' must be specified."):
        _make_dataset(
            sample_compositions,
            sample_descriptors,
            task_configs_no_t_spec,
            sample_attributes_df,
            dataset_name="test_no_t_column_spec",
        )
