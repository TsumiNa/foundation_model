import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from loguru import logger as loguru_logger  # ADDED for loguru bridge

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    RegressionTaskConfig,
    TaskType,
)


def _expected_swapped_indices(train_idx, val_idx, swap_ratio, seed):
    """Replicate the swapping logic used by CompoundDataModule."""
    if swap_ratio <= 0.0:
        return pd.Index(train_idx), pd.Index(val_idx)

    train_idx = pd.Index(train_idx)
    val_idx = pd.Index(val_idx)

    n_swap = int(min(len(train_idx), len(val_idx)) * swap_ratio)
    if n_swap == 0:
        return train_idx, val_idx

    rng = np.random.default_rng(seed)
    train_swap = pd.Index(rng.choice(train_idx.to_numpy(), size=n_swap, replace=False))
    val_swap = pd.Index(rng.choice(val_idx.to_numpy(), size=n_swap, replace=False))

    train_swap_set = set(train_swap)
    val_swap_set = set(val_swap)

    train_remaining = [idx for idx in train_idx if idx not in train_swap_set]
    val_remaining = [idx for idx in val_idx if idx not in val_swap_set]

    new_train = pd.Index(train_remaining + list(val_swap))
    new_val = pd.Index(val_remaining + list(train_swap))
    return new_train, new_val


# --- Fixtures ---
@pytest.fixture
def base_formula_df():  # 20 samples s0-s19
    return pd.DataFrame({"f1": np.random.rand(20), "f2": np.random.rand(20)}, index=[f"s{i}" for i in range(20)])


@pytest.fixture
def formula_df_subset():  # 15 samples, s0-s14
    return pd.DataFrame({"f1": np.random.rand(15), "f2": np.random.rand(15)}, index=[f"s{i}" for i in range(15)])


@pytest.fixture
def attributes_df_full_match():  # 20 samples s0-s19, with split col
    data = {
        "task1_regression_value": np.random.rand(20),
        "task_cls_classification_value": np.random.randint(0, 3, size=20),
        "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
    }
    return pd.DataFrame(data, index=[f"s{i}" for i in range(20)])


@pytest.fixture
def attributes_df_partial_match():  # 18 samples, s2-s19, with split col
    data = {
        "task1_regression_value": np.random.rand(18),
        "task_cls_classification_value": np.random.randint(0, 3, size=18),
        "split": ["train"] * 9 + ["val"] * 5 + ["test"] * 4,
    }
    return pd.DataFrame(data, index=[f"s{i}" for i in range(2, 20)])


@pytest.fixture
def attributes_df_no_split_full_match():  # 20 samples, s0-s19, no split col
    data = {
        "task1_regression_value": np.random.rand(20),
        "task_cls_classification_value": np.random.randint(0, 3, size=20),
    }
    return pd.DataFrame(data, index=[f"s{i}" for i in range(20)])


@pytest.fixture
def sample_task_configs_dm():
    return [
        RegressionTaskConfig(
            name="task1", type=TaskType.REGRESSION, data_column="task1_regression_value", dims=[256, 128, 1]
        ),
        ClassificationTaskConfig(
            name="task_cls", type=TaskType.CLASSIFICATION, data_column="task_cls_classification_value", num_classes=3
        ),
    ]


@pytest.fixture
def sample_task_configs_no_seq_dm():
    return [
        RegressionTaskConfig(
            name="task1", type=TaskType.REGRESSION, data_column="task1_regression_value", dims=[256, 128, 1]
        ),
        ClassificationTaskConfig(
            name="task_cls", type=TaskType.CLASSIFICATION, data_column="task_cls_classification_value", num_classes=3
        ),
    ]


@pytest.fixture
def temp_files_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# --- Test Cases ---


def test_datamodule_init_with_dfs(base_formula_df, attributes_df_full_match, sample_task_configs_dm):
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_df_full_match,
        task_configs=sample_task_configs_dm,
    )
    assert dm.formula_df is not None
    assert dm.attributes_df is not None
    assert len(dm.formula_df) == 20
    assert len(dm.attributes_df) == 20
    assert dm.formula_df.index.equals(attributes_df_full_match.index)


def test_datamodule_alignment_formula_master(base_formula_df, attributes_df_partial_match, sample_task_configs_dm):
    """Test alignment when formula_df is master and attributes_df is partial."""
    # base_formula_df (s0-s19)
    # attributes_df_partial_match (s2-s19)
    # Expected final common index: s2-s19
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df.copy(),
        attributes_source=attributes_df_partial_match.copy(),
        task_configs=sample_task_configs_dm,
    )
    expected_final_index = base_formula_df.index.intersection(attributes_df_partial_match.index)  # s2-s19
    assert dm.formula_df.index.equals(expected_final_index)
    assert dm.attributes_df.index.equals(expected_final_index)
    assert len(dm.formula_df) == 18
    assert len(dm.attributes_df) == 18


def test_datamodule_init_attributes_none_non_sequence(base_formula_df, sample_task_configs_no_seq_dm):
    """Test init with attributes_source=None and only non-sequence tasks."""
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=None,
        task_configs=sample_task_configs_no_seq_dm,
    )
    assert dm.formula_df is not None
    assert dm.attributes_df is None
    assert len(dm.formula_df) == 20


def test_datamodule_setup_split_column(base_formula_df, attributes_df_full_match, sample_task_configs_dm):
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_df_full_match,
        task_configs=sample_task_configs_dm,
    )
    dm.setup(stage="fit")
    assert len(dm.train_idx) == 10
    assert len(dm.val_idx) == 5
    dm.setup(stage="test")
    assert len(dm.test_idx) == 5


def test_datamodule_setup_random_split(base_formula_df, attributes_df_no_split_full_match, sample_task_configs_dm):
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_df_no_split_full_match,
        task_configs=sample_task_configs_dm,
        val_split=0.2,
        test_split=0.1,
    )
    dm.setup(stage="fit")  # test_split=0.1 -> 2 test, 18 train_val. val_split=0.2 -> 0.2/0.9 * 18 = 4 val. 14 train.
    assert len(dm.test_idx) == 2
    assert len(dm.val_idx) == 4
    assert len(dm.train_idx) == 14


def test_datamodule_setup_attributes_none_splitting(base_formula_df, sample_task_configs_no_seq_dm):
    """Test splitting when attributes_source is None (all data should go to train if not test_all)."""
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=None,
        task_configs=sample_task_configs_no_seq_dm,
        val_split=0.1,  # These should be ignored as no basis for random split from attributes
        test_split=0.1,
    )
    dm.setup("fit")
    assert len(dm.train_idx) == 20  # All data from formula_df
    assert dm.val_idx.empty
    assert dm.test_idx.empty

    dm_test_all = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=None,
        task_configs=sample_task_configs_no_seq_dm,
        test_all=True,
    )
    dm_test_all.setup("test")
    assert len(dm_test_all.test_idx) == 20
    assert dm_test_all.train_idx.empty
    assert dm_test_all.val_idx.empty


def test_swap_train_val_split_applied(base_formula_df, attributes_df_full_match, sample_task_configs_dm):
    swap_ratio = 0.4
    swap_seed = 123

    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_df_full_match,
        task_configs=sample_task_configs_dm,
        swap_train_val_split=swap_ratio,
        swap_train_val_seed=swap_seed,
    )

    original_train = attributes_df_full_match[attributes_df_full_match["split"] == "train"].index
    original_val = attributes_df_full_match[attributes_df_full_match["split"] == "val"].index
    expected_train, expected_val = _expected_swapped_indices(original_train, original_val, swap_ratio, swap_seed)

    dm.setup(stage="fit")

    assert list(dm.train_idx) == list(expected_train)
    assert list(dm.val_idx) == list(expected_val)

    # Ensure split labels were updated accordingly
    assert all(dm.attributes_df.loc[idx, "split"] == "train" for idx in dm.train_idx)
    assert all(dm.attributes_df.loc[idx, "split"] == "val" for idx in dm.val_idx)
    assert all(dm.attributes_df.loc[idx, "split"] == "test" for idx in dm.test_idx)


def test_swap_train_val_split_zero_no_change(base_formula_df, attributes_df_full_match, sample_task_configs_dm):
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_df_full_match,
        task_configs=sample_task_configs_dm,
        swap_train_val_split=0.0,
    )

    original_train = attributes_df_full_match[attributes_df_full_match["split"] == "train"].index
    original_val = attributes_df_full_match[attributes_df_full_match["split"] == "val"].index

    dm.setup(stage="fit")

    assert list(dm.train_idx) == list(original_train)
    assert list(dm.val_idx) == list(original_val)

def test_datamodule_setup_with_user_predict_idx(base_formula_df, sample_task_configs_no_seq_dm, caplog):
    """Test setup(stage='predict') with user-provided predict_idx."""

    # --- Loguru to caplog bridge (local to this test) ---
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = loguru_logger.add(PropagateHandler(), format="{message}", level="WARNING")
    # --- End bridge ---

    try:
        user_indices = pd.Index([f"s{i}" for i in range(5)])  # s0-s4
        dm = CompoundDataModule(
            formula_desc_source=base_formula_df,  # s0-s19
            attributes_source=None,  # Test with no attributes
            task_configs=sample_task_configs_no_seq_dm,
            predict_idx=user_indices,
        )
        dm.setup(stage="predict")
        assert dm.predict_idx.equals(user_indices)
        assert len(dm.predict_dataset) == 5

        # Test with partially valid predict_idx
        user_indices_mixed = pd.Index(
            [f"s{i}" for i in range(3)] + ["non_existent_1", "s19"]
        )  # s0,s1,s2,non_existent_1,s19
        expected_valid_mixed = pd.Index(["s0", "s1", "s2", "s19"])
        caplog.clear()
        dm_mixed = CompoundDataModule(
            formula_desc_source=base_formula_df,
            attributes_source=None,
            task_configs=sample_task_configs_no_seq_dm,
            predict_idx=user_indices_mixed,
        )
        dm_mixed.setup(stage="predict")
        assert dm_mixed.predict_idx.equals(expected_valid_mixed)
        assert len(dm_mixed.predict_dataset) == 4
        assert any(
            "User-provided predict_idx contains indices not found" in rec.message
            for rec in caplog.records  # Corrected from 'record' to 'rec' if that was the var name
            if rec.levelname == "WARNING"
        )

        # Test with all invalid predict_idx
        user_indices_invalid = pd.Index(["invalid1", "invalid2"])
        caplog.clear()
        dm_invalid = CompoundDataModule(
            formula_desc_source=base_formula_df,
            attributes_source=None,
            task_configs=sample_task_configs_no_seq_dm,
            predict_idx=user_indices_invalid,
        )
        dm_invalid.setup(stage="predict")
        assert dm_invalid.predict_idx.empty
        assert dm_invalid.predict_dataset is None
        assert any(
            "User-provided predict_idx resulted in an empty set" in rec.message
            for rec in caplog.records  # Corrected from 'record' to 'rec' if that was the var name
            if rec.levelname == "WARNING"
        )
    finally:
        loguru_logger.remove(handler_id)  # Clean up the handler


def test_datamodule_setup_predict_idx_fallback(
    base_formula_df, attributes_df_no_split_full_match, sample_task_configs_dm
):
    """Test predict_idx fallback logic when init_predict_idx is None."""
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_df_no_split_full_match,  # s0-s19
        task_configs=sample_task_configs_dm,
        test_split=0.1,  # test_idx will be 2 samples
        val_split=0,  # ensure train_val is not further split for simplicity here
    )
    dm.setup("test")  # Populates test_idx
    dm.setup("predict")
    assert dm.predict_idx.equals(dm.test_idx)
    assert len(dm.predict_idx) == 2  # 0.1 * 20

    dm_no_test = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_df_no_split_full_match,
        task_configs=sample_task_configs_dm,
        test_split=0.0,  # test_idx will be empty
    )
    dm_no_test.setup("test")
    dm_no_test.setup("predict")
    assert dm_no_test.predict_idx.equals(base_formula_df.index)  # Fallback to full_idx (from formula_df)
    assert len(dm_no_test.predict_idx) == 20


def test_datamodule_init_empty_formula_raises_error(sample_task_configs_dm):
    with pytest.raises(ValueError, match="formula_desc_source must be successfully loaded and cannot be empty."):
        CompoundDataModule(formula_desc_source=pd.DataFrame(), task_configs=sample_task_configs_dm)


def test_datamodule_init_attributes_provided_but_empty_raises_error(base_formula_df, sample_task_configs_dm):
    with pytest.raises(ValueError, match="attributes_source was provided but could not be loaded or is empty."):
        CompoundDataModule(
            formula_desc_source=base_formula_df,
            attributes_source=pd.DataFrame(),  # Provided but empty
            task_configs=sample_task_configs_dm,
        )


def test_datamodule_dataloaders_created(base_formula_df, attributes_df_full_match, sample_task_configs_dm):
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_df_full_match,
        task_configs=sample_task_configs_dm,
    )
    dm.setup()  # Setup all stages
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
    assert dm.test_dataloader() is not None
    dm.setup(stage="predict")  # Ensure predict_dataset is created
    assert dm.predict_dataloader() is not None


# Keep other existing tests like load_data_from_paths, load_data_numpy_array, etc.
# They might need fixture updates if they used base_attributes_df directly.
# For brevity, I'm focusing on the core logic changes.
# The test_datamodule_empty_dataset_returns_none_dataloader should still be relevant.
# test_datamodule_task_masking_ratios_propagation needs attributes_df_full_match.
# test_datamodule_structure_usage_in_dataset needs attributes_df_full_match.


def test_datamodule_load_data_from_paths(
    temp_files_dir, base_formula_df, attributes_df_full_match, sample_task_configs_dm
):
    formula_pkl_path = os.path.join(temp_files_dir, "formula.pkl")
    attributes_csv_path = os.path.join(temp_files_dir, "attributes.csv")
    base_formula_df.to_pickle(formula_pkl_path)
    attributes_df_full_match.to_csv(attributes_csv_path)

    dm = CompoundDataModule(
        formula_desc_source=formula_pkl_path,
        attributes_source=attributes_csv_path,
        task_configs=sample_task_configs_dm,
    )
    assert dm.formula_df is not None
    pd.testing.assert_frame_equal(dm.formula_df, base_formula_df, check_dtype=False)
    assert dm.attributes_df is not None
    # Check that the loaded attributes_df matches the original
    pd.testing.assert_frame_equal(dm.attributes_df, attributes_df_full_match, check_dtype=False)


def test_datamodule_task_masking_ratios_propagation(base_formula_df, attributes_df_full_match, sample_task_configs_dm):
    mask_ratios = {"task1": 0.5}
    dm = CompoundDataModule(
        formula_desc_source=base_formula_df,
        attributes_source=attributes_df_full_match,
        task_configs=sample_task_configs_dm,
        task_masking_ratios=mask_ratios,
    )
    dm.setup(stage="fit")
    assert dm.train_dataset is not None
    assert dm.train_dataset.task_masking_ratios == mask_ratios
    assert dm.val_dataset is not None
    assert dm.val_dataset.task_masking_ratios is None
    dm.setup(stage="test")
    assert dm.test_dataset is not None
    assert dm.test_dataset.task_masking_ratios is None
    dm.setup(stage="predict")
    assert dm.predict_dataset is not None
    assert dm.predict_dataset.task_masking_ratios is None


def test_datamodule_empty_dataset_returns_none_dataloader(base_formula_df, sample_task_configs_no_seq_dm, caplog):
    # --- Loguru to caplog bridge (local to this test) ---
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = loguru_logger.add(PropagateHandler(), format="{message}", level="INFO")
    # --- End bridge ---

    try:
        # Initialize DM with valid, non-empty formula_df to pass __init__
        dm = CompoundDataModule(
            formula_desc_source=base_formula_df,
            attributes_source=None,  # attributes_source can be None
            task_configs=sample_task_configs_no_seq_dm,
        )

        # Manually set dataset attributes to None to directly test dataloader guard clauses.
        # This bypasses the setup logic for creating these datasets, focusing only on the dataloader methods' behavior.
        dm.train_dataset = None
        dm.val_dataset = None
        dm.test_dataset = None
        dm.predict_dataset = None  # predict_dataset is also an attribute that can be None

        assert dm.train_dataloader() is None, "Train dataloader should be None when train_dataset is None"
        assert any(
            "train_dataloader: Train dataset is None or not initialized" in rec.message
            for rec in caplog.records
            if rec.levelname == "WARNING"
        ), "Expected log for None train_dataset not found"
        caplog.clear()

        assert dm.val_dataloader() is None, "Validation dataloader should be None when val_dataset is None"
        assert any(
            "val_dataloader: Validation dataset is None or not initialized" in rec.message
            for rec in caplog.records
            if rec.levelname == "INFO"
        ), "Expected log for None val_dataset not found"
        caplog.clear()

        assert dm.test_dataloader() is None, "Test dataloader should be None when test_dataset is None"
        assert any(
            "test_dataloader: Test dataset is None or not initialized" in rec.message
            for rec in caplog.records
            if rec.levelname == "INFO"
        ), "Expected log for None test_dataset not found"
        caplog.clear()

        assert dm.predict_dataloader() is None, "Predict dataloader should be None when predict_dataset is None"
        assert any(
            "predict_dataloader: Predict dataset is None or not initialized" in rec.message
            for rec in caplog.records
            if rec.levelname == "INFO"
        ), "Expected log for None predict_dataset not found"
    finally:
        loguru_logger.remove(handler_id)  # Clean up the handler
