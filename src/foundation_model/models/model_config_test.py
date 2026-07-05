# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-task data-source fields added to BaseTaskConfig (composition refactor PR1)."""

import pytest

from foundation_model.models.model_config import (
    PREDICT_IDX_LITERALS,
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
    RegressionTaskConfig,
    TaskType,
)

# Subclasses whose extra positional fields all have defaults, so they can be built with
# just ``name`` and exercise the inherited BaseTaskConfig.__post_init__.
ALL_TASK_CONFIG_CLASSES = [
    RegressionTaskConfig,
    ClassificationTaskConfig,
    KernelRegressionTaskConfig,
]


@pytest.mark.parametrize("config_cls", ALL_TASK_CONFIG_CLASSES)
def test_new_fields_have_expected_defaults(config_cls):
    """New per-task data-source fields default to inert values across all task types."""
    cfg = config_cls(name="t")
    assert cfg.data_files == ()
    assert cfg.composition_column is None
    assert cfg.split_column == "split"
    assert cfg.task_masking_ratio is None
    assert cfg.predict_idx is None


def test_data_files_single_string_is_wrapped_in_tuple():
    cfg = RegressionTaskConfig(name="t", data_files="a.parquet")
    assert cfg.data_files == ("a.parquet",)


def test_data_files_sequence_is_normalized_to_tuple_of_str():
    cfg = RegressionTaskConfig(name="t", data_files=["a.csv", "b.pd.xz"])
    assert cfg.data_files == ("a.csv", "b.pd.xz")
    assert isinstance(cfg.data_files, tuple)


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_task_masking_ratio_accepts_unit_interval(ratio):
    cfg = RegressionTaskConfig(name="t", task_masking_ratio=ratio)
    assert cfg.task_masking_ratio == ratio


@pytest.mark.parametrize("ratio", [-0.1, 1.0001, 2.0])
def test_task_masking_ratio_rejects_out_of_range(ratio):
    with pytest.raises(ValueError, match="task_masking_ratio must be in"):
        RegressionTaskConfig(name="t", task_masking_ratio=ratio)


@pytest.mark.parametrize("literal", sorted(PREDICT_IDX_LITERALS))
def test_predict_idx_accepts_known_literals(literal):
    cfg = RegressionTaskConfig(name="t", predict_idx=literal)
    assert cfg.predict_idx == literal


def test_predict_idx_rejects_unknown_string():
    with pytest.raises(ValueError, match="predict_idx string must be one of"):
        RegressionTaskConfig(name="t", predict_idx="holdout")


def test_predict_idx_sequence_is_normalized_to_tuple_of_str():
    cfg = RegressionTaskConfig(name="t", predict_idx=["comp_a", "comp_b"])
    assert cfg.predict_idx == ("comp_a", "comp_b")
    assert isinstance(cfg.predict_idx, tuple)


def test_predict_idx_none_stays_none():
    cfg = RegressionTaskConfig(name="t", predict_idx=None)
    assert cfg.predict_idx is None


def test_predict_idx_non_iterable_raises_type_error():
    with pytest.raises(TypeError, match="predict_idx must be None"):
        RegressionTaskConfig(name="t", predict_idx=5)  # type: ignore[arg-type]


def test_existing_construction_pattern_still_works():
    """Pre-refactor positional/keyword usage must remain valid (backward compatibility)."""
    cfg = RegressionTaskConfig(name="bandgap", data_column="bandgap", enabled=True)
    assert cfg.name == "bandgap"
    assert cfg.data_column == "bandgap"
    assert cfg.type == TaskType.REGRESSION
    # New fields are inert for old-style configs.
    assert cfg.data_files == ()
    assert cfg.predict_idx is None
