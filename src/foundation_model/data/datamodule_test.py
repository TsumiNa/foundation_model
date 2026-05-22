# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the composition-keyed CompoundDataModule (refactor PR3)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from torch.utils.data.distributed import DistributedSampler

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.model_config import (
    ClassificationTaskConfig,
    RegressionTaskConfig,
    _AEConfig,
)

COMPOSITIONS = [f"s{i}" for i in range(20)]


@pytest.fixture
def loguru_messages():
    """Capture loguru INFO+ messages (caplog only sees stdlib logging)."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def make_descriptor_fn(descriptor_df, call_log=None):
    """Build a descriptor_fn that looks up rows from a precomputed descriptor frame."""

    def fn(compositions):
        if call_log is not None:
            call_log.append(list(compositions))
        present = [c for c in compositions if c in descriptor_df.index]
        return descriptor_df.loc[present]

    return fn


@pytest.fixture
def descriptors_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f1": rng.random(20), "f2": rng.random(20)}, index=list(COMPOSITIONS))


@pytest.fixture
def reg_cls_configs():
    return [
        RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 16, 1]),
        ClassificationTaskConfig(name="task_cls", data_column="task_cls", num_classes=3, dims=[2, 16, 3]),
    ]


def _reg_cls_frames(split=None):
    rng = np.random.default_rng(1)
    data1 = {"task1": rng.random(20)}
    data_cls = {"task_cls": rng.integers(0, 3, size=20)}
    if split is not None:
        data1["split"] = split
        data_cls["split"] = split
    return {
        "task1": pd.DataFrame(data1, index=list(COMPOSITIONS)),
        "task_cls": pd.DataFrame(data_cls, index=list(COMPOSITIONS)),
    }


def build_dm(descriptors_df, task_frames=None, configs=None, call_log=None, **kwargs):
    return CompoundDataModule(
        task_configs=configs,
        descriptor_fn=make_descriptor_fn(descriptors_df, call_log),
        task_frames=task_frames,
        **kwargs,
    )


# --- init validation --------------------------------------------------------


def test_init_requires_descriptor_fn(reg_cls_configs):
    with pytest.raises(ValueError, match="descriptor_fn must be provided"):
        CompoundDataModule(task_configs=reg_cls_configs, descriptor_fn=None)  # type: ignore[arg-type]


def test_init_requires_task_configs(descriptors_df):
    with pytest.raises(ValueError, match="task_configs cannot be empty"):
        CompoundDataModule(task_configs=[], descriptor_fn=make_descriptor_fn(descriptors_df))


def test_init_rejects_bad_swap_ratio(descriptors_df, reg_cls_configs):
    with pytest.raises(ValueError, match="swap_train_val_split"):
        build_dm(descriptors_df, configs=reg_cls_configs, swap_train_val_split=1.5)


@pytest.mark.parametrize("kwargs", [{"val_split": -0.1}, {"test_split": 1.5}, {"val_split": 0.6, "test_split": 0.6}])
def test_init_rejects_bad_split_bounds(descriptors_df, reg_cls_configs, kwargs):
    with pytest.raises(ValueError, match="split"):
        build_dm(descriptors_df, configs=reg_cls_configs, **kwargs)


def test_in_memory_frame_dedupes_duplicate_compositions(descriptors_df):
    configs = [RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 16, 1])]
    # Duplicate composition "s0" with conflicting values; keep-first must win.
    frame = pd.DataFrame({"task1": [10.0, 99.0, 20.0]}, index=pd.Index(["s0", "s0", "s1"], name="composition"))
    dm = build_dm(descriptors_df, task_frames={"task1": frame}, configs=configs, test_all=True)
    dm.setup(stage="test")
    assert dm._task_frames["task1"].loc["s0", "task1"] == 10.0
    assert not dm._task_frames["task1"].index.duplicated().any()


# --- sources & descriptors --------------------------------------------------


def test_in_memory_task_frames_drive_setup(descriptors_df, reg_cls_configs):
    dm = build_dm(descriptors_df, task_frames=_reg_cls_frames(), configs=reg_cls_configs, test_all=True)
    dm.setup(stage="test")
    assert dm.descriptors is not None
    assert len(dm.descriptors) == 20
    assert len(dm.test_idx) == 20
    assert dm.test_dataset is not None and len(dm.test_dataset) == 20


def test_descriptor_fn_caches_across_stages(descriptors_df, reg_cls_configs):
    call_log: list[list[str]] = []
    dm = build_dm(descriptors_df, task_frames=_reg_cls_frames(), configs=reg_cls_configs, call_log=call_log)
    dm.setup(stage="fit")
    dm.setup(stage="test")
    dm.setup(stage="predict")
    # Sources are prepared once; descriptor_fn invoked a single time over the universe.
    assert len(call_log) == 1
    assert sorted(call_log[0]) == sorted(COMPOSITIONS)


def test_dropped_composition_without_descriptor(descriptors_df, reg_cls_configs, loguru_messages):
    # Descriptor frame missing s19 -> it should be dropped from the master index.
    partial = descriptors_df.drop(index=["s19"])
    dm = build_dm(partial, task_frames=_reg_cls_frames(), configs=reg_cls_configs, test_all=True)
    dm.setup(stage="test")
    assert "s19" not in dm.master_index
    assert len(dm.master_index) == 19
    assert any("dropped" in m for m in loguru_messages)


def test_data_files_loading(tmp_path, descriptors_df):
    frame = pd.DataFrame({"composition": list(COMPOSITIONS), "task1": np.arange(20.0)})
    path = tmp_path / "task1.parquet"
    frame.to_parquet(path)
    configs = [RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 16, 1], data_files=str(path))]
    dm = build_dm(descriptors_df, configs=configs, test_all=True)
    dm.setup(stage="test")
    assert "task1" in dm._task_frames
    assert len(dm._task_frames["task1"]) == 20


def test_default_data_files_shared_across_tasks(tmp_path, descriptors_df):
    """A single shared file fills in tasks that declare no data_files of their own."""
    shared = pd.DataFrame({"composition": list(COMPOSITIONS), "task1": np.arange(20.0), "task2": np.arange(20.0)})
    path = tmp_path / "shared.parquet"
    shared.to_parquet(path)
    configs = [
        RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 16, 1]),
        RegressionTaskConfig(name="task2", data_column="task2", dims=[2, 16, 1]),
    ]
    dm = CompoundDataModule(
        task_configs=configs,
        descriptor_fn=make_descriptor_fn(descriptors_df),
        default_data_files=str(path),
        test_all=True,
    )
    dm.setup(stage="test")
    assert set(dm._task_frames) == {"task1", "task2"}
    assert len(dm._task_frames["task1"]) == 20


# --- splitting --------------------------------------------------------------


def test_datamodule_normalizes_heterogeneous_compositions():
    """Task frames and descriptors that spell the same compositions differently join by default."""
    from foundation_model.data.composition_sources import lookup_descriptor_fn, normalize_composition

    descs = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]}, index=pd.Index(["Fe2O3", "H2O"]))
    frames = {
        "task1": pd.DataFrame({"task1": [1.0, 2.0]}, index=pd.Index(["Fe2O3", "H2O"])),  # plain
        # Same two compositions, spelled with float amounts / reversed order.
        "task_cls": pd.DataFrame({"task_cls": [0, 1]}, index=pd.Index(["O3.0Fe2.0", "H2.0O1.0"])),
    }
    configs = [
        RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 8, 1]),
        ClassificationTaskConfig(name="task_cls", data_column="task_cls", num_classes=2, dims=[2, 8, 2]),
    ]
    dm = CompoundDataModule(
        task_configs=configs, descriptor_fn=lookup_descriptor_fn(descs), task_frames=frames, test_all=True
    )
    dm.setup(stage="test")
    # Both spellings collapse to the same two canonical keys (universe stays 2, nothing dropped).
    assert set(dm.master_index) == {normalize_composition("Fe2O3"), normalize_composition("H2O")}


def test_datamodule_disable_normalizer_keeps_raw_keys(descriptors_df, reg_cls_configs):
    """composition_normalizer=None preserves raw string keys (synthetic 's0' fixtures unchanged)."""
    dm = CompoundDataModule(
        task_configs=reg_cls_configs,
        descriptor_fn=make_descriptor_fn(descriptors_df),
        task_frames=_reg_cls_frames(),
        composition_normalizer=None,
    )
    dm.setup(stage="fit")
    assert set(dm.master_index) == set(COMPOSITIONS)


def test_datamodule_syncs_normalizer_into_precomputed_source(reg_cls_configs):
    """The DataModule's opt-out propagates into a PrecomputedDescriptorSource (single source of truth)."""
    from foundation_model.data.composition_sources import PrecomputedDescriptorSource

    source = PrecomputedDescriptorSource("unused.parquet")  # defaults to normalization ON
    CompoundDataModule(task_configs=reg_cls_configs, descriptor_fn=source, composition_normalizer=None)
    assert source._composition_normalizer is None


def test_split_column_resolution(descriptors_df, reg_cls_configs):
    split = ["train"] * 10 + ["val"] * 5 + ["test"] * 5
    dm = build_dm(descriptors_df, task_frames=_reg_cls_frames(split=split), configs=reg_cls_configs)
    dm.setup(stage="fit")
    assert len(dm.train_idx) == 10
    assert len(dm.val_idx) == 5
    dm.setup(stage="test")
    assert len(dm.test_idx) == 5


def test_random_split_counts(descriptors_df, reg_cls_configs):
    dm = build_dm(
        descriptors_df,
        task_frames=_reg_cls_frames(),  # no split column -> random fallback
        configs=reg_cls_configs,
        val_split=0.2,
        test_split=0.1,
        random_seed=42,
    )
    dm.setup(stage="fit")
    assert len(dm.test_idx) == 2  # round(20 * 0.1)
    assert len(dm.val_idx) == 4  # round(20 * 0.2)
    assert len(dm.train_idx) == 14


def test_swap_train_val_applied(descriptors_df, reg_cls_configs):
    split = ["train"] * 10 + ["val"] * 5 + ["test"] * 5
    dm = build_dm(
        descriptors_df,
        task_frames=_reg_cls_frames(split=split),
        configs=reg_cls_configs,
        swap_train_val_split=0.4,
        random_seed=123,
    )
    dm.setup(stage="fit")
    # Sizes are preserved by the swap; contents change.
    assert len(dm.train_idx) == 10
    assert len(dm.val_idx) == 5
    assert set(dm.train_idx).isdisjoint(set(dm.test_idx))


def test_swap_zero_no_change(descriptors_df, reg_cls_configs):
    split = ["train"] * 10 + ["val"] * 5 + ["test"] * 5
    dm = build_dm(
        descriptors_df, task_frames=_reg_cls_frames(split=split), configs=reg_cls_configs, swap_train_val_split=0.0
    )
    dm.setup(stage="fit")
    assert list(dm.train_idx) == COMPOSITIONS[:10]
    assert list(dm.val_idx) == COMPOSITIONS[10:15]


# --- masking ----------------------------------------------------------------


def test_masking_ratio_from_config(descriptors_df):
    configs = [
        RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 16, 1], task_masking_ratio=0.5),
        ClassificationTaskConfig(name="task_cls", data_column="task_cls", num_classes=3, dims=[2, 16, 3]),
    ]
    dm = build_dm(
        descriptors_df, task_frames=_reg_cls_frames(), configs=configs, test_all=False, test_split=0.0, val_split=0.0
    )
    dm.setup(stage="fit")
    assert dm.train_dataset is not None
    assert dm.train_dataset.task_masking_ratios == {"task1": 0.5}
    # Validation/test/predict never apply training masking.
    dm.setup(stage="test")
    if dm.test_dataset is not None:
        assert dm.test_dataset.task_masking_ratios is None


# --- prediction -------------------------------------------------------------


def test_predict_default_is_test_split(descriptors_df, reg_cls_configs):
    split = ["train"] * 10 + ["val"] * 5 + ["test"] * 5
    dm = build_dm(descriptors_df, task_frames=_reg_cls_frames(split=split), configs=reg_cls_configs)
    dm.setup(stage="fit")  # populate splits
    dm.setup(stage="predict")
    assert sorted(dm.predict_compositions) == sorted(COMPOSITIONS[15:20])
    assert dm.predict_dataset is not None and len(dm.predict_dataset) == 5


def test_predict_literal_all(descriptors_df):
    configs = [RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 16, 1], predict_idx="all")]
    dm = build_dm(descriptors_df, task_frames={"task1": _reg_cls_frames()["task1"]}, configs=configs)
    dm.setup(stage="predict")
    assert sorted(dm.predict_compositions) == sorted(COMPOSITIONS)


def test_predict_explicit_subset_union(descriptors_df):
    configs = [
        RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 16, 1], predict_idx=["s0", "s1"]),
        RegressionTaskConfig(name="task2", data_column="task2", dims=[2, 16, 1], predict_idx=["s2", "s3"]),
    ]
    frames = {
        "task1": pd.DataFrame({"task1": np.arange(20.0)}, index=list(COMPOSITIONS)),
        "task2": pd.DataFrame({"task2": np.arange(20.0)}, index=list(COMPOSITIONS)),
    }
    dm = build_dm(descriptors_df, task_frames=frames, configs=configs)
    dm.setup(stage="predict")
    # Union of the two per-task subsets, in master order.
    assert dm.predict_compositions == ["s0", "s1", "s2", "s3"]


def test_predict_explicit_subset_with_predict_only_composition(descriptors_df):
    # s99 has no task target but is a requested predict composition; it must still get a descriptor.
    desc = pd.concat([descriptors_df, pd.DataFrame({"f1": [0.5], "f2": [0.5]}, index=["s99"])])
    configs = [RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 16, 1], predict_idx=["s0", "s99"])]
    dm = build_dm(desc, task_frames={"task1": _reg_cls_frames()["task1"]}, configs=configs)
    dm.setup(stage="predict")
    assert set(dm.predict_compositions) == {"s0", "s99"}


# --- dataloaders ------------------------------------------------------------


def test_dataloaders_created(descriptors_df, reg_cls_configs):
    split = ["train"] * 10 + ["val"] * 5 + ["test"] * 5
    dm = build_dm(descriptors_df, task_frames=_reg_cls_frames(split=split), configs=reg_cls_configs)
    dm.setup()
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
    assert dm.test_dataloader() is not None
    dm.setup(stage="predict")
    assert dm.predict_dataloader() is not None


def test_empty_dataset_returns_none_dataloader(descriptors_df, reg_cls_configs):
    dm = build_dm(descriptors_df, task_frames=_reg_cls_frames(), configs=reg_cls_configs)
    dm.train_dataset = None
    dm.val_dataset = None
    dm.test_dataset = None
    dm.predict_dataset = None
    assert dm.train_dataloader() is None
    assert dm.val_dataloader() is None
    assert dm.test_dataloader() is None
    assert dm.predict_dataloader() is None


# --- task type behavior -----------------------------------------------------


def test_autoencoder_rides_along_with_supervised(descriptors_df):
    configs = [
        _AEConfig(dims=[2, 4, 2], name="ae_task"),
        RegressionTaskConfig(name="reg_task", data_column="reg_task", dims=[2, 16, 1]),
    ]
    frames = {"reg_task": pd.DataFrame({"reg_task": np.arange(20.0)}, index=list(COMPOSITIONS))}
    dm = build_dm(descriptors_df, task_frames=frames, configs=configs, test_all=True)
    dm.setup(stage="test")
    loader = dm.test_dataloader()
    x, y_dict, masks, _ = next(iter(loader))
    assert x.shape[1] == 2
    # AE task contributes no target/mask; the supervised task does.
    assert "reg_task" in y_dict and "ae_task" not in y_dict
    assert "reg_task" in masks and "ae_task" not in masks


def test_mixed_tasks_batch_structure(descriptors_df):
    configs = [
        _AEConfig(dims=[2, 4, 2], name="ae_task"),
        RegressionTaskConfig(name="reg_task", data_column="reg_task", dims=[2, 16, 1]),
    ]
    frames = {"reg_task": pd.DataFrame({"reg_task": np.arange(20.0)}, index=list(COMPOSITIONS))}
    dm = build_dm(descriptors_df, task_frames=frames, configs=configs, batch_size=4, test_all=True)
    dm.setup(stage="test")
    x, y_dict, masks, t_seqs = next(iter(dm.test_dataloader()))
    assert x.shape == (4, 2)
    assert "reg_task" in y_dict


# --- DistributedSampler coverage --------------------------------------------


def _ddp_dm(descriptors_df):
    split = ["train"] * 10 + ["val"] * 5 + ["test"] * 5
    return build_dm(
        descriptors_df,
        task_frames=_reg_cls_frames(split=split),
        configs=[RegressionTaskConfig(name="task1", data_column="task1", dims=[2, 16, 1])],
        batch_size=4,
    )


def test_single_gpu_no_distributed_sampler(descriptors_df):
    with patch("torch.distributed.is_available", return_value=False):
        with patch("torch.distributed.is_initialized", return_value=False):
            dm = _ddp_dm(descriptors_df)
            dm.setup(stage="fit")
            loader = dm.train_dataloader()
            assert not isinstance(loader.sampler, DistributedSampler)


def test_multi_gpu_uses_distributed_sampler(descriptors_df):
    with patch("foundation_model.data.datamodule.torch.distributed.is_available", return_value=True):
        with patch("foundation_model.data.datamodule.torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_rank", return_value=0):
                with patch("torch.distributed.get_world_size", return_value=2):
                    dm = _ddp_dm(descriptors_df)
                    dm.setup(stage="fit")
                    loader = dm.train_dataloader()
                    assert isinstance(loader.sampler, DistributedSampler)
                    assert loader.sampler.shuffle is True
                    assert loader.sampler.drop_last is False


def test_multi_gpu_val_sampler_no_shuffle(descriptors_df):
    with patch("foundation_model.data.datamodule.torch.distributed.is_available", return_value=True):
        with patch("foundation_model.data.datamodule.torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_rank", return_value=0):
                with patch("torch.distributed.get_world_size", return_value=2):
                    dm = _ddp_dm(descriptors_df)
                    dm.setup(stage="fit")
                    loader = dm.val_dataloader()
                    assert isinstance(loader.sampler, DistributedSampler)
                    assert loader.sampler.shuffle is False
