# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for composition-keyed data-source helpers (refactor PR2)."""

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from foundation_model.data.composition_sources import (
    DescriptorCache,
    build_composition_universe,
    load_task_frame,
    read_data_file,
    resolve_splits,
)


@pytest.fixture
def loguru_warnings():
    """Capture loguru WARNING-level messages (caplog only sees stdlib logging)."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


# --- read_data_file ---------------------------------------------------------


@pytest.fixture
def sample_frame():
    return pd.DataFrame({"composition": ["H2O", "CO2", "NaCl"], "y": [1.0, 2.0, 3.0]})


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".pd.xz"])
def test_read_data_file_roundtrip(tmp_path, sample_frame, suffix):
    path = tmp_path / f"data{suffix}"
    if suffix == ".csv":
        sample_frame.to_csv(path, index=False)
    elif suffix == ".parquet":
        sample_frame.to_parquet(path)
    else:
        sample_frame.to_pickle(path)
    loaded = read_data_file(str(path))
    assert list(loaded["composition"]) == ["H2O", "CO2", "NaCl"]
    assert list(loaded["y"]) == [1.0, 2.0, 3.0]


def test_read_data_file_rejects_unknown_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_data_file(str(path))


# --- load_task_frame --------------------------------------------------------


def test_load_task_frame_indexes_by_composition(tmp_path, sample_frame):
    path = tmp_path / "t.csv"
    sample_frame.to_csv(path, index=False)
    frame = load_task_frame([str(path)], "composition", task_name="t")
    assert frame.index.name == "composition"
    assert list(frame.index) == ["H2O", "CO2", "NaCl"]
    assert "composition" not in frame.columns
    assert list(frame["y"]) == [1.0, 2.0, 3.0]


def test_load_task_frame_concatenates_multiple_files(tmp_path):
    a = pd.DataFrame({"composition": ["A", "B"], "y": [1.0, 2.0]})
    b = pd.DataFrame({"composition": ["C"], "y": [3.0]})
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)
    frame = load_task_frame([str(pa), str(pb)], "composition")
    assert list(frame.index) == ["A", "B", "C"]


def test_load_task_frame_dedupes_keep_first_with_warning(tmp_path, loguru_warnings):
    a = pd.DataFrame({"composition": ["A", "B"], "y": [1.0, 2.0]})
    b = pd.DataFrame({"composition": ["B", "C"], "y": [99.0, 3.0]})
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)
    frame = load_task_frame([str(pa), str(pb)], "composition", task_name="dup")
    assert list(frame.index) == ["A", "B", "C"]
    assert frame.loc["B", "y"] == 2.0  # first occurrence kept
    assert any("duplicate composition" in m for m in loguru_warnings)


def test_load_task_frame_missing_composition_column_raises(tmp_path):
    df = pd.DataFrame({"name": ["A"], "y": [1.0]})
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="composition column 'composition' not found"):
        load_task_frame([str(path)], "composition", task_name="t")


def test_load_task_frame_empty_files_raises():
    with pytest.raises(ValueError, match="data_files is empty"):
        load_task_frame([], "composition", task_name="t")


# --- build_composition_universe ---------------------------------------------


def test_build_composition_universe_unions_and_preserves_order():
    frames = {
        "t1": pd.DataFrame({"y": [1, 2]}, index=pd.Index(["A", "B"], name="composition")),
        "t2": pd.DataFrame({"y": [1, 2]}, index=pd.Index(["B", "C"], name="composition")),
    }
    assert build_composition_universe(frames) == ["A", "B", "C"]


def test_build_composition_universe_includes_extras():
    frames = {"t1": pd.DataFrame({"y": [1]}, index=pd.Index(["A"], name="composition"))}
    assert build_composition_universe(frames, extra_compositions=["A", "Z"]) == ["A", "Z"]


# --- DescriptorCache --------------------------------------------------------


def _make_counting_fn(calls):
    def fn(comps):
        calls.append(list(comps))
        return pd.DataFrame({"d0": [float(len(c)) for c in comps]}, index=list(comps))

    return fn


def test_descriptor_cache_computes_each_composition_once():
    calls = []
    cache = DescriptorCache(_make_counting_fn(calls))
    first = cache.compute(["a", "bb"])
    second = cache.compute(["bb", "ccc"])
    assert calls == [["a", "bb"], ["ccc"]]  # "bb" not recomputed
    assert list(first.index) == ["a", "bb"]
    assert list(second.index) == ["bb", "ccc"]
    assert second.loc["ccc", "d0"] == 3.0


def test_descriptor_cache_preserves_requested_order():
    cache = DescriptorCache(_make_counting_fn([]))
    out = cache.compute(["ccc", "a", "bb"])
    assert list(out.index) == ["ccc", "a", "bb"]


def test_descriptor_cache_resolve_drops_missing_compositions():
    def fn(comps):
        keep = [c for c in comps if c != "bad"]
        return pd.DataFrame({"d0": [1.0] * len(keep)}, index=keep)

    cache = DescriptorCache(fn)
    valid, dropped = cache.resolve(["a", "bad", "b"])
    assert list(valid.index) == ["a", "b"]
    assert dropped == ["bad"]


def test_descriptor_cache_resolve_drops_all_nan_rows():
    def fn(comps):
        return pd.DataFrame({"d0": [np.nan if c == "x" else 1.0 for c in comps]}, index=list(comps))

    cache = DescriptorCache(fn)
    valid, dropped = cache.resolve(["x", "y"])
    assert list(valid.index) == ["y"]
    assert dropped == ["x"]


def test_descriptor_cache_does_not_retry_known_missing():
    calls = []

    def fn(comps):
        calls.append(list(comps))
        keep = [c for c in comps if c != "bad"]
        return pd.DataFrame({"d0": [1.0] * len(keep)}, index=keep)

    cache = DescriptorCache(fn)
    cache.resolve(["a", "bad"])
    cache.resolve(["bad", "a"])  # "bad" already attempted; "a" cached
    assert calls == [["a", "bad"]]


def test_descriptor_cache_rejects_non_dataframe():
    cache = DescriptorCache(lambda comps: [1, 2, 3])  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError, match="must return a pandas DataFrame"):
        cache.compute(["a"])


# --- resolve_splits ---------------------------------------------------------


def _frame_with_split(index, labels):
    return pd.DataFrame({"split": labels}, index=pd.Index(index, name="composition"))


def test_resolve_splits_reads_split_column():
    frames = {"t1": _frame_with_split(["a", "b", "c"], ["train", "val", "test"])}
    out = resolve_splits(frames, ["a", "b", "c"], {"t1": "split"}, val_split=0.0, test_split=0.0)
    assert out.to_dict() == {"a": "train", "b": "val", "c": "test"}


def test_resolve_splits_conflict_uses_precedence(loguru_warnings):
    frames = {
        "t1": _frame_with_split(["a"], ["train"]),
        "t2": _frame_with_split(["a"], ["test"]),
    }
    out = resolve_splits(frames, ["a"], {"t1": "split", "t2": "split"})
    assert out["a"] == "test"  # test > train
    assert any("conflicting split labels" in m for m in loguru_warnings)


def test_resolve_splits_invalid_label_raises():
    frames = {"t1": _frame_with_split(["a"], ["holdout"])}
    with pytest.raises(ValueError, match="invalid value 'holdout'"):
        resolve_splits(frames, ["a"], {"t1": "split"})


def test_resolve_splits_random_fallback_is_deterministic():
    comps = [f"c{i}" for i in range(10)]
    frames = {"t1": pd.DataFrame({"y": range(10)}, index=pd.Index(comps, name="composition"))}
    kwargs = dict(val_split=0.2, test_split=0.2, random_seed=42)
    out1 = resolve_splits(frames, comps, {"t1": "split"}, **kwargs)
    out2 = resolve_splits(frames, comps, {"t1": "split"}, **kwargs)
    assert out1.equals(out2)
    counts = out1.value_counts().to_dict()
    assert counts["test"] == 2
    assert counts["val"] == 2
    assert counts["train"] == 6


def test_resolve_splits_mixes_labeled_and_random():
    frames = {"t1": _frame_with_split(["a", "b", "c"], ["train", None, "test"])}
    out = resolve_splits(frames, ["a", "b", "c"], {"t1": "split"}, val_split=0.0, test_split=0.0, random_seed=0)
    assert out["a"] == "train"
    assert out["c"] == "test"
    assert out["b"] in {"train", "val", "test"}  # unlabeled -> random fallback


def test_resolve_splits_test_all_overrides_everything():
    frames = {"t1": _frame_with_split(["a", "b"], ["train", "val"])}
    out = resolve_splits(frames, ["a", "b"], {"t1": "split"}, test_all=True)
    assert out.to_dict() == {"a": "test", "b": "test"}


def test_resolve_splits_ignores_tasks_without_split_column():
    frames = {"t1": pd.DataFrame({"y": [1, 2]}, index=pd.Index(["a", "b"], name="composition"))}
    out = resolve_splits(frames, ["a", "b"], {"t1": "split"}, val_split=0.5, test_split=0.0, random_seed=1)
    assert set(out.unique()) <= {"train", "val", "test"}
    assert len(out) == 2
