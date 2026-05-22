# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for composition-keyed data-source helpers (refactor PR2)."""

import joblib
import numpy as np
import pandas as pd
import pytest
from loguru import logger

from foundation_model.data.composition_sources import (
    DescriptorCache,
    PrecomputedDescriptorSource,
    build_composition_universe,
    load_task_frame,
    lookup_descriptor_fn,
    normalize_composition,
    read_data_file,
    resolve_splits,
)


# --- normalize_composition --------------------------------------------------


def test_normalize_composition_formula_and_order_invariant():
    # Integer vs decimal spelling and element ordering all collapse to one canonical key.
    assert normalize_composition("Fe3O2") == normalize_composition("Fe3.0O2.0")
    assert normalize_composition("Fe2O3") == normalize_composition("O3Fe2")
    assert normalize_composition("Fe2O3") == "Fe2 O3"  # readable pymatgen .formula
    # Amounts are NOT reduced: absolute stoichiometry is preserved.
    assert normalize_composition("Fe2O3") != normalize_composition("Fe4O6")


def test_normalize_composition_accepts_mapping_dropping_none():
    # The qc 'composition' column stores every element, mostly None.
    sparse = {"Fe": 2.0, "O": 3.0, "Na": None, "Cl": 0.0}
    assert normalize_composition(sparse) == "Fe2 O3"


def test_normalize_composition_invalid_returns_none():
    assert normalize_composition({}) is None
    assert normalize_composition("not-a-formula!!") is None


def test_normalize_composition_fast_path_rejects_non_formula_strings():
    # Non-formula join keys (MP IDs, SMILES) are rejected without a pymatgen parse.
    assert normalize_composition("mp-1234") is None
    assert normalize_composition("*CC*") is None
    assert normalize_composition("") is None


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


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".pd", ".pd.z", ".pd.xz", ".pkl"])
def test_read_data_file_roundtrip(tmp_path, sample_frame, suffix):
    path = tmp_path / f"data{suffix}"
    if suffix == ".csv":
        sample_frame.to_csv(path, index=False)
    elif suffix == ".parquet":
        sample_frame.to_parquet(path)
    elif suffix == ".pkl":
        joblib.dump(sample_frame, path)
    else:  # .pd / .pd.z / .pd.xz pickled frames
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


def test_load_task_frame_uses_index_when_named_like_composition(tmp_path):
    """A file already indexed by the composition column (e.g. parquet) is accepted."""
    df = pd.DataFrame({"y": [1.0, 2.0]}, index=pd.Index(["A", "B"], name="id"))
    path = tmp_path / "t.parquet"
    df.to_parquet(path)
    frame = load_task_frame([str(path)], "id", task_name="t")
    assert list(frame.index) == ["A", "B"]
    assert list(frame["y"]) == [1.0, 2.0]


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


@pytest.mark.parametrize("extras", [pd.Index(["A", "Z"]), np.array(["A", "Z"])])
def test_build_composition_universe_accepts_array_like_extras(extras):
    """pd.Index / np.ndarray extras must not trip ambiguous-truthiness errors."""
    frames = {"t1": pd.DataFrame({"y": [1]}, index=pd.Index(["A"], name="composition"))}
    assert build_composition_universe(frames, extra_compositions=extras) == ["A", "Z"]


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


# --- PrecomputedDescriptorSource --------------------------------------------


@pytest.mark.parametrize("comp_col", [None, "id"])
def test_precomputed_descriptor_source_indexed_file(tmp_path, comp_col):
    """Looks up descriptors from a composition-indexed file; missing comps omitted."""
    df = pd.DataFrame({"d0": [1.0, 2.0, 3.0]}, index=pd.Index(["A", "B", "C"], name="id"))
    path = tmp_path / "desc.parquet"
    df.to_parquet(path)
    # composition_normalizer=None: look up by the raw (stringified) index.
    source = PrecomputedDescriptorSource(str(path), composition_column=comp_col, composition_normalizer=None)
    out = source(["C", "A", "missing"])
    assert list(out.index) == ["C", "A"]
    assert out.loc["C", "d0"] == 3.0


def test_precomputed_descriptor_source_column_file(tmp_path):
    """Looks up descriptors from a file where composition is a column."""
    df = pd.DataFrame({"id": ["A", "B"], "d0": [1.0, 2.0]})
    path = tmp_path / "desc.csv"
    df.to_csv(path, index=False)
    source = PrecomputedDescriptorSource(str(path), composition_column="id", composition_normalizer=None)
    out = source(["A"])
    assert list(out.index) == ["A"]
    assert out.loc["A", "d0"] == 1.0


def test_precomputed_descriptor_source_normalizes_by_default(tmp_path):
    """By default the index is canonicalized, so heterogeneous spellings of a query still hit."""
    df = pd.DataFrame({"d0": [1.0, 2.0]}, index=pd.Index(["Fe2O3", "H2O"], name="composition"))
    path = tmp_path / "desc.parquet"
    df.to_parquet(path)
    source = PrecomputedDescriptorSource(str(path), composition_column="composition")
    # Queried with a different spelling of Fe2O3 (decimal amounts, reversed order).
    out = source(["O3.0Fe2.0", "missing"])
    assert list(out.index) == [normalize_composition("Fe2O3")]
    assert out.iloc[0]["d0"] == 1.0


def test_lookup_descriptor_fn_aligns_heterogeneous_spellings():
    features = pd.DataFrame({"d0": [1.0, 2.0]}, index=pd.Index(["Fe2O3", "NaCl"]))
    fn = lookup_descriptor_fn(features)
    # Descriptor frame spelled "Fe2O3"; query spelled with float amounts -> still matches.
    out = fn([normalize_composition("Fe2.0O3.0"), "missing"])
    assert list(out.index) == [normalize_composition("Fe2O3")]
    assert out.iloc[0]["d0"] == 1.0


def test_lookup_descriptor_fn_dedupes_colliding_labels(loguru_warnings):
    # Duplicate raw labels / spellings that canonicalize to the same key must not crash the
    # length-matched re-index; they collapse keep-first with a warning (Codex P1 regression).
    features = pd.DataFrame({"d0": [1.0, 9.0, 2.0]}, index=pd.Index(["Fe2O3", "Fe2.0O3.0", "NaCl"]))
    fn = lookup_descriptor_fn(features)
    out = fn([normalize_composition("Fe2O3"), normalize_composition("NaCl")])
    assert list(out.index) == [normalize_composition("Fe2O3"), normalize_composition("NaCl")]
    assert out.loc[normalize_composition("Fe2O3"), "d0"] == 1.0  # first occurrence kept
    assert any("collapsed to a duplicate" in m for m in loguru_warnings)


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
