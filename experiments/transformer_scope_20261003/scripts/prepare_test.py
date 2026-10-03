import json
import numpy as np
import pandas as pd
import pytest
from prepare import DATE, SOURCE, TARGET, digest, isolate, prepare


def test_alias_split_precedence_preserves_original_formula():
    f = isolate(pd.DataFrame({"composition": ["Fe4 O6", "Fe2 O3", "NaCl"], "split": ["train", "test", "val"]}))
    assert f.composition.tolist() == ["Fe4 O6", "NaCl"] and f.split.tolist() == ["test", "val"]


def test_hash_splits_row_independent_and_nested_subsets():
    from prepare import partitions, TARGET

    f = pd.DataFrame(dict(identity=[f"id{i}" for i in range(1000)]))
    for n in TARGET:
        f[n] = np.arange(1000, dtype=float)
    a = partitions(f, 20261011)
    b = partitions(f.sample(frac=1, random_state=7), 20261011).sort_index()
    pd.testing.assert_frame_equal(a, b)
    assert not a.split.equals(partitions(f, 20261012).split)
    for j in range(4):
        one = a[f"target{j}_f001_train"]
        ten = a[f"target{j}_f010_train"]
        all_ = a[f"target{j}_f100_train"]
        assert (one <= ten).all() and (ten <= all_).all() and (all_ == a.split.eq("train")).all()


def test_prepare_writes_aligned_split_files_and_manifest(tmp_path):
    f = pd.DataFrame(dict(composition=[f"Fe{i} O{201 - i}" for i in range(1, 201)], split=["train"] * 200))
    for j, name in enumerate(SOURCE + TARGET):
        f[name] = np.arange(200, dtype=float) + j
    path = tmp_path / "input.parquet"
    f.to_parquet(path, index=False)
    out = tmp_path / "prepared"
    m = prepare(path, out)
    assert m["rows"] == 200 and len(m["files"]) == 4 and m["input_sha256"] == digest(path)
    assert json.loads((out / "manifest.json").read_text()) == m
    for name, checksum in m["files"].items():
        assert digest(out / name) == checksum
    features = np.load(out / f"descriptors_{DATE}.npz")
    assert features["composition"].shape == (200, 94) and features["kmd"].shape == (200, 464)
    assert np.allclose(features["composition"].sum(1), 1)
    for file in out.glob("matrix*.parquet"):
        frame = pd.read_parquet(file)
        assert frame.composition.tolist() == f.composition.tolist() and frame.identity.is_unique
    f[SOURCE[-1]] = np.nan
    f.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="coverage"):
        prepare(path, tmp_path / "bad")


def test_malformed_split_and_missing_column_fail(tmp_path):
    with pytest.raises(ValueError, match="split"):
        isolate(pd.DataFrame(dict(composition=["NaCl"], split=["unknown"])))
    path = tmp_path / "missing.parquet"
    pd.DataFrame(dict(composition=["NaCl"])).to_parquet(path, index=False)
    with pytest.raises((KeyError, ValueError)):
        prepare(path, tmp_path / "out")
