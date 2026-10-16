"""Data-level acceptance: isolate aliases while preserving labels and old files."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "repair_task_datasets", Path(__file__).with_name("repair_task_datasets_20261016.py")
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source_folder(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        {
            "composition": ["Fe2O3", "Fe4O6", "AmO2", "NaCl"],
            "split": ["train", "test", "train", "val"],
            "y": [1.0, 9.0, 7.0, np.nan],
            "curve": [np.array([1.0, 2.0]), np.array([4.0, 5.0]), None, None],
        }
    ).to_parquet(source / "tasks.parquet")
    pd.DataFrame({"composition": ["AmO2"], "raw_y": [7.0]}).to_parquet(source / "tasks_raw.parquet")
    pd.DataFrame(
        [
            dict(task="y", kind="regression", file="tasks.parquet", column="y", raw_file="tasks_raw.parquet"),
            dict(task="curve", kind="curve", file="tasks.parquet", column="curve", raw_file="tasks_raw.parquet"),
        ]
    ).to_csv(source / "task_index.csv", index=False)
    return source


def test_dated_data_preserves_records_and_labels(tmp_path):
    source = source_folder(tmp_path)
    old_hashes = {p.name: module.sha256(p) for p in source.iterdir()}
    destination = tmp_path / "new_20261016"
    result = module.repair_dataset_folder(source, destination)
    assert result["retained_records"] == 3 and result["rejected_records"] == 1
    assert result["cross_split_groups"] == 0
    out = pd.read_parquet(destination / "tasks.parquet")
    assert out.composition.tolist() == ["Fe2O3", "Fe4O6", "NaCl"]
    assert out.split.tolist() == ["test", "test", "val"]
    assert out.y.iloc[:2].tolist() == [1.0, 9.0]
    np.testing.assert_array_equal(out.curve.iloc[1], [4.0, 5.0])
    index = pd.read_csv(destination / "task_index.csv")
    assert index.n_total.tolist() == [2, 2] and index.n_train.tolist() == [0, 0]
    manifest = pd.read_csv(destination / "split_manifest_20261016.csv")
    assert len(manifest) == 2 and manifest.composition_group.is_unique
    assert module.sha256(destination / "tasks_raw.parquet") == old_hashes["tasks_raw.parquet"]
    assert old_hashes == {p.name: module.sha256(p) for p in source.iterdir()}
    with pytest.raises(FileExistsError):
        module.repair_dataset_folder(source, destination)


def test_invalid_split_fails_before_creating_output(tmp_path):
    source = source_folder(tmp_path)
    frame = pd.read_parquet(source / "tasks.parquet")
    frame.loc[0, "split"] = "unknown"
    frame.to_parquet(source / "tasks.parquet")
    with pytest.raises(ValueError, match="invalid"):
        module.repair_dataset_folder(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_all_invalid_inputs_fail_without_partial_output(tmp_path):
    source = source_folder(tmp_path)
    f = pd.read_parquet(source / "tasks.parquet")
    f["composition"] = "AmO2"
    f.to_parquet(source / "tasks.parquet")
    with pytest.raises(ValueError, match="No supported"):
        module.repair_dataset_folder(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
