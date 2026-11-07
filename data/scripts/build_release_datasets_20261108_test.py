"""The release folder drops unresolved MP ids everywhere and names the binary classes."""

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "build_release_datasets", Path(__file__).with_name("build_release_datasets_20261108.py")
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source_folder(tmp_path: Path) -> Path:
    src = tmp_path / "fm_task_datasets_20261016"
    src.mkdir()
    pd.DataFrame(
        {
            "id": ["mp-1", "mp-2", "mp-3"],
            "MP id": ["mp-1", None, "mp-3"],
            "composition": ["NaCl", "Fe2O3", "Si"],
            "split": ["train", "test", "val"],
            "Is metal": [False, True, np.nan],
            "Is metal (label)": [0.0, 1.0, np.nan],
            "Is gap direct": [True, False, True],
            "Is gap direct (label)": [1.0, 0.0, 1.0],
        }
    ).to_parquet(src / "mp_properties.parquet")
    pd.DataFrame(
        {"id": ["mp-2", "qa-9"], "composition": ["Fe2O3", "AlCu"], "split": ["test", "train"], "y": [1.0, 2.0]}
    ).to_parquet(src / "other.parquet")
    pd.DataFrame({"composition_group": ["a"], "split": ["train"]}).to_csv(
        src / "split_manifest_20261016.csv", index=False
    )
    (src / "provenance_20261016.json").write_text('{"date": "20261016"}')
    pd.DataFrame(
        [
            dict(task="is_metal", file="mp_properties.parquet", column="Is metal (label)", n_total=2),
            dict(task="y", file="other.parquet", column="y", n_total=2),
        ]
    ).to_csv(src / "task_index.csv", index=False)
    return src


def test_unresolved_ids_dropped_from_every_table(tmp_path):
    src = source_folder(tmp_path)
    out = tmp_path / "release"
    result = module.build(src, out)
    assert result["dropped_rows"] == {"mp_properties.parquet": 1, "other.parquet": 1}
    assert "mp-2" not in set(pd.read_parquet(out / "mp_properties.parquet").id)
    assert set(pd.read_parquet(out / "other.parquet").id) == {"qa-9"}
    index = pd.read_csv(out / "task_index.csv").set_index("task")
    assert index.loc["is_metal", "n_total"] == 1 and index.loc["y", "n_total"] == 1
    assert (out / "split_manifest_20261016.csv").read_bytes() == (src / "split_manifest_20261016.csv").read_bytes()
    assert not out.with_name(out.name + ".partial").exists()


def test_label_encoders_reproduce_stored_codes(tmp_path):
    src = source_folder(tmp_path)
    out = tmp_path / "release"
    module.build(src, out)
    enc = joblib.load(out / "label_encoders_20261108.pkl.z")
    assert [str(c) for c in enc["is_metal_label_encoder"].classes_] == ["False", "True"]
    assert sorted(enc) == ["is_gap_direct_label_encoder", "is_metal_label_encoder"]


def test_inconsistent_labels_rejected(tmp_path):
    src = source_folder(tmp_path)
    mp = pd.read_parquet(src / "mp_properties.parquet")
    mp["Is gap direct (label)"] = [0.0, 1.0, 0.0]
    mp.to_parquet(src / "mp_properties.parquet")
    with pytest.raises(ValueError, match="is_gap_direct"):
        module.build(src, tmp_path / "release")


def test_existing_output_refused(tmp_path):
    src = source_folder(tmp_path)
    (tmp_path / "release").mkdir()
    with pytest.raises(FileExistsError):
        module.build(src, tmp_path / "release")
