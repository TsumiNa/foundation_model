import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from prepare import DATE, SOURCE, TARGET, prepare
from foundation_model.utils.kmd_plus import KMD, element_features, formula_to_composition


def fixture_frame():
    frame = pd.DataFrame(
        {
            "composition": [f"Fe{i + 1} O1" for i in range(36)],
            "split": ["train"] * 12 + ["val"] * 12 + ["test"] * 12,
        }
    )
    for j, name in enumerate(SOURCE + TARGET):
        frame[name] = np.arange(36, dtype=float) + j
    return frame


def test_prepare_alignment_masks_subsets_and_manifest(tmp_path):
    frame = fixture_frame()
    frame.loc[0, TARGET[0]] = np.inf
    frame.loc[1, SOURCE[0]] = np.nan
    input_path = tmp_path / "input.parquet"
    frame.to_parquet(input_path, index=False)
    output = tmp_path / "prepared"
    manifest = prepare(input_path, output)
    saved = pd.read_parquet(output / f"matrix_{DATE}.parquet")
    with np.load(output / f"descriptors_{DATE}.npz") as data:
        x = data["x"]
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert manifest["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    for name, checksum in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    assert manifest["rows"] == 36 and manifest["input_dim"] == 464
    assert saved.composition.tolist() == frame.composition.tolist()
    expected = KMD(element_features.to_numpy(), n_grids=8).transform(
        np.stack([formula_to_composition(c) for c in saved.composition])
    )
    np.testing.assert_allclose(x, expected, rtol=1e-6, atol=1e-7)
    assert np.isfinite(x).all() and x.dtype == np.float32
    assert np.isnan(saved.loc[0, TARGET[0]]) and np.isnan(saved.loc[1, SOURCE[0]])
    assert manifest["counts"][TARGET[0]] == {"train": 11, "val": 12, "test": 12}
    for j, name in enumerate(TARGET):
        subset = saved[f"target{j}_low_train"]
        eligible = saved.split.eq("train") & saved[name].notna()
        assert subset.sum() == int(np.ceil(eligible.sum() * 0.1))
        assert not (subset & ~eligible).any()
    second = tmp_path / "second"
    prepare(input_path, second)
    pd.testing.assert_frame_equal(saved, pd.read_parquet(second / f"matrix_{DATE}.parquet"))


@pytest.mark.parametrize("failure", ["split", "coverage"])
def test_prepare_rejects_invalid_partition_before_writing(tmp_path, failure):
    frame = fixture_frame()
    if failure == "split":
        frame.loc[0, "split"] = "unknown"
        message = "Explicit train/val/test"
    else:
        frame.loc[frame.split.eq("val"), TARGET[0]] = np.nan
        message = "insufficient partition coverage"
    input_path = tmp_path / "input.parquet"
    frame.to_parquet(input_path, index=False)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match=message):
        prepare(input_path, output)
    assert not output.exists()
