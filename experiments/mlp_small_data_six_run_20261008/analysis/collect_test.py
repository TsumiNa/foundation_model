"""Two pinned campaigns, no duplicate fits, mean-level R² usability filter."""

import importlib.util
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("six_collect", Path(__file__).with_name("collect.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_usability_applies_to_both_means_not_individual_runs():
    rows = []
    for task, a, b in [
        ("usable", [0.1, 0.3, 0.5], [0.3, 0.4, 0.5]),
        ("bad_scratch", [0, 0.1, 0.2], [0.5, 0.6, 0.7]),
        ("bad_transfer", [0.5] * 3, [0.1] * 3),
        ("boundary", [0.2] * 3, [0.2] * 3),
    ]:
        for arm, values in [("scratch", a), ("transfer", b)]:
            rows += [dict(task=task, n=10, arm=arm, r2=v) for v in values]
    result = module.usability_counts(pd.DataFrame(rows))[0]
    assert result == {"n": 10, "evaluated": 4, "retained": 2, "excluded": 2, "better": 1, "similar": 1, "worse": 0}


def case_fixture(tmp_path, monkeypatch):
    old = {"case": 0, "task": "x", "n": 10, "seed": 0, "checkpoint": "a.pt"}
    new = {"case": 0, "task": "x", "n": 10, "seed": 1, "checkpoint": "b.pt"}
    identity = {
        "manifest": "old",
        "revision": "630e973dfa51bde36c0ce8dbaee34919ad9f73ed",
        "image_hash": "f90adc81f1148db2fff0ae94551b751e77f82c5d52d032ea54641f746dbe2f67",
        "smoke": False,
        "script_sha256": {"run.py": "r", "array.sbatch": "a"},
    }
    inputs = {
        "recipes": {},
        "auxiliary": [],
        "checkpoints": [
            {"file": "a.pt", "sha256": "a"},
            {"file": "b.pt", "sha256": "b"},
            {"file": "c.pt", "sha256": "c"},
        ],
        "script_sha256": identity["script_sha256"],
    }
    inputs.update(cases=[old], subsets=[], source_hashes={})
    old_hash = hashlib.sha256(json.dumps(inputs, indent=2).encode()).hexdigest()
    monkeypatch.setattr(module, "REUSE_MANIFEST_SHA256", old_hash)
    identity["manifest"] = old_hash
    manifest = {
        **inputs,
        "reuse_input_manifest": inputs,
        "reuse_identity": identity,
        "reuse_cases": [old],
        "cases": [new],
        "all_cases": [old, new],
        "paired_cases": 2,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    root = tmp_path / "old" / "case000"
    root.mkdir(parents=True)
    (root / "done.json").write_text(json.dumps({"case": old, "identity": identity}))
    for arm in ["scratch", "transfer"]:
        (root / arm).mkdir()
        record = {
            "case": old,
            "identity": identity,
            "arm": arm,
            "seconds": 1,
            "head_initialization": {"sha256": "head"},
            "checkpoint_hash": "a",
            "metrics": {"r2": 0.4, "mae": 0.2, "rmse": 0.3, "steps": 3, "test_hash": "test"},
        }
        (root / arm / "done.json").write_text(json.dumps(record))
    return path, root


def test_partial_reuse_and_reject_mismatch(tmp_path, monkeypatch):
    path, root = case_fixture(tmp_path, monkeypatch)
    summary = module.collect(tmp_path / "new", tmp_path / "old", path, "n" * 40, tmp_path / "out")
    assert summary["partial"] and summary["complete_pairs"] == 1 and summary["points"][0]["r2_std"] is None
    record = json.loads((root / "transfer/done.json").read_text())
    record["head_initialization"]["sha256"] = "changed"
    (root / "transfer/done.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="head"):
        module.collect(tmp_path / "new", tmp_path / "old", path, "n" * 40, tmp_path / "out")


@pytest.mark.parametrize("failure", ["smoke", "case", "checkpoint", "steps", "test"])
def test_invalid_reuse_records(tmp_path, monkeypatch, failure):
    path, root = case_fixture(tmp_path, monkeypatch)
    if failure in ["smoke", "case"]:
        marker = json.loads((root / "done.json").read_text())
        if failure == "smoke":
            marker["identity"]["smoke"] = True
        else:
            marker["case"]["n"] = 20
        (root / "done.json").write_text(json.dumps(marker))
    else:
        record = json.loads((root / "transfer/done.json").read_text())
        if failure == "checkpoint":
            record["checkpoint_hash"] = "wrong"
        elif failure == "steps":
            record["metrics"]["steps"] = 0
        else:
            record["metrics"]["test_hash"] = "wrong"
        (root / "transfer/done.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        module.collect(tmp_path / "new", tmp_path / "old", path, "n" * 40, tmp_path / "out")
