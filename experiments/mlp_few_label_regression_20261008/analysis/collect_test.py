"""Collector refuses smoke/mixed holdouts and reports incomplete seed coverage."""

import json

import pytest

from collect import collect


def test_paired_collection_and_mismatch(tmp_path):
    c = {"case": 0, "task": "x", "n": 10, "seed": 0, "checkpoint": "source.pt"}
    identity = {"smoke": False, "manifest": "m"}
    lane = tmp_path / "root" / "case000"
    lane.mkdir(parents=True)
    (lane / "done.json").write_text(json.dumps({"case": c, "identity": identity}))
    for arm in ("scratch", "transfer"):
        (lane / arm).mkdir()
        p = {
            "case": c,
            "identity": identity,
            "arm": arm,
            "seconds": 1,
            "metrics": {"r2": 0.4, "mae": 0.2, "rmse": 0.3, "test_hash": "h"},
        }
        (lane / arm / "done.json").write_text(json.dumps(p))
    s = collect(tmp_path / "root", tmp_path / "out")
    assert s["partial"] and s["complete_pairs"] == 1 and s["points"][0]["seed_count"] == 1
    p["metrics"]["test_hash"] = "other"
    (lane / "transfer" / "done.json").write_text(json.dumps(p))
    with pytest.raises(ValueError, match="Unpaired"):
        collect(tmp_path / "root", tmp_path / "out")
