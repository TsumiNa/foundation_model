import json
from pathlib import Path
import tomllib
import pytest

from prepare import TARGET, digest
from select_pilot import select
from study import cases


def test_selection_uses_complete_validation_only_matrix(tmp_path):
    config = Path(__file__).parents[1] / "configs/study.toml"
    raw = tomllib.loads(config.read_text())
    for i, case in enumerate(cases(raw, "pilot")):
        lane = tmp_path / f"case{i:03d}"
        lane.mkdir()
        identity = {
            "phase": "pilot",
            "case": case,
            "config": digest(config),
            "smoke": False,
            "cpu_test": False,
            "manifest": raw["study"]["data_manifest_sha256"],
            "revision": "revision",
        }
        (lane / "identity.json").write_text(json.dumps(identity))
        metrics = [
            {
                "target": t,
                "fraction": 1.0,
                "mode": "ridge",
                "val_mse": 1.0 + raw["arms"][case["arm"]]["learning_rates"].index(case["lr"]),
            }
            for t in TARGET
        ]
        (lane / "done.json").write_text(json.dumps({"case": case, "metrics": metrics}))
    result = select(tmp_path, config)
    assert result["learning_rates"] == {a: v["learning_rates"][0] for a, v in raw["arms"].items()}
    (tmp_path / "case023" / "done.json").unlink()
    with pytest.raises(FileNotFoundError):
        select(tmp_path, config)
