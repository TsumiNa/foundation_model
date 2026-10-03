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
            "revision": "a" * 40,
            "script": digest(Path(__file__).with_name("study.py")),
            "preparation_script": digest(Path(__file__).with_name("prepare.py")),
            "package": raw["study"]["package_version"],
            "selection": None,
        }
        (lane / "identity.json").write_text(json.dumps(identity))
        runtime = dict(
            device="cuda",
            sif_sha256=raw["study"]["sif_sha256"],
            image_revision=raw["study"]["image_revision"],
            package_path="/opt/lib/site-packages/foundation_model/__init__.py",
        )
        (lane / "run_provenance.json").write_text(json.dumps(dict(identity=identity, runtime=runtime)))
        metrics = [
            {
                "target": t,
                "fraction": 1.0,
                "mode": "ridge",
                "val_mse": 1.0 + raw["arms"][case["arm"]]["learning_rates"].index(case["lr"]),
            }
            for t in TARGET
        ]
        (lane / "done.json").write_text(
            json.dumps({"case": case, "metrics": metrics, "source": {"steps": raw["study"]["pilot_steps"]}})
        )
    result = select(tmp_path, config)
    assert result["learning_rates"] == {a: v["learning_rates"][0] for a, v in raw["arms"].items()}
    p = tmp_path / "case000/identity.json"
    original = p.read_text()
    wrong = json.loads(original)
    wrong["script"] = "wrong"
    p.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="identity"):
        select(tmp_path, config)
    p.write_text(original)
    p = tmp_path / "case000/run_provenance.json"
    original = p.read_text()
    wrong = json.loads(original)
    wrong["runtime"]["sif_sha256"] = "wrong"
    p.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="identity"):
        select(tmp_path, config)
    p.write_text(original)
    (tmp_path / "case023" / "done.json").unlink()
    with pytest.raises(FileNotFoundError):
        select(tmp_path, config)
