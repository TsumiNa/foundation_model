import json
from pathlib import Path
import runpy
import subprocess
import sys
import tomllib

import pytest

from benchmark import DATE


@pytest.mark.parametrize("cpu", [False, True])
def test_inverse_cli_generates_supported_accelerator_config(tmp_path, monkeypatch, cpu):
    script = Path(__file__).with_name("inverse_smoke.py")
    protocol = script.parents[1] / "configs/protocol.toml"
    manifest = {
        "source": {"file": "source.parquet"},
        "targets": [
            {"task": "dielectric_total", "seed": 0, "fraction": 1, "file": "target.parquet", "scaler": "target.joblib"}
        ],
    }
    (tmp_path / f"manifest_{DATE}.json").write_text(json.dumps(manifest))
    output = tmp_path / "out"
    args = [
        str(script),
        "--protocol",
        str(protocol),
        "--data-dir",
        str(tmp_path),
        "--checkpoint",
        str(tmp_path / "model.pt"),
        "--output",
        str(output),
    ]
    if cpu:
        args.append("--cpu-smoke")
    monkeypatch.setattr(sys, "argv", args)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    runpy.run_path(str(script), run_name="__main__")
    config = tomllib.loads((output / "inverse.toml").read_text())
    assert config["inverse"]["accelerator"] == ("cpu" if cpu else "auto")
    assert {p["method"] for p in config["inverse"]["paths"]} == {"latent", "composition"}
    assert calls == [["fm", "inverse", "--config", str(output.resolve() / "inverse.toml")]]
