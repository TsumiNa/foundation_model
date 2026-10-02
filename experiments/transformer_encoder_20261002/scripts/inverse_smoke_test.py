import json
from pathlib import Path
import runpy
import subprocess
import sys
import tomllib

import pytest
import torch

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
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: not cpu)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0 if cpu else 1)
    monkeypatch.setattr(sys, "argv", args)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    runpy.run_path(str(script), run_name="__main__")
    config = tomllib.loads((output / "inverse.toml").read_text())
    assert config["inverse"]["accelerator"] == ("cpu" if cpu else "auto")
    assert {p["method"] for p in config["inverse"]["paths"]} == {"latent", "composition"}
    assert calls == [["fm", "inverse", "--config", str(output.resolve() / "inverse.toml")]]


@pytest.mark.parametrize("job_id,cuda,count", [(None, True, 1), ("12345", False, 0), ("12345", True, 2)])
def test_gpu_inverse_rejects_missing_allocation_or_cuda(tmp_path, monkeypatch, job_id, cuda, count):
    script = Path(__file__).with_name("inverse_smoke.py")
    if job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--protocol",
            str(tmp_path),
            "--data-dir",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(RuntimeError, match="Slurm allocation"):
        runpy.run_path(str(script), run_name="__main__")
    assert not (tmp_path / "out").exists()
