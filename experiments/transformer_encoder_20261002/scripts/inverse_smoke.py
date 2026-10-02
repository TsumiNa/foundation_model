"""Run both inverse workflows on a retained scalar-target smoke checkpoint."""

import argparse
import os
from pathlib import Path
import subprocess

import torch

from foundation_model.workflows.inverse import build_inverse_config

from benchmark import workflow_config, write_toml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="grouped_concat")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if not args.cpu_smoke:
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("GPU inverse smoke requires a Slurm allocation with exactly one visible CUDA GPU")
    raw = workflow_config(
        args.protocol.resolve(),
        args.data_dir.resolve(),
        args.arm,
        0,
        target="dielectric_total",
        fraction=1,
        mode="full",
    )
    raw.pop("training")
    raw["inverse"] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "steps": 2,
        "lr": 0.01,
        "animation_formats": [],
        "record_trajectory": True,
        "accelerator": "cpu" if args.cpu_smoke else "auto",
        "seeds": {"strategy": "random", "n": 2, "split": "train"},
        "scenarios": [{"name": "dielectric_high", "targets": [{"task": "dielectric_total", "direction": "high"}]}],
        "paths": [
            {"name": "latent", "method": "latent"},
            {"name": "composition", "method": "composition", "init": "seed"},
        ],
    }
    raw["output"] = {"dir": str(args.output.resolve())}
    build_inverse_config(raw)
    config_path = args.output / "inverse.toml"
    write_toml(raw, config_path)
    subprocess.run(["fm", "inverse", "--config", str(config_path)], check=True)
