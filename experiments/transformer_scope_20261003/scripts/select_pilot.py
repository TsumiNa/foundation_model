"""Require the full registered pilot and select source LRs using target validation only."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import tomllib
import numpy as np
from study import cases, atomic
from prepare import digest, TARGET


def select(root: Path, config: Path) -> dict:
    raw = tomllib.loads(config.read_text())
    expected = cases(raw, "pilot")
    scores = {}
    manifest = None
    revision = None
    for i, case in enumerate(expected):
        lane = root / f"case{i:03d}"
        identity = json.loads((lane / "identity.json").read_text())
        result = json.loads((lane / "done.json").read_text())
        if (
            identity["phase"] != "pilot"
            or identity["case"] != case
            or identity["config"] != digest(config)
            or identity["manifest"] != raw["study"]["data_manifest_sha256"]
            or result["case"] != case
            or identity["smoke"]
            or identity["cpu_test"]
        ):
            raise ValueError("Invalid pilot identity")
        if manifest is None:
            manifest = identity["manifest"]
            revision = identity["revision"]
        if identity["manifest"] != manifest or identity["revision"] != revision:
            raise ValueError("Mixed pilot provenance")
        metrics = result["metrics"]
        if (
            len(metrics) != len(TARGET)
            or {m["target"] for m in metrics} != set(TARGET)
            or any(m["mode"] != "ridge" or m["fraction"] != 1.0 for m in metrics)
        ):
            raise ValueError("Incomplete pilot endpoint matrix")
        weights = {TARGET[0]: 1 / 3, TARGET[1]: 1 / 6, TARGET[2]: 1 / 6, TARGET[3]: 1 / 3}
        score = float(sum(weights[m["target"]] * m["val_mse"] for m in metrics))
        if not np.isfinite(score):
            raise ValueError("Nonfinite pilot score")
        scores.setdefault(case["arm"], {}).setdefault(case["lr"], []).append(score)
    selected = {arm: min(values, key=lambda lr: float(np.mean(values[lr]))) for arm, values in scores.items()}
    return dict(
        learning_rates=selected,
        criterion="Equal-family target-validation standardized MSE of full-training frozen ridge; two pilot seeds on first split; no test labels",
        scores=scores,
        config_sha256=digest(config),
        manifest_sha256=manifest,
        pilot_revision=revision,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["root", "config", "output"]:
        p.add_argument(f"--{k}", type=Path, required=True)
    a = p.parse_args()
    r = select(a.root, a.config)
    atomic(a.output, r)
    print(json.dumps(r, indent=2))
