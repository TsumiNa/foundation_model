"""Collect complete scientific lanes and paired seed-level transfer contrasts."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: np.ndarray) -> tuple[float, float, float]:
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite paired effects required")
    means = np.random.default_rng(20261003).choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def collect(root: Path, config: Path, selection: Path, output: Path) -> dict:
    raw = tomllib.loads(config.read_text())
    chosen = json.loads(selection.read_text())["learning_rates"]
    expected = [
        dict(arm=a, seed=s, condition=c, lr=chosen[a])
        for a in raw["arms"]
        for s in raw["study"]["seeds"]
        for c in ["random", "real1", "real3", "real7", "shuffled7"]
    ]
    rows = []
    missing = []
    common = None
    for i, case in enumerate(expected):
        lane = root / f"case{i:03d}"
        if not (lane / "done.json").exists():
            missing.append(i)
            continue
        identity = json.loads((lane / "identity.json").read_text())
        done = json.loads((lane / "done.json").read_text())
        if (
            identity["case"] != case
            or done["case"] != case
            or identity["phase"] != "formal"
            or identity["smoke"]
            or identity["cpu_test"]
            or identity["config"] != file_hash(config)
            or identity["selection"] != file_hash(selection)
            or identity["manifest"] != raw["study"]["data_manifest_sha256"]
        ):
            raise ValueError(f"Invalid scientific identity: case {i}")
        provenance = tuple(identity[k] for k in ["revision", "script", "preparation_script", "manifest", "package"])
        if common is None:
            common = provenance
        if common != provenance:
            raise ValueError("Mixed provenance")
        metrics = done["metrics"]
        keys = {(m["target"], m["fraction"], m["mode"]) for m in metrics}
        endpoints = {
            (t, f, m)
            for t in ["Dielectric total", "Bulk modulus", "Shear modulus", "Piezoelectric max"]
            for f in [0.1, 1.0]
            for m in ["ridge", "frozen", "full"]
        }
        if len(metrics) != 24 or keys != endpoints:
            raise ValueError("Incomplete endpoint matrix")
        for m in metrics:
            if not np.isfinite([m[k] for k in ["rmse", "mae", "standardized_rmse", "val_mse"]]).all():
                raise ValueError("Nonfinite metrics")
            rows.append({**case, **m})
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "complete_lanes": len(expected) - len(missing),
        "expected_lanes": len(expected),
        "metrics": len(rows),
        "expected_metrics": 4800,
        "missing_cases": missing,
        "partial": bool(missing),
        "common_provenance": common,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    if not rows:
        return audit
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    summary = frame.groupby(["arm", "condition", "target", "fraction", "mode"]).rmse.agg(["mean", "std", "count"])
    summary.to_csv(output / "summary.csv")
    contrasts = []
    # All comparisons use paired optimization seeds; compositions are not replicates.
    for key, g in frame.groupby(["arm", "target", "fraction", "mode"]):
        wide = g.pivot(index="seed", columns="condition", values="standardized_rmse")
        for a, b in [("real7", "random"), ("real7", "real1"), ("real7", "real3"), ("real7", "shuffled7")]:
            if a not in wide or b not in wide:
                continue
            paired = wide[[a, b]].dropna()
            v = (paired[a] - paired[b]).to_numpy()
            if not len(v):
                continue
            mean, lo, hi = interval(v)
            contrasts.append(
                dict(zip(["arm", "target", "fraction", "mode"], key, strict=True))
                | dict(
                    contrast=f"{a} minus {b}",
                    mean=mean,
                    lo95=lo,
                    hi95=hi,
                    n_seeds=len(v),
                    partial=len(v) != len(raw["study"]["seeds"]),
                )
            )
    pd.DataFrame(contrasts).to_csv(output / "paired_contrasts.csv", index=False)
    interactions = []
    for key, g in frame.groupby(["target", "fraction", "mode"]):
        wide = g.pivot(index="seed", columns=["arm", "condition"], values="standardized_rmse")
        for baseline in ["mlp", "mlp_large", "no_attention"]:
            columns = [("transformer", "real7"), ("transformer", "random"), (baseline, "real7"), (baseline, "random")]
            if not all(c in wide for c in columns):
                continue
            q = wide[columns].dropna()
            v = (q[columns[0]] - q[columns[1]] - q[columns[2]] + q[columns[3]]).to_numpy()
            if not len(v):
                continue
            mean, lo, hi = interval(v)
            interactions.append(
                dict(zip(["target", "fraction", "mode"], key, strict=True))
                | dict(
                    baseline=baseline,
                    mean=mean,
                    lo95=lo,
                    hi95=hi,
                    n_seeds=len(v),
                    partial=len(v) != len(raw["study"]["seeds"]),
                )
            )
    pd.DataFrame(interactions).to_csv(output / "encoder_pretraining_interactions.csv", index=False)
    (output / "README.md").write_text(
        "Negative contrasts mean lower error for real7. Units are target-training-standard-deviation-normalized RMSE differences, not percentages. Negative interaction means the Transformer gains more from pretraining than the comparison encoder. Bootstrap intervals resample paired seeds, are pointwise/exploratory, and do not adjust for multiple endpoints. Partial rows are not final evidence. Always inspect original-unit metrics and MLP scratch (random/full) alongside frozen random-feature comparisons.\n"
    )
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["root", "config", "selection", "output"]:
        p.add_argument(f"--{k}", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(collect(a.root, a.config, a.selection, a.output), indent=2))
