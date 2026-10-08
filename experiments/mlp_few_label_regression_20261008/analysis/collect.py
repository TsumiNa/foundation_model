"""Collect complete paired cases without mixing smoke or checkpoint cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect(root: Path, output: Path) -> dict:
    rows = []
    identities = []
    for marker in sorted(root.glob("case*/done.json")):
        case = json.loads(marker.read_text())
        if case["identity"]["smoke"]:
            raise ValueError("Smoke results are not scientific fits")
        identities.append(case["identity"])
        pair = [json.loads((marker.parent / a / "done.json").read_text()) for a in ("scratch", "transfer")]
        if any(p["identity"] != case["identity"] or p["case"] != case["case"] for p in pair):
            raise ValueError("Fit identity mismatch")
        if pair[0]["metrics"]["test_hash"] != pair[1]["metrics"]["test_hash"]:
            raise ValueError("Unpaired test labels")
        if pair[0]["head_initialization"] != pair[1]["head_initialization"]:
            raise ValueError("Unpaired target-head initialization")
        for p in pair:
            if not np.isfinite([p["metrics"][k] for k in ("r2", "mae", "rmse")]).all():
                raise ValueError("Nonfinite metrics")
            rows.append({**p["case"], "arm": p["arm"], **p["metrics"], "seconds": p["seconds"]})
    if any(i != identities[0] for i in identities):
        raise ValueError("Mixed campaign identities")
    frame = pd.DataFrame(rows)
    if len(frame) and frame.duplicated(["task", "n", "seed", "arm"]).any():
        raise ValueError("Duplicate fit identity")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics.csv", index=False)
    summary = {
        "complete_pairs": len(rows) // 2,
        "expected_pairs": 156,
        "partial": len(rows) != 312,
        "identity": identities[0] if identities else None,
        "points": [],
    }
    if len(frame):
        for (task, n, arm), g in frame.groupby(["task", "n", "arm"]):
            summary["points"].append(
                {
                    "task": task,
                    "n": int(n),
                    "arm": arm,
                    "seeds": g.seed.tolist(),
                    "seed_count": len(g),
                    **{f"{k}_mean": float(g[k].mean()) for k in ("r2", "mae", "rmse")},
                    **{f"{k}_std": float(g[k].std()) if len(g) > 1 else None for k in ("r2", "mae", "rmse")},
                }
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(collect(args.root, args.output), indent=2))
