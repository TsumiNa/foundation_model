#!/usr/bin/env python3
"""Collect a stage's grid runs into one tidy CSV.

Each run directory holds `training/metrics_table.csv` (one row per evaluated task per step) and
`run_provenance.json` (the fully resolved config). This flattens the LAST step of every run into
one row per (runid, task), joined with the knobs that run actually used — so the ranking table
never depends on parsing the runid.

    python .../collect.py <outroot> -o <results>/stage_a.csv

`<outroot>` is the directory the array job wrote into (one subdirectory per runid). Runs without
a DONE marker are reported on stderr and skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Resolved-config knobs worth carrying into the ranking table.
MODEL_KEYS = [
    "latent_dim",
    "encoder_hidden_dims",
    "head_hidden_dims",
    "kr_x_hidden_dims",
    "kr_t_hidden_dims",
    "n_kernel",
]
TRAINING_KEYS = [
    "encoder_lr",
    "head_lr",
    "kr_lr",
    "kr_weight_decay",
    "ae_lr",
    "max_epochs",
    "seed",
]
METRIC_KEYS = ["r2", "mae", "accuracy", "macro_f1", "primary", "samples", "points", "epochs_run"]


def run_rows(run_dir: Path) -> list[dict]:
    table = run_dir / "training" / "metrics_table.csv"
    prov = run_dir / "run_provenance.json"
    if not table.exists():
        print(f"skip {run_dir.name}: no metrics_table.csv", file=sys.stderr)
        return []

    rows = list(csv.DictReader(open(table)))
    if not rows:
        print(f"skip {run_dir.name}: empty metrics_table.csv", file=sys.stderr)
        return []
    last = max(int(r["step"]) for r in rows)

    knobs: dict[str, object] = {}
    if prov.exists():
        cfg = json.load(open(prov)).get("resolved_config", {})
        model, training = cfg.get("model", {}), cfg.get("training", {})
        catalog = cfg.get("catalog", {})
        knobs = {f"model.{k}": model.get(k) for k in MODEL_KEYS}
        knobs |= {f"training.{k}": training.get(k) for k in TRAINING_KEYS}
        knobs["data.batch_size"] = (catalog.get("data") or {}).get("batch_size")
        knobs["descriptor.n_grids"] = (catalog.get("descriptor") or {}).get("n_grids")
        knobs["task_sequence"] = ",".join(cfg.get("task_sequence") or [])

    out = []
    for r in rows:
        if int(r["step"]) != last:
            continue
        row = {"runid": run_dir.name, "step": last, "new_task": r.get("new_task"), "task": r["task"]}
        row |= {k: r.get(k, "") for k in METRIC_KEYS}
        row |= knobs
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outroot", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--prefix", default="", help="only collect runids starting with this")
    ap.add_argument("--allow-incomplete", action="store_true", help="also collect runs without DONE")
    args = ap.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(p for p in args.outroot.iterdir() if p.is_dir()):
        if args.prefix and not run_dir.name.startswith(args.prefix):
            continue
        if not (run_dir / "DONE").exists() and not args.allow_incomplete:
            print(f"skip {run_dir.name}: no DONE marker", file=sys.stderr)
            continue
        rows += run_rows(run_dir)

    if not rows:
        raise SystemExit("no rows collected")

    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.output}  ({len(rows)} rows, {len({r['runid'] for r in rows})} runs)")


if __name__ == "__main__":
    main()
