#!/usr/bin/env python3
"""Collect the warm-restart control results into results/warm_restart.csv.

The control (jobs/warm_restart_control.sbatch) trains each task alone on its FIXED full training
data: it00 = single-task pretrain (early stopping), it01..it10 = chained full-model retrains with
fresh optimizers. This file records the per-iteration test primary metric; the analysis uses the
best over all iterations as the task's true single-task baseline.

Usage: python collect_warm_restart.py [MIRROR_DIR]   (default: <repo>/artifacts/warm_restart)
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIRROR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "artifacts/warm_restart"
OUT = HERE.parent / "results/warm_restart.csv"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


rows = []
for task_dir in sorted(p for p in MIRROR.iterdir() if p.is_dir()):
    task = task_dir.name
    it0 = task_dir / "it00/training/metrics_table.csv"
    if not it0.exists():
        print(f"{task}: it00 missing — skipped")
        continue
    mt = list(csv.DictReader(open(it0)))
    mx = max(int(r["step"]) for r in mt)
    rows.append({"task": task, "iteration": 0, "primary": fnum(next(r["primary"] for r in mt if int(r["step"]) == mx and r["task"] == task))})
    for i in range(1, 11):
        p = task_dir / f"it{i:02d}/training/finetune_summary.json"
        if p.exists():
            rows.append({"task": task, "iteration": i, "primary": fnum(json.load(open(p))["metrics_after"][task]["primary"])})

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["task", "iteration", "primary"])
    w.writeheader()
    w.writerows(rows)
n_tasks = len({r["task"] for r in rows})
print(f"wrote {OUT}: {len(rows)} rows across {n_tasks} tasks")
