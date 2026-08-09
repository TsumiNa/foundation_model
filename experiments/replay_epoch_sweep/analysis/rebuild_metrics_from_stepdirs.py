#!/usr/bin/env python3
"""Rebuild full metrics_table.csv files from per-step metric JSONs.

Why: ``fm pretrain --resume`` writes metrics_table.csv from the resuming process's in-memory
records only, so a run recovered from a walltime kill gets a PARTIAL table (only the resumed
steps). The per-step ``training/stepNN_<task>/<task>_metrics.json`` files are written as each
step completes and are authoritative. This script reassembles the complete long-format table
for the affected runs and overwrites the collected copy under results/.

Caveat: ``epochs_run`` is not recorded in the step JSONs and is left empty in rebuilt rows.
(Known upstream issue — the recorder should reload prior records on resume; needs a src/ PR.)
"""

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parents[1] / "results"

RUNS = {
    "artifacts/replay_sweep_epoch_p24/replay_n100_epoch_p24": "mt_n100_epoch_p24.csv",
    "artifacts/replay_sweep_epoch_p24/replay_n200_epoch_p24": "mt_n200_epoch_p24.csv",
    "artifacts/replay_sweep_epoch_p24/replay_n500_epoch_p24": "mt_n500_epoch_p24.csv",
    "artifacts/replay_sweep_epoch_m150/replay_n1500_epoch_m150": "mt_n1500_epoch_m150.csv",
    "artifacts/replay_sweep_epoch_m150/replay_n2000_epoch_m150": "mt_n2000_epoch_m150.csv",
    "artifacts/replay_sweep_epoch_m150/replay_n2500_epoch_m150": "mt_n2500_epoch_m150.csv",
    "artifacts/replay_sweep_epoch_m150/replay_0p10_epoch_m150": "mt_0p10_epoch_m150.csv",
    "artifacts/replay_sweep_epoch_m150/replay_0p20_epoch_m150": "mt_0p20_epoch_m150.csv",
}
FIELDS = ["step", "new_task", "epochs_run", "task", "r2", "mae", "samples", "primary",
          "points", "accuracy", "macro_f1"]

for run_rel, out_name in RUNS.items():
    tdir = REPO / run_rel / "training"
    rows = []
    for d in sorted(tdir.glob("step*_*")):
        m = re.match(r"step(\d+)_(.+)", d.name)
        step, new_task = int(m.group(1)), m.group(2)
        for jf in sorted(d.glob("*_metrics.json")):
            task = jf.name[: -len("_metrics.json")]
            metrics = json.load(open(jf))
            row = {"step": step, "new_task": new_task, "epochs_run": "", "task": task}
            row.update({k: metrics.get(k, "") for k in FIELDS[4:]})
            rows.append(row)
    rows.sort(key=lambda r: (r["step"], r["task"]))
    out = RESULTS / out_name
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    steps = {r["step"] for r in rows}
    print(f"{out_name}: {len(rows)} rows, steps {min(steps)}..{max(steps)}")
