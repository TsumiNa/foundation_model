#!/usr/bin/env python3
"""Warm-restart control curves: retraining with a fresh optimizer does not improve performance.

Reads results/warm_restart.csv (24 tasks × iterations it00..it10; it00 = the single-task
baseline training, it01+ = chained full-model retrainings with fresh optimizers on unchanged
data). Left: absolute primary metric per task. Right: change relative to it00 — the flat lines
are the point. Emits warm_restart_flat.png.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results/warm_restart.csv"
MUT, GRID, ACC = "#6b7280", "#e5e7eb", "#0077BB"

by_task: dict[str, dict[int, float]] = defaultdict(dict)
for r in csv.DictReader(open(RES)):
    by_task[r["task"]][int(r["iteration"])] = float(r["primary"])

# magnetic_susceptibility (58 labels) is degenerate: it00 overfits to R^2 = -5.2 and every retrain
# lands at ~0 (a trivial mean predictor). That jump is not a restart gain — excluded with a note.
DEGENERATE = {"magnetic_susceptibility"}
excluded = {k: by_task.pop(k) for k in list(by_task) if k in DEGENERATE}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
deltas_last = []
for task, vals in sorted(by_task.items()):
    its = sorted(vals)
    y = [vals[i] for i in its]
    axes[0].plot(its, y, color="#9ca3af", lw=0.9, alpha=0.6)
    axes[1].plot(its, [v - y[0] for v in y], color="#9ca3af", lw=0.9, alpha=0.6)
    deltas_last.append(vals[max(its)] - y[0])

# mean delta line
all_its = sorted({i for v in by_task.values() for i in v})
mean_delta = [float(np.mean([v[i] - v[0] for v in by_task.values() if i in v and 0 in v])) for i in all_its]
axes[1].plot(all_its, mean_delta, color=ACC, lw=2.4)
axes[1].axhline(0, color="#444", lw=1.0, ls="--")

axes[0].set_ylabel("primary metric (R² / accuracy)", fontsize=9.5)
axes[0].set_title("absolute — one line per task", fontsize=10)
axes[1].set_ylabel("change vs it00 (the initial training)", fontsize=9.5)
axes[1].set_title("relative — flat ⇒ restarts add nothing", fontsize=10)
axes[1].set_ylim(-0.25, 0.25)
for ax in axes:
    ax.set_xlabel("retraining iteration (fresh optimizer each time, data unchanged)", fontsize=9.5)
    ax.set_xticks(all_its)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUT, labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.suptitle("Warm-restart control: 10 chained fresh-optimizer retrainings per task — no improvement", fontsize=12, y=1.02)
fig.text(0.01, -0.04,
         "excluded as degenerate: magnetic_susceptibility (58 labels) — single-task it00 overfits to R²=−5.2, every retrain lands at ≈0 (trivial mean predictor); not a restart gain",
         fontsize=8, color="#6b7280")
fig.legend(
    handles=[
        Line2D([], [], color="#9ca3af", lw=1, label="individual tasks (23; degenerate case excluded)"),
        Line2D([], [], color=ACC, lw=2.4, label="mean change"),
    ],
    loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.97),
)
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = HERE / "warm_restart_flat.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}; mean final delta = {np.mean(deltas_last):+.4f}, max |delta| = {max(abs(d) for d in deltas_last):.3f}")
