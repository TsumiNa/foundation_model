#!/usr/bin/env python3
"""Residual forgetting at the largest tested replay budget vs task size — the case for ratio replay.

Model-free reading: for every task and arm (step-p8 / epoch-p8 / epoch-p24), plot the残差 gap
(at-intro mean − final) at n = 2500 — the largest fixed count tested — against the task's own
train-set size. n = 2500 means FULL replay for every task below 2,500 labels but only ~11% of a
~23k-label task, so if the residual gap grows with n_train, a fixed count structurally
under-serves the big tasks and ratio-parameterized replay matches the requirement.

Outputs: replay_requirement_vs_size.png + a per-task table on stdout.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES_EPOCH = HERE.parent / "results"
RES_STEP = HERE.parent.parent / "rikyu_replay_sweep" / "results"
OUT = HERE / "replay_requirement_vs_size.png"

N_TRAIN = {
    "density": 23678, "efermi": 23668, "final_energy": 23678, "total_magnetization": 23678,
    "volume": 23678, "dielectric_total": 3124, "dielectric_ionic": 3124, "dielectric_electronic": 3124,
    "magnetization": 1160, "curie": 6272, "neel": 3466, "kp": 3875,
    "magnetic_susceptibility": 58, "zt": 3445, "power_factor": 3638, "thermal_conductivity": 4272,
    "electrical_resistivity": 5051, "dos_density": 7009, "seebeck": 8072,
    "formation_energy": 23180, "magnetic_moment": 851, "tc": 7207, "klat": 3863, "material_type": 33556,
}  # fmt: skip
N_REF = 2500
BLUE, RED, GREEN, MUTED, GRID = "#0077BB", "#CC3311", "#009E73", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    mx = max(int(r["step"]) for r in rows)
    final = {r["task"]: fnum(r["primary"]) for r in rows if int(r["step"]) == mx}
    intro = {r["task"]: fnum(r["primary"]) for r in rows if r["new_task"] == r["task"]}
    return final, intro


ARMS = (
    ("step-p8", RES_STEP / f"mt_n{N_REF}.csv", BLUE, "o"),
    ("epoch-p8", RES_EPOCH / f"mt_n{N_REF}_epoch.csv", RED, "D"),
    ("epoch-p24", RES_EPOCH / f"mt_n{N_REF}_epoch_p24.csv", GREEN, "s"),
)

gaps: dict[str, dict[str, float]] = {t: {} for t in N_TRAIN}
for arm, path, *_ in ARMS:
    if not path.exists():
        continue
    final, intro = load(path)
    for task in N_TRAIN:
        f, c = final.get(task), intro.get(task)
        if f is not None and c is not None:
            gaps[task][arm] = c - f

print(f"{'task':24} {'n_train':>7} {'replay@2500':>11} " + " ".join(f"{a:>10}" for a, *_ in ARMS)
      + "   (residual gap at n=2500; negative = ends above at-intro)")
print("-" * 90)
for task, ntr in sorted(N_TRAIN.items(), key=lambda kv: -kv[1]):
    cov = f"{min(N_REF, ntr) / ntr:>10.0%}"
    row = " ".join(f"{gaps[task][a]:>+10.3f}" if a in gaps[task] else f"{'-':>10}" for a, *_ in ARMS)
    print(f"{task:24} {ntr:>7} {cov:>11} {row}")

fig, ax = plt.subplots(figsize=(9.6, 6.8), dpi=150)
ax.axhline(0, color=MUTED, lw=1.2)
ax.axvline(N_REF, color=MUTED, ls=":", lw=1.2)
ax.annotate("← n=2500 is FULL replay   |   only partial replay →", (N_REF, 0.185),
            fontsize=9, color=MUTED, ha="center")
for arm, _, color, marker in ARMS:
    pts = [(N_TRAIN[t], gaps[t][arm], t) for t in N_TRAIN if arm in gaps[t]]
    xs, ys, names = zip(*pts)
    ax.scatter(xs, ys, s=54, marker=marker, color=color, edgecolor="white", lw=0.7, zorder=3, label=arm)
    for x, y, name in pts:
        if x > 20000 and arm == "step-p8":
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 3), fontsize=7.5, color=MUTED)
ax.set_xscale("log")
ax.set_xlabel("task's own training-set size, n_train (log)")
ax.set_ylabel(f"residual gap at n={N_REF}:  at-intro − final  (positive = not recovered)")
ax.set_title("At the largest fixed replay count, only the data-rich tasks stay unrecovered", fontsize=12.5)
ax.grid(True, which="both", color=GRID, lw=0.5, zorder=0)
ax.legend(frameon=False, fontsize=10, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")
