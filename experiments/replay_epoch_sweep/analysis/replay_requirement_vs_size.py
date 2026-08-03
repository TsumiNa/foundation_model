#!/usr/bin/env python3
"""Distance to the single-task ceiling across every replay budget, by task size.

One panel per task-size group; x = replay count n; y = mean deficit to the task's TRUE ceiling
(single-task baseline from the L2 warm-restart control), i.e. single − final. Two anchors:
  y = 0        — the single-task ceiling (never crossed in any prior experiment)
  grey dashed  — the group's at-intro cost (single − at-intro): the deficit that remains even
                 with zero forgetting; anything between dashed and 0 means replay ended ABOVE
                 the task's own introduction level.
Four arms (step-p8 / epoch-p8 / epoch-p24 / epoch-m150) show the full n-evolution per group.

Excluded: material_type (accuracy metric), magnetic_susceptibility (degenerate single-task
baseline — 58 labels, see the L2 notes). at-intro reference per task = mean over all runs of
all arms (per-run intro values are near-identical by design).

Outputs: replay_requirement_vs_size.png + a group × arm × n table on stdout.
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
    "zt": 3445, "power_factor": 3638, "thermal_conductivity": 4272,
    "electrical_resistivity": 5051, "dos_density": 7009, "seebeck": 8072,
    "formation_energy": 23180, "magnetic_moment": 851, "tc": 7207, "klat": 3863,
}  # fmt: skip  (material_type and magnetic_susceptibility excluded — see docstring)
COUNTS = [100, 200, 500, 1000, 1500, 2000, 2500]
BLUE, RED, GREEN, ORANGE, MUTED, GRID = "#0077BB", "#CC3311", "#009E73", "#EE7733", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})

GROUPS = (
    ("big tasks — ≥20k labels (6)", [t for t, n in N_TRAIN.items() if n >= 20000]),
    ("mid tasks — 3k–8.1k labels (14)", [t for t, n in N_TRAIN.items() if 3000 <= n < 20000]),
    ("small tasks — ≤1.2k labels (2)", [t for t, n in N_TRAIN.items() if n < 3000]),
)
ARMS = (
    ("step-p8", RES_STEP, "", BLUE, "o"),
    ("epoch-p8", RES_EPOCH, "_epoch", RED, "D"),
    ("epoch-p24", RES_EPOCH, "_epoch_p24", GREEN, "s"),
    ("epoch-m150", RES_EPOCH, "_epoch_m150", ORANGE, "^"),
)


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


def single_task_baseline() -> dict[str, float]:
    best: dict[str, float] = {}
    for r in csv.DictReader(open(RES_STEP / "warm_restart.csv")):
        v = fnum(r["primary"])
        if v is not None:
            t = r["task"]
            best[t] = v if t not in best else max(best[t], v)
    return best


SINGLE = single_task_baseline()

finals: dict[str, dict[int, dict[str, float]]] = {}
intro_acc: dict[str, list[float]] = {t: [] for t in N_TRAIN}
for arm, res_dir, suffix, *_ in ARMS:
    finals[arm] = {}
    for n in COUNTS:
        p = res_dir / f"mt_n{n}{suffix}.csv"
        if not p.exists():
            continue
        final, intro = load(p)
        finals[arm][n] = final
        for t in N_TRAIN:
            if intro.get(t) is not None:
                intro_acc[t].append(intro[t])
INTRO_REF = {t: float(np.mean(v)) for t, v in intro_acc.items() if v}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), dpi=150, sharey=True)
print(f"{'group':34} {'arm':>10} " + " ".join(f"{n:>7}" for n in COUNTS) + "   (mean single − final)")
for ax, (glabel, tasks) in zip(axes, GROUPS):
    intro_cost = float(np.mean([SINGLE[t] - INTRO_REF[t] for t in tasks]))
    ax.axhline(0, color=MUTED, lw=1.6, zorder=2)
    ax.axhline(intro_cost, color=MUTED, ls="--", lw=1.4, zorder=2)
    ax.axhspan(0, intro_cost, color="#f0fdf4", zorder=0)
    for arm, _, _, color, marker in ARMS:
        xs, ys = [], []
        for n in COUNTS:
            f = finals[arm].get(n)
            vals = [SINGLE[t] - f[t] for t in tasks if f and f.get(t) is not None]
            if vals:
                xs.append(n)
                ys.append(float(np.mean(vals)))
        ax.plot(xs, ys, color=color, lw=2.2, marker=marker, ms=5.5, mec="white", label=arm, zorder=3)
        print(f"{glabel:34} {arm:>10} " + " ".join(f"{y:>7.3f}" for y in ys))
    frac = np.mean([min(2500, N_TRAIN[t]) / N_TRAIN[t] for t in tasks])
    ax.set_title(f"{glabel}\nn=2500 replays {frac:.0%} of own data", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(COUNTS)
    ax.set_xticklabels(["100", "200", "500", "1000", "1500", "2k", "2.5k"], fontsize=8)
    ax.minorticks_off()
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("deficit to the single-task ceiling:  single − final")
axes[0].annotate("single-task ceiling", (COUNTS[0], 0), textcoords="offset points",
                 xytext=(2, -11), fontsize=8.5, color=MUTED)
axes[0].annotate("at-intro level (multi-task cost — green zone = ends ABOVE own introduction)",
                 (COUNTS[0], axes[0].get_ylim()[1]), fontsize=8.5, color=MUTED, va="top",
                 xytext=(2, -2), textcoords="offset points")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.985))
fig.suptitle("How close does replay get to the single-task ceiling? — every budget, every arm, by task size",
             fontsize=13, y=1.03)
fig.supxlabel("replay labels per old task per step, n (log)", fontsize=11)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")
