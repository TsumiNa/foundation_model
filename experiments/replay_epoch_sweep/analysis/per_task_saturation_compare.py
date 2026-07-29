#!/usr/bin/env python3
"""Per-task replay saturation, step vs epoch resampling — final metric vs labels replayed per step.

Same panel layout as ../../rikyu_replay_sweep/analysis/per_task_saturation.py, restricted to the
fixed-count family (n = 100..2500), with BOTH resample modes per panel:
  blue circles  = frozen subset per step (historical runs, rikyu GB200)
  red diamonds  = per-epoch resampling (this sweep, ism A100)
Reference lines: green = single-task baseline (unchanged, no replay involved); grey dashed =
at-intro mean of the step runs; red dashed = at-intro mean of the epoch runs.

Outputs: per_task_saturation_compare.png, per_task_saturation_compare/<task>.png,
and a per-task/per-n delta table on stdout.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES_EPOCH = HERE.parent / "results"
RES_STEP = HERE.parent.parent / "rikyu_replay_sweep" / "results"
OUT = HERE / "per_task_saturation_compare.png"
SINGLE_DIR = HERE / "per_task_saturation_compare"

# n_valid_train per task (train-split valid labels) — same table as the step-sweep analysis.
N_TRAIN = {
    "density": 23678, "efermi": 23668, "final_energy": 23678, "total_magnetization": 23678,
    "volume": 23678, "dielectric_total": 3124, "dielectric_ionic": 3124, "dielectric_electronic": 3124,
    "magnetization": 1160, "curie": 6272, "neel": 3466, "kp": 3875,
    "magnetic_susceptibility": 58, "zt": 3445, "power_factor": 3638, "thermal_conductivity": 4272,
    "electrical_resistivity": 5051, "dos_density": 7009, "seebeck": 8072,
    "formation_energy": 23180, "magnetic_moment": 851, "tc": 7207, "klat": 3863, "material_type": 33556,
}  # fmt: skip
ORDER = list(N_TRAIN)
COUNTS = [100, 200, 500, 1000, 1500, 2000, 2500]

BLUE, RED, GREEN = "#0077BB", "#CC3311", "#009E73"
MUTED, VLINE, GRID = "#6b7280", "#9ca3af", "#e5e7eb"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})


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


def load_mode(res_dir: Path, suffix: str):
    """{n: {final, intro}} for every existing fixed-count CSV of one resample mode."""
    out = {}
    for n in COUNTS:
        p = res_dir / f"mt_n{n}{suffix}.csv"
        if p.exists():
            final, intro = load(p)
            out[n] = dict(final=final, intro=intro)
    return out


def load_single_task_baseline() -> dict[str, float]:
    p = RES_STEP / "warm_restart.csv"
    best: dict[str, float] = {}
    if p.exists():
        for r in csv.DictReader(open(p)):
            v = fnum(r["primary"])
            if v is not None:
                t = r["task"]
                best[t] = v if t not in best else max(best[t], v)
    return best


STEP = load_mode(RES_STEP, "")
EPOCH = load_mode(RES_EPOCH, "_epoch")
SINGLE_BASE = load_single_task_baseline()
if not STEP:
    raise SystemExit(f"no step-mode CSVs under {RES_STEP}")
if not EPOCH:
    print(f"WARNING: no epoch-mode CSVs under {RES_EPOCH} yet — plotting step only")


def legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", ls="-", color=BLUE, mec="white", ms=7, lw=1.2,
               label='resample="step" (frozen subset; rikyu GB200)'),
        Line2D([], [], marker="D", ls="-", color=RED, mec="white", ms=6.5, lw=1.2,
               label='resample="epoch" (redrawn every epoch; ism A100)'),
        Line2D([], [], ls="-", color=GREEN, lw=1.8, label="single-task baseline (best self-warm-restart)"),
        Line2D([], [], ls="--", color=MUTED, lw=1.4, label="at-intro level (step runs)"),
        Line2D([], [], ls="--", color=RED, lw=1.1, alpha=0.6, label="at-intro level (epoch runs)"),
        Line2D([], [], ls=":", color=VLINE, lw=1.4, label="this task's full train-set size"),
    ]


def series(mode: dict, task: str):
    xs, ys = [], []
    for n in COUNTS:
        d = mode.get(n)
        y = d and d["final"].get(task)
        if y is not None:
            xs.append(float(min(n, N_TRAIN[task])))
            ys.append(y)
    return np.array(xs), np.array(ys)


def intro_mean(mode: dict, task: str):
    vals = [d["intro"][task] for d in mode.values() if d["intro"].get(task) is not None]
    return float(np.mean(vals)) if vals else None


def draw_panel(ax, task: str, *, tick_size: float = 7.5) -> str:
    n_train = N_TRAIN[task]
    row = [f"{task:24} {n_train:>7}"]
    for mode, color, marker, ms in ((STEP, BLUE, "o", 34), (EPOCH, RED, "D", 30)):
        xs, ys = series(mode, task)
        if len(xs):
            ax.plot(xs, ys, color=color, lw=1.1, alpha=0.55, zorder=2)
            ax.scatter(xs, ys, s=ms, marker=marker, color=color, edgecolor="white", lw=0.6, zorder=3)
    ci_step, ci_epoch = intro_mean(STEP, task), intro_mean(EPOCH, task)
    if ci_step is not None:
        ax.axhline(ci_step, color=MUTED, ls="--", lw=1.1, zorder=1)
    if ci_epoch is not None:
        ax.axhline(ci_epoch, color=RED, ls="--", lw=0.9, alpha=0.55, zorder=1)
    if task in SINGLE_BASE:
        ax.axhline(SINGLE_BASE[task], color=GREEN, ls="-", lw=1.5, alpha=0.9, zorder=1)
    ax.axvline(n_train, color=VLINE, lw=1.2, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_title(f"{task}  (n_train={n_train:,})", fontsize=9 if tick_size < 9 else 12)
    ax.grid(True, which="both", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=tick_size)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # stdout: per-n delta (epoch - step)
    for n in COUNTS:
        ys_s = STEP.get(n, {}).get("final", {}).get(task)
        ys_e = EPOCH.get(n, {}).get("final", {}).get(task)
        row.append(f"{ys_e - ys_s:+.3f}" if (ys_s is not None and ys_e is not None) else "     -")
    return " ".join(row)


SINGLE_DIR.mkdir(exist_ok=True)
fig, axes = plt.subplots(4, 6, figsize=(24, 14.5), dpi=140)
print(f"{'task':24} {'n_train':>7} " + " ".join(f"{'Δn' + str(n):>6}" for n in COUNTS) + "   (Δ = epoch − step, final primary)")
print("-" * 110)
for ax, task in zip(axes.flat, ORDER):
    print(draw_panel(ax, task))

# all-task mean delta per n
means = []
for n in COUNTS:
    ds = [EPOCH[n]["final"][t] - STEP[n]["final"][t] for t in ORDER
          if n in EPOCH and n in STEP and EPOCH[n]["final"].get(t) is not None and STEP[n]["final"].get(t) is not None]
    means.append(f"{np.mean(ds):+.3f}" if ds else "     -")
print("-" * 110)
print(f"{'MEAN Δ (24 tasks)':32} " + " ".join(f"{m:>6}" for m in means))

fig.suptitle("Per-task replay saturation — frozen subset (step) vs per-epoch resampling (epoch)", fontsize=14, y=0.995)
fig.legend(handles=legend_handles(), loc="upper center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.978))
fig.supxlabel("labels replayed per step for this task (log scale)", fontsize=11)
fig.supylabel("final primary metric after all 24 steps (test R²; accuracy for material_type)", fontsize=11)
fig.tight_layout(rect=(0.01, 0.01, 1, 0.94))
fig.savefig(OUT, bbox_inches="tight")
print(f"\nsaved {OUT}")

for task in ORDER:
    fig1, ax1 = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    draw_panel(ax1, task, tick_size=9.5)
    ax1.set_xlabel("labels replayed per step for this task (log scale)")
    ax1.set_ylabel("final primary metric (test)")
    fig1.legend(handles=legend_handles(), loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
    fig1.tight_layout(rect=(0, 0, 1, 0.85))
    fig1.savefig(SINGLE_DIR / f"{task}.png", bbox_inches="tight")
    plt.close(fig1)
print(f"saved 24 per-task figures to {SINGLE_DIR}/")
