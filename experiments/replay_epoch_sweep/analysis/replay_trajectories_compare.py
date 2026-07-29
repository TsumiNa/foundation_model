#!/usr/bin/env python3
"""Per-task metric trajectories across replay events — step (blues) vs epoch (reds) resampling.

Same layout as ../../rikyu_replay_sweep/analysis/replay_trajectories.py, fixed-count family only:
for each task, x = replay events since introduction, y = primary test metric; blues = frozen
subset per step (light 100 → dark 2500), reds = per-epoch resampling (same shading). Shows
whether per-epoch resampling removes the small-n step-like collapses (event-driven forgetting).

Outputs: replay_trajectories_compare.png + replay_trajectories_compare/<task>.png
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
OUT = HERE / "replay_trajectories_compare.png"
SINGLE_DIR = HERE / "replay_trajectories_compare"

COUNTS = [100, 200, 500, 1000, 1500, 2000, 2500]
MUTED, GRID = "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})
LEGEND_TITLE = "labels replayed per task per step (blues = frozen subset/step, reds = per-epoch resampling)"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path: Path):
    series: dict[str, dict[int, float]] = {}
    intro: dict[str, int] = {}
    order: list[str] = []
    for r in csv.DictReader(open(path)):
        step, task = int(r["step"]), r["task"]
        series.setdefault(task, {})[step] = fnum(r["primary"])
        if r["new_task"] == task and task not in intro:
            intro[task] = step
            order.append(task)
    return series, intro, order


data: dict[tuple[str, int], dict] = {}  # (mode, n) -> {series, intro}
task_order: list[str] = []
for mode, res_dir, suffix in (("step", RES_STEP, ""), ("epoch", RES_EPOCH, "_epoch")):
    for n in COUNTS:
        p = res_dir / f"mt_n{n}{suffix}.csv"
        if not p.exists():
            continue
        series, intro, order = load(p)
        data[(mode, n)] = dict(series=series, intro=intro)
        if not task_order:
            task_order = order
        elif order != task_order:
            raise SystemExit(f"{mode} n{n} has a different task order — shared-order assumption broken")
if not any(m == "epoch" for m, _ in data):
    print(f"WARNING: no epoch CSVs under {RES_EPOCH} yet — plotting step only")

TASKS = task_order[:20]
blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(COUNTS)))
reds = plt.cm.Reds(np.linspace(0.35, 0.95, len(COUNTS)))
COLOR = {("step", n): blues[i] for i, n in enumerate(COUNTS)}
COLOR |= {("epoch", n): reds[i] for i, n in enumerate(COUNTS)}


def draw_panel(ax, task: str, *, tick_size: float = 7.5) -> None:
    for (mode, n), d in data.items():
        if task not in d["intro"]:
            continue
        s0 = d["intro"][task]
        steps = sorted(st for st in d["series"][task] if st >= s0 and d["series"][task][st] is not None)
        ax.plot(
            [st - s0 for st in steps],
            [d["series"][task][st] for st in steps],
            color=COLOR[(mode, n)],
            lw=1.4,
            marker="o",
            ms=2.2,
            label=f"{mode[0]}{n}",
        )
    s0 = next(iter(data.values()))["intro"][task]
    ax.set_title(f"{task}  (learned at step {s0})", fontsize=9 if tick_size < 9 else 12)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=tick_size)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


SINGLE_DIR.mkdir(exist_ok=True)
fig, axes = plt.subplots(4, 5, figsize=(22, 14), dpi=140)
for ax, task in zip(axes.flat, TASKS):
    draw_panel(ax, task)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.suptitle("Per-task metric across replay events — step vs epoch resampling, first 20 tasks", fontsize=14, y=0.995)
fig.legend(handles, labels, loc="upper center", ncol=len(labels) // 2 + 1, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 0.982), title=LEGEND_TITLE, title_fontsize=9.5)
fig.supxlabel("replay events since the task was introduced (0 = at introduction)", fontsize=11)
fig.supylabel("primary test metric (R²)", fontsize=11)
fig.tight_layout(rect=(0.01, 0.01, 1, 0.918))
fig.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")

for task in task_order:
    fig1, ax1 = plt.subplots(figsize=(7.6, 6.0), dpi=150)
    draw_panel(ax1, task, tick_size=9.5)
    ax1.set_xlabel("replay events since the task was introduced (0 = at introduction)")
    ax1.set_ylabel("primary test metric")
    h1, l1 = ax1.get_legend_handles_labels()
    fig1.legend(h1, l1, loc="upper center", ncol=7, frameon=False, fontsize=8,
                bbox_to_anchor=(0.5, 0.985), title=LEGEND_TITLE, title_fontsize=7.5)
    fig1.tight_layout(rect=(0, 0, 1, 0.845))
    fig1.savefig(SINGLE_DIR / f"{task}.png", bbox_inches="tight")
    plt.close(fig1)
print(f"saved {len(task_order)} per-task figures to {SINGLE_DIR}/")
