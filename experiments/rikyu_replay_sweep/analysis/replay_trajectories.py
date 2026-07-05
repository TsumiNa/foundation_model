#!/usr/bin/env python3
"""Per-task metric trajectories across replay events, one panel per task.

For each of the first 20 tasks in the (fixed, shared) 24-task sequence: x = how many replay
steps the task has been through since it was introduced (0 = the step it was learned; with
interval=1 every later step replays it once), y = the task's primary test metric at that step.
One line per run — fixed-count runs in blues (light 100 → dark 2500), fraction runs in oranges
(light 0.05 → dark 0.20). Shows directly how each replay amount slows (or doesn't) the decay of
each individual task, and where the drops happen.

Output: replay_trajectories.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE / "replay_trajectories.png"

RUNS = {  # tag -> (family, value, label)
    "n100": ("count", 100, "100"), "n200": ("count", 200, "200"), "n500": ("count", 500, "500"),
    "n1000": ("count", 1000, "1000"), "n1500": ("count", 1500, "1500"),
    "n2000": ("count", 2000, "2000"), "n2500": ("count", 2500, "2500"),
    "base0p05": ("frac", 0.05, "0.05*"), "0p10": ("frac", 0.10, "0.10"),
    "0p15": ("frac", 0.15, "0.15"), "0p20": ("frac", 0.20, "0.20"),
}  # fmt: skip
COUNT_TAGS = [t for t, (f, _, _) in RUNS.items() if f == "count"]
FRAC_TAGS = [t for t, (f, _, _) in RUNS.items() if f == "frac"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path: Path):
    """{task: {step: primary}} + {task: intro_step} + the introduction order."""
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


data = {}
task_order: list[str] = []
for tag in RUNS:
    p = RES / f"mt_{tag}.csv"
    if not p.exists():
        continue
    series, intro, order = load(p)
    data[tag] = dict(series=series, intro=intro)
    if not task_order:
        task_order = order
    elif order != task_order:
        raise SystemExit(f"run {tag} has a different task order — the shared-order assumption is broken")

TASKS = task_order[:20]  # the late tail (steps 21-24) has too few replay events to be interesting

blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(COUNT_TAGS)))
oranges = plt.cm.Oranges(np.linspace(0.35, 0.95, len(FRAC_TAGS)))
COLOR = {tag: blues[i] for i, tag in enumerate(COUNT_TAGS)}
COLOR |= {tag: oranges[i] for i, tag in enumerate(FRAC_TAGS)}

MUTED, GRID = "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})
SINGLE_DIR = HERE / "replay_trajectories"
SINGLE_DIR.mkdir(exist_ok=True)
LEGEND_TITLE = "replay per task per step (blues = fixed count, oranges = fraction of the task's train set)"


def draw_panel(ax, task: str, *, tick_size: float = 7.5) -> None:
    for tag in RUNS:
        d = data.get(tag)
        if d is None or task not in d["intro"]:
            continue
        s0 = d["intro"][task]
        steps = sorted(st for st in d["series"][task] if st >= s0 and d["series"][task][st] is not None)
        ax.plot(
            [st - s0 for st in steps],
            [d["series"][task][st] for st in steps],
            color=COLOR[tag],
            lw=1.5,
            marker="o",
            ms=2.4,
            label=RUNS[tag][2],
        )
    s0 = next(iter(data.values()))["intro"][task]
    ax.set_title(f"{task}  (learned at step {s0})", fontsize=9 if tick_size < 9 else 12)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=tick_size)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, axes = plt.subplots(4, 5, figsize=(22, 14), dpi=140)
for ax, task in zip(axes.flat, TASKS):
    draw_panel(ax, task)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.suptitle("Per-task metric across replay events — first 20 tasks of the sequence", fontsize=14, y=0.995)
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels),
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.978),
    title=LEGEND_TITLE,
    title_fontsize=10,
)
fig.supxlabel("replay events since the task was introduced (0 = at introduction)", fontsize=11)
fig.supylabel("primary test metric (R²)", fontsize=11)
fig.tight_layout(rect=(0.01, 0.01, 1, 0.925))
fig.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")

for task in task_order:  # singles for ALL 24 tasks (the grid keeps the first 20)
    fig1, ax1 = plt.subplots(figsize=(7.6, 6.0), dpi=150)
    draw_panel(ax1, task, tick_size=9.5)
    ax1.set_xlabel("replay events since the task was introduced (0 = at introduction)")
    ax1.set_ylabel("primary test metric")
    h1, l1 = ax1.get_legend_handles_labels()
    fig1.legend(
        h1,
        l1,
        loc="upper center",
        ncol=6,  # 11 entries -> two rows
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.985),
        title=LEGEND_TITLE,
        title_fontsize=7.5,
    )
    fig1.tight_layout(rect=(0, 0, 1, 0.845))
    fig1.savefig(SINGLE_DIR / f"{task}.png", bbox_inches="tight")
    plt.close(fig1)
print(f"saved {len(task_order)} per-task figures to {SINGLE_DIR}/")
