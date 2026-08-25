#!/usr/bin/env python3
"""Stage-B figures: what per-task head tuning bought, and what tuning in isolation cost.

* ``stage_b_gains.png`` — one bar per task, the relative gain of that task's own tuned head over
  the untuned head. Sorted by gain, coloured by head family, each bar labelled with the metric it
  was ranked on (tasks differ: saturated ones fall back from R² to MAE, classification uses
  macro-F1), because a mixed-unit axis would otherwise be unreadable. Gains are relative for the
  same reason — the tasks' metrics do not share a scale.
* ``stage_b_pertask_vs_joint.png`` — the control arm. Per-task tuning against one jointly tuned
  shared head on the same multi-task probe, with the untuned head as the common reference.

    python .../plot_stage_b.py ../results/head_winners.json --mt ../results/bmt.csv \\
        --base bmtreg_H64_L0p005 --joint <best> --pertask mt_pertask_reg
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

BLUE, RED, GREEN, ORANGE, PURPLE = "#0077BB", "#CC3311", "#009E73", "#EE7733", "#AA3377"
INK, MUTED, GRID, SURFACE = "#1f2937", "#6b7280", "#e5e7eb", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})

# Fixed hue order by head family — assigned by identity, never cycled.
KIND_COLOR = {"breg": BLUE, "bkr": GREEN, "bclf": PURPLE}
KIND_LABEL = {"breg": "regression", "bkr": "kernel-regression", "bclf": "classification"}
ARM_COLOR = {"untuned": MUTED, "joint": ORANGE, "per-task": BLUE}
ARM_MARKER = {"untuned": "o", "joint": "D", "per-task": "s"}


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gains_figure(winners: dict, out: Path) -> None:
    rows = []
    for task, w in winners.items():
        base, best = w.get("baseline_value"), w.get("value")
        if base in (None, 0) or best is None:
            continue
        sign = -1.0 if w["metric"] == "mae" else 1.0
        rows.append((sign * (best - base) / abs(base), task, w["kind"], w["metric"], bool(w["override"])))
    if not rows:
        raise SystemExit("no tasks with a baseline grid point")
    rows.sort()

    fig, ax = plt.subplots(figsize=(9.0, 0.30 * len(rows) + 1.9), constrained_layout=True)
    y = np.arange(len(rows))
    ax.barh(
        y, [r[0] for r in rows], height=0.66,
        color=[KIND_COLOR[r[2]] for r in rows], zorder=3,
    )
    ax.set_yticks(y, [r[1] for r in rows], fontsize=8)
    # Labels always start just right of zero (or of a positive bar's end) — a negative bar is
    # short, and left-hand labels would run straight into the task names.
    for i, (gain, _, _, metric, changed) in enumerate(rows):
        ax.text(
            max(gain, 0.0) + 0.004, i, f"{gain:+.1%}  ({metric}{'' if changed else ', kept default'})",
            va="center", ha="left", fontsize=7.5, color=INK,
        )
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_xlabel("relative gain of the task's own tuned head over the untuned head")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.set_xlim(min(0.0, min(r[0] for r in rows)) * 1.35 - 0.005, max(r[0] for r in rows) * 1.55 + 0.01)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KIND_COLOR.values()]
    ax.legend(handles, KIND_LABEL.values(), loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Stage B · what per-task head tuning bought", fontsize=11.5, color=INK, x=0.008, y=0.99, ha="left")
    fig.text(
        0.008, 0.955,
        "each task tuned on its own single-task probe; the ranking metric is chosen per task and printed on the bar",
        fontsize=8, color=MUTED, ha="left",
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def control_figure(csv_path: Path, base: str, joint: str, pertask: str, metric: str, out: Path) -> None:
    value: dict[str, dict[str, float]] = defaultdict(dict)
    for r in csv.DictReader(open(csv_path)):
        v = fnum(r.get(metric))
        if v is not None:
            value[r["runid"]][r["task"]] = v
    arms = [("untuned", base), ("joint", joint), ("per-task", pertask)]
    missing = [label for label, runid in arms if runid not in value]
    if missing:
        raise SystemExit(f"missing arm(s) {missing} in {csv_path}")
    tasks = sorted(set.intersection(*(set(value[r]) for _, r in arms)))

    fig, ax = plt.subplots(figsize=(1.9 * len(tasks) + 3.2, 3.9), constrained_layout=True)
    width = 0.26
    x = np.arange(len(tasks))
    for k, (label, runid) in enumerate(arms):
        offs = (k - 1) * (width + 0.02)
        vals = [value[runid][t] for t in tasks]
        ax.bar(x + offs, vals, width, label=label, color=ARM_COLOR[label], zorder=3)
        for xi, v in zip(x + offs, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, tasks)
    ax.set_ylabel(metric)
    ax.set_ylim(bottom=min(min(value[r][t] for t in tasks) for _, r in arms) * 0.95)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper center")
    fig.suptitle(
        "Stage B control · per-task tuning vs one jointly tuned shared head",
        fontsize=11.5, color=INK, x=0.008, y=0.99, ha="left",
    )
    fig.text(
        0.008, 0.945,
        "all three arms measured on the same multi-task probe; 'untuned' is the common reference",
        fontsize=8, color=MUTED, ha="left",
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("winners", type=Path, help="JSON from analysis/pick_heads.py")
    ap.add_argument("--mt", type=Path, help="collected CSV covering the three control arms")
    ap.add_argument("--base"), ap.add_argument("--joint"), ap.add_argument("--pertask")
    ap.add_argument("--metric", default="r2")
    ap.add_argument("--suffix", default="", help="appended to the control figure's filename")
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    gains_figure(json.loads(args.winners.read_text()), args.outdir / "stage_b_gains.png")
    if args.mt:
        if not (args.base and args.joint and args.pertask):
            raise SystemExit("--mt needs --base, --joint and --pertask")
        control_figure(
            args.mt, args.base, args.joint, args.pertask, args.metric,
            args.outdir / f"stage_b_pertask_vs_joint{args.suffix}.png",
        )


if __name__ == "__main__":
    main()
