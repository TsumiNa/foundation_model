#!/usr/bin/env python3
"""Stage-B figures: what per-task head tuning bought, and what tuning in isolation cost.

* ``stage_b_gains.png`` — one bar per task: its measured gain divided by its own seed band, so
  the confirmation rule is a single vertical line at 1.0 and 24 incompatible metric scales become
  one comparable axis. Bars below the line are faded — those gains are not supported by repetition.
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
    """Per-task gain expressed in units of that task's own seed band.

    Plotting gain and band as two absolute quantities does not work here: the tasks' metrics live
    on wildly different scales, and one degenerate task (magnetic_susceptibility, 58 labels, mean
    R² ≈ 0.05) has a relative band above 1000%, which flattens every other task to a hairline.
    The ratio gain/band is the quantity the decision actually uses, it is dimensionless, and the
    rule becomes one vertical line: a task keeps its tuned head iff its bar crosses 1.0.
    """
    rows = []
    for task, w in winners.items():
        band, gain = w.get("band"), w.get("confirmed_gain")
        if band is None or gain is None or not band:
            continue
        rows.append((gain / band, task, w["kind"], w["metric"], bool(w["confirmed"]),
                     gain, band, w.get("mean_base")))
    if not rows:
        raise SystemExit("winners JSON has no confirmation fields — run analysis/confirm_heads.py")
    rows.sort()

    fig, ax = plt.subplots(figsize=(9.8, 0.32 * len(rows) + 2.1), constrained_layout=True)
    for i, (ratio, _task, kind, _metric, ok, *_rest) in enumerate(rows):
        ax.barh(i, ratio, height=0.62, color=KIND_COLOR[kind], alpha=1.0 if ok else 0.35, zorder=3)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8)
    for i, (ratio, _t, _k, metric, ok, gain, band, base) in enumerate(rows):
        scale = f"{gain:+.4f} vs {band:.4f}" if base is None else f"{gain:+.4f} vs band {band:.4f}"
        ax.text(max(ratio, 0) + 0.04, i, f"{scale}  ({metric})" + ("   KEEP" if ok else ""),
                va="center", ha="left", fontsize=7.5, color=INK, fontweight="bold" if ok else "normal")
    ax.axvline(0, color=INK, lw=1.0)
    ax.axvline(1.0, color=RED, lw=1.6, ls="--", zorder=5)
    ax.text(1.0, len(rows) - 0.2, "  confirmation threshold: gain = band", color=RED, fontsize=8, va="top")
    ax.set_xlabel("measured gain of the tuned head ÷ that task's own seed band  (3 seeds per arm)")
    ax.set_xlim(min(0.0, min(r[0] for r in rows)) * 1.25 - 0.05, max(1.35, max(r[0] for r in rows) * 1.05) + 1.35)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KIND_COLOR.values()]
    ax.legend(handles, list(KIND_LABEL.values()), loc="lower right", frameon=False, fontsize=8)
    kept = sum(1 for r in rows if r[4])
    fig.suptitle("Stage B · per-task head tuning, measured against its own noise",
                 fontsize=11.5, color=INK, x=0.008, y=0.99, ha="left")
    fig.text(0.008, 0.958,
             f"winner and untuned baseline each re-run at 3 seeds; only {kept}/{len(rows)} gains "
             "exceed the task's seed band (solid bars). Labels give the raw gain and band.",
             fontsize=8, color=MUTED, ha="left")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.945))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def control_figure(csv_path: Path, base: str, joint: str, pertask: str, metric: str, out: Path) -> None:
    """Both tuning strategies as a difference from the untuned arm.

    Plotting the three arms' raw values shares one y-axis across tasks whose levels differ by
    0.25 (formation_energy sits at 0.991, magnetization at 0.75), which visually flattens the very
    differences the control exists to show. Differencing against the untuned arm puts every task on
    a common, meaningful zero and makes each strategy's purchase directly readable.
    """
    value: dict[str, dict[str, float]] = defaultdict(dict)
    for r in csv.DictReader(open(csv_path)):
        v = fnum(r.get(metric))
        if v is not None:
            value[r["runid"]][r["task"]] = v
    arms = [("joint", joint), ("per-task", pertask)]
    missing = [label for label, runid in [("untuned", base), *arms] if runid not in value]
    if missing:
        raise SystemExit(f"missing arm(s) {missing} in {csv_path}")
    tasks = sorted(set(value[base]) & set(value[joint]) & set(value[pertask]))

    fig, ax = plt.subplots(figsize=(1.85 * len(tasks) + 3.4, 4.2), constrained_layout=True)
    width = 0.3
    x = np.arange(len(tasks))
    for k, (label, runid) in enumerate(arms):
        offs = (k - 0.5) * (width + 0.03)
        deltas = [value[runid][t] - value[base][t] for t in tasks]
        ax.bar(x + offs, deltas, width, label=label, color=ARM_COLOR[label], zorder=3)
        for xi, d in zip(x + offs, deltas):
            ax.text(xi, d, f"{d:+.4f}", ha="center", va="bottom" if d >= 0 else "top",
                    fontsize=8, color=INK)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(x, tasks)
    ax.set_ylabel(f"Δ {metric} vs the untuned shared head")
    ax.margins(y=0.22)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncols=2, loc="upper right")

    mj = np.mean([value[joint][t] - value[base][t] for t in tasks])
    mp = np.mean([value[pertask][t] - value[base][t] for t in tasks])
    fig.suptitle("Stage B control · per-task tuning vs one jointly tuned shared head",
                 fontsize=11.5, color=INK, x=0.008, y=0.985, ha="left")
    fig.text(0.008, 0.925,
             f"same multi-task probe, zero = the untuned shared head  ·  mean Δ: joint {mj:+.4f}, "
             f"per-task {mp:+.4f}  ·  single seed, so read the sign, not the magnitude",
             fontsize=8, color=MUTED, ha="left")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("winners", type=Path, help="JSON from analysis/confirm_heads.py")
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
