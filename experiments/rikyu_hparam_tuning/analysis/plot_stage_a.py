#!/usr/bin/env python3
"""Stage-A figures: the encoder grid, and which knob actually moved it.

Two figures, each answering one question:

* ``stage_a_grid.png`` — small-multiple heatmaps, one panel per ``encoder_lr``, cells = the grid's
  ``latent_dim`` x ``encoder_hidden_dims`` plane. The quantity is *signed* (improvement over the
  untuned baseline), so the ramp is diverging — cool = better, warm = worse, neutral grey exactly
  at zero — and every cell carries its own value, so the figure doubles as the results table and
  no reading depends on colour alone.
* ``stage_a_marginals.png`` — for each knob, the spread of the score over every grid point at each
  of its levels. This is what separates "this knob matters" from "this one point happened to win".

    python .../plot_stage_a.py ../results/stage_a.csv --baseline a1_L128_H256_E0p005
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent

# House palette, shared with experiments/replay_epoch_sweep/analysis (Paul Tol bright; validated:
# lightness band / chroma floor / normal-vision separation all pass, orange-green sits in the
# 6-8 CVD floor band which is why every categorical series also carries a distinct marker).
BLUE, RED, GREEN, ORANGE, PURPLE = "#0077BB", "#CC3311", "#009E73", "#EE7733", "#AA3377"
INK, MUTED, GRID, SURFACE = "#1f2937", "#6b7280", "#e5e7eb", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED})

# Diverging ramp: two hues around a neutral grey midpoint — never a hue at zero.
DIVERGING = LinearSegmentedColormap.from_list("worse_better", [RED, "#ece9e4", BLUE])

MARKERS = ["o", "D", "s", "^", "v"]


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_scores(path: Path, baseline: str, metric: str) -> tuple[dict[str, float], list[str]]:
    """Mean relative improvement over the baseline run, per config. Scale-free, so the three probe
    tasks contribute equally regardless of their metric's magnitude."""
    per_task: dict[str, dict[str, float]] = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        v = fnum(r.get(metric))
        if v is not None:
            per_task[r["runid"]][r["task"]] = v
    if baseline not in per_task:
        raise SystemExit(f"baseline {baseline!r} not in {path} ({len(per_task)} configs)")
    base = per_task[baseline]
    sign = -1.0 if metric == "mae" else 1.0
    scores = {}
    for runid, values in per_task.items():
        shared = [t for t in values if t in base and base[t]]
        if shared:
            scores[runid] = statistics.fmean(sign * (values[t] - base[t]) / abs(base[t]) for t in shared)
    return scores, sorted(base)


def hidden_order(spec: str) -> tuple[int, int]:
    """Order width lists by depth, then by first width — never lexically ('1024-512' < '512-256')."""
    parts = spec.split("-")
    return len(parts), int(parts[0])


def parse(runid: str) -> tuple[int, str, float]:
    _, latent, hidden, lr = runid.split("_", 3)
    return int(latent[1:]), hidden[1:], float(lr[1:].replace("p", "."))


def grid_figure(scores: dict[str, float], out: Path, metric: str, tasks: list[str]) -> None:
    points = {parse(r): v for r, v in scores.items() if r.startswith("a1_")}
    latents = sorted({k[0] for k in points})
    hiddens = sorted({k[1] for k in points}, key=hidden_order)
    lrs = sorted({k[2] for k in points})
    best = max(points.items(), key=lambda kv: kv[1])[0] if points else None

    limit = max(abs(v) for v in points.values()) or 1e-6
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(1, len(lrs), figsize=(2.9 * len(lrs) + 1.6, 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, lr in zip(axes, lrs):
        matrix = np.full((len(latents), len(hiddens)), np.nan)
        for i, latent in enumerate(latents):
            for j, hidden in enumerate(hiddens):
                if (latent, hidden, lr) in points:
                    matrix[i, j] = points[(latent, hidden, lr)]
        ax.imshow(matrix, cmap=DIVERGING, norm=norm, aspect="auto")
        for i in range(len(latents)):
            for j in range(len(hiddens)):
                if np.isnan(matrix[i, j]):
                    continue
                is_best = best == (latents[i], hiddens[j], lr)
                ax.text(
                    j, i, f"{matrix[i, j]:+.1%}".replace("%", ""),
                    ha="center", va="center", fontsize=8,
                    color=INK, fontweight="bold" if is_best else "normal",
                )
                if is_best:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=2.0))
        ax.set_xticks(range(len(hiddens)), [h.replace("-", "→") for h in hiddens], rotation=35, ha="right")
        ax.set_yticks(range(len(latents)), latents, fontsize=8)
        ax.set_title(f"encoder_lr {lr:g}", fontsize=10, color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("latent_dim")
    fig.supxlabel("encoder_hidden_dims", fontsize=9, color=MUTED)
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), ax=axes.tolist(),
        fraction=0.035, pad=0.01,
    )
    bar.set_label(f"mean relative {metric} improvement", fontsize=8, color=MUTED)
    bar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    bar.ax.tick_params(labelsize=8, length=0)
    fig.suptitle("Stage A · encoder grid, 3-task probe", fontsize=11.5, color=INK, x=0.008, y=0.985, ha="left")
    fig.text(
        0.008, 0.925,
        f"cells = mean relative {metric} improvement (%) over the untuned baseline · probe: "
        f"{' / '.join(tasks)} · boxed cell = best",
        fontsize=8, color=MUTED, ha="left",
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def marginals_figure(scores: dict[str, float], out: Path, metric: str) -> None:
    points = {parse(r): v for r, v in scores.items() if r.startswith("a1_")}
    knobs = [
        ("latent_dim", lambda k: k[0], lambda v: str(v)),
        ("encoder_hidden_dims", lambda k: k[1], lambda v: v.replace("-", "→")),
        ("encoder_lr", lambda k: k[2], lambda v: f"{v:g}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), constrained_layout=True)
    for ax, (name, key, label), colour, marker in zip(axes, knobs, [BLUE, GREEN, PURPLE], MARKERS):
        groups: dict = defaultdict(list)
        for k, v in points.items():
            groups[key(k)].append(v)
        levels = sorted(groups, key=hidden_order if isinstance(next(iter(groups)), str) else None)
        for i, level in enumerate(levels):
            values = groups[level]
            ax.scatter([i] * len(values), values, s=16, color=colour, marker=marker, alpha=0.45, lw=0)
            ax.plot([i - 0.28, i + 0.28], [statistics.fmean(values)] * 2, color=INK, lw=2.0, zorder=3)
        ax.axhline(0, color=MUTED, lw=1.0, ls="--")
        ax.set_xticks(range(len(levels)), [label(v) for v in levels], rotation=25, ha="right")
        ax.set_title(name, fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(f"mean rel. {metric} improvement")
    for ax in axes:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    fig.suptitle("Stage A · marginal effect of each encoder knob", fontsize=11.5, color=INK, x=0.008, y=0.985, ha="left")
    fig.text(0.01, 0.905, "every grid point plotted; horizontal bar = level mean; dashed line = untuned baseline",
             fontsize=8, color=MUTED, ha="left")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.87))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--baseline", default="a1_L128_H256_E0p005")
    ap.add_argument("--metric", default="mae")
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    scores, tasks = load_scores(args.csv, args.baseline, args.metric)
    args.outdir.mkdir(parents=True, exist_ok=True)
    grid_figure(scores, args.outdir / "stage_a_grid.png", args.metric, tasks)
    marginals_figure(scores, args.outdir / "stage_a_marginals.png", args.metric)


if __name__ == "__main__":
    main()
