#!/usr/bin/env python3
"""Task-increment scaling curves: target test R² vs number of pretraining tasks k.

Reads results/scaling.csv (from collect.py). One panel per target + a mean panel; solid lines =
warm-start, dashed = frozen-encoder finetune; thin lines = the 3 task orders, thick = their mean;
grey band = the from-scratch (k=0) baseline range over its 3 seeds. One figure per replay branch
present in the data (n1000 / n1500), plus per-target singles under scaling_curves/.
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
RES = HERE.parent / "results/scaling.csv"
SINGLE_DIR = HERE / "scaling_curves"
SINGLE_DIR.mkdir(exist_ok=True)

TARGETS = ["dielectric_total", "dielectric_ionic", "dielectric_electronic"]
WS_C, FT_C, MUTED, GRID = "#0077BB", "#EE7733", "#6b7280", "#e5e7eb"

rows = list(csv.DictReader(open(RES)))
for r in rows:
    r["k"] = int(r["k"])
    r["r2"] = float(r["r2"]) if r["r2"] else None
    r["replay_n"] = int(r["replay_n"])

branches = sorted({r["replay_n"] for r in rows if r["mode"] in ("ws", "ft")})


def series(branch: int, mode: str, target: str, ord_id: str) -> tuple[list[int], list[float]]:
    pts = sorted(
        (r["k"], r["r2"])
        for r in rows
        if r["replay_n"] == branch and r["mode"] == mode and r["target"] == target and r["ord"] == ord_id and r["r2"] is not None
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def scratch_range(branch: int, target: str | None) -> tuple[float, float]:
    vals = [
        r["r2"]
        for r in rows
        if r["replay_n"] == branch and r["mode"] == "scratch" and (target is None or r["target"] == target) and r["r2"] is not None
    ]
    if target is None:  # mean over targets per seed
        by_seed: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r["replay_n"] == branch and r["mode"] == "scratch" and r["r2"] is not None:
                by_seed[r["ord"]].append(r["r2"])
        vals = [float(np.mean(v)) for v in by_seed.values()]
    return (min(vals), max(vals)) if vals else (np.nan, np.nan)


def draw_panel(ax, branch: int, target: str | None, title: str) -> None:
    orders = sorted({r["ord"] for r in rows if r["replay_n"] == branch and r["mode"] == "ws"})
    for mode, color, ls in (("ws", WS_C, "-"), ("ft", FT_C, "--")):
        mean_by_k: dict[int, list[float]] = defaultdict(list)
        for o in orders:
            if target is None:
                per_k: dict[int, list[float]] = defaultdict(list)
                for t in TARGETS:
                    ks, vs = series(branch, mode, t, o)
                    for k, v in zip(ks, vs):
                        per_k[k].append(v)
                ks = sorted(per_k)
                vs = [float(np.mean(per_k[k])) for k in ks]
            else:
                ks, vs = series(branch, mode, target, o)
            if not ks:
                continue
            ax.plot(ks, vs, color=color, ls=ls, lw=0.8, alpha=0.45)
            for k, v in zip(ks, vs):
                mean_by_k[k].append(v)
        ks = sorted(mean_by_k)
        ax.plot(ks, [float(np.mean(mean_by_k[k])) for k in ks], color=color, ls=ls, lw=2.2)
    lo, hi = scratch_range(branch, target)
    if np.isfinite(lo):
        ax.axhspan(lo, hi, color="#9ca3af", alpha=0.25, zorder=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([1, 5, 9, 13, 17, 21])
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


HANDLES = [
    Line2D([], [], color=WS_C, ls="-", lw=2.2, label="warm-start (encoder unfrozen, replay)"),
    Line2D([], [], color=FT_C, ls="--", lw=2.2, label="finetune (encoder frozen)"),
    Line2D([], [], color="#9ca3af", lw=7, alpha=0.4, label="from-scratch baseline (k=0, 3 seeds)"),
    Line2D([], [], color=MUTED, ls="-", lw=0.8, alpha=0.6, label="thin = individual task orders"),
]

for branch in branches:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), dpi=150)
    for ax, tgt in zip(axes, [*TARGETS, None]):
        draw_panel(ax, branch, tgt, tgt if tgt else "mean over the 3 targets")
    axes[0].set_ylabel("final test R²")
    fig.suptitle(f"Task-increment scaling — dielectric targets vs number of pretraining tasks k (replay n={branch})", fontsize=12.5, y=1.04)
    fig.legend(handles=HANDLES, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.supxlabel("k = number of tasks in the pretrained checkpoint", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    out = HERE / f"scaling_n{branch}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    for tgt in TARGETS:
        fig1, ax1 = plt.subplots(figsize=(7.4, 5.4), dpi=150)
        draw_panel(ax1, branch, tgt, f"{tgt}  (replay n={branch})")
        ax1.set_xlabel("k = number of tasks in the pretrained checkpoint")
        ax1.set_ylabel("final test R²")
        fig1.legend(handles=HANDLES, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.99))
        fig1.tight_layout(rect=(0, 0, 1, 0.86))
        fig1.savefig(SINGLE_DIR / f"n{branch}_{tgt}.png", bbox_inches="tight")
        plt.close(fig1)
print(f"saved per-target singles to {SINGLE_DIR}/")
