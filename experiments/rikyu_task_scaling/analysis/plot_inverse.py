#!/usr/bin/env python3
"""Inverse-design quality vs number of pretraining tasks k.

Reads results/inverse.csv. Top row: mean objective score (lower = better; Σ weighted target
losses, model-self-scored) vs k, one panel per optimisation path, warm-start vs finetune models,
thin per-order lines + thick mean, dashed grey = the seed-pool baseline. Bottom row: achieved
per-target channels vs k (formation_energy → −1, dielectric mean → +1). One figure per replay
branch in the data.
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
RES = HERE.parent / "results/inverse.csv"

WS_C, FT_C, MUTED, GRID = "#0077BB", "#EE7733", "#6b7280", "#e5e7eb"
PATHS = [("latent_default", "latent path (AE-aligned)"), ("comp_k4_lowdiv", "composition path (≤4 elements, low diversity)")]
DIEL = ["dielectric_total_after_mean", "dielectric_ionic_after_mean", "dielectric_electronic_after_mean"]

rows = list(csv.DictReader(open(RES)))
for r in rows:
    r["k"] = int(r["k"])
    r["replay_n"] = int(r["replay_n"])
    for key in ("objective_mean", "seed_objective_mean", "formation_energy_after_mean", *DIEL):
        r[key] = float(r[key]) if r.get(key) else None

branches = sorted({r["replay_n"] for r in rows})


def mode_series(branch, mode, path, value_fn):
    per_ord: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r["replay_n"] == branch and r["mode"] == mode and r["path"] == path:
            v = value_fn(r)
            if v is not None:
                per_ord[r["ord"]][r["k"]] = v
    return per_ord


def draw(ax, branch, path, value_fn, ylabel, seed_fn=None):
    for mode, color in (("ws", WS_C), ("ft", FT_C)):
        per_ord = mode_series(branch, mode, path, value_fn)
        mean_by_k: dict[int, list[float]] = defaultdict(list)
        for o, s in per_ord.items():
            ks = sorted(s)
            ax.plot(ks, [s[k] for k in ks], color=color, lw=0.8, alpha=0.45)
            for k in ks:
                mean_by_k[k].append(s[k])
        ks = sorted(mean_by_k)
        ax.plot(ks, [float(np.mean(mean_by_k[k])) for k in ks], color=color, lw=2.2)
    if seed_fn is not None:
        seeds = [seed_fn(r) for r in rows if r["replay_n"] == branch and r["path"] == path and seed_fn(r) is not None]
        if seeds:
            ax.axhline(float(np.mean(seeds)), color="#9ca3af", ls="--", lw=1.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks([1, 5, 9, 13, 17, 21])
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


HANDLES = [
    Line2D([], [], color=WS_C, lw=2.2, label="on warm-start models"),
    Line2D([], [], color=FT_C, lw=2.2, label="on finetune models"),
    Line2D([], [], color="#9ca3af", ls="--", lw=1.4, label="seed-pool baseline objective"),
]

for branch in branches:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), dpi=150)
    for col, (path, title) in enumerate(PATHS):
        draw(axes[0][col], branch, path, lambda r: r["objective_mean"], "objective score (lower = better)", seed_fn=lambda r: r["seed_objective_mean"])
        axes[0][col].set_title(title, fontsize=10.5)
        draw(
            axes[1][col],
            branch,
            path,
            lambda r: None if r["formation_energy_after_mean"] is None else r["formation_energy_after_mean"],
            "achieved formation_energy (target −1)",
        )
        axes[1][col].axhline(-1.0, color="#C44E52", ls=":", lw=1.2)
    fig.suptitle(f"Inverse-design quality vs pretraining task count k (replay n={branch}); objective is model-self-scored", fontsize=12, y=1.0)
    fig.legend(handles=HANDLES, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.supxlabel("k = number of tasks in the pretrained checkpoint", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    out = HERE / f"inverse_n{branch}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
