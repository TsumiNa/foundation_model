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
PATHS = [("latent_default", "-"), ("comp_k4", "--")]
SCENARIOS = [("fe_down_total_up", "dielectric_total"), ("fe_down_ionic_up", "dielectric_ionic"), ("fe_down_electronic_up", "dielectric_electronic")]
DIEL = ["dielectric_total_after_mean", "dielectric_ionic_after_mean", "dielectric_electronic_after_mean"]

rows = list(csv.DictReader(open(RES)))
for r in rows:
    r["k"] = int(r["k"])
    r["replay_n"] = int(r["replay_n"])
    for key in ("objective_mean", "seed_objective_mean", "formation_energy_after_mean", *DIEL):
        r[key] = float(r[key]) if r.get(key) else None

branches = sorted({r["replay_n"] for r in rows})


def mode_series(branch, mode, path, scen, value_fn):
    per_ord: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r["replay_n"] == branch and r["mode"] == mode and r["path"] == path and r.get("scenario") == scen:
            v = value_fn(r)
            if v is not None:
                per_ord[r["ord"]][r["k"]] = v
    return per_ord


def draw(ax, branch, scen, value_fn, ylabel, seed_fn=None):
    for mode, color in (("ws", WS_C), ("ft", FT_C)):
        for path, ls in PATHS:
            per_ord = mode_series(branch, mode, path, scen, value_fn)
            mean_by_k: dict[int, list[float]] = defaultdict(list)
            for o, s in per_ord.items():
                for k in sorted(s):
                    mean_by_k[k].append(s[k])
            ks = sorted(mean_by_k)
            if ks:
                ax.plot(ks, [float(np.mean(mean_by_k[k])) for k in ks], color=color, ls=ls, lw=2.0)
    if seed_fn is not None:
        seeds = [seed_fn(r) for r in rows if r["replay_n"] == branch and r.get("scenario") == scen and seed_fn(r) is not None]
        if seeds:
            ax.axhline(float(np.mean(seeds)), color="#9ca3af", ls=":", lw=1.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks([1, 5, 9, 13, 17, 21])
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


HANDLES = [
    Line2D([], [], color=WS_C, lw=2.0, label="warm-start models"),
    Line2D([], [], color=FT_C, lw=2.0, label="finetune models"),
    Line2D([], [], color=MUTED, ls="-", lw=2, label="latent path"),
    Line2D([], [], color=MUTED, ls="--", lw=2, label="composition path (≤4 elem)"),
    Line2D([], [], color="#9ca3af", ls=":", lw=1.4, label="seed-pool baseline"),
]

for branch in branches:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.2), dpi=150)
    for col, (scen, diel) in enumerate(SCENARIOS):
        draw(axes[0][col], branch, scen, lambda r: r["objective_mean"], "objective score (lower = better)" if col == 0 else "", seed_fn=lambda r: r["seed_objective_mean"])
        axes[0][col].set_title(f"{scen}\n(FE→−1 + {diel}→+1)", fontsize=9.5)
        draw(
            axes[1][col],
            branch,
            scen,
            lambda r, d=diel: r[f"{d}_after_mean"],
            "achieved dielectric channel (target +1)" if col == 0 else "",
        )
        axes[1][col].axhline(1.0, color="#C44E52", ls=":", lw=1.2)
    fig.suptitle(f"Inverse-design quality vs pretraining task count k (replay n={branch}); objective is model-self-scored", fontsize=12, y=1.0)
    fig.legend(handles=HANDLES, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.supxlabel("k = number of tasks in the pretrained checkpoint", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    out = HERE / f"inverse_n{branch}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
