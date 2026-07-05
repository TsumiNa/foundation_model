#!/usr/bin/env python3
"""Composition diversity of the 20 inverse-design candidates vs pretraining task count k.

Usage: python plot_diversity.py [MIRROR_DIR] [REPLAY_N]

For every (mode, order, k, path): at the FINAL optimisation step, count the distinct top-3
element systems among the 20 candidates and their mean pairwise L1 distance. Plots both vs k
(one panel per metric, line style by path, color by mode, thin = orders, thick = mean).
"""

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
MIRROR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "artifacts/task_scaling"
REPLAY_N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TAG = "" if REPLAY_N == 1000 else f"_n{REPLAY_N}"
SCENARIO = "fe_down_diel_up"
PATHS = {"latent_default": "-", "comp_k4_lowdiv": "--"}
WS_C, FT_C, MUTED, GRID = "#0077BB", "#EE7733", "#6b7280", "#e5e7eb"

stats: dict = defaultdict(dict)  # (mode, path, ord) -> {k: (systems, l1)}
for mode in ("ws", "ft"):
    for ord_id in range(3):
        for k in range(1, 22):
            for path in PATHS:
                p = MIRROR / f"{mode}{TAG}_o{ord_id}/k{k:02d}/inverse/{SCENARIO}/trajectories/{path}.npz"
                if not p.exists():
                    continue
                z = np.load(p, allow_pickle=False)
                w = z["weights"]
                if w.size == 0:
                    continue
                final = w[-1]
                n = final.shape[0]
                systems = {tuple(sorted(np.argsort(final[b])[::-1][:3].tolist())) for b in range(n)}
                l1 = float(np.mean([np.abs(final[a] - final[c]).sum() for a, c in combinations(range(n), 2)]))
                stats[(mode, path, str(ord_id))][k] = (len(systems), l1)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=150)
for ax, idx, ylabel in ((axes[0], 0, "distinct top-3 element systems (of 20)"), (axes[1], 1, "mean pairwise L1 distance")):
    for mode, color in (("ws", WS_C), ("ft", FT_C)):
        for path, ls in PATHS.items():
            mean_by_k: dict[int, list[float]] = defaultdict(list)
            for o in ("0", "1", "2"):
                s = stats.get((mode, path, o), {})
                ks = sorted(s)
                if not ks:
                    continue
                ax.plot(ks, [s[k][idx] for k in ks], color=color, ls=ls, lw=0.7, alpha=0.35)
                for k in ks:
                    mean_by_k[k].append(s[k][idx])
            ks = sorted(mean_by_k)
            ax.plot(ks, [float(np.mean(mean_by_k[k])) for k in ks], color=color, ls=ls, lw=2.2)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xticks([1, 5, 9, 13, 17, 21])
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

HANDLES = [
    Line2D([], [], color=WS_C, lw=2.2, label="warm-start models"),
    Line2D([], [], color=FT_C, lw=2.2, label="finetune models"),
    Line2D([], [], color=MUTED, ls="-", lw=2, label="latent path"),
    Line2D([], [], color=MUTED, ls="--", lw=2, label="composition path (≤4 elem, low diversity)"),
]
fig.suptitle(f"Diversity of the 20 designed candidates at the final step vs k (replay n={REPLAY_N})", fontsize=12, y=1.02)
fig.legend(handles=HANDLES, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.97))
fig.supxlabel("k = number of tasks in the pretrained checkpoint", fontsize=10.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.88))
out = HERE / f"diversity_n{REPLAY_N}.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
