#!/usr/bin/env python3
"""Do checkpoint (k), eval mode (ws/ft), or task order systematically shape the designed candidates?

Usage: python candidate_regularity.py [MIRROR_DIR] [REPLAY_N]

Because the weighted_random seed set is model-independent, every run optimises the SAME 20 seed
compositions — so per-seed differences between final compositions isolate the model's influence.
For each path (latent / composition), per k we compute mean per-seed L1 distances between final
compositions:

  Δk        — same order & mode, consecutive checkpoints k vs k+1 (does one more pretraining
              task move the answers?)
  ws vs ft  — same order & k, across eval modes
  Δorder    — same mode & k, across the 3 task orders
  |final−seed| — displacement from the seed (how far optimisation travels at all)

If Δk ≈ Δorder ≈ ws-vs-ft ≈ displacement scale, candidates carry no checkpoint fingerprint;
values well below the displacement mean the answers are largely model-independent. Emits
candidate_regularity_n<REPLAY_N>.png + a printed summary table.
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
SCENARIOS = ["fe_down_total_up", "fe_down_ionic_up", "fe_down_electronic_up"]
PATHS = ["latent_default", "comp_k4"]
MUT, GRID = "#6b7280", "#e5e7eb"

# finals[(path, scen, mode, ord, k)] = (20, 94) final weights; seed0 = (20, 94) seed weights
finals: dict = {}
seed0 = None
for mode in ("ws", "ft"):
    for o in range(3):
        for k in range(1, 22):
            for scen in SCENARIOS:
                for path in PATHS:
                    p = MIRROR / f"{mode}{TAG}_o{o}/k{k:02d}/inverse3/{scen}/trajectories/{path}.npz"
                    if not p.exists():
                        continue
                    z = np.load(p, allow_pickle=False)
                    w = z["weights"]
                    if w.size == 0:
                        continue
                    finals[(path, scen, mode, o, k)] = w[-1]
                    if seed0 is None:
                        seed0 = w[0]


def pair_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-seed L1 between two (20, 94) ensembles (same seed index compared)."""
    return float(np.abs(a - b).sum(axis=1).mean())


# per path/k: aggregate each comparison over scenarios (+ orders/modes as applicable)
series: dict = defaultdict(lambda: defaultdict(list))  # (path, metric) -> {k: [values]}
for path in PATHS:
    for scen in SCENARIOS:
        for mode in ("ws", "ft"):
            for o in range(3):
                for k in range(1, 22):
                    f = finals.get((path, scen, mode, o, k))
                    if f is None:
                        continue
                    series[(path, "disp")][k].append(pair_l1(f, seed0))
                    nxt = finals.get((path, scen, mode, o, k + 1))
                    if nxt is not None:
                        series[(path, "dk")][k].append(pair_l1(f, nxt))
        for k in range(1, 22):
            for o in range(3):
                a, b = finals.get((path, scen, "ws", o, k)), finals.get((path, scen, "ft", o, k))
                if a is not None and b is not None:
                    series[(path, "mode")][k].append(pair_l1(a, b))
            for mode in ("ws", "ft"):
                for o1, o2 in combinations(range(3), 2):
                    a, b = finals.get((path, scen, mode, o1, k)), finals.get((path, scen, mode, o2, k))
                    if a is not None and b is not None:
                        series[(path, "order")][k].append(pair_l1(a, b))

METRICS = [
    ("disp", "#9ca3af", "--", "|final − seed| (displacement scale)"),
    ("dk", "#0077BB", "-", "Δk: consecutive checkpoints (k vs k+1)"),
    ("mode", "#EE7733", "-", "ws vs ft (same order, same k)"),
    ("order", "#55A868", "-", "Δorder: across task orders (same mode, k)"),
]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), dpi=150)
print(f"mean per-seed L1 distances between FINAL candidate ensembles (replay n={REPLAY_N}):")
for ax, path in zip(axes, PATHS):
    for key, color, ls, _ in METRICS:
        s = series[(path, key)]
        ks = sorted(s)
        if ks:
            ax.plot(ks, [float(np.mean(s[k])) for k in ks], color=color, ls=ls, lw=2.0)
    row = {key: float(np.mean([v for vals in series[(path, key)].values() for v in vals])) for key, *_ in METRICS}
    print(f"  {path:16} disp={row['disp']:.3f}  Δk={row['dk']:.3f}  ws-vs-ft={row['mode']:.3f}  Δorder={row['order']:.3f}")
    ax.set_title(f"{path}", fontsize=10.5)
    ax.set_xlabel("k")
    ax.set_ylabel("mean per-seed L1 (0 = identical, 2 = disjoint)", fontsize=9)
    ax.set_xticks([1, 5, 9, 13, 17, 21])
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(colors=MUT, labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.suptitle(
    f"Checkpoint / mode / order fingerprint on designed candidates (replay n={REPLAY_N}; identical seeds everywhere)",
    fontsize=12, y=1.03,
)
fig.legend(
    handles=[Line2D([], [], color=c, ls=ls, lw=2, label=la) for _, c, ls, la in METRICS],
    loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.97),
)
fig.tight_layout(rect=(0, 0, 1, 0.88))
out = HERE / f"candidate_regularity_n{REPLAY_N}.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
