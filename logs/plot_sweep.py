"""Visualise sweep_tau_schedule_results.json as heatmaps + scatter."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "logs/sweep_tau_schedule_results.json"
OUT_PNG = REPO / "logs/sweep_tau_schedule.png"

TAU_STARTS = [1.0, 2.0, 5.0, 10.0, 20.0]
SCHEDULES = ["geometric", "linear", "cosine"]


def _target_distance(achieved: dict, targets: dict) -> float:
    """Mean |achieved - target| across regression objectives. Lower is better."""
    return float(np.mean([abs(achieved[t]["mean"] - v) for t, v in targets.items()]))


def main() -> None:
    data = json.loads(RESULTS.read_text())
    target_sets = {
        "2T": {"formation_energy": -2.0, "magnetization": 2.0},
        "1T": {"magnetization": 2.0},
    }
    Ks = [3, 5]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), squeeze=False)

    # Top row: target-distance heatmaps (lower is better, blue = good)
    # Bottom row: QC-probability heatmaps (higher is better, blue = good)
    for col, (tset_name, tgt) in enumerate(target_sets.items()):
        for k_idx, K in enumerate(Ks):
            col_idx = col * 2 + k_idx
            # Build (5, 3) grid: τ_start × schedule
            dist_grid = np.full((len(TAU_STARTS), len(SCHEDULES)), np.nan)
            qc_grid = np.full((len(TAU_STARTS), len(SCHEDULES)), np.nan)
            for r in data:
                if r["target_set"] != tset_name or r["K"] != K:
                    continue
                if r["schedule"] not in SCHEDULES or r["tau_start"] not in TAU_STARTS:
                    continue
                i = TAU_STARTS.index(r["tau_start"])
                j = SCHEDULES.index(r["schedule"])
                dist_grid[i, j] = _target_distance(r["achieved"], tgt)
                qc_grid[i, j] = r["qc_after"]

            # Get baselines for annotation
            base = next(r for r in data if r["target_set"] == tset_name and r["schedule"] == "baseline")
            base_dist = _target_distance(base["achieved"], tgt)
            base_qc = base["qc_after"]
            const_hi = next(r for r in data if r["target_set"] == tset_name
                            and r["K"] == K and r["schedule"] == "const_t1.0")
            post_dist = _target_distance(const_hi["achieved"], tgt)
            post_qc = const_hi["qc_after"]

            # Top row: target distance
            ax = axes[0, col_idx]
            im = ax.imshow(dist_grid, aspect="auto", cmap="RdYlBu_r", origin="lower")
            ax.set_xticks(range(len(SCHEDULES)))
            ax.set_xticklabels(SCHEDULES)
            ax.set_yticks(range(len(TAU_STARTS)))
            ax.set_yticklabels([str(t) for t in TAU_STARTS])
            ax.set_xlabel("schedule")
            ax.set_ylabel("τ_start")
            ax.set_title(f"{tset_name}  K={K}\ntarget-dist  (baseline={base_dist:.2f}, post-hoc={post_dist:.2f})")
            for i in range(len(TAU_STARTS)):
                for j in range(len(SCHEDULES)):
                    val = dist_grid[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val > np.nanmedian(dist_grid) else "black", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Bottom row: QC
            ax = axes[1, col_idx]
            im = ax.imshow(qc_grid, aspect="auto", cmap="RdYlBu", origin="lower", vmin=0.5, vmax=1.0)
            ax.set_xticks(range(len(SCHEDULES)))
            ax.set_xticklabels(SCHEDULES)
            ax.set_yticks(range(len(TAU_STARTS)))
            ax.set_yticklabels([str(t) for t in TAU_STARTS])
            ax.set_xlabel("schedule")
            ax.set_ylabel("τ_start")
            ax.set_title(f"{tset_name}  K={K}\nQC prob  (baseline={base_qc:.2f}, post-hoc={post_qc:.2f})")
            for i in range(len(TAU_STARTS)):
                for j in range(len(SCHEDULES)):
                    val = qc_grid[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val < 0.75 else "black", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "max_elements sweep: τ_start × schedule × K × target_set\n"
        "top row: |achieved − target| averaged across regression objectives (LOWER is better, blue)\n"
        "bottom row: P(material_type ∈ QC_classes) (HIGHER is better, blue)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
