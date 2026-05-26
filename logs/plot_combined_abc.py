"""Visualise the A+B+C comprehensive evaluation as a comparison bar chart."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import seed_everything

from foundation_model.scripts.continual_rehearsal_demo import (
    QC_CLASSES,
    ContinualRehearsalConfig,
    ContinualRehearsalRunner,
)
from foundation_model.scripts.eval_inverse_methods import _qc_prob, _seed_weights_from_compositions

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "samples/continual_rehearsal_demo_config_inverse_baseline.toml"
CKPT = REPO / "artifacts/inverse_design_run/finetune/final_model.pt"
SCENARIO = "scenario1_fe_down_magnetic_up"
OUT_PNG = REPO / "logs/combined_abc_comparison.png"


def _build():
    raw = tomllib.loads(CFG_PATH.read_text(encoding="utf-8"))
    scenarios = raw.pop("inverse_scenarios", [])
    sc = next(s for s in scenarios if s["name"] == SCENARIO)
    raw["inverse_reg_tasks"] = sc["reg_tasks"]
    raw["inverse_reg_targets"] = sc["reg_targets"]
    config = ContinualRehearsalConfig(**raw)
    seed_everything(config.random_seed, workers=True)
    runner = ContinualRehearsalRunner(config)
    model = runner._build_full_model()
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    kernel = runner._kmd.kernel_torch(device=device, dtype=dtype)
    return runner, model, kernel, device


def main() -> None:
    runner, model, kernel, device = _build()
    def _qc_fn(x): return _qc_prob(model, x)
    seeds = runner._select_seeds(model, device, _qc_fn)[:8]
    w_seed = _seed_weights_from_compositions(seeds, n_components=kernel.shape[0])
    common = dict(
        task_targets={"formation_energy": -2.0, "magnetization": 2.0},
        class_targets={"material_type": QC_CLASSES},
        class_target_weight=5.0,
        initial_weights=w_seed,
        seed_blend=0.95,
        steps=300,
        lr=0.05,
    )

    configs = [
        ("baseline", {}),
        ("A: K=3", {"max_elements": 3}),
        ("B: fix Au=.65 Ga=.20", {"fixed_amounts": {"Au": 0.65, "Ga": 0.20}}),
        ("C: floor=.10", {"min_nonzero_weight": 0.10}),
        ("A+B: K=4 + fix", {"max_elements": 4, "fixed_amounts": {"Au": 0.65, "Ga": 0.20}}),
        ("A+C: K=5 + floor", {"max_elements": 5, "min_nonzero_weight": 0.10}),
        ("B+C: fix Au=.30 Ga=.20 + floor=.10",
         {"fixed_amounts": {"Au": 0.30, "Ga": 0.20}, "min_nonzero_weight": 0.10}),
        ("A+B+C: K=4 + fix + floor",
         {"max_elements": 4, "fixed_amounts": {"Au": 0.30, "Ga": 0.20}, "min_nonzero_weight": 0.10}),
        ("A+B+C, scale=0.8",
         {"max_elements": 4, "fixed_amounts": {"Au": 0.30, "Ga": 0.20},
          "min_nonzero_weight": 0.10, "annealing_scale": 0.8}),
    ]

    results = []
    for label, extras in configs:
        torch.manual_seed(0)
        res = model.optimize_composition(kernel, **common, **extras)
        w = res.optimized_weights
        results.append({
            "label": label,
            "fe_mean": float(res.optimized_target[:, 0].mean()),
            "fe_std": float(res.optimized_target[:, 0].std()),
            "mag_mean": float(res.optimized_target[:, 1].mean()),
            "mag_std": float(res.optimized_target[:, 1].std()),
            "qc": float(_qc_prob(model, res.optimized_descriptor).mean()),
            "nz_mean": float((w > 1e-6).sum(dim=-1).float().mean()),
        })

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), squeeze=False)
    axes = axes[0]
    labels = [r["label"] for r in results]
    colors = ["#888"] + ["#2563EB"] * (len(results) - 1)
    x = np.arange(len(results))

    # Panel 1: FE
    ax = axes[0]
    fe_means = [r["fe_mean"] for r in results]
    fe_stds = [r["fe_std"] for r in results]
    ax.bar(x, fe_means, yerr=fe_stds, color=colors, capsize=4)
    ax.axhline(-2.0, color="red", linestyle="--", label="target -2.0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("formation_energy")
    ax.set_title("Formation energy (↓)")
    ax.legend()

    # Panel 2: Mag
    ax = axes[1]
    mag_means = [r["mag_mean"] for r in results]
    mag_stds = [r["mag_std"] for r in results]
    ax.bar(x, mag_means, yerr=mag_stds, color=colors, capsize=4)
    ax.axhline(2.0, color="red", linestyle="--", label="target +2.0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("magnetization")
    ax.set_title("Magnetization (↑)")
    ax.legend()

    # Panel 3: QC
    ax = axes[2]
    qcs = [r["qc"] for r in results]
    ax.bar(x, qcs, color=colors)
    ax.axhline(1.0, color="red", linestyle="--", label="target 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P(material_type ∈ QC)")
    ax.set_ylim(0, 1)
    ax.set_title("QC probability (↑)")
    ax.legend()

    # Panel 4: nz
    ax = axes[3]
    nzs = [r["nz_mean"] for r in results]
    ax.bar(x, nzs, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean # non-zero elements")
    ax.set_title("Recipe complexity (lower = simpler)")
    ax.set_yscale("symlog")

    fig.suptitle(
        "A (max_elements) + B (fixed_amounts) + C (min_nonzero_weight) — combined evaluation\n"
        "scenario1 (FE↓, Mag↑) · 8 seeded starts · 300 steps",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    print(f"saved: {OUT_PNG}")
    # also dump JSON for replay
    (REPO / "logs/combined_abc_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
