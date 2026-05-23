# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Paper-grade comparison of inverse-design methods on a single trained checkpoint.

Orchestrates a full sweep that ``eval_inverse_methods`` can do piecewise, and writes everything
(the model checkpoint, the seed list, the raw per-seed JSON, and the figures) into one folder
ready to drop into a paper draft. Reuses the per-method helpers from
``eval_inverse_methods`` so the methodology is identical.

The study covers:

* **Latent method** with AE-cycle weight λ ∈ {0, 0.1, 0.5, 1, 2, 5}.
* **Composition method** (differentiable KMD) under five configurations chosen to expose how
  ``seed_blend``, the element whitelist, and seeding strategy affect novelty / diversity:
    1. ``seed_blend = 1.0`` — strict seed init (the original behaviour, baseline for "no new
       elements can enter the support set");
    2. ``seed_blend = 0.95`` — new default; non-seed-element logits become reachable by Adam,
       letting the optimiser introduce elements outside the seed when helpful;
    3. (2) + ``allowed_elements`` restricted to a feasible alloy palette;
    4. (3) + ``entropy_weight`` (formerly ``sparsity_weight``) to softly prefer few-element
       formulas;
    5. Random initialisation (``initial_weights=None``, ``n_starts=B``) — completely free
       exploration, no seed bias at all (Scheme D control).

    python -m foundation_model.scripts.paper_inverse_comparison \\
        --config-file samples/continual_rehearsal_demo_config_inverse_baseline.toml \\
        --checkpoint artifacts/inverse_heads_finetuned/final_model.pt \\
        --output-dir artifacts/paper_inverse_design
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import seed_everything
from loguru import logger

from foundation_model.scripts.continual_rehearsal_demo import (
    QC_CLASSES,
    ContinualRehearsalConfig,
    ContinualRehearsalRunner,
)
from foundation_model.scripts.eval_inverse_methods import (
    _format_weights,
    _qc_prob,
    _reg_preds,
    _run_latent_method,
    _seed_weights_from_compositions,
)

# Default feasible alloy palette for the constrained-composition runs. These are the metals most
# commonly used to form quasicrystals experimentally; the model is free to pick any blend within
# this set while exploration of e.g. lanthanides / actinides is suppressed.
DEFAULT_ALLOY_PALETTE = ["Mg", "Al", "Cu", "Ni", "Zn", "Ag", "Pd", "Co", "Fe", "Re", "Ga", "In"]

# Composition-method configurations. Each row produces one bar in the comparison plot. The first
# two isolate the seed_blend effect; the next two layer on element constraints; the last drops the
# seed entirely (random init) as the no-seed-bias control (Scheme D).
COMPOSITION_CONFIGS: list[dict[str, Any]] = [
    {"label": "comp\n(strict seed)", "init": "seed", "blend": 1.0, "allowed": "all", "scale": 1.0, "entropy": 0.0},
    {"label": "comp\n(blended seed)", "init": "seed", "blend": 0.95, "allowed": "all", "scale": 1.0, "entropy": 0.0},
    {
        "label": "comp\n(alloy palette)",
        "init": "seed",
        "blend": 0.95,
        "allowed": DEFAULT_ALLOY_PALETTE,
        "scale": 1.0,
        "entropy": 0.0,
    },
    {
        "label": "comp\n(alloy + entropy)",
        "init": "seed",
        "blend": 0.95,
        "allowed": DEFAULT_ALLOY_PALETTE,
        "scale": 1.0,
        "entropy": 0.5,
    },
    {"label": "comp\n(random init)", "init": "random", "blend": 0.95, "allowed": "all", "scale": 1.0, "entropy": 0.0},
]
LATENT_CYCLE_WEIGHTS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]


def _plot_comparison(results: list[dict[str, Any]], reg_targets: dict[str, float], out_path: Path) -> None:
    """Three-panel comparison: QC probability + each regression target across all methods."""
    n_panels = 1 + len(reg_targets)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 5.6), squeeze=False)
    axes = axes[0]
    # Single-line labels so rotated x-ticks don't collide.
    labels = [r["label"].replace("\n", " ") for r in results]
    colors = ["#55A868" if r["method"] == "latent" else "#2563EB" for r in results]
    x = np.arange(len(results))

    def _set_xticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    # Panel 1: QC probability.
    qc_means = [float(np.mean(r["qc_after_decode"])) for r in results]
    qc_stds = [float(np.std(r["qc_after_decode"])) for r in results]
    axes[0].bar(x, qc_means, yerr=qc_stds, color=colors, capsize=3)
    axes[0].axhline(1.0, color="#C44E52", ls="--", lw=1.4, label="target = 1.0")
    _set_xticks(axes[0])
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("P(quasicrystal)")
    axes[0].set_title("Quasicrystal Probability (primary)")
    axes[0].legend(fontsize=9, loc="lower right")

    # Remaining panels: regression targets.
    for ax, (t, tgt) in zip(axes[1:], reg_targets.items()):
        means = [float(np.mean(r["reg_after_decode"][t])) for r in results]
        stds = [float(np.std(r["reg_after_decode"][t])) for r in results]
        ax.bar(x, means, yerr=stds, color=colors, capsize=3)
        ax.axhline(tgt, color="#C44E52", ls="--", lw=1.4, label=f"target = {tgt:+.1f}")
        _set_xticks(ax)
        ax.set_ylabel("Predicted value")
        ax.set_title(t)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Inverse-design comparison: latent (cycle sweep) vs differentiable KMD (configs)", y=1.00)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote comparison plot to {out_path}")


def _summarise(results: list[dict[str, Any]], reg_targets: dict[str, float]) -> list[dict[str, Any]]:
    summary = []
    for r in results:
        row = {
            "label": r["label"].replace("\n", " "),
            "method": r["method"],
            "cycle_weight": r.get("cycle_weight"),
            "config": r.get("config"),
            "elapsed_s": round(r["elapsed_s"], 2),
            "qc_after_mean": round(float(np.mean(r["qc_after_decode"])), 4),
            "qc_after_std": round(float(np.std(r["qc_after_decode"])), 4),
        }
        for t in reg_targets:
            row[f"{t}_after_mean"] = round(float(np.mean(r["reg_after_decode"][t])), 3)
            row[f"{t}_after_std"] = round(float(np.std(r["reg_after_decode"][t])), 3)
        summary.append(row)
    return summary


def run(config: ContinualRehearsalConfig, ckpt_path: Path) -> None:
    seed_everything(config.random_seed, workers=True)
    runner = ContinualRehearsalRunner(config)

    # Load the trained model exactly as we built it during training (same task_sequence).
    model = runner._build_full_model()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy the checkpoint so this folder is a self-contained paper artefact (skip when
    # the source and destination resolve to the same file — happens on idempotent reruns).
    dst = out_dir / "final_model.pt"
    if ckpt_path.resolve() != dst.resolve():
        shutil.copy2(ckpt_path, dst)

    device = next(model.parameters()).device

    def _qc_prob_fn(x: torch.Tensor) -> np.ndarray:
        return _qc_prob(model, x)

    seeds = runner._select_seeds(model, device, _qc_prob_fn)
    if not seeds:
        raise RuntimeError("No seed compositions selected.")
    x_seed, seeds = runner._descriptor_tensor(seeds, device)
    (out_dir / "seeds.json").write_text(json.dumps({"seeds": list(seeds)}, indent=2), encoding="utf-8")
    logger.info(f"Selected {len(seeds)} seed compositions (saved to seeds.json)")

    reg_targets = {t: v for t, v in zip(config.inverse_reg_tasks, config.inverse_reg_targets)}
    results: list[dict[str, Any]] = []

    # Latent method: cycle weight sweep.
    for lam in LATENT_CYCLE_WEIGHTS:
        logger.info(f"--- Latent method, ae_cycle_weight = {lam} ---")
        r = _run_latent_method(
            runner,
            model,
            seeds,
            x_seed,
            reg_targets,
            class_weight=config.inverse_class_weight,
            cycle_weight=lam,
            steps=config.inverse_steps,
            lr=config.inverse_lr,
        )
        r["label"] = f"latent\nλ={lam:g}"
        r["config"] = {"ae_cycle_weight": lam}
        results.append(r)

    # Composition method: walk through the configuration matrix.
    for cfg in COMPOSITION_CONFIGS:
        logger.info(f"--- {cfg['label'].replace(chr(10), ' ')} ---")
        r = _run_composition_config(
            runner,
            model,
            seeds,
            reg_targets,
            class_weight=config.inverse_class_weight,
            steps=config.inverse_steps,
            lr=config.inverse_lr,
            cfg=cfg,
        )
        r["label"] = cfg["label"]
        r["config"] = {k: cfg[k] for k in ("init", "blend", "allowed", "scale", "entropy")}
        results.append(r)

    summary = _summarise(results, reg_targets)
    logger.info("=== Summary ===")
    for row in summary:
        logger.info(row)

    (out_dir / "results.json").write_text(
        json.dumps({"reg_targets": reg_targets, "results": results, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    _plot_comparison(results, reg_targets, out_dir / "comparison.png")
    _write_readme(out_dir, summary, reg_targets, ckpt_path)
    logger.info(f"Paper materials written to {out_dir}")


def _run_composition_config(
    runner: ContinualRehearsalRunner,
    model,
    seeds: list[str],
    reg_targets: dict[str, float],
    *,
    class_weight: float,
    steps: int,
    lr: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run :meth:`optimize_composition` under one config row (handles seed/random init both)."""
    import time

    from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    kernel = runner._kmd.kernel_torch(device=device, dtype=dtype)

    if cfg["init"] == "seed":
        w_seed = _seed_weights_from_compositions(seeds, n_components=len(DEFAULT_ELEMENTS))
        init_kwargs = {"initial_weights": w_seed, "seed_blend": cfg["blend"]}
    elif cfg["init"] == "random":
        # n_starts matches the seed count so per-row aggregation lines up with the latent runs.
        init_kwargs = {"initial_weights": None, "n_starts": len(seeds)}
    else:
        raise ValueError(f"Unknown init mode in config: {cfg['init']!r}")

    t0 = time.perf_counter()
    res = model.optimize_composition(
        kernel,
        task_targets=reg_targets,
        class_targets={"material_type": QC_CLASSES},
        class_target_weight=class_weight,
        entropy_weight=cfg["entropy"],
        allowed_elements=cfg["allowed"],
        element_step_scale=cfg["scale"],
        steps=steps,
        lr=lr,
        **init_kwargs,
    )
    elapsed = time.perf_counter() - t0

    reg_names = list(reg_targets)
    optimized_desc = res.optimized_descriptor
    return {
        "method": "composition",
        "cycle_weight": None,
        "elapsed_s": elapsed,
        # For random init the "seeds" entry is informational only — there's no per-row correspondence.
        "seeds": list(seeds) if cfg["init"] == "seed" else [f"random_start_{i}" for i in range(len(seeds))],
        "qc_after_decode": _qc_prob(model, optimized_desc).tolist(),
        "reg_achieved_latent": {t: res.optimized_target.cpu().numpy()[:, j].tolist() for j, t in enumerate(reg_names)},
        "reg_after_decode": {t: _reg_preds(model, optimized_desc, [t])[t].tolist() for t in reg_names},
        "decoded_composition": _format_weights(res.optimized_weights.cpu().numpy()),
    }


def _write_readme(out_dir: Path, summary: list[dict[str, Any]], reg_targets: dict[str, float], ckpt_path: Path) -> None:
    lines = [
        "# Inverse-design method comparison — paper materials",
        "",
        f"Trained model: `final_model.pt` (copied from `{ckpt_path}`).",
        "Seed compositions: top-QC training compositions, listed in `seeds.json`.",
        f"Targets: QC probability → 1.0; {', '.join(f'{t} → {v:+.1f}' for t, v in reg_targets.items())}.",
        "",
        "Raw per-seed JSON: `results.json` (one entry per method+config).",
        "Comparison figure: `comparison.png`.",
        "",
        "## Summary (mean ± std across seeds)",
        "",
        "| label | QC after | " + " | ".join(f"{t} after" for t in reg_targets) + " | elapsed (s) |",
        "| --- | --- | " + " | ".join("---" for _ in reg_targets) + " | --- |",
    ]
    for row in summary:
        qc_cell = f"{row['qc_after_mean']:.3f} ± {row['qc_after_std']:.3f}"
        reg_cells = [f"{row[f'{t}_after_mean']:+.2f} ± {row[f'{t}_after_std']:.2f}" for t in reg_targets]
        lines.append(f"| {row['label']} | {qc_cell} | " + " | ".join(reg_cells) + f" | {row['elapsed_s']} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> tuple[ContinualRehearsalConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Paper-grade inverse-design comparison.")
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    import tomllib

    data = tomllib.loads(args.config_file.read_text(encoding="utf-8"))
    data["output_dir"] = str(args.output_dir)
    field_names = set(ContinualRehearsalConfig.__dataclass_fields__)
    path_fields = {
        "qc_data_path",
        "qc_preprocessing_path",
        "superconductor_path",
        "magnetic_path",
        "phonix_path",
        "output_dir",
    }
    kwargs: dict[str, object] = {}
    for key, value in data.items():
        if key not in field_names:
            continue
        kwargs[key] = Path(value) if key in path_fields and value is not None else value
    return ContinualRehearsalConfig(**kwargs), args


def main(argv: list[str] | None = None) -> None:
    config, args = _parse_args(argv)
    run(config, args.checkpoint)


if __name__ == "__main__":
    main()
