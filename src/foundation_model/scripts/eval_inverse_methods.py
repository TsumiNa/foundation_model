# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Compare two inverse-design methods on a single trained checkpoint.

Method A — latent-space optimisation with cycle-consistency
    optimize_latent(optimize_space="latent", class_target_weight=…, cycle_consistency_weight=λ).
    The optimised latent is decoded back to a descriptor through the AE; the heads' values at
    the **decoded** descriptor are reported (so "round-trip drift" is the key failure mode and
    cycle-consistency is the proposed mitigation, swept over λ).

Method B — composition-space optimisation via differentiable KMD
    optimize_composition(kmd_kernel, class_target_weight=…). The optimisation variable IS the
    element-weight recipe ``w``; descriptor is ``w @ K``; there is no AE in the loop.

Both methods run on the **same model**, **same seed compositions**, and **same targets** so the
two columns are directly comparable. Output is a JSON summary + a comparison PNG.

This script is independent of the rehearsal demo — its own CLI, own output dir, no rehearsal.

    python -m foundation_model.scripts.eval_inverse_methods \\
        --config-file samples/continual_rehearsal_demo_config_inverse_baseline.toml \\
        --checkpoint artifacts/inverse_heads_finetuned/final_model.pt \\
        --output-dir artifacts/inverse_methods_eval \\
        --cycle-weights 0,0.1,0.5,1,2,5
"""

from __future__ import annotations

import argparse
import json
import time
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
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, formula_to_composition


# --- Helpers ------------------------------------------------------------------


def _qc_prob(model, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        h = torch.tanh(model.encoder(x))
        probs = torch.softmax(model.task_heads["material_type"](h), dim=-1)
        return probs[:, QC_CLASSES].sum(dim=-1).cpu().numpy()


def _reg_preds(model, x: torch.Tensor, tasks: list[str]) -> dict[str, np.ndarray]:
    with torch.no_grad():
        h = torch.tanh(model.encoder(x))
        return {t: model.task_heads[t](h).squeeze(-1).cpu().numpy() for t in tasks}


def _seed_weights_from_compositions(seeds: list[str], n_components: int) -> torch.Tensor:
    """Element-weight tensor (B, n_components) for ``optimize_composition`` seeding."""
    rows = []
    for c in seeds:
        w = formula_to_composition(c)
        if w is None:
            raise ValueError(f"Cannot parse seed composition '{c}' to element weights.")
        rows.append(np.asarray(w, dtype=np.float64))
    return torch.tensor(np.stack(rows), dtype=torch.float64)


def _decode_latent_path(kmd, descriptors: np.ndarray) -> list[str]:
    """Latent path's composition output: AE-decoded descriptor → KMD.inverse → formula string."""
    try:
        weights = kmd.inverse(descriptors)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"KMD.inverse failed ({exc}); skipping composition decoding.")
        return ["<undecodable>"] * descriptors.shape[0]
    return _format_weights(weights)


def _format_weights(weights: np.ndarray, top_k: int = 6, eps: float = 1e-3) -> list[str]:
    """Render element-weight rows as compact formula strings (top-K elements above ``eps``)."""
    out: list[str] = []
    for row in weights:
        order = np.argsort(row)[::-1]
        parts = [f"{DEFAULT_ELEMENTS[i]}{row[i]:.3f}" for i in order[:top_k] if row[i] > eps]
        out.append(" ".join(parts) if parts else "<empty>")
    return out


# --- Methods ------------------------------------------------------------------


def _run_latent_method(
    runner: ContinualRehearsalRunner,
    model,
    seeds: list[str],
    x_seed: torch.Tensor,
    reg_targets: dict[str, float],
    class_weight: float,
    cycle_weight: float,
    steps: int,
    lr: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    t0 = time.perf_counter()
    res = model.optimize_latent(
        initial_input=x_seed,
        task_targets=reg_targets,
        class_targets={"material_type": QC_CLASSES},
        class_target_weight=class_weight,
        cycle_consistency_weight=cycle_weight,
        optimize_space="latent",
        steps=steps,
        lr=lr,
    )
    elapsed = time.perf_counter() - t0

    reg_names = list(reg_targets.keys())
    achieved_latent = res.optimized_target[:, 0, :].cpu().numpy()  # (B, T) in reg_targets order
    optimized_desc = res.optimized_input[:, 0, :]  # (B, x_dim) — AE-decoded descriptor
    after_qc = _qc_prob(model, optimized_desc)
    after_reg = _reg_preds(model, optimized_desc, reg_names)
    decoded = _decode_latent_path(runner._kmd, optimized_desc.detach().cpu().numpy())

    return {
        "method": "latent",
        "cycle_weight": cycle_weight,
        "elapsed_s": elapsed,
        "seeds": list(seeds),
        "qc_after_decode": after_qc.tolist(),
        "reg_achieved_latent": {t: achieved_latent[:, j].tolist() for j, t in enumerate(reg_names)},
        "reg_after_decode": {t: after_reg[t].tolist() for t in reg_names},
        "decoded_composition": decoded,
    }


def _run_composition_method(
    runner: ContinualRehearsalRunner,
    model,
    seeds: list[str],
    reg_targets: dict[str, float],
    class_weight: float,
    steps: int,
    lr: float,
    allowed_elements: torch.Tensor | None = None,
    element_step_scale: torch.Tensor | None = None,
) -> dict[str, Any]:
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    kernel = runner._kmd.kernel_torch(device=device, dtype=dtype)
    w_seed = _seed_weights_from_compositions(seeds, n_components=len(DEFAULT_ELEMENTS))

    t0 = time.perf_counter()
    res = model.optimize_composition(
        kernel,
        initial_weights=w_seed,
        task_targets=reg_targets,
        class_targets={"material_type": QC_CLASSES},
        class_target_weight=class_weight,
        allowed_elements=allowed_elements,
        element_step_scale=element_step_scale,
        steps=steps,
        lr=lr,
    )
    elapsed = time.perf_counter() - t0

    reg_names = list(reg_targets.keys())
    achieved = res.optimized_target.cpu().numpy()  # (B, T)
    optimized_desc = res.optimized_descriptor  # (B, x_dim) — w @ K, no decode
    final_qc = _qc_prob(model, optimized_desc)
    final_reg = _reg_preds(model, optimized_desc, reg_names)
    w_final = res.optimized_weights.cpu().numpy()

    return {
        "method": "composition",
        "cycle_weight": None,
        "elapsed_s": elapsed,
        "seeds": list(seeds),
        # In composition space there is no "after-decode" drift — the model values AT the optimised
        # ``w`` are the same as at the descriptor ``w @ K``. We still report both for symmetry.
        "qc_after_decode": final_qc.tolist(),
        "reg_achieved_latent": {t: achieved[:, j].tolist() for j, t in enumerate(reg_names)},
        "reg_after_decode": {t: final_reg[t].tolist() for t in reg_names},
        "decoded_composition": _format_weights(w_final),
    }


# --- Plot ---------------------------------------------------------------------


def _plot_summary(results: list[dict[str, Any]], reg_targets: dict[str, float], out_path: Path) -> None:
    """Side-by-side: QC prob and each regression target across methods (mean ± seeds)."""
    fig, axes = plt.subplots(1, 1 + len(reg_targets), figsize=(4.6 * (1 + len(reg_targets)), 4.2), squeeze=False)
    axes = axes[0]
    labels = [f"latent (λ={r['cycle_weight']})" if r["method"] == "latent" else "composition" for r in results]

    # QC probability
    qc_means = [float(np.mean(r["qc_after_decode"])) for r in results]
    qc_stds = [float(np.std(r["qc_after_decode"])) for r in results]
    x = np.arange(len(results))
    axes[0].bar(x, qc_means, yerr=qc_stds, color="#55A868", capsize=3)
    axes[0].axhline(1.0, color="#C44E52", ls="--", lw=1.4, label="target = 1.0")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("P(quasicrystal)")
    axes[0].set_title("Quasicrystal Probability (primary)")
    axes[0].legend(fontsize=9, loc="lower right")

    for ax, (t, tgt) in zip(axes[1:], reg_targets.items()):
        means = [float(np.mean(r["reg_after_decode"][t])) for r in results]
        stds = [float(np.std(r["reg_after_decode"][t])) for r in results]
        ax.bar(x, means, yerr=stds, color="#4C72B0", capsize=3)
        ax.axhline(tgt, color="#C44E52", ls="--", lw=1.4, label=f"target = {tgt:+.1f}")
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel("Predicted value")
        ax.set_title(f"{t}")
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Inverse-design methods compared (same model, same seeds, same targets)", y=1.04)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Main flow ----------------------------------------------------------------


def _resolve_element_constraints(
    allowed_syms: list[str] | None,
    locked_syms: list[str] | None,
    step_value: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Convert symbol lists to (allowed_elements bool mask, element_step_scale tensor)."""
    n = len(DEFAULT_ELEMENTS)
    sym_to_idx = {sym: i for i, sym in enumerate(DEFAULT_ELEMENTS)}

    def _to_idx(symbols: list[str]) -> list[int]:
        bad = [s for s in symbols if s not in sym_to_idx]
        if bad:
            raise ValueError(f"Unknown element symbol(s): {bad}. Valid: e.g. {DEFAULT_ELEMENTS[:8]}…")
        return [sym_to_idx[s] for s in symbols]

    allowed_mask = None
    if allowed_syms:
        allowed_mask = torch.zeros(n, dtype=torch.bool)
        allowed_mask[_to_idx(allowed_syms)] = True

    step_scale = None
    if locked_syms:
        step_scale = torch.ones(n)
        step_scale[_to_idx(locked_syms)] = step_value
    return allowed_mask, step_scale


def evaluate(
    config: ContinualRehearsalConfig,
    ckpt_path: Path,
    cycle_weights: list[float],
    allowed_elements: torch.Tensor | None = None,
    element_step_scale: torch.Tensor | None = None,
) -> None:
    seed_everything(config.random_seed, workers=True)
    runner = ContinualRehearsalRunner(config)
    model = runner._build_full_model()

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()

    # Deterministic seed compositions: same set for both methods. We reuse the demo's "top-QC
    # training composition" selector so this matches what users see from continual_rehearsal_demo.
    device = next(model.parameters()).device

    def _qc_prob_fn(x: torch.Tensor) -> np.ndarray:
        return _qc_prob(model, x)

    seeds = runner._select_seeds(model, device, _qc_prob_fn)
    if not seeds:
        raise RuntimeError("No seed compositions selected (check inverse_seed_strategy / data).")
    x_seed, seeds = runner._descriptor_tensor(seeds, device)
    logger.info(f"Selected {len(seeds)} seed compositions")

    reg_targets = {t: v for t, v in zip(config.inverse_reg_tasks, config.inverse_reg_targets)}

    results: list[dict[str, Any]] = []

    # Method A: latent-space, sweep cycle weight.
    for lam in cycle_weights:
        logger.info(f"--- Latent method, cycle_consistency_weight = {lam} ---")
        results.append(
            _run_latent_method(
                runner,
                model,
                seeds,
                x_seed,
                reg_targets,
                class_weight=config.inverse_class_weight,
                cycle_weight=float(lam),
                steps=config.inverse_steps,
                lr=config.inverse_lr,
            )
        )

    # Method B: differentiable KMD, single run (no λ). Element constraints (if any) only apply here.
    logger.info("--- Composition method (differentiable KMD) ---")
    if allowed_elements is not None:
        logger.info(f"  allowed_elements: {int(allowed_elements.sum())} of {len(DEFAULT_ELEMENTS)} elements")
    if element_step_scale is not None:
        locked = [DEFAULT_ELEMENTS[i] for i in (element_step_scale == 0).nonzero(as_tuple=True)[0].tolist()]
        if locked:
            logger.info(f"  locked elements (step_scale=0): {locked}")
    results.append(
        _run_composition_method(
            runner,
            model,
            seeds,
            reg_targets,
            class_weight=config.inverse_class_weight,
            steps=config.inverse_steps,
            lr=config.inverse_lr,
            allowed_elements=allowed_elements,
            element_step_scale=element_step_scale,
        )
    )

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compact human-readable summary alongside the full per-seed JSON.
    summary = []
    for r in results:
        row = {
            "label": f"latent λ={r['cycle_weight']}" if r["method"] == "latent" else "composition",
            "elapsed_s": round(r["elapsed_s"], 2),
            "qc_after_mean": round(float(np.mean(r["qc_after_decode"])), 4),
        }
        for t in reg_targets:
            row[f"{t}_after_mean"] = round(float(np.mean(r["reg_after_decode"][t])), 3)
        summary.append(row)
    logger.info("=== Summary ===")
    for row in summary:
        logger.info(row)

    (out_dir / "eval_inverse_methods.json").write_text(
        json.dumps({"reg_targets": reg_targets, "results": results, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    _plot_summary(results, reg_targets, out_dir / "eval_inverse_methods.png")
    logger.info(f"Wrote {out_dir / 'eval_inverse_methods.json'} and the comparison plot.")


def _parse_args(argv: list[str] | None = None) -> tuple[ContinualRehearsalConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Compare inverse-design methods on a trained checkpoint.")
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cycle-weights",
        type=str,
        default="0,0.1,0.5,1,2,5",
        help="Comma-separated λ values for cycle_consistency_weight in the latent method.",
    )
    parser.add_argument(
        "--allowed-elements",
        type=str,
        default="",
        help=(
            "Comma-separated element symbols the composition method is allowed to use (hard "
            "whitelist; e.g. 'Mg,Al,Cu,Ni,Zn,Ag'). Empty means every element allowed."
        ),
    )
    parser.add_argument(
        "--locked-elements",
        type=str,
        default="",
        help=(
            "Comma-separated element symbols whose composition weight is frozen at the seed "
            "value (sets element_step_scale to --locked-step-scale; default 0 = fully locked)."
        ),
    )
    parser.add_argument(
        "--locked-step-scale",
        type=float,
        default=0.0,
        help="Gradient multiplier for locked elements (0 = fully locked; 0.1 = slow drift).",
    )
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
    cycle_weights = [float(x) for x in args.cycle_weights.split(",") if x.strip()]
    allowed_syms = [s.strip() for s in args.allowed_elements.split(",") if s.strip()]
    locked_syms = [s.strip() for s in args.locked_elements.split(",") if s.strip()]
    allowed_mask, step_scale = _resolve_element_constraints(allowed_syms, locked_syms, args.locked_step_scale)
    evaluate(
        config,
        args.checkpoint,
        cycle_weights,
        allowed_elements=allowed_mask,
        element_step_scale=step_scale,
    )


if __name__ == "__main__":
    main()
