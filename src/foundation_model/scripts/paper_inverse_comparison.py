# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Paper-grade comparison of inverse-design methods on a single trained checkpoint.

Orchestrates a full sweep that ``eval_inverse_methods`` can do piecewise, and writes everything
(the model checkpoint, the seed list, the raw per-seed JSON, and the figures) into one folder
ready to drop into a paper draft. Reuses the per-method helpers from
``eval_inverse_methods`` so the methodology is identical.

The study covers:

* **Latent method** with AE-alignment scale α ∈ {0, 0.25, 1.0} — failure-mode baseline, a useful
  intermediate, and the [0, 1] upper bound. (Earlier runs swept finer; the three points are enough
  to show the qualitative plateau.)
* **Composition method** (differentiable KMD) under five configurations chosen to expose how
  ``seed_blend``, the element whitelist, and seeding strategy affect novelty / diversity. Labels
  follow a "describe the config in the label" convention:
    1. ``comp (seed)`` — ``seed_blend = 1.0`` (strict seed, support set frozen);
    2. ``comp (seed, 5% all)`` — ``seed_blend = 0.95`` (5 % uniform mixed in, all 94 elements
       reachable but no whitelist);
    3. ``comp (seed, 5% all, element list)`` — (2) + ``allowed_elements = ALLOY_PALETTE``;
    4. ``comp (seed, 5% all, element list, low diversity)`` — (3) + ``diversity_scale = 0`` so
       per-output entropy is penalised → peaky few-element recipes (ablation);
    5. ``comp (random)`` — ``initial_weights=None``, no seed bias.

    python -m foundation_model.scripts.paper_inverse_comparison \\
        --config-file samples/continual_rehearsal_demo_config_inverse_baseline.toml \\
        --checkpoint artifacts/inverse_heads_finetuned/final_model.pt \\
        --output-dir artifacts/paper_inverse_design
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
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

# Feasible alloy palette for the constrained-composition runs. Designed per the plan in
# docs/continual_rehearsal_full_PLAN.md §5: light alkaline-earth + group 13/14 + the full 4th/5th
# period transition metals (Tc excluded for radioactivity) + Au (needed for Au-Ga-RE seeds) +
# accessible lanthanides (Pm radioactive, Tm/Lu scarce). 41 symbols total — wide enough to expose
# multiple QC-prone basins, narrow enough to suppress Pu/F/Cs/Tm-style non-physical model bias.
DEFAULT_ALLOY_PALETTE = [
    "Mg",
    "Ca",
    "B",
    "Al",
    "Ga",
    "In",
    "Tl",
    "Si",
    "Ge",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Au",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Yb",
]
assert len(DEFAULT_ALLOY_PALETTE) == 41

# Composition-method configurations. Each row produces one bar in the comparison plot. The first
# two isolate the seed_blend effect; the next two layer on element constraints; the last drops the
# seed entirely (random init) as the no-seed-bias control (Scheme D).
COMPOSITION_CONFIGS: list[dict[str, Any]] = [
    # diversity = 1.0 = no entropy penalty (default user-facing behaviour).
    # Labels follow the "describe the config" convention: each comma-separated phrase names a
    # knob that's been turned on relative to the previous row.
    {"label": "comp\n(seed)", "init": "seed", "blend": 1.0, "allowed": "all", "scale": 1.0, "diversity": 1.0},
    {"label": "comp\n(seed, 5% all)", "init": "seed", "blend": 0.95, "allowed": "all", "scale": 1.0, "diversity": 1.0},
    {
        "label": "comp\n(seed, 5% all, element list)",
        "init": "seed",
        "blend": 0.95,
        "allowed": DEFAULT_ALLOY_PALETTE,
        "scale": 1.0,
        "diversity": 1.0,
    },
    {
        # Ablation: clamp diversity to 0 → max entropy penalty → forced peaky few-element recipes.
        "label": "comp\n(seed, 5% all,\nelement list, low diversity)",
        "init": "seed",
        "blend": 0.95,
        "allowed": DEFAULT_ALLOY_PALETTE,
        "scale": 1.0,
        "diversity": 0.0,
    },
    {"label": "comp\n(random)", "init": "random", "blend": 0.95, "allowed": "all", "scale": 1.0, "diversity": 1.0},
]
LATENT_ALIGN_SCALES = [0.0, 0.25, 1.0]  # ae_align_scale ∈ [0, 1] — three points: failure / mid / max


#: Per-task display title with units and a directional arrow that points the way the optimiser
#: should drive the value. Defaults applied for the two tasks the plan §5 scenarios use. The
#: lookup falls back to the raw task name if a task isn't in the map (so the plot still works
#: when scenarios 1 / 2 add ``magnetic_moment`` / ``tc``).
REG_TASK_TITLES: dict[str, str] = {
    "formation_energy": "Formation energy [eV/atom] ↓",
    "klat": "klat [W/mK] ↑",
    "magnetic_moment": "Magnetic moment [μB/f.u.] ↑",
    "tc": "Critical temperature [K] ↑",
}


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

    # Panel 1: QC probability. The arrow makes the optimisation direction explicit at a glance.
    qc_means = [float(np.mean(r["qc_after_decode"])) for r in results]
    qc_stds = [float(np.std(r["qc_after_decode"])) for r in results]
    axes[0].bar(x, qc_means, yerr=qc_stds, color=colors, capsize=3)
    axes[0].axhline(1.0, color="#C44E52", ls="--", lw=1.4, label="target = 1.0")
    _set_xticks(axes[0])
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("P(quasicrystal)")
    axes[0].set_title("P(quasicrystal) ↑")
    axes[0].legend(fontsize=9, loc="lower right")

    # Remaining panels: regression targets. Title pulled from REG_TASK_TITLES with the unit and
    # an arrow indicating whether the target is below (↓) or above (↑) the model's baseline.
    for ax, (t, tgt) in zip(axes[1:], reg_targets.items()):
        means = [float(np.mean(r["reg_after_decode"][t])) for r in results]
        stds = [float(np.std(r["reg_after_decode"][t])) for r in results]
        ax.bar(x, means, yerr=stds, color=colors, capsize=3)
        ax.axhline(tgt, color="#C44E52", ls="--", lw=1.4, label=f"target = {tgt:+.1f}")
        _set_xticks(ax)
        ax.set_ylabel("Predicted value")
        ax.set_title(REG_TASK_TITLES.get(t, t))
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Inverse-design comparison: latent (ae_align_scale sweep) vs differentiable KMD (configs)", y=1.00)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote comparison plot to {out_path}")


#: Discovered-element x-tick colour: bright orange. High contrast against the heatmap's Blues
#: cmap, and visually distinct from the project's #2563EB / #55A868 / #C44E52 palette so readers
#: don't have to re-map colour meaning. Synced with the matching helper in
#: ``continual_rehearsal_full.py``.
DISCOVERED_ELEMENT_COLOR = "#E67E22"

# Element-symbol grouping regex used both here and in seed parsing — capital + optional lowercase.
_COMP_RE = re.compile(r"([A-Z][a-z]?)([\d.]*)")


def _element_set(formula: str) -> frozenset[str]:
    """Set of element symbols in a composition string (ignoring stoichiometry)."""
    return frozenset(el for el, _ in _COMP_RE.findall(formula) if el)


def _plot_element_frequency_heatmap(
    results: list[dict[str, Any]],
    seeds: list[str],
    out_path: Path,
    *,
    top_k: int = 25,
) -> None:
    """Per-method × top-K-element occurrence heatmap.

    For each method we count how many of its B decoded recipes contain each element (i.e.
    ``element_symbol`` appears anywhere in the formatted ``decoded_composition`` string). The
    top ``top_k`` elements globally are shown as columns; methods are rows. Elements absent
    from every seed in ``seeds`` are highlighted on the x-axis as **bold orange** — the
    inverse-design *element-discovery* signal. No underline (visually noisy under tight
    rotated labels); bold + a distinct colour is enough.
    """
    n = len(results)
    labels = [r["label"].replace("\n", " ") for r in results]

    # Seed element multiplicity — used to decide which elements are "new" (0 in seeds).
    seed_cnt = Counter()
    for s in seeds:
        for el in _element_set(s):
            seed_cnt[el] += 1

    # Per-method element-presence counts.
    per_method = []
    for r in results:
        c = Counter()
        for d in r["decoded_composition"]:
            for el in _element_set(d):
                c[el] += 1
        per_method.append(c)

    # Globally top elements (rank by sum-of-top-8-per-method so single-method blow-ups don't
    # dominate). Matches the ranking the standalone post-hoc script used.
    global_cnt = Counter()
    for c in per_method:
        for el, k in c.most_common(8):
            global_cnt[el] += k
    top_elems = [e for e, _ in global_cnt.most_common(top_k)]
    if not top_elems:
        logger.warning("No elements found in decoded_composition; skipping heatmap.")
        return

    n_per_method = len(results[0]["decoded_composition"]) if results else 20
    mat = np.zeros((n, len(top_elems)), dtype=int)
    for i, c in enumerate(per_method):
        for j, el in enumerate(top_elems):
            mat[i, j] = c[el]

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=n_per_method)
    ax.set_xticks(range(len(top_elems)))
    ax.set_xticklabels(top_elems, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(
        f"Element appearance counts per method (top {len(top_elems)})\n"
        f"Bold orange element symbols = NOT in any of the {len(seeds)} seeds (introduced by the optimiser)",
        fontsize=11,
        pad=12,
    )
    # Bold + orange for discovered elements; everything else stays in the default style.
    for tick_label, el in zip(ax.get_xticklabels(), top_elems):
        if seed_cnt[el] == 0:
            tick_label.set_fontweight("bold")
            tick_label.set_color(DISCOVERED_ELEMENT_COLOR)
    # Cell annotations.
    for i in range(n):
        for j in range(len(top_elems)):
            if mat[i, j]:
                ax.text(
                    j,
                    i,
                    str(mat[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if mat[i, j] > n_per_method * 0.5 else "#333",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label(f"appearance count (out of {n_per_method} outputs)")
    # The shared demo style sets ``axes.grid = True`` globally, which on an ``imshow`` heatmap
    # draws grid lines through every cell centre (major ticks coincide with cell centres). Turn
    # the grid off here so the cells stay clean — matches what continual_rehearsal_full.py does.
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote element-frequency heatmap to {out_path}")


def _summarise(results: list[dict[str, Any]], reg_targets: dict[str, float]) -> list[dict[str, Any]]:
    summary = []
    for r in results:
        row = {
            "label": r["label"].replace("\n", " "),
            "method": r["method"],
            "align_scale": r.get("align_scale"),
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

    # Latent method: ae_align_scale sweep over [0, 1].
    for lam in LATENT_ALIGN_SCALES:
        logger.info(f"--- Latent method, ae_align_scale = {lam} ---")
        r = _run_latent_method(
            runner,
            model,
            seeds,
            x_seed,
            reg_targets,
            class_weight=config.inverse_class_weight,
            align_scale=lam,
            steps=config.inverse_steps,
            lr=config.inverse_lr,
        )
        r["label"] = f"latent\nα={lam:g}"
        r["config"] = {"ae_align_scale": lam}
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
        r["config"] = {k: cfg[k] for k in ("init", "blend", "allowed", "scale", "diversity")}
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
    # Per-method × top-25-element occurrence heatmap. Always written so the discovered-element
    # signal (bold orange on the x-axis) is part of every paper-comparison output — the slide
    # author / downstream reader doesn't need to find or rerun a separate post-hoc script.
    _plot_element_frequency_heatmap(results, list(seeds), out_dir / "element_frequency_heatmap.png")
    # The auto-generated README is a compact summary table only. It writes to ``SUMMARY.md``
    # (not ``README.md``) so a user-written index — pointing to every figure, file, and the
    # full ANALYSIS.md — can live at ``README.md`` without being overwritten on rerun.
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
        diversity_scale=cfg["diversity"],
        allowed_elements=cfg["allowed"],
        element_step_scale=cfg["scale"],
        steps=steps,
        lr=lr,
        **init_kwargs,
    )
    elapsed = time.perf_counter() - t0

    reg_names = list(reg_targets)
    optimized_desc = res.optimized_descriptor
    w_final = res.optimized_weights.cpu().numpy()
    return {
        "method": "composition",
        "align_scale": None,
        "elapsed_s": elapsed,
        # For random init the "seeds" entry is informational only — there's no per-row correspondence.
        "seeds": list(seeds) if cfg["init"] == "seed" else [f"random_start_{i}" for i in range(len(seeds))],
        "qc_after_decode": _qc_prob(model, optimized_desc).tolist(),
        "reg_achieved_latent": {t: res.optimized_target.cpu().numpy()[:, j].tolist() for j, t in enumerate(reg_names)},
        "reg_after_decode": {t: _reg_preds(model, optimized_desc, [t])[t].tolist() for t in reg_names},
        "decoded_composition": _format_weights(w_final),
        # Raw arrays — keep so future replots (per-element bar charts, similarity matrices, etc.)
        # don't have to re-run the optimisation. ``optimized_weights`` is (B, n_components),
        # ``optimized_descriptor`` is (B, x_dim); element order matches DEFAULT_ELEMENTS.
        "optimized_descriptor": optimized_desc.detach().cpu().numpy().tolist(),
        "optimized_weights": w_final.tolist(),
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
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
