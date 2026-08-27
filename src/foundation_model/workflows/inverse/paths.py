# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Running one seed set down one path, and what a result looks like.

The two paths differ in what they optimise — a descriptor through the autoencoder, or element
weights straight through the KMD transform — and agree on what they return, which is what lets the
scenario loop treat them interchangeably and the report compare them side by side.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS  # noqa: E402

from .. import inverse_trajectory  # noqa: E402
from ..task_catalog import TaskCatalog  # noqa: E402
from .config import InverseConfig, PathConfig, ScenarioConfig, TargetSpec  # noqa: E402
from .seeds import _evaluate, _format_weights, _seed_weights  # noqa: E402


def _run_latent_path(
    model: Any,
    catalog: TaskCatalog,
    seeds: list[str],
    x_seed: torch.Tensor,
    path: PathConfig,
    scenario: ScenarioConfig,
    *,
    steps: int,
    lr: float,
    record_trajectory: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = model.optimize_latent(
        initial_input=x_seed,
        targets=scenario.model_targets,
        ae_align_scale=path.ae_align_scale,
        optimize_space="latent",
        steps=steps,
        lr=lr,
        record_input_trajectory=record_trajectory,
    )
    elapsed = time.perf_counter() - t0
    kmd = catalog.kmd()
    achieved = res.optimized_target[:, 0, :].cpu().numpy()  # channels at the final LATENT h
    optimized_desc = res.optimized_input[:, 0, :]
    channels_after, objective_after = _evaluate(model, optimized_desc, scenario.targets)
    desc_np = optimized_desc.detach().cpu().numpy()
    weights = kmd.inverse(desc_np) if kmd is not None else np.zeros((desc_np.shape[0], len(DEFAULT_ELEMENTS)))
    out = _result_dict(
        path, "latent", seeds, channels_after, objective_after, achieved, scenario.targets, weights, elapsed
    )
    if record_trajectory:
        out["trajectory_targets"] = res.trajectory[:, 0, :, :].cpu().numpy().transpose(1, 0, 2)
        if kmd is not None and res.input_trajectory is not None:
            steps_in = res.input_trajectory[:, 0, :, :].cpu().numpy().transpose(1, 0, 2)
            out["trajectory_weights"] = np.stack([kmd.inverse(steps_in[s]) for s in range(steps_in.shape[0])], axis=0)
        else:
            out["trajectory_weights"] = np.zeros((0, 0, 0))
    return out


def _run_composition_path(
    model: Any,
    catalog: TaskCatalog,
    seeds: list[str],
    path: PathConfig,
    scenario: ScenarioConfig,
    *,
    steps: int,
    lr: float,
    record_trajectory: bool,
) -> dict[str, Any]:
    kmd = catalog.kmd()
    assert kmd is not None  # guaranteed by config validation
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    kernel = kmd.kernel_torch(device=device, dtype=dtype)

    if path.init == "seed":
        init_kwargs: dict[str, Any] = {"initial_weights": _seed_weights(seeds), "seed_blend": path.seed_blend}
        n_rows = len(seeds)
    else:
        n_rows = path.n_starts or len(seeds)  # random init yields n_starts rows, not len(seeds)
        init_kwargs = {"initial_weights": None, "n_starts": n_rows}

    t0 = time.perf_counter()
    res = model.optimize_composition(
        kernel,
        targets=scenario.model_targets,
        diversity_scale=path.diversity_scale,
        allowed_elements=path.allowed_elements,
        element_step_scale=path.element_step_scale,
        fixed_amounts=path.fixed_amounts or None,
        max_elements=path.max_elements,
        annealing_scale=path.annealing_scale,
        annealing_schedule=path.annealing_schedule,
        steps=steps,
        lr=lr,
        record_weights_trajectory=record_trajectory,
        **init_kwargs,
    )
    elapsed = time.perf_counter() - t0
    optimized_desc = res.optimized_descriptor
    weights = res.optimized_weights.cpu().numpy()
    achieved = res.optimized_target.cpu().numpy()
    channels_after, objective_after = _evaluate(model, optimized_desc, scenario.targets)
    seed_labels = list(seeds) if path.init == "seed" else [f"random_start_{i}" for i in range(n_rows)]
    out = _result_dict(
        path, "composition", seed_labels, channels_after, objective_after, achieved, scenario.targets, weights, elapsed
    )
    if record_trajectory:
        out["trajectory_targets"] = res.trajectory.cpu().numpy()
        out["trajectory_weights"] = (
            res.weights_trajectory.cpu().numpy() if res.weights_trajectory is not None else np.zeros((0, 0, 0))
        )
    return out


def _result_dict(
    path: PathConfig,
    method: str,
    seeds: list[str],
    channels_after: dict[str, np.ndarray],
    objective_after: np.ndarray,
    achieved: np.ndarray,
    specs: Sequence[TargetSpec],
    weights: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "path": path.name,
        "method": method,
        "ae_align_scale": path.ae_align_scale if method == "latent" else None,
        "elapsed_s": elapsed,
        "seeds": seeds,
        # Per-seed objective score (lower = better) + per-target channels, both computed on the
        # final decoded descriptor (for the latent path this is AFTER the AE round-trip).
        "objective_after_decode": objective_after.tolist(),
        "channels_after_decode": {s.task: channels_after[s.task].tolist() for s in specs},
        # Channels at the optimiser's own final state (latent path: pre-decode latent h;
        # composition path: same descriptor, so this matches channels_after_decode).
        "channels_optimized": {s.task: achieved[:, j].tolist() for j, s in enumerate(specs)},
        "decoded_composition": _format_weights(weights),
        "optimized_weights": np.asarray(weights).tolist(),
    }


def _resolve_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _emit_trajectory(
    result: dict[str, Any],
    targets: np.ndarray,
    weights: np.ndarray,
    scenario: ScenarioConfig,
    seed_channels: Mapping[str, np.ndarray],
    cfg: InverseConfig,
    traj_dir: Path,
) -> None:
    """Write the static trajectory plot (+ requested animations) for one path."""
    if targets.size == 0:
        return
    metas = scenario.target_metas
    progress = inverse_trajectory.normalize_target_trajectories(targets, metas, seed_channels)
    inverse_trajectory.plot_trajectory_static(
        progress, traj_dir / f"{result['path']}_trajectory.png", title=result["path"]
    )

    if cfg.animation_formats and weights.size:
        # Representative seed for the composition animation = the best final objective score.
        best = min(int(np.argmin(result["objective_after_decode"])), weights.shape[1] - 1)
        out_paths = {fmt: traj_dir / f"{result['path']}_trajectory.{fmt}" for fmt in cfg.animation_formats}
        inverse_trajectory.plot_trajectory_animation(
            progress, weights[:, best, :], list(DEFAULT_ELEMENTS), out_paths, title=result["path"]
        )

    if cfg.per_seed_trajectories:
        per_dir = traj_dir / f"{result['path']}_per_seed"
        per_dir.mkdir(exist_ok=True)
        matrix = inverse_trajectory.target_progress_matrix(targets, metas, seed_channels)
        for i in range(min(targets.shape[1], 20)):  # cap the per-seed fan-out
            ps = {label: mat[:, i] for label, mat in matrix.items()}
            inverse_trajectory.plot_trajectory_static(
                ps, per_dir / f"seed{i:02d}.png", title=f"{result['path']} · seed {i}"
            )
