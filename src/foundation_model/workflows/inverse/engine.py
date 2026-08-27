# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``run``: the scenario loop.

Loads the checkpoint, validates that every task a scenario names has a head, selects seeds once,
then drives each configured path and hands the results to the report. The thin layer that composes
:mod:`.config`, :mod:`.seeds`, :mod:`.paths` and :mod:`.report`; everything it does is in one of
those four.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning import seed_everything  # noqa: E402
from loguru import logger  # noqa: E402


from .._engine import descriptor_tensor, resolve_device  # noqa: E402
from ..recording import RunRecorder  # noqa: E402
from ..task_catalog import TaskCatalog  # noqa: E402
from .config import InverseConfig, InverseMethod, ScenarioConfig, target_label, TargetSpec  # noqa: E402
from .paths import _emit_trajectory, _run_composition_path, _run_latent_path  # noqa: E402
from .report import (  # noqa: E402
    _plot_comparison,
    _plot_element_frequency,
    _plot_objective_vs_targets,
    _plot_seed_to_optimized,
    _write_root_summary,
    _write_scenario_md,
)
from .seeds import _evaluate, _rebuild_model, select_seeds  # noqa: E402


def run(
    cfg: InverseConfig, recorder: RunRecorder | None = None, *, only_scenarios: Sequence[str] | None = None
) -> dict[str, Any]:
    """Run inverse design for every scenario × path. Returns the nested all-scenario summary."""

    catalog = TaskCatalog(cfg.catalog)
    owns_recorder = recorder is None
    rec = recorder or RunRecorder(cfg.output_dir)
    seed_everything(cfg.seed, workers=True)

    try:
        model, ckpt_tasks = _rebuild_model(cfg, catalog)
        _validate_heads(model, cfg)
        device = resolve_device(cfg.accelerator)
        model.to(device)

        # Apply the --scenario filter FIRST, then select seeds using the first *selected* scenario's
        # classification objective (so a filtered run doesn't depend on unrelated config order).
        scenarios = [s for s in cfg.scenarios if only_scenarios is None or s.name in set(only_scenarios)]
        if not scenarios:
            raise ValueError(f"no scenarios match the filter {list(only_scenarios or [])}.")

        seed_scn = scenarios[0]
        seeds = select_seeds(catalog, model, cfg.seeds, targets=seed_scn.targets, device=device)
        if not seeds:
            raise RuntimeError("no seed compositions selected.")
        x_seed, seeds = descriptor_tensor(catalog, seeds, device)
        (rec.paths.root / "seeds.json").write_text(json.dumps({"seeds": list(seeds)}, indent=2), encoding="utf-8")
        logger.info(f"Selected {len(seeds)} seeds.")

        all_summary: dict[str, Any] = {}
        for scenario in scenarios:
            logger.info(f"=== scenario '{scenario.name}' ({len(cfg.paths)} paths) ===")
            summary = _run_scenario(cfg, catalog, model, scenario, seeds, x_seed, rec)
            all_summary[scenario.name] = summary

        (rec.paths.root / "inverse_design.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
        _write_root_summary(rec.paths.root, all_summary, cfg)
        return all_summary
    finally:
        if owns_recorder:
            rec.close()


def _validate_heads(model: Any, cfg: InverseConfig) -> None:
    heads = set(model.task_heads)
    for scenario in cfg.scenarios:
        missing = sorted(set(scenario.task_names) - heads)
        if missing:
            raise ValueError(
                f"scenario '{scenario.name}': checkpoint is missing head(s) {missing} (have {sorted(heads)})."
            )


def _run_scenario(
    cfg: InverseConfig,
    catalog: TaskCatalog,
    model: Any,
    scenario: ScenarioConfig,
    seeds: list[str],
    x_seed: torch.Tensor,
    rec: RunRecorder,
) -> list[dict[str, Any]]:
    sc_dir = rec.paths.root / scenario.name
    sc_dir.mkdir(parents=True, exist_ok=True)
    seed_channels, seed_objective = _evaluate(model, x_seed, scenario.targets)

    results: list[dict[str, Any]] = []
    for path in cfg.paths:
        if path.method is InverseMethod.LATENT:
            r = _run_latent_path(
                model,
                catalog,
                seeds,
                x_seed,
                path,
                scenario,
                steps=cfg.steps,
                lr=cfg.lr,
                record_trajectory=cfg.record_trajectory,
            )
        else:
            r = _run_composition_path(
                model,
                catalog,
                seeds,
                path,
                scenario,
                steps=cfg.steps,
                lr=cfg.lr,
                record_trajectory=cfg.record_trajectory,
            )
        results.append(r)

    summary = _summarise(results, scenario.targets)
    target_dump = [t.dump() for t in scenario.targets]

    # Trajectory outputs: static plot + requested animations, then externalize arrays to .npz.
    if cfg.record_trajectory:
        traj_dir = sc_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        labels = np.array([target_label(t) for t in scenario.targets])
        for r in results:
            if "trajectory_targets" not in r:
                continue
            targets = np.asarray(r["trajectory_targets"], dtype=np.float32)
            weights = np.asarray(r["trajectory_weights"], dtype=np.float32)
            _emit_trajectory(r, targets, weights, scenario, seed_channels, cfg, traj_dir)
            npz = traj_dir / f"{r['path']}.npz"
            np.savez_compressed(npz, targets=targets, weights=weights, labels=labels)
            r["trajectory_file"] = str(npz.relative_to(sc_dir))
            del r["trajectory_targets"]
            del r["trajectory_weights"]

    (sc_dir / "scenario.json").write_text(
        json.dumps(
            {
                "name": scenario.name,
                "targets": target_dump,
                "checkpoint": str(cfg.checkpoint),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (sc_dir / "results.json").write_text(
        json.dumps(
            {
                "targets": target_dump,
                "seed_predictions": {
                    "channels": {t: v.tolist() for t, v in seed_channels.items()},
                    "objective": seed_objective.tolist(),
                },
                "results": results,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (sc_dir / "targets.json").write_text(json.dumps(target_dump, indent=2), encoding="utf-8")
    (sc_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_scenario_md(sc_dir, scenario, summary)

    # figures
    rel = scenario.name
    _plot_comparison(results, scenario, rec, f"{rel}/comparison.png")
    _plot_objective_vs_targets(
        results, scenario, seed_channels, seed_objective, rec, f"{rel}/objective_vs_targets_scatter.png"
    )
    _plot_element_frequency(results, list(seeds), rec, f"{rel}/element_frequency_heatmap.png")
    for r in results:
        if r["method"] == "composition" and r["path"].endswith("random"):
            continue  # random init: no per-seed correspondence
        _plot_seed_to_optimized(
            list(seeds),
            r,
            scenario.targets,
            seed_channels,
            seed_objective,
            rec,
            f"{rel}/seed_to_optimized__{r['path']}.png",
        )
    return summary


def _summarise(results: list[dict[str, Any]], specs: Sequence[TargetSpec]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "path": r["path"],
            "method": r["method"],
            "ae_align_scale": r["ae_align_scale"],
            "elapsed_s": round(r["elapsed_s"], 2),
            "objective_mean": round(float(np.mean(r["objective_after_decode"])), 4),
            "objective_std": round(float(np.std(r["objective_after_decode"])), 4),
        }
        for s in specs:
            vals = np.asarray(r["channels_after_decode"][s.task], dtype=float)
            row[f"{s.task}_after_mean"] = round(float(vals.mean()), 3)
            row[f"{s.task}_after_std"] = round(float(vals.std()), 3)
        rows.append(row)
    return rows
