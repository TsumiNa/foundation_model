# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Run the paper-grade inverse-design comparison across multiple scenarios on a single checkpoint.

This is a thin orchestrator around :mod:`paper_inverse_comparison`. The TOML config is expected to
contain a ``[[inverse_scenarios]]`` array of tables (see plan §5), each entry overriding
``reg_tasks`` / ``reg_targets`` for one scenario. The script loops over the scenarios and writes
each one's outputs into ``<output-dir>/<scenario.name>/`` so the per-scenario files (figures, raw
arrays, summary) stay isolated.

Layout::

    <output-dir>/
        scenario1_fe_down_magnetic_up/
            final_model.pt        # copy of the input checkpoint (self-contained)
            seeds.json
            results.json          # per-seed raw arrays for all 11 paths (latent α-sweep + 5 comp)
            comparison.png        # headline 3-panel bar chart
            SUMMARY.md
            scenario.json         # this scenario's reg_tasks/reg_targets
        scenario2_fe_down_tc_up_magnetic_up/
            ...
        scenario3_fe_down_klat_up/
            ...
        README.md                 # cross-scenario summary index (hand-written downstream)

The trained model has to expose every regression head listed in any scenario's ``reg_tasks``;
otherwise the per-scenario run will fail loudly at the model side. ``material_type`` (the
classification head) is implicit and always required for the QC primary objective.

Run:
    python -m foundation_model.scripts.paper_inverse_3scenarios \\
        --config-file samples/continual_rehearsal_demo_config_inverse_baseline.toml \\
        --checkpoint artifacts/inverse_design_run/finetune/final_model.pt \\
        --output-dir artifacts/inverse_design_run/inverse_design
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tomllib
from pathlib import Path
from typing import Any

from loguru import logger

from foundation_model.scripts.continual_rehearsal_demo import ContinualRehearsalConfig
from foundation_model.scripts.paper_inverse_comparison import _parse_args as _paper_parse_args
from foundation_model.scripts.paper_inverse_comparison import run as paper_run


def _load_scenarios(config_file: Path) -> list[dict[str, Any]]:
    """Pull the ``[[inverse_scenarios]]`` array out of the TOML and validate it."""
    raw = tomllib.loads(config_file.read_text(encoding="utf-8"))
    scenarios = raw.get("inverse_scenarios", [])
    if not scenarios:
        raise ValueError(
            f"No [[inverse_scenarios]] array found in {config_file}. "
            "Add the array (with name/reg_tasks/reg_targets) per plan §5 first."
        )
    for sc in scenarios:
        missing = {"name", "reg_tasks", "reg_targets"} - set(sc)
        if missing:
            raise ValueError(f"Scenario missing required fields {sorted(missing)}: {sc!r}.")
        if len(sc["reg_tasks"]) != len(sc["reg_targets"]):
            raise ValueError(
                f"reg_tasks and reg_targets length mismatch in scenario {sc['name']!r}: "
                f"{len(sc['reg_tasks'])} vs {len(sc['reg_targets'])}."
            )
    return scenarios


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-grade inverse-design comparison across multiple scenarios.")
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Parent folder; each scenario writes into <output-dir>/<scenario.name>/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    scenarios = _load_scenarios(args.config_file)
    logger.info(f"Loaded {len(scenarios)} inverse-design scenarios from {args.config_file}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build a baseline config once by re-using the single-scenario parser. We then ``replace`` it
    # per-scenario to override ``inverse_reg_tasks`` / ``inverse_reg_targets`` / ``output_dir``.
    paper_argv = [
        "--config-file",
        str(args.config_file),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(args.output_dir / scenarios[0]["name"]),  # placeholder; overridden below
    ]
    base_config, _ = _paper_parse_args(paper_argv)

    for sc in scenarios:
        sc_dir = args.output_dir / sc["name"]
        sc_config: ContinualRehearsalConfig = dataclasses.replace(
            base_config,
            inverse_reg_tasks=list(sc["reg_tasks"]),
            inverse_reg_targets=list(sc["reg_targets"]),
            output_dir=sc_dir,
        )
        logger.info(f"=== Scenario {sc['name']} ===")
        logger.info(f"  reg_tasks   : {sc['reg_tasks']}")
        logger.info(f"  reg_targets : {sc['reg_targets']}")
        logger.info(f"  output      : {sc_dir}")
        paper_run(sc_config, args.checkpoint)
        # Drop a per-scenario meta file so future readers don't need to chase results.json's
        # `config` block to learn what this folder represents.
        (sc_dir / "scenario.json").write_text(
            json.dumps(
                {
                    "name": sc["name"],
                    "reg_tasks": list(sc["reg_tasks"]),
                    "reg_targets": list(sc["reg_targets"]),
                    "primary_objective": "P(material_type = QC) ↑",
                    "checkpoint": str(args.checkpoint),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"=== {sc['name']} done ===")


if __name__ == "__main__":
    main()
