# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``fm inverse`` — scenario × path inverse-design engine.

Each scenario is a fully user-specified set of objective targets — regression tasks toward a value
or a direction (higher/lower), kernel-regression tasks toward a target curve ``{(t_i, y_i)}``,
classification tasks pushing the probability of chosen label(s) high or low — each with its own
weight. The engine selects seed compositions once per run, then optimises them along each
configured *path*: latent-space optimisation with AE alignment, or differentiable
composition-space optimisation over element weights.

WHAT LIVES WHERE

This was one 1,445-line module, and its parts already formed a strict hierarchy — measured, not
guessed: the cross-references between these five groups run one way, with no cycles at all.

* :mod:`.config`  — the TOML schema: targets, scenarios, paths, seeds, and the builders that
  validate them  (depends on nothing here)
* :mod:`.seeds`   — choosing which compositions to start from  (depends on config)
* :mod:`.paths`   — running one seed set down one path, and what a result looks like
* :mod:`.report`  — figures and markdown; the leaf that everything else feeds
* :mod:`.trajectory` — per-step analytics, plots and animations; a second leaf, deliberately
  knowing nothing about the config that feeds it
* :mod:`.engine`  — ``run``: the scenario loop that drives the others

The names below are re-exported, so the CLI's ``from foundation_model.workflows.inverse import
InverseConfig, build_inverse_config, run`` is unchanged. Only the public surface is: a private
helper is imported from the module that owns it, which is what makes it obvious where to go and
what widening this list would cost.
"""

from .config import (
    DEFAULT_ALLOY_PALETTE,
    InverseConfig,
    InverseMethod,
    PathConfig,
    ScenarioConfig,
    SeedConfig,
    SeedStrategy,
    TargetKind,
    TargetSpec,
    build_inverse_config,
    target_label,
)
from .engine import run
from .seeds import select_seeds

__all__ = [
    "DEFAULT_ALLOY_PALETTE",
    "InverseConfig",
    "InverseMethod",
    "PathConfig",
    "ScenarioConfig",
    "SeedConfig",
    "SeedStrategy",
    "TargetKind",
    "TargetSpec",
    "build_inverse_config",
    "run",
    "select_seeds",
    "target_label",
]
