# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""The ``fm inverse`` TOML schema: what to search for, from where, and along which path.

A *scenario* is a set of weighted objective targets; a *path* is one way of pursuing them
(latent-space or composition-space, with its own knobs); a *seed* policy decides which
compositions the search starts from. This module is those three plus the builders that turn a
parsed TOML table into them, rejecting unknown keys by name.

The leaf of the package: it imports nothing from its siblings, and all four of them import it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")


from foundation_model.models.flexible_multi_task_model import OptimizationTarget  # noqa: E402

from .. import inverse_trajectory  # noqa: E402
from .._sections import ModelSectionConfig, build_model_section, reject_unknown  # noqa: E402
from ..task_catalog import TaskCatalogConfig, TaskKind, TaskSpec, build_task_catalog_config  # noqa: E402


_ANIMATION_FORMATS = {"gif", "html", "svg"}


_ACCELERATORS = {"auto", "cpu"}


_ELEMENT_TOKEN = re.compile(r"[A-Z][a-z]?")


# 48-element feasible alloy palette (copied verbatim from paper_inverse_comparison.py).
DEFAULT_ALLOY_PALETTE: list[str] = [
    "Mg", "Ca", "B", "Al", "Ga", "In", "Tl", "Si", "Ge", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re",
    "Os", "Ir", "Pt", "Au", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Yb",
]  # fmt: skip


assert len(DEFAULT_ALLOY_PALETTE) == 48


_INVERSE_ROOT_KEYS = {"data", "descriptor", "datasets", "tasks", "model", "inverse", "output"}


_CATALOG_KEYS = {"data", "descriptor", "datasets", "tasks"}


class InverseMethod(str, Enum):
    LATENT = "latent"
    COMPOSITION = "composition"


class SeedStrategy(str, Enum):
    TOP_OBJECTIVE = "top_objective"  # rank candidates by the scenario's objective score
    WEIGHTED_RANDOM = "weighted_random"  # sample; selection probability rises with a task's true label
    RANDOM = "random"
    EXPLICIT = "explicit"


@dataclass(kw_only=True)
class SeedConfig:
    strategy: SeedStrategy | str = SeedStrategy.TOP_OBJECTIVE  # str accepted; coerced in __post_init__
    n: int = 20
    split: str = "test"
    # weighted_random only: the regression task whose TRUE labels weight the sampling — candidates
    # are drawn without replacement with probability proportional to a rank score, so the pool
    # keeps variety while favoring seeds that match the exploration intent:
    #   weight_direction = "high"  → higher label = more likely (default)
    #   weight_direction = "low"   → lower label = more likely
    #   weight_value = <float>     → closer to that label value = more likely
    weight_task: str | None = None
    weight_direction: str | None = None
    weight_value: float | None = None
    explicit: list[str] = field(default_factory=list)
    explicit_append: list[str] = field(default_factory=list)
    dedup_by_element_system: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, SeedStrategy):
            self.strategy = SeedStrategy(str(self.strategy))
        if self.n < 1:
            raise ValueError(f"seeds.n must be >= 1, got {self.n}.")
        if self.split not in {"train", "val", "test", "all"}:
            raise ValueError(f"seeds.split must be train/val/test/all, got {self.split!r}.")
        if self.strategy is SeedStrategy.EXPLICIT and not self.explicit:
            raise ValueError("seeds.strategy='explicit' requires a non-empty seeds.explicit list.")
        if self.strategy is SeedStrategy.WEIGHTED_RANDOM and not self.weight_task:
            raise ValueError("seeds.strategy='weighted_random' requires seeds.weight_task.")
        weight_keys_set = (
            self.weight_task is not None or self.weight_direction is not None or self.weight_value is not None
        )
        if self.strategy is not SeedStrategy.WEIGHTED_RANDOM and weight_keys_set:
            raise ValueError(
                "seeds.weight_task/weight_direction/weight_value only apply to strategy='weighted_random'."
            )
        if self.weight_direction is not None and self.weight_value is not None:
            raise ValueError("seeds.weight_direction and seeds.weight_value are mutually exclusive.")
        if self.weight_direction is not None and self.weight_direction not in {"high", "low"}:
            raise ValueError(f"seeds.weight_direction must be 'high' or 'low', got {self.weight_direction!r}.")
        if (
            self.strategy is SeedStrategy.WEIGHTED_RANDOM
            and self.weight_direction is None
            and self.weight_value is None
        ):
            self.weight_direction = "high"


class TargetKind(str, Enum):
    VALUE = "value"  # regression toward a value
    DIRECTION = "direction"  # regression pushed higher/lower (no fixed value)
    CURVE = "curve"  # kernel-regression toward a {(t, y)} curve
    CLASS = "class"  # classification label(s) probability pushed high/low


@dataclass(kw_only=True)
class TargetSpec:
    """One ``[[inverse.scenarios.targets]]`` entry. The kind derives from the task's catalog kind:

    - regression → exactly one of ``value`` / ``direction`` (``"high"`` | ``"low"``);
    - kernel_regression → ``points`` = target curve ``[[t, y], ...]``;
    - classification → ``classes`` (strict subset of the label indices) + optional ``direction``
      (default ``"high"``).

    ``weight`` (> 0, default 1.0) scales this term against the scenario's other targets.
    Direction targets have no stationary point — the achieved magnitude scales with
    ``steps × lr``; use ``weight`` to balance them against the bounded terms.
    """

    task: str
    value: float | None = None
    direction: str | None = None
    points: list[list[float]] | None = None
    classes: list[int] | None = None
    weight: float = 1.0
    kind: TargetKind = field(init=False, default=TargetKind.VALUE)

    def __post_init__(self) -> None:
        if float(self.weight) <= 0:
            raise ValueError(f"target '{self.task}': weight must be > 0, got {self.weight}.")
        if self.direction is not None and self.direction not in {"high", "low"}:
            raise ValueError(f"target '{self.task}': direction must be 'high' or 'low', got {self.direction!r}.")
        if self.points is not None:
            pairs = [list(p) for p in self.points]
            if not pairs or any(len(p) != 2 for p in pairs):
                raise ValueError(f"target '{self.task}': points must be a non-empty list of [t, y] pairs.")
            self.points = [[float(t), float(y)] for t, y in pairs]
        if self.classes is not None:
            if not self.classes:
                raise ValueError(f"target '{self.task}': classes must be non-empty when given.")
            self.classes = [int(c) for c in self.classes]

    def resolve_kind(self, spec: TaskSpec) -> None:
        """Cross-validate the fields against the task's catalog kind and set :attr:`kind`."""
        name = self.task
        if spec.kind is TaskKind.REGRESSION:
            if self.points is not None or self.classes is not None:
                raise ValueError(
                    f"target '{name}' is a regression task: it accepts value/direction, not points/classes."
                )
            if (self.value is None) == (self.direction is None):
                raise ValueError(f"target '{name}' (regression) needs exactly one of value or direction.")
            self.kind = TargetKind.VALUE if self.value is not None else TargetKind.DIRECTION
        elif spec.kind is TaskKind.KERNEL_REGRESSION:
            if self.value is not None or self.direction is not None or self.classes is not None:
                raise ValueError(f"target '{name}' is a kernel-regression task: it accepts points only.")
            if not self.points:
                raise ValueError(f"target '{name}' (kernel_regression) needs a non-empty points list of [t, y] pairs.")
            self.kind = TargetKind.CURVE
        else:  # classification
            if self.value is not None or self.points is not None:
                raise ValueError(f"target '{name}' is a classification task: it accepts classes (+ direction) only.")
            if not self.classes:
                raise ValueError(f"target '{name}' (classification) needs a non-empty classes list.")
            if self.direction is None:
                self.direction = "high"
            n_cls = spec.num_classes
            if n_cls is not None:
                if any(not 0 <= c < n_cls for c in self.classes):
                    raise ValueError(
                        f"target '{name}': classes {self.classes} out of range for a {n_cls}-class task; "
                        f"valid indices are [0, {n_cls})."
                    )
                if len(set(self.classes)) >= n_cls:
                    raise ValueError(
                        f"target '{name}': classes {self.classes} covers every class of a {n_cls}-class task; "
                        "use a strict subset (the objective is otherwise constant/undefined)."
                    )
            self.kind = TargetKind.CLASS

    def to_model_target(self) -> OptimizationTarget:
        return OptimizationTarget(
            task=self.task,
            value=self.value,
            direction=self.direction,
            points=self.points,
            classes=self.classes,
            weight=self.weight,
        )

    def dump(self) -> dict[str, Any]:
        """JSON-ready provenance record (omits unset fields)."""
        out: dict[str, Any] = {"task": self.task, "kind": self.kind.value, "weight": self.weight}
        for key in ("value", "direction", "points", "classes"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        out["label"] = target_label(self)
        return out


def target_label(spec: TargetSpec) -> str:
    """Human-readable one-liner for a target — used as plot legend / progress-dict key."""
    if spec.kind is TargetKind.VALUE:
        return f"{spec.task}→{spec.value:g}"
    if spec.kind is TargetKind.DIRECTION:
        return f"{spec.task}{'↑' if spec.direction == 'high' else '↓'}"
    if spec.kind is TargetKind.CURVE:
        return f"{spec.task}~curve({len(spec.points or [])}pts)"
    classes = ",".join(str(c) for c in (spec.classes or []))
    return f"P({spec.task}∈{{{classes}}}){'↑' if spec.direction == 'high' else '↓'}"


@dataclass(kw_only=True)
class ScenarioConfig:
    name: str
    targets: list[TargetSpec]

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError(f"scenario '{self.name}': needs at least one [[inverse.scenarios.targets]] entry.")
        names = [t.task for t in self.targets]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"scenario '{self.name}': duplicate target task(s): {dupes}.")

    @property
    def task_names(self) -> list[str]:
        return [t.task for t in self.targets]

    @property
    def model_targets(self) -> list[OptimizationTarget]:
        return [t.to_model_target() for t in self.targets]

    @property
    def target_metas(self) -> list[inverse_trajectory.TargetMeta]:
        return [
            inverse_trajectory.TargetMeta(
                task=t.task,
                kind=t.kind.value,
                label=target_label(t),
                value=t.value,
                class_high=t.direction != "low",
            )
            for t in self.targets
        ]


# Composition-path field defaults (used to reject latent-only / composition-only key misuse).
_COMP_FIELD_DEFAULTS: dict[str, Any] = {
    "init": "seed",
    "seed_blend": 0.95,
    "allowed_elements": "all",
    "diversity_scale": 1.0,
    "max_elements": None,
    "element_step_scale": 1.0,
    "annealing_scale": 0.5,
    "annealing_schedule": None,
    "n_starts": None,
}


_LATENT_DEFAULT_AE_ALIGN = 0.5


@dataclass(kw_only=True)
class PathConfig:
    name: str
    method: InverseMethod
    # latent-only:
    ae_align_scale: float = _LATENT_DEFAULT_AE_ALIGN
    # composition-only:
    init: str = "seed"
    n_starts: int | None = None
    seed_blend: float = 0.95
    allowed_elements: list[str] | str = "all"
    diversity_scale: float = 1.0
    max_elements: int | None = None
    element_step_scale: float | dict[str, float] = 1.0
    fixed_amounts: dict[str, float] = field(default_factory=dict)
    annealing_scale: float = 0.5
    annealing_schedule: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, InverseMethod):
            self.method = InverseMethod(str(self.method))
        if self.method is InverseMethod.LATENT:
            # Reject explicitly-set composition-only keys on a latent path.
            bad = [k for k, dflt in _COMP_FIELD_DEFAULTS.items() if getattr(self, k) != dflt]
            if self.fixed_amounts:
                bad.append("fixed_amounts")
            if bad:
                raise ValueError(f"path '{self.name}' (latent): composition-only keys set: {sorted(bad)}.")
            if self.init not in ("seed", "random"):
                raise ValueError(f"path '{self.name}': init must be 'seed' or 'random'.")
        else:
            if self.ae_align_scale != _LATENT_DEFAULT_AE_ALIGN:
                raise ValueError(f"path '{self.name}' (composition): latent-only key 'ae_align_scale' set.")
            if self.init not in ("seed", "random"):
                raise ValueError(f"path '{self.name}': init must be 'seed' or 'random'.")


def _default_paths() -> list[PathConfig]:
    """The 11 legacy DEFAULT_PATHS: 3 latent + 8 composition."""
    P = DEFAULT_ALLOY_PALETTE
    latent = [
        PathConfig(name=f"latent_align{a:g}".replace(".", "p"), method=InverseMethod.LATENT, ae_align_scale=a)
        for a in (0.0, 0.25, 1.0)
    ]
    comp = [
        PathConfig(name="comp_seed", method=InverseMethod.COMPOSITION, init="seed", seed_blend=1.0),
        PathConfig(name="comp_seed_blend95", method=InverseMethod.COMPOSITION, init="seed", seed_blend=0.95),
        PathConfig(
            name="comp_seed_blend95_elemlist",
            method=InverseMethod.COMPOSITION,
            init="seed",
            seed_blend=0.95,
            allowed_elements=P,
        ),
        PathConfig(
            name="comp_seed_blend95_elemlist_lowdiv",
            method=InverseMethod.COMPOSITION,
            init="seed",
            seed_blend=0.95,
            allowed_elements=P,
            diversity_scale=0.0,
        ),
        PathConfig(name="comp_random", method=InverseMethod.COMPOSITION, init="random", seed_blend=0.95),
        PathConfig(
            name="comp_seed_elemlist_k3",
            method=InverseMethod.COMPOSITION,
            init="seed",
            seed_blend=0.95,
            allowed_elements=P,
            max_elements=3,
        ),
        PathConfig(
            name="comp_seed_elemlist_k5",
            method=InverseMethod.COMPOSITION,
            init="seed",
            seed_blend=0.95,
            allowed_elements=P,
            max_elements=5,
        ),
        PathConfig(
            name="comp_seed_elemlist_k5_linear",
            method=InverseMethod.COMPOSITION,
            init="seed",
            seed_blend=0.95,
            allowed_elements=P,
            max_elements=5,
            annealing_scale=0.715,
            annealing_schedule={"step": [1.0], "scale": [0.0], "annealing_func": ["linear"]},
        ),
    ]
    return latent + comp


@dataclass(kw_only=True)
class InverseConfig:
    catalog: TaskCatalogConfig
    model: ModelSectionConfig
    checkpoint: Path
    seeds: SeedConfig
    scenarios: list[ScenarioConfig]
    paths: list[PathConfig]
    output_dir: Path
    steps: int = 300
    lr: float = 0.05
    record_trajectory: bool = True
    per_seed_trajectories: bool = False
    animation_formats: list[str] = field(default_factory=lambda: ["gif"])
    seed: int = 2025
    accelerator: str = "auto"

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint)
        self.output_dir = Path(self.output_dir)
        if not self.scenarios:
            raise ValueError("at least one [[inverse.scenarios]] is required.")
        names = [s.name for s in self.scenarios]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate scenario names: {dupes}.")
        if not self.paths:
            self.paths = _default_paths()
        bad_fmt = [f for f in self.animation_formats if f not in _ANIMATION_FORMATS]
        if bad_fmt:
            raise ValueError(f"animation_formats must be a subset of {sorted(_ANIMATION_FORMATS)}, got {bad_fmt}.")
        if self.accelerator not in _ACCELERATORS:
            raise ValueError(f"inverse.accelerator must be one of {sorted(_ACCELERATORS)}, got {self.accelerator!r}.")
        # composition paths require an invertible KMD descriptor
        if self.catalog.descriptor.kind != "kmd" and any(p.method is InverseMethod.COMPOSITION for p in self.paths):
            raise ValueError("composition paths require descriptor.kind == 'kmd' (an invertible KMD descriptor).")
        if self.seeds.weight_task is not None:
            spec = next((s for s in self.catalog.tasks if s.name == self.seeds.weight_task), None)
            if spec is None:
                raise ValueError(f"seeds.weight_task '{self.seeds.weight_task}' is not a catalog task.")
            if spec.kind is not TaskKind.REGRESSION:
                raise ValueError("seeds.weight_task must be a regression task (its scalar labels weight the sampling).")


def _build_seed_config(raw: Mapping[str, Any]) -> SeedConfig:
    data = dict(raw)
    reject_unknown("inverse.seeds", data, set(SeedConfig.__dataclass_fields__))
    return SeedConfig(**data)


def _build_target(raw: Mapping[str, Any], scenario_name: str, idx: int, specs: Mapping[str, TaskSpec]) -> TargetSpec:
    data = dict(raw)
    where = f"inverse.scenarios.{scenario_name}.targets[{idx}]"
    reject_unknown(where, data, {"task", "value", "direction", "points", "classes", "weight"})
    if "task" not in data:
        raise ValueError(f"{where}: 'task' is required.")
    target = TargetSpec(**data)
    if target.task not in specs:
        raise ValueError(f"{where}: unknown task '{target.task}' (known tasks: {sorted(specs)}).")
    target.resolve_kind(specs[target.task])
    return target


def _build_scenario(raw: Mapping[str, Any], specs: Mapping[str, TaskSpec]) -> ScenarioConfig:
    data = dict(raw)
    name = str(data.get("name", "?"))
    reject_unknown(f"inverse.scenarios.{name}", data, {"name", "targets"})
    if "name" not in data:
        raise ValueError("every [[inverse.scenarios]] entry needs a 'name'.")
    targets = [_build_target(t, name, i, specs) for i, t in enumerate(data.get("targets", []))]
    return ScenarioConfig(name=name, targets=targets)


def _build_path(raw: Mapping[str, Any]) -> PathConfig:
    data = dict(raw)
    reject_unknown(f"inverse.paths.{data.get('name', '?')}", data, set(PathConfig.__dataclass_fields__))
    return PathConfig(**data)


def build_inverse_config(
    raw: Mapping[str, Any], *, output_dir: str | Path | None = None, checkpoint: str | Path | None = None
) -> InverseConfig:
    """Normalize a parsed-TOML tree into an :class:`InverseConfig`."""

    reject_unknown("<root>", raw, _INVERSE_ROOT_KEYS)
    catalog = build_task_catalog_config({k: raw[k] for k in _CATALOG_KEYS if k in raw})
    model = build_model_section(raw.get("model", {}))

    inv_raw = dict(raw.get("inverse", {}))
    reject_unknown(
        "inverse",
        inv_raw,
        {
            "checkpoint",
            "steps",
            "lr",
            "record_trajectory",
            "per_seed_trajectories",
            "animation_formats",
            "seed",
            "accelerator",
            "seeds",
            "scenarios",
            "paths",
        },
    )
    resolved_checkpoint = checkpoint if checkpoint is not None else inv_raw.get("checkpoint")
    if resolved_checkpoint is None:
        raise ValueError("checkpoint must be given via --checkpoint or [inverse].checkpoint.")
    resolved_output = output_dir if output_dir is not None else raw.get("output", {}).get("dir")
    if resolved_output is None:
        raise ValueError("output directory must be given via --output-dir or [output].dir.")

    seeds = _build_seed_config(inv_raw.get("seeds", {}))
    task_specs = {t.name: t for t in catalog.tasks}
    scenarios = [_build_scenario(s, task_specs) for s in inv_raw.get("scenarios", [])]
    paths = [_build_path(p) for p in inv_raw.get("paths", [])]

    return InverseConfig(
        catalog=catalog,
        model=model,
        checkpoint=Path(resolved_checkpoint),
        seeds=seeds,
        scenarios=scenarios,
        paths=paths,
        output_dir=Path(resolved_output),
        steps=int(inv_raw.get("steps", 300)),
        lr=float(inv_raw.get("lr", 0.05)),
        record_trajectory=bool(inv_raw.get("record_trajectory", True)),
        per_seed_trajectories=bool(inv_raw.get("per_seed_trajectories", False)),
        animation_formats=list(inv_raw.get("animation_formats", ["gif"])),
        seed=int(inv_raw.get("seed", 2025)),
        accelerator=str(inv_raw.get("accelerator", "auto")),
    )
