# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``fm inverse`` — scenario × path inverse-design engine.

Each scenario is a fully user-specified set of objective targets — regression tasks toward a
value or a direction (higher/lower), kernel-regression tasks toward a target curve
``{(t_i, y_i)}``, classification tasks pushing the probability of chosen label(s) high or low —
each with its own weight. The engine selects seed compositions once per run, then optimises them
along each configured *path* — either latent-space optimisation with AE alignment
(``optimize_latent``) or differentiable composition-space optimisation over element weights
(``optimize_composition``). Trajectory analytics live in
:mod:`foundation_model.workflows.inverse_trajectory`.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning import seed_everything  # noqa: E402
from loguru import logger  # noqa: E402

from foundation_model.models.flexible_multi_task_model import OptimizationTarget  # noqa: E402
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, formula_to_composition  # noqa: E402

from . import inverse_trajectory  # noqa: E402
from ._engine import build_model_for_checkpoint, checkpoint_task_order  # noqa: E402
from ._sections import ModelSectionConfig, build_model_section, reject_unknown  # noqa: E402
from .plots import DISCOVERED_ELEMENT_COLOR, SCATTER_COLOR  # noqa: E402
from .recording import RunRecorder, load_checkpoint_state  # noqa: E402
from .task_catalog import TaskCatalog, TaskCatalogConfig, TaskKind, TaskSpec, build_task_catalog_config  # noqa: E402

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


# --- config dataclasses -------------------------------------------------------------------


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


# --- builder ------------------------------------------------------------------------------


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


# --- prediction helpers -------------------------------------------------------------------


def _evaluate(model: Any, x: torch.Tensor, specs: Sequence[TargetSpec]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-target channels (keyed by task name) + per-sample objective score for descriptors ``x``.

    Thin wrapper over :meth:`FlexibleMultiTaskModel.evaluate_targets` — the same terms the
    optimisers minimise, so baselines, after-decode stats and seed ranking cannot drift from the
    optimisation objective.
    """
    channels, objective = model.evaluate_targets(x, [s.to_model_target() for s in specs])
    ch = channels.cpu().numpy()
    return {s.task: ch[:, j] for j, s in enumerate(specs)}, objective.cpu().numpy()


def _seed_weights(seeds: Sequence[str]) -> torch.Tensor:
    rows = []
    for comp in seeds:
        w = formula_to_composition(comp)
        if w is None:
            raise ValueError(f"cannot parse seed composition '{comp}' to element weights.")
        rows.append(np.asarray(w, dtype=np.float64))
    return torch.tensor(np.stack(rows), dtype=torch.float64)


def _format_weights(weights: np.ndarray, *, top_k: int = 6, eps: float = 1e-3) -> list[str]:
    out = []
    for row in np.asarray(weights):
        order = np.argsort(row)[::-1]
        parts = [f"{DEFAULT_ELEMENTS[int(i)]}{row[int(i)]:.3f}" for i in order[:top_k] if row[int(i)] > eps]
        out.append(" ".join(parts) if parts else "<empty>")
    return out


def _element_system(composition: str) -> frozenset[str]:
    return frozenset(_ELEMENT_TOKEN.findall(composition))


# --- model rebuild + seed selection -------------------------------------------------------


def _rebuild_model(cfg: InverseConfig, catalog: TaskCatalog) -> tuple[Any, list[str]]:
    state = load_checkpoint_state(cfg.checkpoint)
    ckpt_tasks = checkpoint_task_order(state)
    catalog_tasks = {t.name for t in cfg.catalog.tasks}
    missing = [t for t in ckpt_tasks if t not in catalog_tasks]
    if missing:
        raise ValueError(f"checkpoint tasks {missing} are not in the catalog (have {sorted(catalog_tasks)}).")

    model = build_model_for_checkpoint(catalog, cfg.model, ckpt_tasks)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, ckpt_tasks


def _dedup_by_system(candidates: Sequence[str], n: int, *, enabled: bool) -> list[str]:
    if not enabled:
        return list(candidates)[:n]
    seen: set[frozenset[str]] = set()
    out: list[str] = []
    for comp in candidates:
        key = _element_system(comp)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(comp)
        if len(out) >= n:
            break
    return out


def select_seeds(
    catalog: TaskCatalog,
    model: Any,
    seed_cfg: SeedConfig,
    *,
    targets: Sequence[TargetSpec],
    device: torch.device,
) -> list[str]:
    """Select seed compositions per :class:`SeedConfig`.

    The candidate pool is the ordered union of the scenario's target tasks' data frames (filtered
    to ``seed_cfg.split``). ``top_objective`` ranks the pool by the scenario's objective score
    (lower = closer to the targets) via :meth:`FlexibleMultiTaskModel.evaluate_targets` — the
    exact quantity the optimisers minimise.
    """

    descriptor_fn = catalog.descriptor_fn()

    def _has_descriptor(comp: str) -> bool:
        return not descriptor_fn([comp]).empty

    appended: list[str] = []
    for raw in seed_cfg.explicit_append:
        if not _has_descriptor(raw):
            raise ValueError(f"seeds.explicit_append entry {raw!r} has no computable descriptor.")
        appended.append(raw)
    appended = _dedup_by_system(appended, len(appended), enabled=seed_cfg.dedup_by_element_system)
    n_strategy = max(0, seed_cfg.n - len(appended))

    def _merge(strategy_seeds: Sequence[str]) -> list[str]:
        seen = {_element_system(c) for c in appended}
        kept = [c for c in strategy_seeds if _element_system(c) not in seen]
        return kept[:n_strategy] + appended

    if seed_cfg.strategy is SeedStrategy.EXPLICIT:
        pool = [c for c in seed_cfg.explicit if _has_descriptor(c)]
        return _merge(_dedup_by_system(pool, n_strategy, enabled=seed_cfg.dedup_by_element_system))

    if seed_cfg.strategy is SeedStrategy.WEIGHTED_RANDOM:
        # Pool = the weight task's rows in the chosen split with a valid label; draw a full
        # weighted permutation without replacement, probability proportional to the rank of a
        # score encoding the exploration intent (scale-free — z-scored/negative labels are fine),
        # then dedup/merge as usual.
        assert seed_cfg.weight_task is not None  # guaranteed by SeedConfig validation
        frame = catalog.task_frames([seed_cfg.weight_task])[seed_cfg.weight_task]
        spec = catalog.task_spec(seed_cfg.weight_task)
        if seed_cfg.split == "all" or "split" not in frame.columns:
            sub = frame
        else:
            sub = frame[frame["split"] == seed_cfg.split]
        labels = sub[spec.column].astype(float)
        sub = sub[labels.notna()]
        pairs = [(str(c), float(v)) for c, v in zip(sub.index, labels[labels.notna()]) if _has_descriptor(str(c))]
        if not pairs:
            return appended
        vals = np.array([v for _, v in pairs])
        if seed_cfg.weight_value is not None:  # closer to the requested value = more likely
            score = -np.abs(vals - seed_cfg.weight_value)
        elif seed_cfg.weight_direction == "low":  # lower label = more likely
            score = -vals
        else:  # "high": higher label = more likely
            score = vals
        ranks = score.argsort().argsort() + 1  # 1 = worst match … N = best match
        probs = ranks / ranks.sum()
        rng = np.random.default_rng(0)
        perm = rng.choice(len(pairs), size=len(pairs), replace=False, p=probs)
        ordered = [pairs[i][0] for i in perm]
        return _merge(_dedup_by_system(ordered, n_strategy, enabled=seed_cfg.dedup_by_element_system))

    # Candidate pool: ordered union across the scenario's target tasks (target order, then frame
    # row order) — no dependence on any particular head kind.
    frames = catalog.task_frames([t.task for t in targets])
    index: list[str] = []
    for t in targets:
        frame = frames[t.task]
        if seed_cfg.split == "all" or "split" not in frame.columns:
            index.extend(frame.index)
        else:
            index.extend(frame.index[frame["split"] == seed_cfg.split])
    index = list(dict.fromkeys(str(c) for c in index))
    pool = [c for c in index if _has_descriptor(c)]
    if not pool:
        return appended

    if seed_cfg.strategy is SeedStrategy.RANDOM:
        rng = np.random.default_rng(0)
        shuffled = [pool[i] for i in rng.permutation(len(pool))]
        return _merge(_dedup_by_system(shuffled, n_strategy, enabled=seed_cfg.dedup_by_element_system))

    # top_objective — chunked no-grad scoring, stable ascending sort (lower score = better seed).
    x, pool = _descriptor_tensor(catalog, pool, device)
    model_targets = [t.to_model_target() for t in targets]
    scores = [
        model.evaluate_targets(x[i : i + 4096], model_targets)[1].cpu().numpy() for i in range(0, len(pool), 4096)
    ]
    objective = np.concatenate(scores) if scores else np.zeros(0)
    ranked = [pool[i] for i in np.argsort(objective, kind="stable")]
    return _merge(_dedup_by_system(ranked, n_strategy, enabled=seed_cfg.dedup_by_element_system))


def _descriptor_tensor(
    catalog: TaskCatalog, comps: Sequence[str], device: torch.device
) -> tuple[torch.Tensor, list[str]]:
    desc = catalog.descriptor_fn()(list(comps))
    kept = [c for c in comps if c in desc.index]
    return torch.tensor(desc.loc[kept].values, dtype=torch.float32, device=device), kept


# --- per-path dispatch --------------------------------------------------------------------


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


# --- engine -------------------------------------------------------------------------------


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
        device = _resolve_device(cfg.accelerator)
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
        x_seed, seeds = _descriptor_tensor(catalog, seeds, device)
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
        _plot_seed_to_optimized(list(seeds), r, seed_objective, rec, f"{rel}/seed_to_optimized__{r['path']}.png")
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


# --- figures (compact reimplementations) --------------------------------------------------


def _method_color(method: str) -> str:
    return "#55A868" if method == "latent" else SCATTER_COLOR


def _target_ref_line(spec: TargetSpec) -> float | None:
    """The dashed goal line for a target's channel panel (None = no fixed goal)."""
    if spec.kind is TargetKind.VALUE:
        return float(spec.value)  # type: ignore[arg-type]
    if spec.kind is TargetKind.CLASS:
        return 1.0 if spec.direction == "high" else 0.0
    if spec.kind is TargetKind.CURVE:
        return 0.0  # channel is RMSE-to-curve
    return None  # direction: unbounded


def _plot_comparison(results: list[dict[str, Any]], scenario: ScenarioConfig, rec: RunRecorder, rel: str) -> None:
    specs = scenario.targets
    panels = ["objective", *[s.task for s in specs]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 5.0), squeeze=False)
    labels = [r["path"] for r in results]
    colors = [_method_color(r["method"]) for r in results]
    x = np.arange(len(results))
    for ax, panel in zip(axes[0], panels):
        if panel == "objective":
            means = [float(np.mean(r["objective_after_decode"])) for r in results]
            stds = [float(np.std(r["objective_after_decode"])) for r in results]
            ax.set_title("objective score  (lower = better)")
        else:
            spec = next(s for s in specs if s.task == panel)
            means = [float(np.mean(r["channels_after_decode"][panel])) for r in results]
            stds = [float(np.std(r["channels_after_decode"][panel])) for r in results]
            ref = _target_ref_line(spec)
            if ref is not None:
                ax.axhline(ref, color="#C44E52", ls="--", lw=1.0)
            ax.set_title(target_label(spec))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.set_xticks(x, labels, rotation=75, ha="right", fontsize=7)
    fig.suptitle("Inverse-design paths — achieved objectives", y=1.02)
    rec.save_figure(rel, fig)
    plt.close(fig)


def _plot_objective_vs_targets(
    results: list[dict[str, Any]],
    scenario: ScenarioConfig,
    seed_channels: dict[str, np.ndarray],
    seed_objective: np.ndarray,
    rec: RunRecorder,
    rel: str,
) -> None:
    specs = scenario.targets
    fig, axes = plt.subplots(1, len(specs), figsize=(5.2 * len(specs), 5.0), squeeze=False)
    for ax, spec in zip(axes[0], specs):
        task = spec.task
        ax.scatter(
            seed_objective,
            seed_channels[task],
            marker="*",
            s=70,
            color=DISCOVERED_ELEMENT_COLOR,
            label="seed",
            zorder=1,
        )
        for r in results:
            ax.scatter(
                r["objective_after_decode"],
                r["channels_after_decode"][task],
                s=18,
                alpha=0.6,
                color=_method_color(r["method"]),
                zorder=2,
            )
        ref = _target_ref_line(spec)
        if ref is not None:
            ax.axhline(ref, color="#C44E52", ls="--", lw=1.0)
        ax.set_xlabel("objective score  (lower = better)")
        ax.set_ylabel(target_label(spec))
    fig.suptitle("Objective score vs per-target channels", y=1.02)
    rec.save_figure(rel, fig)
    plt.close(fig)


def _plot_seed_to_optimized(
    seeds: list[str], result: dict[str, Any], seed_objective: np.ndarray, rec: RunRecorder, rel: str
) -> None:
    decoded = result["decoded_composition"]
    opt_obj = np.asarray(result["objective_after_decode"])
    n = min(len(seeds), len(decoded))
    fig, ax = plt.subplots(figsize=(9.0, max(3.0, 0.32 * n + 1)))
    ax.axis("off")
    ax.set_title(f"Seed → optimised · {result['path']}  (obj = objective score, lower = better)", fontsize=11)
    for i in range(n):
        y = n - i
        ax.text(0.01, y, f"{seeds[i]}  (obj={seed_objective[i]:.2f})", fontsize=8, family="monospace", va="center")
        ax.text(0.42, y, "→", fontsize=10, va="center")
        ax.text(0.47, y, f"{decoded[i]}  (obj={opt_obj[i]:.2f})", fontsize=8, family="monospace", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 1)
    rec.save_figure(rel, fig)
    plt.close(fig)


def _plot_element_frequency(results: list[dict[str, Any]], seeds: list[str], rec: RunRecorder, rel: str) -> None:
    seed_elements = set().union(*[_element_system(s) for s in seeds]) if seeds else set()
    counts: dict[str, dict[str, int]] = {}
    all_elems: dict[str, int] = {}
    for r in results:
        c: dict[str, int] = {}
        for formula in r["decoded_composition"]:
            for el in _element_system(formula):
                c[el] = c.get(el, 0) + 1
                all_elems[el] = all_elems.get(el, 0) + 1
        counts[r["path"]] = c
    top = [el for el, _ in sorted(all_elems.items(), key=lambda kv: -kv[1])[:25]]
    matrix = np.array([[counts[r["path"]].get(el, 0) for el in top] for r in results], dtype=float)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(top)), max(4.0, 0.4 * len(results))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(top)), top, rotation=90, fontsize=7)
    for tick, el in zip(ax.get_xticklabels(), top):
        if el not in seed_elements:  # discovered elements (not in any seed) highlighted
            tick.set_color(DISCOVERED_ELEMENT_COLOR)
    ax.set_yticks(range(len(results)), [r["path"] for r in results], fontsize=7)
    fig.colorbar(im, ax=ax, label="occurrences", fraction=0.03)
    ax.set_title("Element frequency across paths (orange = discovered)")
    rec.save_figure(rel, fig)
    plt.close(fig)


def _write_scenario_md(sc_dir: Path, scenario: ScenarioConfig, summary: list[dict[str, Any]]) -> None:
    lines = [f"# Inverse design — {scenario.name}", "", "Targets:"]
    lines.extend(f"- {target_label(t)}  (weight {t.weight:g})" for t in scenario.targets)
    lines.append("")
    lines.append("| path | method | objective (mean±std, lower = better) | elapsed(s) |")
    lines.append("|---|---|---|---|")
    for row in summary:
        lines.append(
            f"| {row['path']} | {row['method']} | {row['objective_mean']}±{row['objective_std']} | {row['elapsed_s']} |"
        )
    (sc_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_summary(root: Path, all_summary: Mapping[str, Any], cfg: InverseConfig) -> None:
    lines = ["# Inverse design — all scenarios", "", f"Checkpoint: `{cfg.checkpoint}`", ""]
    for name, summary in all_summary.items():
        best = min(summary, key=lambda row: row["objective_mean"]) if summary else None
        best_str = f"best path by objective: **{best['path']}** ({best['objective_mean']})" if best else "(no paths)"
        lines.append(f"- **{name}** — {best_str}")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
