# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``fm inverse`` — scenario × path inverse-design engine.

For each scenario (a set of regression targets + a classification objective) the engine selects
seed compositions once per run, then optimises them along each configured *path* — either
latent-space optimisation with AE alignment (``optimize_latent``) or differentiable
composition-space optimisation over element weights (``optimize_composition``). Migrated from
``scripts/paper_inverse_comparison.py`` (per-path engine + figures), ``paper_inverse_3scenarios.py``
(per-scenario loop) and the seed selection in ``continual_rehearsal_demo._select_seeds``.
Trajectory analytics live in :mod:`foundation_model.workflows.inverse_trajectory`.
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

from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, formula_to_composition  # noqa: E402

from . import inverse_trajectory  # noqa: E402
from ._engine import build_model_for_checkpoint  # noqa: E402
from ._sections import ModelSectionConfig, build_model_section, reject_unknown  # noqa: E402
from .plots import DISCOVERED_ELEMENT_COLOR, SCATTER_COLOR  # noqa: E402
from .recording import RunRecorder, load_checkpoint_state  # noqa: E402
from .task_catalog import TaskCatalog, TaskCatalogConfig, build_task_catalog_config  # noqa: E402

# Reserved AE head name.
_AE_NAME = "__reconstruction__"
# Default QC class indices for the classification objective when class_target is unset.
_DEFAULT_QC_CLASSES = [1]
_ANIMATION_FORMATS = {"gif", "html", "svg"}
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
    TOP_QC = "top_qc"
    RANDOM = "random"
    EXPLICIT = "explicit"


# --- config dataclasses -------------------------------------------------------------------


@dataclass(kw_only=True)
class SeedConfig:
    strategy: SeedStrategy = SeedStrategy.TOP_QC
    n: int = 20
    split: str = "test"
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


@dataclass(kw_only=True)
class ScenarioConfig:
    name: str
    reg_tasks: list[str]
    reg_targets: list[float]
    class_task: str = "material_type"
    class_target: int | None = None  # None → legacy QC default class index

    def __post_init__(self) -> None:
        if not self.reg_tasks:
            raise ValueError(f"scenario '{self.name}': reg_tasks must be non-empty.")
        if len(self.reg_tasks) != len(self.reg_targets):
            raise ValueError(
                f"scenario '{self.name}': reg_tasks ({len(self.reg_tasks)}) and reg_targets "
                f"({len(self.reg_targets)}) must have equal length."
            )

    @property
    def reg_target_map(self) -> dict[str, float]:
        return {t: float(v) for t, v in zip(self.reg_tasks, self.reg_targets)}

    @property
    def class_indices(self) -> list[int]:
        return _DEFAULT_QC_CLASSES if self.class_target is None else [int(self.class_target)]


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
    class_weight: float = 5.0
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
        # composition paths require an invertible KMD descriptor
        if self.catalog.descriptor.kind != "kmd" and any(p.method is InverseMethod.COMPOSITION for p in self.paths):
            raise ValueError("composition paths require descriptor.kind == 'kmd' (an invertible KMD descriptor).")


# --- builder ------------------------------------------------------------------------------


def _build_seed_config(raw: Mapping[str, Any]) -> SeedConfig:
    data = dict(raw)
    reject_unknown("inverse.seeds", data, set(SeedConfig.__dataclass_fields__))
    return SeedConfig(**data)


def _build_scenario(raw: Mapping[str, Any]) -> ScenarioConfig:
    data = dict(raw)
    reject_unknown(f"inverse.scenarios.{data.get('name', '?')}", data, set(ScenarioConfig.__dataclass_fields__))
    return ScenarioConfig(**data)


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
            "class_weight",
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
    scenarios = [_build_scenario(s) for s in inv_raw.get("scenarios", [])]
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
        class_weight=float(inv_raw.get("class_weight", 5.0)),
        record_trajectory=bool(inv_raw.get("record_trajectory", True)),
        per_seed_trajectories=bool(inv_raw.get("per_seed_trajectories", False)),
        animation_formats=list(inv_raw.get("animation_formats", ["gif"])),
        seed=int(inv_raw.get("seed", 2025)),
        accelerator=str(inv_raw.get("accelerator", "auto")),
    )


# --- prediction helpers -------------------------------------------------------------------


def _qc_prob(model: Any, x: torch.Tensor, class_task: str, class_indices: Sequence[int]) -> np.ndarray:
    with torch.no_grad():
        h = torch.tanh(model.encoder(x))
        probs = torch.softmax(model.task_heads[class_task](h), dim=-1)
        return probs[:, list(class_indices)].sum(dim=-1).cpu().numpy()


def _reg_preds(model: Any, x: torch.Tensor, tasks: Sequence[str]) -> dict[str, np.ndarray]:
    with torch.no_grad():
        h = torch.tanh(model.encoder(x))
        return {t: model.task_heads[t](h).squeeze(-1).cpu().numpy() for t in tasks}


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
    ckpt_tasks = list(state.get("task_sequence") or _task_names_from_state(state["model"]))
    catalog_tasks = {t.name for t in cfg.catalog.tasks}
    missing = [t for t in ckpt_tasks if t not in catalog_tasks]
    if missing:
        raise ValueError(f"checkpoint tasks {missing} are not in the catalog (have {sorted(catalog_tasks)}).")

    model = build_model_for_checkpoint(catalog, cfg.model, ckpt_tasks)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, ckpt_tasks


def _task_names_from_state(state_dict: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for key in state_dict:
        if key.startswith("task_heads."):
            name = key.split(".", 2)[1]
            if name != _AE_NAME and name not in names:
                names.append(name)
    return names


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
    class_task: str,
    class_indices: Sequence[int],
    device: torch.device,
) -> list[str]:
    """Select seed compositions per :class:`SeedConfig` (mirrors the legacy ``_select_seeds``)."""

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

    frame = catalog.task_frames([class_task])[class_task]
    if seed_cfg.split == "all" or "split" not in frame.columns:
        index = list(frame.index)
    else:
        index = list(frame.index[frame["split"] == seed_cfg.split])
    pool = [c for c in index if _has_descriptor(c)]
    if not pool:
        return appended

    if seed_cfg.strategy is SeedStrategy.RANDOM:
        rng = np.random.default_rng(0)
        shuffled = [pool[i] for i in rng.permutation(len(pool))]
        return _merge(_dedup_by_system(shuffled, n_strategy, enabled=seed_cfg.dedup_by_element_system))

    # top_qc
    x, pool = _descriptor_tensor(catalog, pool, device)
    probs = _qc_prob(model, x, class_task, class_indices)
    ranked = [pool[i] for i in np.argsort(probs)[::-1]]
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
    class_weight: float,
    steps: int,
    lr: float,
    record_trajectory: bool,
) -> dict[str, Any]:
    reg_targets = scenario.reg_target_map
    reg_names = list(reg_targets)
    t0 = time.perf_counter()
    res = model.optimize_latent(
        initial_input=x_seed,
        task_targets=reg_targets,
        class_targets={scenario.class_task: scenario.class_indices},
        class_target_weight=class_weight,
        ae_align_scale=path.ae_align_scale,
        optimize_space="latent",
        steps=steps,
        lr=lr,
        record_input_trajectory=record_trajectory,
    )
    elapsed = time.perf_counter() - t0
    kmd = catalog.kmd()
    achieved = res.optimized_target[:, 0, :].cpu().numpy()
    optimized_desc = res.optimized_input[:, 0, :]
    after_qc = _qc_prob(model, optimized_desc, scenario.class_task, scenario.class_indices)
    after_reg = _reg_preds(model, optimized_desc, reg_names)
    desc_np = optimized_desc.detach().cpu().numpy()
    weights = kmd.inverse(desc_np) if kmd is not None else np.zeros((desc_np.shape[0], len(DEFAULT_ELEMENTS)))
    out = _result_dict(path, "latent", seeds, after_qc, achieved, after_reg, reg_names, weights, desc_np, elapsed)
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
    class_weight: float,
    steps: int,
    lr: float,
    record_trajectory: bool,
) -> dict[str, Any]:
    reg_targets = scenario.reg_target_map
    reg_names = list(reg_targets)
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
        task_targets=reg_targets,
        class_targets={scenario.class_task: scenario.class_indices},
        class_target_weight=class_weight,
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
    after_qc = _qc_prob(model, optimized_desc, scenario.class_task, scenario.class_indices)
    after_reg = _reg_preds(model, optimized_desc, reg_names)
    seed_labels = list(seeds) if path.init == "seed" else [f"random_start_{i}" for i in range(n_rows)]
    out = _result_dict(
        path,
        "composition",
        seed_labels,
        after_qc,
        achieved,
        after_reg,
        reg_names,
        weights,
        optimized_desc.detach().cpu().numpy(),
        elapsed,
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
    after_qc: np.ndarray,
    achieved: np.ndarray,
    after_reg: dict[str, np.ndarray],
    reg_names: list[str],
    weights: np.ndarray,
    descriptor: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "path": path.name,
        "method": method,
        "ae_align_scale": path.ae_align_scale if method == "latent" else None,
        "elapsed_s": elapsed,
        "seeds": seeds,
        "qc_after_decode": after_qc.tolist(),
        "reg_achieved_latent": {t: achieved[:, j].tolist() for j, t in enumerate(reg_names)},
        "reg_after_decode": {t: after_reg[t].tolist() for t in reg_names},
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


def _reg_progress(
    targets: np.ndarray, reg_names: Sequence[str], reg_targets: Mapping[str, float], seed_reg: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-step mean progress (0 = seed baseline, 1 = target) for each regression task."""
    progress: dict[str, np.ndarray] = {}
    for j, task in enumerate(reg_names):
        baseline = np.asarray(seed_reg[task], dtype=float)  # (B,)
        denom = float(reg_targets[task]) - baseline
        denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
        traj = targets[:, :, j]  # (steps, B)
        progress[task] = ((traj - baseline[None, :]) / denom[None, :]).mean(axis=1)
    return progress


def _emit_trajectory(
    result: dict[str, Any],
    targets: np.ndarray,
    weights: np.ndarray,
    reg_targets: Mapping[str, float],
    seed_reg: Mapping[str, np.ndarray],
    cfg: InverseConfig,
    traj_dir: Path,
) -> None:
    """Write the static trajectory plot (+ requested animations) for one path."""
    reg_names = list(reg_targets)
    if targets.size == 0:
        return
    progress = _reg_progress(targets, reg_names, reg_targets, seed_reg)
    inverse_trajectory.plot_trajectory_static(
        progress, traj_dir / f"{result['path']}_trajectory.png", title=result["path"]
    )

    if cfg.animation_formats and weights.size:
        qc_final = np.asarray(result["qc_after_decode"])
        reg_final = {t: np.asarray(result["reg_after_decode"][t]) for t in reg_names}
        best = min(
            inverse_trajectory.best_seed_by_target_distance(qc_final, reg_final, reg_targets), weights.shape[1] - 1
        )
        out_paths = {fmt: traj_dir / f"{result['path']}_trajectory.{fmt}" for fmt in cfg.animation_formats}
        inverse_trajectory.plot_trajectory_animation(
            progress, weights[:, best, :], list(DEFAULT_ELEMENTS), out_paths, title=result["path"]
        )

    if cfg.per_seed_trajectories:
        per_dir = traj_dir / f"{result['path']}_per_seed"
        per_dir.mkdir(exist_ok=True)
        for i in range(min(targets.shape[1], 20)):  # cap the per-seed fan-out
            ps: dict[str, np.ndarray] = {}
            for j, task in enumerate(reg_names):
                baseline = float(np.asarray(seed_reg[task])[i])
                denom = float(reg_targets[task]) - baseline
                ps[task] = (targets[:, i, j] - baseline) / (denom if abs(denom) >= 1e-9 else 1.0)
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
        seeds = select_seeds(
            catalog,
            model,
            cfg.seeds,
            class_task=seed_scn.class_task,
            class_indices=seed_scn.class_indices,
            device=device,
        )
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
        needed = set(scenario.reg_tasks) | {scenario.class_task}
        missing = sorted(needed - heads)
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
    reg_targets = scenario.reg_target_map
    seed_qc = _qc_prob(model, x_seed, scenario.class_task, scenario.class_indices)
    seed_reg = _reg_preds(model, x_seed, list(reg_targets))

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
                class_weight=cfg.class_weight,
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
                class_weight=cfg.class_weight,
                steps=cfg.steps,
                lr=cfg.lr,
                record_trajectory=cfg.record_trajectory,
            )
        results.append(r)

    summary = _summarise(results, reg_targets)

    # Trajectory outputs: static plot + requested animations, then externalize arrays to .npz.
    if cfg.record_trajectory:
        traj_dir = sc_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        for r in results:
            if "trajectory_targets" not in r:
                continue
            targets = np.asarray(r["trajectory_targets"], dtype=np.float32)
            weights = np.asarray(r["trajectory_weights"], dtype=np.float32)
            _emit_trajectory(r, targets, weights, reg_targets, seed_reg, cfg, traj_dir)
            npz = traj_dir / f"{r['path']}.npz"
            np.savez_compressed(npz, targets=targets, weights=weights)
            r["trajectory_file"] = str(npz.relative_to(sc_dir))
            del r["trajectory_targets"]
            del r["trajectory_weights"]

    (sc_dir / "scenario.json").write_text(
        json.dumps(
            {
                "name": scenario.name,
                "reg_tasks": scenario.reg_tasks,
                "reg_targets": scenario.reg_targets,
                "class_task": scenario.class_task,
                "class_indices": scenario.class_indices,
                "primary_objective": f"P({scenario.class_task} in {scenario.class_indices}) ↑",
                "checkpoint": str(cfg.checkpoint),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (sc_dir / "results.json").write_text(
        json.dumps(
            {
                "reg_targets": reg_targets,
                "seed_predictions": {"qc": seed_qc.tolist(), "reg": {t: v.tolist() for t, v in seed_reg.items()}},
                "results": results,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (sc_dir / "targets.json").write_text(json.dumps(reg_targets, indent=2), encoding="utf-8")
    (sc_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_scenario_md(sc_dir, scenario, summary)

    # figures
    rel = scenario.name
    _plot_comparison(results, reg_targets, rec, f"{rel}/comparison.png")
    _plot_qc_vs_secondary(results, reg_targets, seed_qc, seed_reg, rec, f"{rel}/qc_vs_secondary_scatter.png")
    _plot_element_frequency(results, list(seeds), rec, f"{rel}/element_frequency_heatmap.png")
    for r in results:
        if r["method"] == "composition" and r["path"].endswith("random"):
            continue  # random init: no per-seed correspondence
        _plot_seed_to_optimized(list(seeds), r, seed_qc, rec, f"{rel}/seed_to_optimized__{r['path']}.png")
    return summary


def _summarise(results: list[dict[str, Any]], reg_targets: Mapping[str, float]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "path": r["path"],
            "method": r["method"],
            "ae_align_scale": r["ae_align_scale"],
            "elapsed_s": round(r["elapsed_s"], 2),
            "qc_after_mean": round(float(np.mean(r["qc_after_decode"])), 4),
            "qc_after_std": round(float(np.std(r["qc_after_decode"])), 4),
        }
        for t in reg_targets:
            vals = np.asarray(r["reg_after_decode"][t], dtype=float)
            row[f"{t}_after_mean"] = round(float(vals.mean()), 3)
            row[f"{t}_after_std"] = round(float(vals.std()), 3)
        rows.append(row)
    return rows


# --- figures (compact reimplementations) --------------------------------------------------


def _method_color(method: str) -> str:
    return "#55A868" if method == "latent" else SCATTER_COLOR


def _plot_comparison(
    results: list[dict[str, Any]], reg_targets: Mapping[str, float], rec: RunRecorder, rel: str
) -> None:
    panels = ["QC", *reg_targets]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 5.0), squeeze=False)
    labels = [r["path"] for r in results]
    colors = [_method_color(r["method"]) for r in results]
    x = np.arange(len(results))
    for ax, panel in zip(axes[0], panels):
        if panel == "QC":
            means = [float(np.mean(r["qc_after_decode"])) for r in results]
            stds = [float(np.std(r["qc_after_decode"])) for r in results]
            ax.axhline(1.0, color="#C44E52", ls="--", lw=1.0)
            ax.set_title("P(QC)")
        else:
            means = [float(np.mean(r["reg_after_decode"][panel])) for r in results]
            stds = [float(np.std(r["reg_after_decode"][panel])) for r in results]
            ax.axhline(float(reg_targets[panel]), color="#C44E52", ls="--", lw=1.0)
            ax.set_title(panel)
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.set_xticks(x, labels, rotation=75, ha="right", fontsize=7)
    fig.suptitle("Inverse-design paths — achieved objectives", y=1.02)
    rec.save_figure(rel, fig)
    plt.close(fig)


def _plot_qc_vs_secondary(
    results: list[dict[str, Any]],
    reg_targets: Mapping[str, float],
    seed_qc: np.ndarray,
    seed_reg: dict[str, np.ndarray],
    rec: RunRecorder,
    rel: str,
) -> None:
    reg_names = list(reg_targets)
    fig, axes = plt.subplots(1, len(reg_names), figsize=(5.2 * len(reg_names), 5.0), squeeze=False)
    for ax, task in zip(axes[0], reg_names):
        ax.scatter(seed_qc, seed_reg[task], marker="*", s=70, color=DISCOVERED_ELEMENT_COLOR, label="seed", zorder=1)
        for r in results:
            ax.scatter(
                r["qc_after_decode"],
                r["reg_after_decode"][task],
                s=18,
                alpha=0.6,
                color=_method_color(r["method"]),
                zorder=2,
            )
        ax.axvline(1.0, color="#C44E52", ls="--", lw=1.0)
        ax.axhline(float(reg_targets[task]), color="#C44E52", ls="--", lw=1.0)
        ax.set_xlabel("P(QC)")
        ax.set_ylabel(task)
    fig.suptitle("QC probability vs secondary properties", y=1.02)
    rec.save_figure(rel, fig)
    plt.close(fig)


def _plot_seed_to_optimized(
    seeds: list[str], result: dict[str, Any], seed_qc: np.ndarray, rec: RunRecorder, rel: str
) -> None:
    decoded = result["decoded_composition"]
    opt_qc = np.asarray(result["qc_after_decode"])
    n = min(len(seeds), len(decoded))
    fig, ax = plt.subplots(figsize=(9.0, max(3.0, 0.32 * n + 1)))
    ax.axis("off")
    ax.set_title(f"Seed → optimised · {result['path']}", fontsize=11)
    for i in range(n):
        y = n - i
        ax.text(0.01, y, f"{seeds[i]}  (QC={seed_qc[i] * 100:.0f}%)", fontsize=8, family="monospace", va="center")
        ax.text(0.42, y, "→", fontsize=10, va="center")
        ax.text(0.47, y, f"{decoded[i]}  (QC={opt_qc[i] * 100:.0f}%)", fontsize=8, family="monospace", va="center")
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
    lines = [f"# Inverse design — {scenario.name}", ""]
    lines.append(f"Primary objective: P({scenario.class_task} in {scenario.class_indices}) ↑")
    lines.append(f"Regression targets: {scenario.reg_target_map}")
    lines.append("")
    lines.append("| path | method | QC (mean±std) | elapsed(s) |")
    lines.append("|---|---|---|---|")
    for row in summary:
        lines.append(
            f"| {row['path']} | {row['method']} | {row['qc_after_mean']}±{row['qc_after_std']} | {row['elapsed_s']} |"
        )
    (sc_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_summary(root: Path, all_summary: Mapping[str, Any], cfg: InverseConfig) -> None:
    lines = ["# Inverse design — all scenarios", "", f"Checkpoint: `{cfg.checkpoint}`", ""]
    for name, summary in all_summary.items():
        best = max(summary, key=lambda row: row["qc_after_mean"]) if summary else None
        best_str = f"best QC path: **{best['path']}** ({best['qc_after_mean']})" if best else "(no paths)"
        lines.append(f"- **{name}** — {best_str}")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
