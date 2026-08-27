# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Choosing which compositions a scenario starts from.

Seeds are selected once per run and shared by every path, so the paths are compared on the same
starting points rather than on the luck of their draws. The policies — best-QC, random, an explicit
list — plus the element-system dedup that stops a scenario spending all its restarts on variations
of one alloy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, formula_to_composition  # noqa: E402

from .._engine import build_model_for_checkpoint, checkpoint_task_order, descriptor_tensor  # noqa: E402
from ..recording import load_checkpoint_state  # noqa: E402
from ..task_catalog import TaskCatalog  # noqa: E402
from .config import _ELEMENT_TOKEN, InverseConfig, SeedConfig, SeedStrategy, TargetSpec  # noqa: E402


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
    x, pool = descriptor_tensor(catalog, pool, device)
    model_targets = [t.to_model_target() for t in targets]
    scores = [
        model.evaluate_targets(x[i : i + 4096], model_targets)[1].cpu().numpy() for i in range(0, len(pool), 4096)
    ]
    objective = np.concatenate(scores) if scores else np.zeros(0)
    ranked = [pool[i] for i in np.argsort(objective, kind="stable")]
    return _merge(_dedup_by_system(ranked, n_strategy, enabled=seed_cfg.dedup_by_element_system))
