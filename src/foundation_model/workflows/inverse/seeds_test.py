# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for seed selection.

Seeds are chosen once per run and shared by every path, so a bug here biases the whole comparison
rather than one column of it.
"""

from __future__ import annotations

import numpy as np
import torch

from foundation_model.workflows.inverse import SeedConfig, SeedStrategy, select_seeds

from .conftest import _model_with_heads, _spec


def test_weighted_random_selection_deterministic_and_from_pool(data_dir) -> None:
    cat, model = _model_with_heads(data_dir)
    seed_cfg = SeedConfig(strategy="weighted_random", weight_task="a", n=4, split="all")
    specs = [_spec(cat, task="a", value=1.0)]
    s1 = select_seeds(cat, model, seed_cfg, targets=specs, device=torch.device("cpu"))
    s2 = select_seeds(cat, model, seed_cfg, targets=specs, device=torch.device("cpu"))
    assert s1 == s2 and len(s1) == 4
    assert set(s1) <= set(str(c) for c in cat.task_frames(["a"])["a"].index)


def test_weighted_random_direction_and_value_bias(data_dir) -> None:
    """With n = pool_size - 1, the excluded candidate reveals the weighting: the sole miss should
    sit at the unfavored end (statistically; deterministic here via the fixed rng)."""
    cat, model = _model_with_heads(data_dir)
    frame = cat.task_frames(["a"])["a"]
    spec = cat.task_spec("a")
    labels = frame[spec.column].astype(float)
    ordered_by_label = [str(c) for c in labels.sort_values().index]
    n = len(ordered_by_label) - 1
    specs = [_spec(cat, task="a", value=1.0)]

    def picked(**kw):
        cfg = SeedConfig(
            strategy="weighted_random", weight_task="a", n=n, split="all", dedup_by_element_system=False, **kw
        )
        return select_seeds(cat, model, cfg, targets=specs, device=torch.device("cpu"))

    for kw, favored_end in (
        ({"weight_direction": "high"}, ordered_by_label[-1]),
        ({"weight_direction": "low"}, ordered_by_label[0]),
    ):
        got = picked(**kw)
        assert len(got) == n
        assert favored_end in got  # the best-matching candidate must survive an n-1 draw
    v = float(labels.median())
    got = picked(weight_value=v)
    closest = str((labels - v).abs().sort_values().index[0])
    assert closest in got


def test_seed_selection_explicit_verbatim(data_dir) -> None:
    cat, model = _model_with_heads(data_dir)
    seed_cfg = SeedConfig(strategy=SeedStrategy.EXPLICIT, n=2, explicit=["Fe2 O3", "Al2 O3"])
    seeds = select_seeds(cat, model, seed_cfg, targets=[_spec(cat, task="a", value=1.0)], device=torch.device("cpu"))
    assert set(seeds) <= {"Fe2 O3", "Al2 O3"}


def test_seed_selection_explicit_append_reduces_budget(data_dir) -> None:
    cat, model = _model_with_heads(data_dir)
    seed_cfg = SeedConfig(strategy=SeedStrategy.TOP_OBJECTIVE, n=4, split="all", explicit_append=["Fe2 O3"])
    seeds = select_seeds(cat, model, seed_cfg, targets=[_spec(cat, task="a", value=1.0)], device=torch.device("cpu"))
    assert "Fe2 O3" in seeds  # appended seed always survives
    assert len(seeds) <= 4


def test_seed_selection_works_without_classification_target(data_dir) -> None:
    # Regression guard for the QC de-hardcoding: the candidate pool must not depend on any
    # classification head/frame.
    cat, model = _model_with_heads(data_dir)
    seed_cfg = SeedConfig(strategy=SeedStrategy.RANDOM, n=3, split="all")
    seeds = select_seeds(
        cat, model, seed_cfg, targets=[_spec(cat, task="a", direction="low")], device=torch.device("cpu")
    )
    assert len(seeds) == 3


def test_top_objective_ranking_matches_evaluate_targets(data_dir) -> None:
    cat, model = _model_with_heads(data_dir)
    specs = [_spec(cat, task="a", value=5.0)]
    seed_cfg = SeedConfig(strategy=SeedStrategy.TOP_OBJECTIVE, n=3, split="all", dedup_by_element_system=False)
    seeds = select_seeds(cat, model, seed_cfg, targets=specs, device=torch.device("cpu"))

    # Hand-compute the expected ranking with the exact same pool order + scoring call.
    frame = cat.task_frames(["a"])["a"]
    descriptor_fn = cat.descriptor_fn()
    pool = [c for c in frame.index if not descriptor_fn([c]).empty]
    desc = descriptor_fn(pool)
    kept = [c for c in pool if c in desc.index]
    x = torch.tensor(desc.loc[kept].values, dtype=torch.float32)
    _, objective = model.evaluate_targets(x, [s.to_model_target() for s in specs])
    expected = [kept[i] for i in np.argsort(objective.numpy(), kind="stable")][:3]
    assert seeds == expected


def test_the_run_seed_actually_varies_the_draw(data_dir):
    """Two seeds must give different starts; the same seed must repeat.

    Both random strategies used a literal default_rng(0), so [inverse].seed reached
    seed_everything but never reached here — every run of a stochastic search began from exactly
    the same compositions, which is the one thing the knob exists to prevent.
    """
    cat, model = _model_with_heads(data_dir)
    specs = [_spec(cat, task="a", value=1.0)]
    cfg = SeedConfig(strategy=SeedStrategy.RANDOM, n=4, split="all", dedup_by_element_system=False)

    def pick(seed):
        return select_seeds(cat, model, cfg, targets=specs, device=torch.device("cpu"), seed=seed)

    assert pick(2025) == pick(2025), "the same seed must reproduce the same seeds"
    assert pick(2025) != pick(7), "different seeds must give different seeds"


def test_weighted_random_also_honours_the_run_seed(data_dir):
    cat, model = _model_with_heads(data_dir)
    specs = [_spec(cat, task="a", value=1.0)]
    cfg = SeedConfig(strategy="weighted_random", weight_task="a", n=4, split="all", dedup_by_element_system=False)

    def pick(seed):
        return select_seeds(cat, model, cfg, targets=specs, device=torch.device("cpu"), seed=seed)

    assert pick(2025) == pick(2025)
    assert pick(2025) != pick(7)
