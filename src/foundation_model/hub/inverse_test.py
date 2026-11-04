# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""``fm inverse --pretrained``: physical-unit targets and class names, seeds without data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from foundation_model.hub import read_config
from foundation_model.workflows.inverse import build_inverse_config
from foundation_model.workflows.inverse import run as inverse_run
from foundation_model.workflows.inverse.seeds import random_compositions

from .export_test import _checkpoint, _export, catalog_raw  # noqa: F401  (catalog_raw is a fixture)

_SEEDS = ["Fe2 Al3", "Cu2 Ni3", "Mg2 O3", "Ti2 Zn3"]
_PATHS = [
    {"name": "latent", "method": "latent", "ae_align_scale": 0.25},
    {"name": "comp", "method": "composition", "init": "seed", "seed_blend": 0.95},
]


def _inverse(targets: list[dict[str, Any]], out: Path, *, seeds: dict[str, Any], **source: Any) -> dict[str, Any]:
    return {
        "inverse": {
            "steps": 3,
            "record_trajectory": False,
            "animation_formats": [],
            "accelerator": "cpu",
            "seed": 11,
            **source,
            "seeds": seeds,
            "paths": _PATHS,
            "scenarios": [{"name": "s", "targets": targets}],
        },
        "output": {"dir": str(out)},
    }


@pytest.fixture
def bundle_and_ckpt(catalog_raw: dict, tmp_path: Path) -> tuple[Path, Path]:  # noqa: F811
    ckpt = _checkpoint(catalog_raw, tmp_path / "c.pt", 0)
    return _export(catalog_raw, [ckpt], tmp_path / "bundle"), ckpt


def test_bundle_path_equals_catalog_path_after_unit_conversion(
    catalog_raw: dict,  # noqa: F811
    bundle_and_ckpt: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    bundle, ckpt = bundle_and_ckpt
    gap_t = read_config(bundle).head("band_gap").transform
    sb_t = read_config(bundle).head("seebeck").transform
    assert gap_t is not None and sb_t is not None
    physical: list[dict[str, Any]] = [
        {"task": "band_gap", "value": 1.5},
        {"task": "phase", "classes": ["metal"], "weight": 2.0},
        {"task": "seebeck", "points": [[350.0, 2e-4], [450.0, 3e-4]]},
    ]
    normalised: list[dict[str, Any]] = [
        {"task": "band_gap", "value": float(gap_t.transform([1.5])[0])},
        {"task": "phase", "classes": [1], "weight": 2.0},  # metal is index 1 of insulator/metal/semiconductor
        {"task": "seebeck", "points": [[350.0, float(sb_t.transform([2e-4])[0])], [450.0, float(sb_t.transform([3e-4])[0])]]},
    ]  # fmt: skip
    seeds = {"strategy": "explicit", "explicit": _SEEDS, "n": 4, "dedup_by_element_system": False}

    from_bundle = build_inverse_config(_inverse(physical, tmp_path / "a", seeds=seeds, pretrained=str(bundle)))
    assert [t.value for t in from_bundle.scenarios[0].targets[:1]] == [normalised[0]["value"]]
    a = inverse_run(from_bundle)

    raw = {**{k: v for k, v in catalog_raw.items()}, "model": {"latent_dim": 8, "encoder_hidden_dims": [16],
           "head_hidden_dims": [8], "n_kernel": 4}, **_inverse(normalised, tmp_path / "b", seeds=seeds, checkpoint=str(ckpt))}  # fmt: skip
    b = inverse_run(build_inverse_config(raw))

    for row_a, row_b in zip(a["s"], b["s"]):
        assert row_a["objective_mean"] == pytest.approx(row_b["objective_mean"], abs=1e-6)
        assert row_a["band_gap_after_mean"] == pytest.approx(row_b["band_gap_after_mean"], abs=1e-6)
        # Bundle runs also report the regression channel back in eV.
        assert "band_gap_after_mean_physical" in row_a and "band_gap_after_mean_physical" not in row_b
    scenario = json.loads((tmp_path / "a" / "s" / "scenario.json").read_text())
    assert scenario["pretrained"] == str(bundle) and scenario["targets_physical"][1]["classes"] == ["metal"]


def test_element_seeds_need_no_data(bundle_and_ckpt: tuple[Path, Path], tmp_path: Path) -> None:
    bundle, _ = bundle_and_ckpt
    seeds = {"strategy": "elements", "elements": ["Fe", "Ni", "Al"], "n": 5}
    cfg = build_inverse_config(_inverse([{"task": "band_gap", "direction": "high"}], tmp_path / "o", seeds=seeds,
                                        pretrained=str(bundle)))  # fmt: skip
    inverse_run(cfg)
    chosen = json.loads((tmp_path / "o" / "seeds.json").read_text())["seeds"]
    assert len(chosen) == 5 and all(set(c.replace(".", " ").split()) >= {"Fe0", "Ni0", "Al0"} for c in chosen)


def test_random_compositions_are_reproducible_and_positive() -> None:
    a = random_compositions(["Fe", "Co"], 6, seed=3)
    assert a == random_compositions(["Fe", "Co"], 6, seed=3) and len(set(a)) == 6
    for comp in a:
        fractions = [float(part[2:]) for part in comp.split()]
        assert all(f > 0 for f in fractions) and sum(fractions) == pytest.approx(1.0, abs=2e-4)


def test_bundle_rejections(bundle_and_ckpt: tuple[Path, Path], catalog_raw: dict, tmp_path: Path) -> None:  # noqa: F811
    bundle, _ = bundle_and_ckpt
    explicit = {"strategy": "explicit", "explicit": _SEEDS}
    with pytest.raises(ValueError, match="carries no data to draw seeds from"):
        build_inverse_config(_inverse([{"task": "band_gap", "value": 1.0}], tmp_path / "x",
                                      seeds={"strategy": "top_objective"}, pretrained=str(bundle)))  # fmt: skip
    with pytest.raises(ValueError, match="do not apply when designing from a released bundle"):
        build_inverse_config({**catalog_raw, **_inverse([], tmp_path / "x", seeds=explicit, pretrained=str(bundle))})
    with pytest.raises(ValueError, match="unknown class name"):
        build_inverse_config(_inverse([{"task": "phase", "classes": ["glass"]}], tmp_path / "x", seeds=explicit,
                                      pretrained=str(bundle)))  # fmt: skip
    with pytest.raises(ValueError, match="at least two distinct elements"):
        build_inverse_config(_inverse([{"task": "band_gap", "value": 1.0}], tmp_path / "x",
                                      seeds={"strategy": "elements", "elements": ["Fe"]}, pretrained=str(bundle)))  # fmt: skip


def test_shape_head_targets_stay_on_the_shape_scale(bundle_and_ckpt: tuple[Path, Path], tmp_path: Path) -> None:
    bundle, _ = bundle_and_ckpt
    cfg = build_inverse_config(
        _inverse([{"task": "dos", "points": [[0.0, 0.5], [2.0, 1.0]]}], tmp_path / "x",
                 seeds={"strategy": "explicit", "explicit": _SEEDS}, pretrained=str(bundle))
    )  # fmt: skip
    np.testing.assert_allclose(np.array(cfg.scenarios[0].targets[0].points), [[0.0, 0.5], [2.0, 1.0]])
