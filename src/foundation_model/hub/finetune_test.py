# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""End to end: export → fine-tune from the bundle on new data only → derived bundle → load → predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foundation_model.hub import load_pretrained, read_config, read_manifest
from foundation_model.workflows.finetune import build_finetune_config
from foundation_model.workflows.finetune import run as finetune_run

from .export_test import _FORMULAS, _checkpoint, _export, catalog_raw  # noqa: F401  (catalog_raw is a fixture)


@pytest.fixture
def bundle(catalog_raw: dict, tmp_path: Path) -> Path:  # noqa: F811
    return _export(catalog_raw, [_checkpoint(catalog_raw, tmp_path / "c.pt", 1)], tmp_path / "bundle")


@pytest.fixture
def user_data(tmp_path: Path) -> Path:
    """The user's own table, in physical units, with no relation to the pretraining files."""
    rng = np.random.default_rng(7)
    n = len(_FORMULAS)
    path = tmp_path / "mine.parquet"
    pd.DataFrame(
        {
            "composition": _FORMULAS,
            "gap_eV": rng.lognormal(0.3, 0.6, n),  # existing head, physical units
            "phase_name": rng.choice(["insulator", "metal", "semiconductor"], n),  # existing head, by name
            "hardness_GPa": rng.normal(10, 3, n),  # new head
            "split": (["train", "train", "val", "test"] * n)[:n],
        }
    ).to_parquet(path)
    return path


def _raw(user_data: Path, bundle: Path, out: Path, tasks: list[str], **finetune: object) -> dict:
    return {
        "descriptor": {"kind": "kmd", "n_grids": 4},
        "datasets": {"mine": {"path": str(user_data)}},
        "tasks": [
            {"name": "band_gap", "kind": "regression", "dataset": "mine", "column": "gap_eV", "unit": "eV"},
            {"name": "phase", "kind": "classification", "dataset": "mine", "column": "phase_name", "num_classes": 3},
            {"name": "hardness", "kind": "regression", "dataset": "mine", "column": "hardness_GPa", "unit": "GPa"},
        ],
        "training": {"accelerator": "cpu", "seed": 3, "max_epochs": 1},
        "finetune": {"pretrained": str(bundle), "tasks": tasks, "epochs": 1, **finetune},
        "output": {"dir": str(out)},
    }


def test_finetune_from_bundle_writes_a_derived_bundle(bundle: Path, user_data: Path, tmp_path: Path) -> None:
    out = tmp_path / "ft"
    cfg = build_finetune_config(_raw(user_data, bundle, out, ["band_gap", "phase", "hardness"]))
    summary = finetune_run(cfg)
    assert summary["added_tasks"] == ["hardness"] and summary["license"] == "cc-by-nc-sa-4.0"
    assert (out / "training" / "final_model.pt").exists()

    derived = read_config(out / "bundle")
    assert derived.license == "cc-by-nc-sa-4.0"  # ShareAlike: inherited, not chosen
    lineage = derived.provenance["derived_from"]
    assert lineage["source"] == str(bundle) and lineage["member"] == "m000"
    names = [h.name for h in derived.heads]
    assert sorted(names[:-1]) == ["band_gap", "dos", "phase", "seebeck"] and names[-1] == "hardness"
    hardness = derived.head("hardness")
    assert hardness.unit == "GPa" and hardness.transform is not None  # fitted on the user's train split
    # Existing heads keep the source transform, so their outputs stay in the same physical units.
    assert derived.head("band_gap").transform == read_config(bundle).head("band_gap").transform
    assert read_manifest(out / "bundle").member().task_order[-1] == "hardness"

    model = load_pretrained(out / "bundle")
    pred = model.predict(_FORMULAS[:4], ["band_gap", "hardness", "phase"])
    assert pred["hardness"].between(-10, 40).all()  # GPa, not the normalised scale
    assert set(pred["phase"]) <= {"insulator", "metal", "semiconductor"}


def test_derived_bundles_chain_their_lineage(bundle: Path, user_data: Path, tmp_path: Path) -> None:
    first = tmp_path / "ft1"
    finetune_run(build_finetune_config(_raw(user_data, bundle, first, ["hardness"])))
    second = tmp_path / "ft2"
    finetune_run(build_finetune_config(_raw(user_data, first / "bundle", second, ["band_gap"])))
    prov = read_config(second / "bundle").provenance
    assert prov["derived_from"]["source"] == str(first / "bundle")
    assert prov["lineage"][0]["source"] == str(bundle)


def test_save_keeps_licence_and_records_source(bundle: Path, tmp_path: Path) -> None:
    saved = load_pretrained(bundle).save(tmp_path / "copy")
    config = read_config(saved)
    assert config.license == "cc-by-nc-sa-4.0" and config.provenance["derived_from"]["source"] == str(bundle)
    a = load_pretrained(bundle).predict(_FORMULAS[:3], ["band_gap"])
    b = load_pretrained(saved).predict(_FORMULAS[:3], ["band_gap"])
    pd.testing.assert_frame_equal(a, b)


def test_pretrained_config_rejections(bundle: Path, user_data: Path, tmp_path: Path) -> None:
    raw = _raw(user_data, bundle, tmp_path / "x", ["band_gap"])
    with pytest.raises(ValueError, match=r"\[model\] does not apply"):
        build_finetune_config({**raw, "model": {"latent_dim": 4}})
    with pytest.raises(ValueError, match="not both"):
        build_finetune_config(raw, checkpoint="c.pt")
    raw["tasks"][0]["scaler"] = {"path": "s.pkl"}
    with pytest.raises(ValueError, match=r"no \[tasks.scaler\]"):
        finetune_run(build_finetune_config(raw))
    raw = _raw(user_data, bundle, tmp_path / "y", ["band_gap"])
    raw["tasks"][0]["kind"] = "classification"
    raw["tasks"][0]["num_classes"] = 2
    with pytest.raises(ValueError, match="bundle's head is regression"):
        finetune_run(build_finetune_config(raw))
    raw = _raw(user_data, bundle, tmp_path / "z", ["hardness"], add_new_tasks=False)
    with pytest.raises(ValueError, match="not in the bundle and add_new_tasks = false"):
        finetune_run(build_finetune_config(raw))
