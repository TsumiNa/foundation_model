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
    raw = _raw(user_data, bundle, tmp_path / "d", ["band_gap"])
    raw["descriptor"]["n_grids"] = 6
    with pytest.raises(ValueError, match=r"\[descriptor\] must match the bundle.*n_grids = 4"):
        finetune_run(build_finetune_config(raw))
    raw = _raw(user_data, bundle, tmp_path / "z", ["hardness"], add_new_tasks=False)
    with pytest.raises(ValueError, match="not in the bundle and add_new_tasks = false"):
        finetune_run(build_finetune_config(raw))


@pytest.fixture
def curve_data(tmp_path: Path) -> Path:
    """Curves in physical units: heterogeneous lengths, an extended T range, a zero and a missing DOS row."""
    rng = np.random.default_rng(11)
    n = len(_FORMULAS)
    temps = [np.linspace(250 + 5 * i, 900, 4 + (i % 3)) for i in range(n)]  # beyond the source 300–745 K
    dos_e = np.linspace(-6, 10, 7)
    dos: list[np.ndarray | None] = [rng.normal(size=7) for _ in range(n)]
    dos[0] = np.zeros(7)
    dos[1] = None
    path = tmp_path / "curves.parquet"
    pd.DataFrame(
        {
            "composition": _FORMULAS,
            "S_VK": [rng.normal(1e-4, 5e-5, t.size) for t in temps],
            "T_K": temps,
            "dos": dos,
            "E": [dos_e] * n,
            "sigma": [rng.lognormal(10, 1, t.size) for t in temps],
            "split": (["train", "train", "val", "test"] * n)[:n],
        }
    ).to_parquet(path)
    return path


def _curve_raw(data: Path, bundle: Path, out: Path, tasks: list[str], **seebeck: object) -> dict:
    return {
        "descriptor": {"kind": "kmd", "n_grids": 4},
        "datasets": {"c": {"path": str(data)}},
        "tasks": [
            {"name": "seebeck", "kind": "kernel_regression", "dataset": "c", "column": "S_VK", "t_column": "T_K",
             "unit": "V/K", "t_unit": "K", **seebeck},
            {"name": "dos", "kind": "kernel_regression", "dataset": "c", "column": "dos", "t_column": "E"},
            {"name": "sigma", "kind": "kernel_regression", "dataset": "c", "column": "sigma", "t_column": "T_K",
             "unit": "S/m", "t_unit": "K"},
        ],
        "training": {"accelerator": "cpu", "seed": 5},
        "finetune": {"pretrained": str(bundle), "tasks": tasks, "epochs": 1},
        "output": {"dir": str(out)},
    }  # fmt: skip


def test_finetune_curve_heads(bundle: Path, curve_data: Path, tmp_path: Path) -> None:
    out = tmp_path / "ftc"
    finetune_run(build_finetune_config(_curve_raw(curve_data, bundle, out, ["seebeck", "dos", "sigma"])))
    source, derived = read_config(bundle), read_config(out / "bundle")

    seebeck = derived.head("seebeck")  # existing curve head: same transform, metadata covers the new T
    assert seebeck.transform == source.head("seebeck").transform
    assert seebeck.t_range is not None and seebeck.t_range[1] == pytest.approx(900.0)
    assert seebeck.t_range[0] == pytest.approx(min(250.0, source.head("seebeck").t_range[0]))  # type: ignore[index]
    assert seebeck.t_grid[0] == pytest.approx(seebeck.t_range[0]) and seebeck.t_grid[-1] == pytest.approx(900.0)
    assert len(seebeck.t_grid) == len(source.head("seebeck").t_grid)

    dos = derived.head("dos")  # shape head inherited without output_scale in the config
    assert dos.output_scale == "shape" and dos.transform is None

    sigma = derived.head("sigma")  # new curve head: transform fitted on train rows only, grid kept on reload
    assert sigma.transform is not None and sigma.unit == "S/m" and sigma.t_unit == "K"
    frame = pd.read_parquet(curve_data)
    train = np.concatenate(frame.loc[frame.split == "train", "sigma"].to_list())
    np.testing.assert_allclose(np.mean(sigma.transform.transform(train)), 0.0, atol=1e-8)
    model = load_pretrained(out / "bundle")
    curve = model.predict_curve(_FORMULAS[:2], "sigma")
    assert len(curve) == 2 * len(sigma.t_grid) and np.isfinite(curve["sigma"]).all()


def test_existing_head_unit_and_scale_conflicts(bundle: Path, curve_data: Path, tmp_path: Path) -> None:
    raw = _curve_raw(curve_data, bundle, tmp_path / "u", ["seebeck"], unit="µV/K")
    with pytest.raises(ValueError, match="unit 'µV/K' differs from the bundle head's 'V/K'"):
        finetune_run(build_finetune_config(raw))
    raw = _curve_raw(curve_data, bundle, tmp_path / "t", ["seebeck"], t_unit="°C")
    with pytest.raises(ValueError, match="t_unit '°C' differs"):
        finetune_run(build_finetune_config(raw))
    raw = _curve_raw(curve_data, bundle, tmp_path / "s", ["seebeck"], output_scale="shape")
    with pytest.raises(ValueError, match="output_scale 'shape' disagrees"):
        finetune_run(build_finetune_config(raw))


def test_existing_head_honours_task_lr(
    bundle: Path, user_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import foundation_model.workflows.finetune as ft

    seen: dict[str, float] = {}
    real_fit = ft._fit

    def spy(model, catalog, cfg, rec):  # noqa: ANN001
        seen["lr"] = model.task_configs_map["band_gap"].optimizer.lr
        return real_fit(model, catalog, cfg, rec)

    monkeypatch.setattr(ft, "_fit", spy)
    raw = _raw(user_data, bundle, tmp_path / "lr", ["band_gap"])
    raw["tasks"][0]["lr"] = 3.3e-4
    finetune_run(build_finetune_config(raw))
    assert seen["lr"] == pytest.approx(3.3e-4)


def test_shape_input_overrides_a_configured_scaler(tmp_path: Path) -> None:
    import joblib  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

    joblib.dump(StandardScaler().fit([[0.0], [10.0]]), tmp_path / "s.pkl")
    raw = {"descriptor": {"kind": "kmd", "n_grids": 4}, "datasets": {"d": {"path": "x.parquet"}},
           "tasks": [{"name": "c", "kind": "kernel_regression", "dataset": "d", "column": "y", "t_column": "t",
                      "scaler": {"path": str(tmp_path / "s.pkl")}}]}  # fmt: skip
    catalog = TaskCatalog(build_task_catalog_config(raw), target_inputs={"c": "shape"})
    assert catalog.scaler("c") is None
    np.testing.assert_array_equal(catalog.inverse_transform("c", np.array([0.5])), [0.5])
