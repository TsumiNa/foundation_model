# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for ``fm export`` and the bundle round trip (no network, tiny model)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler  # type: ignore[import-untyped]

from foundation_model.cli.main import main
from foundation_model.hub import BundleError, export_runs, load_member, read_config, read_manifest
from foundation_model.hub.bundle import CONFIG_FILE, MEMBERS_DIR, WEIGHTS_FILE, DescriptorSpec
from foundation_model.hub.export import default_grid
from foundation_model.workflows._engine import build_model_for_checkpoint
from foundation_model.workflows._sections import ModelSectionConfig
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

_ELEMENTS = ["Fe", "Al", "Cu", "Ni", "Ti", "Zn", "Mg", "Ca", "Na", "Cl", "O", "Si"]
_FORMULAS = [f"{a}2 {b}3" for i, a in enumerate(_ELEMENTS) for b in _ELEMENTS[i + 1 :]][:30]
_MODEL = ModelSectionConfig(latent_dim=8, encoder_hidden_dims=[16], head_hidden_dims=[8], n_kernel=4)
_DOS_GRID = np.linspace(-6, 10, 7)


@pytest.fixture
def catalog_raw(tmp_path: Path) -> dict:
    rng = np.random.default_rng(0)
    n = len(_FORMULAS)
    gap = rng.lognormal(0.5, 0.5, n)
    labels = rng.choice(["metal", "insulator", "semiconductor"], n)
    encoder = LabelEncoder().fit(labels)
    gap_scaler = make_pipeline(StandardScaler(), PowerTransformer()).fit(gap.reshape(-1, 1))
    curve_t = [np.linspace(300 + 10 * i, 600 + 5 * i, 5 + (i % 2)) for i in range(n)]
    curve_y = [rng.normal(size=t.size) for t in curve_t]
    seebeck_scaler = StandardScaler().fit(np.concatenate(curve_y).reshape(-1, 1))
    frame = pd.DataFrame(
        {
            "composition": _FORMULAS,
            "gap": gap_scaler.transform(gap.reshape(-1, 1)).ravel(),
            "label": encoder.transform(labels),
            "dos": [rng.normal(size=_DOS_GRID.size) for _ in range(n)],
            "dos_e": [_DOS_GRID.copy() for _ in range(n)],
            "seebeck": [seebeck_scaler.transform(y.reshape(-1, 1)).ravel() for y in curve_y],
            "temperature": curve_t,
            "split": (["train", "val", "test"] * n)[:n],
        }
    )
    frame.to_parquet(tmp_path / "d.parquet")
    joblib.dump(
        {"gap_scaler": gap_scaler, "label_encoder": encoder, "seebeck_scaler": seebeck_scaler}, tmp_path / "pre.pkl.z"
    )
    pre = str(tmp_path / "pre.pkl.z")
    return {
        "descriptor": {"kind": "kmd", "n_grids": 4},
        "datasets": {"d": {"path": str(tmp_path / "d.parquet")}},
        "tasks": [
            {"name": "band_gap", "kind": "regression", "dataset": "d", "column": "gap", "unit": "eV",
             "scaler": {"path": pre, "key": "gap_scaler"}},
            {"name": "phase", "kind": "classification", "dataset": "d", "column": "label", "num_classes": 3,
             "scaler": {"path": pre, "key": "label_encoder"}},
            {"name": "dos", "kind": "kernel_regression", "dataset": "d", "column": "dos", "t_column": "dos_e",
             "t_unit": "eV", "output_scale": "shape"},
            {"name": "seebeck", "kind": "kernel_regression", "dataset": "d", "column": "seebeck",
             "t_column": "temperature", "unit": "V/K", "t_unit": "K", "scaler": {"path": pre, "key": "seebeck_scaler"}},
        ],
    }  # fmt: skip


def _checkpoint(raw: dict, path: Path, seed: int, *, tasks: list[str] | None = None) -> Path:
    catalog = TaskCatalog(build_task_catalog_config(raw))
    names = tasks or catalog.task_names
    torch.manual_seed(seed)
    model = build_model_for_checkpoint(catalog, _MODEL, names)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "task_sequence": names[::-1] if seed % 2 else names}, path)
    return path


def _export(raw: dict, runs: list[Path], out: Path, **kw) -> Path:
    return export_runs(build_task_catalog_config(raw), _MODEL, runs, out, license="cc-by-nc-sa-4.0", **kw)


def test_export_round_trip_two_members(catalog_raw: dict, tmp_path: Path) -> None:
    runs = [_checkpoint(catalog_raw, tmp_path / f"run{i}" / "training" / "final_model.pt", i) for i in range(2)]
    out = _export(catalog_raw, [r.parent.parent for r in runs], tmp_path / "bundle")

    config, manifest = read_config(out), read_manifest(out)
    assert config.license == "cc-by-nc-sa-4.0" and config.schema_version == 1
    assert [m.id for m in manifest.members] == ["m000", "m001"] and manifest.default == "m000"
    assert manifest.member("m001").task_order == ("seebeck", "dos", "phase", "band_gap")
    assert (out / MEMBERS_DIR / "m001" / WEIGHTS_FILE).exists()

    gap = config.head("band_gap")
    assert gap.unit == "eV" and gap.output_scale == "physical" and gap.transform is not None
    phase_classes = config.head("phase").classes
    assert phase_classes is not None and phase_classes.classes == ("insulator", "metal", "semiconductor")
    dos = config.head("dos")
    assert dos.output_scale == "shape" and dos.transform is None
    np.testing.assert_allclose(dos.t_grid, _DOS_GRID)  # shared grid kept as is
    seebeck = config.head("seebeck")
    assert seebeck.t_unit == "K" and len(seebeck.t_grid) in (5, 6)  # per-row grids → evenly spaced default
    assert seebeck.t_range is not None and seebeck.t_range[0] == pytest.approx(300.0)

    # No pickle and no catalog needed to load; strict load; bit-identical to the checkpoint.
    model, _, info = load_member(out, "m001")
    state = torch.load(runs[1], map_location="cpu", weights_only=True)["model"]
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, state[key]), key
    assert info.source == str(runs[1])


def test_config_json_is_self_contained(catalog_raw: dict, tmp_path: Path) -> None:
    out = _export(catalog_raw, [_checkpoint(catalog_raw, tmp_path / "c.pt", 0)], tmp_path / "b")
    payload = json.loads((out / CONFIG_FILE).read_text())
    assert {"schema_version", "fm_version", "license", "descriptor", "model", "heads", "provenance"} <= set(payload)
    assert "transformer" not in payload["model"]  # MLP model: no [model.transformer]
    gap = next(h for h in payload["heads"] if h["name"] == "band_gap")
    assert [s["kind"] for s in gap["transform"]["steps"]] == ["standard_scaler", "power_transformer"]
    # Physical value → training scale reproduces the sklearn scaler stored in the pickle.
    pipe = joblib.load(Path(catalog_raw["tasks"][0]["scaler"]["path"]))["gap_scaler"]
    x = np.array([0.5, 1.0, 4.0])
    np.testing.assert_allclose(
        read_config(out).head("band_gap").transform.transform(x),  # type: ignore[union-attr]
        pipe.transform(x.reshape(-1, 1)).ravel(),
        atol=1e-12,
    )


def test_export_requires_a_transform_or_shape(catalog_raw: dict, tmp_path: Path) -> None:
    del catalog_raw["tasks"][0]["scaler"]
    with pytest.raises(BundleError, match="band_gap.*no target transform"):
        _export(catalog_raw, [_checkpoint(catalog_raw, tmp_path / "c.pt", 0)], tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_export_requires_class_names(catalog_raw: dict, tmp_path: Path) -> None:
    del catalog_raw["tasks"][1]["scaler"]
    with pytest.raises(BundleError, match="phase.*class names"):
        _export(catalog_raw, [_checkpoint(catalog_raw, tmp_path / "c.pt", 0)], tmp_path / "b")


def test_members_must_share_heads(catalog_raw: dict, tmp_path: Path) -> None:
    a = _checkpoint(catalog_raw, tmp_path / "a.pt", 0)
    b = _checkpoint(catalog_raw, tmp_path / "b.pt", 1, tasks=["band_gap", "phase", "dos"])
    with pytest.raises(BundleError, match="share one set of heads.*seebeck"):
        _export(catalog_raw, [a, b], tmp_path / "bundle")


def test_member_with_different_widths_is_refused(catalog_raw: dict, tmp_path: Path) -> None:
    good = _checkpoint(catalog_raw, tmp_path / "a.pt", 0)
    state = torch.load(good, weights_only=True)
    wide = ModelSectionConfig(latent_dim=8, encoder_hidden_dims=[32], head_hidden_dims=[8], n_kernel=4)
    other = build_model_for_checkpoint(
        TaskCatalog(build_task_catalog_config(catalog_raw)), wide, state["task_sequence"]
    )
    torch.save({"model": other.state_dict(), "task_sequence": state["task_sequence"]}, tmp_path / "w.pt")
    with pytest.raises(BundleError, match="wrong shape"):
        _export(catalog_raw, [good, tmp_path / "w.pt"], tmp_path / "bundle")


def test_training_only_tensors_are_dropped(catalog_raw: dict, tmp_path: Path) -> None:
    path = _checkpoint(catalog_raw, tmp_path / "c.pt", 0)
    state = torch.load(path, weights_only=True)
    state["model"]["task_log_sigmas.band_gap"] = torch.zeros(1)
    torch.save(state, path)
    load_member(_export(catalog_raw, [path], tmp_path / "b"))  # strict load succeeds


def test_out_must_be_empty(catalog_raw: dict, tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "keep.txt").write_text("x")
    with pytest.raises(BundleError, match="not empty"):
        _export(catalog_raw, [_checkpoint(catalog_raw, tmp_path / "c.pt", 0)], tmp_path / "b")
    assert (tmp_path / "b" / "keep.txt").exists()


def test_newer_schema_and_foreign_descriptor_are_refused(catalog_raw: dict, tmp_path: Path) -> None:
    out = _export(catalog_raw, [_checkpoint(catalog_raw, tmp_path / "c.pt", 0)], tmp_path / "b")
    payload = json.loads((out / CONFIG_FILE).read_text())
    payload["schema_version"] = 99
    (out / CONFIG_FILE).write_text(json.dumps(payload))
    with pytest.raises(BundleError, match="schema 99 is newer"):
        read_config(out)
    payload["schema_version"] = 1
    payload["descriptor"]["features_sha256"] = "0" * 64
    (out / CONFIG_FILE).write_text(json.dumps(payload))
    with pytest.raises(BundleError, match="element_features.csv differs"):
        load_member(out)


def test_descriptor_spec_matches_catalog_width() -> None:
    raw = {"descriptor": {"kind": "kmd", "n_grids": 4}, "datasets": {"d": {"path": "x.parquet"}},
           "tasks": [{"name": "a", "kind": "regression", "dataset": "d", "column": "a"}]}  # fmt: skip
    assert DescriptorSpec.current(4).dim() == TaskCatalog(build_task_catalog_config(raw)).descriptor_dim


def test_default_grid_shared_vs_per_row() -> None:
    shared = [np.array([1.0, 2.0, 3.0])] * 4
    assert default_grid(shared) == ((1.0, 3.0), (1.0, 2.0, 3.0))
    rows = [np.linspace(0, 10, 4), np.linspace(5, 20, 4), np.linspace(2, 8, 3)]
    rng, grid = default_grid(rows)
    assert rng == (0.0, 20.0) and len(grid) == 4 and grid[0] == 0.0 and grid[-1] == 20.0
    with pytest.raises(BundleError, match="no training t values"):
        default_grid([])


def test_catalog_reads_recorded_json_transforms(tmp_path: Path) -> None:
    record = {
        "tasks": {
            "tc (normalized)": {
                "transform": {"steps": [{"kind": "log1p", "clip_lower": 0.0}, {"kind": "standard_scaler", "mean": 1.0, "scale": 2.0}]}
            }
        }
    }  # fmt: skip
    (tmp_path / "t.json").write_text(json.dumps(record))
    raw: dict[str, Any] = {"descriptor": {"kind": "kmd", "n_grids": 4}, "datasets": {"d": {"path": "x.parquet"}},
           "tasks": [{"name": "tc", "kind": "regression", "dataset": "d", "column": "tc (normalized)",
                      "scaler": {"path": str(tmp_path / "t.json"), "key": "tc (normalized)"}}]}  # fmt: skip
    catalog = TaskCatalog(build_task_catalog_config(raw))
    np.testing.assert_allclose(catalog.inverse_transform("tc", np.array([0.0])), [np.expm1(1.0)])
    assert catalog.target_transform("tc") is catalog.scaler("tc")
    raw["tasks"][0]["scaler"]["key"] = "missing"
    with pytest.raises(KeyError, match="transform key 'missing'"):
        TaskCatalog(build_task_catalog_config(raw)).scaler("tc")


def test_output_scale_validation() -> None:
    base: dict[str, Any] = {"name": "a", "kind": "regression", "dataset": "d", "column": "a"}
    cases: list[tuple[dict[str, Any], str]] = [
        ({"output_scale": "absolute"}, "must be 'shape'"),
        ({"output_scale": "shape", "scaler": {"path": "s.pkl"}}, "takes no scaler"),
        ({"t_unit": "K"}, "only valid for kernel_regression"),
    ]
    for extra, match in cases:
        raw = {"descriptor": {"kind": "kmd", "n_grids": 4}, "datasets": {"d": {"path": "x.parquet"}},
               "tasks": [{**base, **extra}]}  # fmt: skip
        with pytest.raises(ValueError, match=match):
            build_task_catalog_config(raw)


def _toml(raw: dict) -> str:
    """The fixture catalog as TOML (scalars, lists and tables only — enough for this config)."""

    def value(v: object) -> str:
        return json.dumps(v)

    lines = [
        "[descriptor]",
        'kind = "kmd"',
        "n_grids = 4",
        "",
        "[datasets.d]",
        f"path = {value(raw['datasets']['d']['path'])}",
    ]
    for task in raw["tasks"]:
        lines += ["", "[[tasks]]"]
        lines += [f"{k} = {value(v)}" for k, v in task.items() if k != "scaler"]
        if "scaler" in task:
            lines += ["[tasks.scaler]"] + [f"{k} = {value(v)}" for k, v in task["scaler"].items()]
    lines += ["", "[model]", "latent_dim = 8", "encoder_hidden_dims = [16]", "head_hidden_dims = [8]", "n_kernel = 4"]
    return "\n".join(lines) + "\n"


def test_fm_export_cli(catalog_raw: dict, tmp_path: Path) -> None:
    cfg = tmp_path / "export.toml"
    cfg.write_text(_toml(catalog_raw))
    ckpt = _checkpoint(catalog_raw, tmp_path / "c.pt", 0)
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--config", str(cfg), "--run", str(ckpt), "--out", str(tmp_path / "b")])
    assert result.exit_code != 0 and "--license" in result.output  # no default licence
    result = runner.invoke(
        main,
        ["export", "--config", str(cfg), "--run", str(ckpt), "--out", str(tmp_path / "b"), "--license", "apache-2.0",
         "--member-id", "base"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert read_config(tmp_path / "b").license == "apache-2.0"
    assert read_manifest(tmp_path / "b").default == "base"
