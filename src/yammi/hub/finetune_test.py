# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""End to end: export → fine-tune from the bundle on new data only → derived bundle → load → predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from yammi.hub import load_pretrained, read_config, read_manifest
from yammi.workflows.finetune import build_finetune_config
from yammi.workflows.finetune import run as finetune_run

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


def test_random_init_is_a_from_scratch_control(bundle: Path, user_data: Path, tmp_path: Path) -> None:
    """init = "random": the bundle's architecture and transforms, fresh weights, only the target heads kept."""
    out = tmp_path / "scratch"
    raw = _raw(user_data, bundle, out, ["band_gap", "hardness"], init="random", freeze_encoder=False)
    summary = finetune_run(build_finetune_config(raw))
    assert summary["init"] == "random" and summary["added_tasks"] == ["hardness"]

    derived = read_config(out / "bundle")
    assert [h.name for h in derived.heads] == ["band_gap", "hardness"]  # untrained heads are not kept
    assert derived.provenance["init"] == "random" and derived.provenance["derived_from"]["init"] == "random"
    # An existing target head keeps the source transform, so a scratch run is scored in the same units.
    assert derived.head("band_gap").transform == read_config(bundle).head("band_gap").transform
    assert read_manifest(out / "bundle").member().task_order == ("band_gap", "hardness")
    pred = load_pretrained(out / "bundle").predict(_FORMULAS[:4], ["band_gap", "hardness"])
    assert list(pred.columns[1:]) == ["band_gap", "hardness"]


def _spy_fit(monkeypatch: pytest.MonkeyPatch, inspect) -> None:  # noqa: ANN001
    """Run ``inspect(model, catalog)`` when training starts, before any optimisation step."""
    import yammi.workflows.finetune as ft

    real = ft._fit

    def spy(model, catalog, cfg, rec):  # noqa: ANN001
        inspect(model, catalog)
        return real(model, catalog, cfg, rec)

    monkeypatch.setattr(ft, "_fit", spy)


def test_random_init_trains_from_fresh_weights(
    bundle: Path, user_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the start of training the encoder is a fresh build for the seed, not the released one."""
    from lightning import seed_everything

    from yammi.hub.bundle import build_model, load_member
    from yammi.workflows._sections import build_training_section

    seen: dict = {}
    _spy_fit(monkeypatch, lambda model, catalog: seen.update(
        {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}))  # fmt: skip
    raw = _raw(user_data, bundle, tmp_path / "s", ["band_gap", "hardness"], init="random", freeze_encoder=False)
    finetune_run(build_finetune_config(raw))

    released = load_member(bundle)[0].encoder.state_dict()
    seed_everything(raw["training"]["seed"], workers=True)
    fresh = build_model(read_config(bundle), build_training_section(raw["training"])).encoder.state_dict()
    assert seen.keys() == fresh.keys() and all(torch.equal(seen[k], fresh[k]) for k in fresh)
    linear = [k for k in fresh if k.endswith("layer.weight")]  # BatchNorm starts at ones / zeros either way
    assert linear and not any(torch.equal(seen[k], released[k]) for k in linear)


def test_random_init_places_an_existing_curve_head_on_the_data(
    bundle: Path, curve_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A curve target the bundle has gets kernels on the training coordinates, as a new curve head does."""
    from yammi.workflows.task_catalog import init_kernel_centers_sigmas

    seen: dict = {}

    def inspect(model, catalog):  # noqa: ANN001
        params = dict(model.task_heads["seebeck"].named_parameters())
        seen["centers"] = next(v for k, v in params.items() if k.endswith("centers")).detach().clone()
        seen["expected"] = init_kernel_centers_sigmas(catalog._train_t_values("seebeck"), seen["centers"].numel())[0]

    _spy_fit(monkeypatch, inspect)
    out = tmp_path / "sc"
    raw = _curve_raw(curve_data, bundle, out, ["seebeck"])
    raw["finetune"].update(init="random", freeze_encoder=False)
    finetune_run(build_finetune_config(raw))
    assert torch.allclose(seen["centers"], torch.as_tensor(seen["expected"], dtype=torch.float32))
    assert [h.name for h in read_config(out / "bundle").heads] == ["seebeck"]
    curve = load_pretrained(out / "bundle").predict_curve(_FORMULAS[:2], "seebeck", t=[300, 600])
    assert len(curve) == 4


def test_random_init_keeps_a_per_task_head_architecture(
    catalog_raw: dict,  # noqa: F811
    curve_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A head the bundle built with its own n_kernel / x_hidden_dims keeps them; the fine-tune need not repeat them."""
    from yammi.workflows.task_catalog import init_kernel_centers_sigmas

    raw = {**catalog_raw, "tasks": [dict(t) for t in catalog_raw["tasks"]]}
    seebeck = next(t for t in raw["tasks"] if t["name"] == "seebeck")
    seebeck.update(n_kernel=7, x_hidden_dims=[12])
    custom = _export(raw, [_checkpoint(raw, tmp_path / "c.pt", 1)], tmp_path / "custom")
    assert read_config(custom).head("seebeck").n_kernel == 7

    seen: dict = {}

    def inspect(model, catalog):  # noqa: ANN001
        params = dict(model.task_heads["seebeck"].named_parameters())
        seen["centers"] = next(v for k, v in params.items() if k.endswith("centers")).detach().clone()
        seen["expected"] = init_kernel_centers_sigmas(catalog._train_t_values("seebeck"), 7)[0]

    _spy_fit(monkeypatch, inspect)
    out = tmp_path / "sc"
    ft = _curve_raw(curve_data, custom, out, ["seebeck"])  # no per-task architecture given here
    ft["finetune"].update(init="random", freeze_encoder=False)
    finetune_run(build_finetune_config(ft))
    assert torch.allclose(seen["centers"], torch.as_tensor(seen["expected"], dtype=torch.float32))
    head = read_config(out / "bundle").head("seebeck")
    assert head.n_kernel == 7 and list(head.x_hidden_dims) == [12]


def test_save_keeps_licence_and_records_source(bundle: Path, tmp_path: Path) -> None:
    saved = load_pretrained(bundle).save(tmp_path / "copy")
    config = read_config(saved)
    assert config.license == "cc-by-nc-sa-4.0" and config.provenance["derived_from"]["source"] == str(bundle)
    a = load_pretrained(bundle).predict(_FORMULAS[:3], ["band_gap"])
    b = load_pretrained(saved).predict(_FORMULAS[:3], ["band_gap"])
    pd.testing.assert_frame_equal(a, b)
    # derived_from() reports where a bundle came from; an original bundle came from nowhere.
    assert load_pretrained(bundle).derived_from() is None
    record = load_pretrained(saved).derived_from()
    assert record is not None and record["source"] == str(bundle) and record["member"] == "m000"


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
    import yammi.workflows.finetune as ft

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

    from yammi.workflows.task_catalog import TaskCatalog, build_task_catalog_config

    joblib.dump(StandardScaler().fit([[0.0], [10.0]]), tmp_path / "s.pkl")
    raw = {"descriptor": {"kind": "kmd", "n_grids": 4}, "datasets": {"d": {"path": "x.parquet"}},
           "tasks": [{"name": "c", "kind": "kernel_regression", "dataset": "d", "column": "y", "t_column": "t",
                      "scaler": {"path": str(tmp_path / "s.pkl")}}]}  # fmt: skip
    catalog = TaskCatalog(build_task_catalog_config(raw), target_inputs={"c": "shape"})
    assert catalog.scaler("c") is None
    np.testing.assert_array_equal(catalog.inverse_transform("c", np.array([0.5])), [0.5])


def test_new_head_transform_is_invertible_everywhere(bundle: Path, user_data: Path, tmp_path: Path) -> None:
    """A heavy right tail would fit Yeo-Johnson lambda < 0; new heads get one whose inverse has no bound."""
    from yammi.data.targets import ClipStep, fit_default
    from yammi.workflows.finetune import _target_inputs

    frame = pd.read_parquet(user_data)
    frame["hardness_GPa"] = np.random.default_rng(5).lognormal(0.0, 2.5, len(frame))
    extreme = frame.index[frame["split"] == "test"][0]
    frame.loc[extreme, "hardness_GPa"] = frame["hardness_GPa"].max() * 50
    frame.to_parquet(user_data)
    train = frame.loc[frame["split"] == "train", "hardness_GPa"].to_numpy()
    assert fit_default(train).inverse_domain() == (-np.inf, np.inf)

    out = tmp_path / "ft"
    cfg = build_finetune_config(_raw(user_data, bundle, out, ["hardness"]))
    labels = _target_inputs(cfg, read_config(bundle))["hardness"]
    assert not any(isinstance(s, ClipStep) for s in labels.steps)  # held-out labels keep their values
    held_out = labels.transform(np.array([frame.loc[extreme, "hardness_GPa"]]))[0]
    np.testing.assert_allclose(labels.inverse(np.array([held_out])), [frame.loc[extreme, "hardness_GPa"]], rtol=1e-8)

    finetune_run(cfg)
    head = read_config(out / "bundle").head("hardness").transform
    assert head is not None and not any(isinstance(s, ClipStep) and s.guard for s in head.steps)  # nothing to guard
    assert np.isfinite(head.inverse(np.array([held_out + 10.0, -50.0]))).all()


def test_num_classes_may_be_left_out(bundle: Path, user_data: Path, tmp_path: Path) -> None:
    """An existing head takes the bundle's class count, a new head counts its labels."""
    data = pd.read_parquet(user_data)
    data["grade"] = (["A", "B"] * len(data))[: len(data)]
    data.to_parquet(user_data)
    raw = _raw(user_data, bundle, tmp_path / "ft", ["phase", "grade"])
    for task in raw["tasks"]:
        task.pop("num_classes", None)
    raw["tasks"].append({"name": "grade", "kind": "classification", "dataset": "mine", "column": "grade"})
    cfg = build_finetune_config(raw)
    counts = {t.name: t.num_classes for t in cfg.catalog.tasks}
    assert counts["phase"] == 3 and counts["grade"] == 2
    finetune_run(cfg)
    derived = load_pretrained(tmp_path / "ft" / "bundle")
    classes = derived.config.head("grade").classes
    assert classes is not None and classes.classes == ("A", "B")


def test_class_count_from_labels_or_indices() -> None:
    from yammi.workflows.finetune import _class_count

    assert _class_count("t", pd.Series(["b", "a", "b"])) == 2
    assert _class_count("t", pd.Series([0, 2, 2.0])) == 3  # class indices: max + 1, not the distinct count
    with pytest.raises(ValueError, match="non-negative integer"):
        _class_count("t", pd.Series([0, 1.5]))


def test_num_classes_inference_accepts_kind_aliases(bundle: Path, user_data: Path, tmp_path: Path) -> None:
    raw = _raw(user_data, bundle, tmp_path / "ft", ["phase"])
    for task in raw["tasks"]:
        if task["name"] == "phase":
            task.pop("num_classes")
            task["kind"] = "CLF"  # the catalog's case-insensitive legacy spelling
    counts = {t.name: t.num_classes for t in build_finetune_config(raw).catalog.tasks}
    assert counts["phase"] == 3


def test_inferred_class_count_follows_the_rows_the_catalog_keeps(bundle: Path, user_data: Path, tmp_path: Path) -> None:
    """A class present only in a row min_elements drops is not counted."""
    from yammi.workflows.finetune import _target_inputs

    data = pd.read_parquet(user_data)
    data["grade"] = (["A", "B"] * len(data))[: len(data)]
    lone = pd.DataFrame({"composition": ["Fe"], "grade": ["C"], "split": ["train"]})
    pd.concat([data, lone], ignore_index=True).to_parquet(user_data)
    raw = _raw(user_data, bundle, tmp_path / "ft", ["grade"])
    raw["datasets"]["mine"]["min_elements"] = 2
    raw["tasks"].append({"name": "grade", "kind": "classification", "dataset": "mine", "column": "grade"})
    cfg = build_finetune_config(raw)
    assert {t.name: t.num_classes for t in cfg.catalog.tasks}["grade"] == 2
    labels = _target_inputs(cfg, read_config(bundle))["grade"]
    assert labels.classes == ("A", "B")
