# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for loading released bundles and predicting in physical units (no network, no datasets)."""

from __future__ import annotations

import fnmatch
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner

from yammi.cli.main import main
from yammi.hub import BundleError, ExtrapolationWarning, SaturationWarning, load_pretrained, load_suite
from yammi.hub import pretrained as pretrained_mod
from yammi.hub.pretrained import parse_source
from yammi.workflows.task_catalog import TaskCatalog, build_task_catalog_config

from .export_test import _checkpoint, _export, catalog_raw  # noqa: F401  (catalog_raw is a fixture)

_COMPS = ["Fe2 Al3", "Cu2 Ni3", "Mg2 O3"]


@pytest.fixture
def bundle(catalog_raw: dict, tmp_path: Path) -> Path:  # noqa: F811
    runs = [_checkpoint(catalog_raw, tmp_path / f"r{i}.pt", i) for i in range(3)]
    path = _export(catalog_raw, runs, tmp_path / "bundle")
    # The catalog's data and pickles are deleted: a bundle must not need them.
    for task in catalog_raw["tasks"]:
        if "scaler" in task:
            Path(task["scaler"]["path"]).unlink(missing_ok=True)
    return path


def test_predict_matches_catalog_path_in_physical_units(catalog_raw: dict, tmp_path: Path) -> None:  # noqa: F811
    ckpt = _checkpoint(catalog_raw, tmp_path / "c.pt", 0)
    out = _export(catalog_raw, [ckpt], tmp_path / "b")
    catalog = TaskCatalog(build_task_catalog_config(catalog_raw))
    model = load_pretrained(out)
    pred = model.predict(_COMPS, ["band_gap"])
    # Reference: raw head output through the catalog's sklearn inverse transform.
    x = torch.tensor(catalog.descriptor_fn()(_COMPS).values, dtype=torch.float32)
    with torch.no_grad():
        raw = model.model.task_heads["band_gap"](torch.tanh(model.model.encoder(x))).squeeze(-1).numpy()
    np.testing.assert_allclose(pred["band_gap"], catalog.inverse_transform("band_gap", raw), rtol=1e-10)
    assert pred.attrs["units"] == {"band_gap": "eV"}


def test_classification_names_and_probabilities(bundle: Path) -> None:
    model = load_pretrained(bundle)
    pred = model.predict(_COMPS, ["phase"], probabilities=True)
    classes = ["insulator", "metal", "semiconductor"]
    assert set(pred["phase"]) <= set(classes)
    probs = pred[[f"phase:{c}" for c in classes]].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
    assert list(pred["phase"]) == [classes[i] for i in probs.argmax(axis=1)]


def test_default_predict_skips_curves(bundle: Path) -> None:
    pred = load_pretrained(bundle).predict(_COMPS)
    assert list(pred.columns) == ["composition", "band_gap", "phase"]


def test_predict_curve_grids_and_extrapolation(bundle: Path) -> None:
    model = load_pretrained(bundle)
    head = model.config.head("seebeck")
    curve = model.predict_curve(_COMPS, "seebeck")
    assert list(curve.columns) == ["composition", "temperature", "seebeck"]
    assert len(curve) == len(_COMPS) * len(head.t_grid)
    assert curve.attrs["units"] == {"seebeck": "V/K", "temperature": "K"}

    shared = model.predict_curve(_COMPS[:1], "seebeck", t=[350.0, 400.0])
    assert list(shared["temperature"]) == [350.0, 400.0]
    per_row = model.predict_curve(_COMPS[:2], "seebeck", t=[[350.0], [360.0, 370.0, 380.0]])
    assert per_row.groupby("composition", sort=False).size().tolist() == [1, 3]
    # Same composition and t → same value whichever grid it came from.
    a = model.predict_curve(["Fe2 Al3"], "seebeck", t=[350.0])["seebeck"].iloc[0]
    assert a == pytest.approx(shared["seebeck"].iloc[0])

    with pytest.warns(ExtrapolationWarning, match="outside the training range"):
        model.predict_curve(_COMPS[:1], "seebeck", t=[10.0, 400.0])
    with pytest.raises(ValueError, match="2 t grids for 3"):
        model.predict_curve(_COMPS, "seebeck", t=[[1.0], [2.0]])


def test_shape_only_head_returns_normalised_units(bundle: Path) -> None:
    model = load_pretrained(bundle)
    assert model.units()["dos"] == "shape"
    curve = model.predict_curve(_COMPS[:1], "dos")
    assert curve.attrs["units"]["dos"] == "shape" and curve.attrs["units"]["dos_e"] == "eV"


def test_wrong_kind_unknown_task_and_bad_composition(bundle: Path) -> None:
    model = load_pretrained(bundle)
    with pytest.raises(ValueError, match="use predict_curve"):
        model.predict(_COMPS, ["seebeck"])
    with pytest.raises(ValueError, match="use predict"):
        model.predict_curve(_COMPS, "band_gap")
    with pytest.raises(KeyError, match="no head 'nope'"):
        model.predict(_COMPS, ["nope"])
    with pytest.raises(ValueError, match="cannot build descriptors for 1 composition"):
        model.predict(["Fe2 Al3", "Xx9"], ["band_gap"])


def test_member_selection_and_suite(bundle: Path) -> None:
    assert load_pretrained(bundle).member.id == "m000"
    m2 = load_pretrained(bundle, member="m002")
    assert m2.member.id == "m002"
    suite = load_suite(bundle)
    assert len(suite) == 3 and suite.member_ids == ["m000", "m001", "m002"]
    assert suite._loaded == {}  # lazy
    preds = [m.predict(_COMPS[:1], ["band_gap"])["band_gap"].iloc[0] for m in suite]
    assert len(set(preds)) == 3  # different weights → different members
    with pytest.raises(KeyError, match="no member 'm009'"):
        load_pretrained(bundle, member="m009")


def test_parse_source() -> None:
    src = parse_source("TsumiNa/yammi-inorganic@v1.0.0#m003")
    assert (src.repo, src.revision, src.member, src.path) == ("TsumiNa/yammi-inorganic", "v1.0.0", "m003", None)
    assert parse_source("TsumiNa/yammi-inorganic", revision="main", member="m001").revision == "main"
    with pytest.raises(BundleError, match="neither an existing bundle directory"):
        parse_source("not a repo")


def test_local_bundle_takes_a_member_suffix(bundle: Path) -> None:
    src = parse_source(f"{bundle}#m002")
    assert (src.path, src.member, src.repo) == (bundle, "m002", None)
    assert load_pretrained(f"{bundle}#m002").member.id == "m002"
    assert parse_source(f"{bundle}#m002", member="m001").member == "m001"  # the keyword wins


def _set_gap_steps(bundle: Path, steps: list[dict]) -> None:
    path = bundle / "config.json"
    cfg = json.loads(path.read_text())
    next(h for h in cfg["heads"] if h["name"] == "band_gap")["transform"] = {"steps": steps}
    path.write_text(json.dumps(cfg))


def test_saturated_predictions_warn_and_return_the_bound(bundle: Path) -> None:
    _set_gap_steps(bundle, [{"kind": "identity"}])  # physical == training scale: shows the raw outputs
    raw = load_pretrained(bundle).predict(_COMPS, ["band_gap"])["band_gap"].to_numpy(dtype=float)
    cap = float(np.sort(raw)[1])
    _set_gap_steps(bundle, [{"kind": "identity"}, {"kind": "clip", "lower": None, "upper": cap}])
    with pytest.warns(SaturationWarning, match="most extreme training value"):
        out = load_pretrained(bundle).predict(_COMPS, ["band_gap"])["band_gap"].to_numpy(dtype=float)
    np.testing.assert_allclose(out, np.minimum(raw, cap))


def test_past_a_guard_is_nan_by_default_or_the_bound_on_request(bundle: Path) -> None:
    _set_gap_steps(bundle, [{"kind": "identity"}])
    raw = load_pretrained(bundle).predict(_COMPS, ["band_gap"])["band_gap"].to_numpy(dtype=float)
    cap = float(np.sort(raw)[1])
    _set_gap_steps(bundle, [{"kind": "identity"}, {"kind": "clip", "lower": None, "upper": cap, "guard": True}])
    with pytest.warns(SaturationWarning, match="are NaN"):
        out = load_pretrained(bundle).predict(_COMPS, ["band_gap"])["band_gap"].to_numpy(dtype=float)
    np.testing.assert_array_equal(np.isnan(out), raw > cap)
    np.testing.assert_allclose(out[raw <= cap], raw[raw <= cap])
    with pytest.warns(SaturationWarning, match="most extreme training value"):
        clipped = load_pretrained(bundle).predict(_COMPS, ["band_gap"], saturated="clip")["band_gap"]
    np.testing.assert_allclose(clipped, np.minimum(raw, cap))
    with pytest.raises(ValueError, match="saturated must be one of"):
        load_pretrained(bundle).predict(_COMPS, ["band_gap"], saturated="drop")


def test_suite_predict_combines_members_and_skips_out_of_range_ones(bundle: Path) -> None:
    suite = load_suite(bundle)
    per = np.stack([m.predict(_COMPS, ["band_gap"])["band_gap"].to_numpy(float) for m in suite])
    probs = np.mean([m.predict(_COMPS, ["phase"], probabilities=True).filter(like="phase:").to_numpy(float)
                     for m in suite], axis=0)  # fmt: skip
    combined = load_suite(bundle).predict(_COMPS, ["band_gap", "phase"], probabilities=True)
    np.testing.assert_allclose(combined["band_gap"], per.mean(axis=0))
    np.testing.assert_allclose(combined["band_gap:std"], per.std(axis=0))
    classes = ["insulator", "metal", "semiconductor"]
    assert list(combined["phase"]) == [classes[i] for i in probs.argmax(axis=1)]
    assert combined.attrs["members"] == ["m000", "m001", "m002"]

    # A guard below some members' outputs: those members drop out of the mean on that row only.
    _set_gap_steps(bundle, [{"kind": "identity"}])
    per = np.stack([m.predict(_COMPS, ["band_gap"])["band_gap"].to_numpy(float) for m in load_suite(bundle)])
    cap = float(np.median(per))
    _set_gap_steps(bundle, [{"kind": "identity"}, {"kind": "clip", "lower": None, "upper": cap, "guard": True}])
    with pytest.warns(SaturationWarning, match="left out of the mean"):
        got = load_suite(bundle).predict(_COMPS, ["band_gap"])["band_gap"].to_numpy(float)
    kept = np.where(per <= cap, per, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_allclose(got, np.nanmean(kept, axis=0))


def test_uninvertible_predictions_warn_instead_of_silent_nan(bundle: Path) -> None:
    # A bundle exported before 1.0.1: Yeo-Johnson lambda = -1 and no guard. Its inverse is defined only
    # below z = 1, i.e. below a training-scale value of 1 - 1000, which every prediction exceeds.
    _set_gap_steps(bundle, [{"kind": "power_transformer", "method": "yeo-johnson", "lmbda": -1.0,
                             "standardize": True, "mean": 1000.0, "scale": 1.0}])  # fmt: skip
    with pytest.warns(SaturationWarning, match="Re-export"):
        out = load_pretrained(bundle).predict(_COMPS, ["band_gap"])["band_gap"]
    assert out.isna().any()


def _fake_hub(monkeypatch: pytest.MonkeyPatch, bundle: Path, tmp_path: Path) -> list[list[str]]:
    """Serve ``bundle`` as a Hub repo: each snapshot copies only the files its patterns allow."""
    calls: list[list[str]] = []
    cache = tmp_path / "hub-cache"

    def commit(repo, revision, token, cache_dir=None):  # noqa: ANN001
        assert repo == "owner/fm" and revision == "v1"
        calls.append(["<resolve v1>"])
        return "c0ffee"

    def snapshot(repo, revision, patterns, token, cache_dir, required=()):  # noqa: ANN001
        assert repo == "owner/fm" and revision == "c0ffee"  # every download uses the pinned commit
        calls.append(list(patterns))
        for file in bundle.rglob("*"):
            rel = file.relative_to(bundle).as_posix()
            if file.is_file() and any(fnmatch.fnmatch(rel, p) for p in patterns):
                (cache / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, cache / rel)
        return cache

    monkeypatch.setattr(pretrained_mod, "_snapshot", snapshot)
    monkeypatch.setattr(pretrained_mod, "_commit", commit)
    return calls


def test_hub_download_fetches_only_the_needed_member(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_hub(monkeypatch, bundle, tmp_path)
    model = load_pretrained("owner/fm@v1")
    assert model.member.id == "m000" and model.revision == "c0ffee"
    assert calls[0] == ["<resolve v1>"] and calls[1] == ["config.json", "manifest.json", "README.md"]
    assert "members/m000/*" in calls[-1] and not any("m001" in p for c in calls for p in c)
    assert not (tmp_path / "hub-cache" / "members" / "m001").exists()

    calls.clear()
    load_pretrained("owner/fm@v1#m002")
    assert any("members/m002/*" in c for c in calls)


def test_hub_suite_downloads_members_on_access(bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_hub(monkeypatch, bundle, tmp_path)
    suite = load_suite("owner/fm@v1")
    assert len(suite) == 3 and suite.member_ids == ["m000", "m001", "m002"]
    assert not any(p.startswith("members/") for c in calls for p in c)  # metadata only so far
    suite["m002"].predict(_COMPS[:1], ["band_gap"])
    fetched = {p for c in calls for p in c if p.startswith("members/")}
    assert fetched == {"members/m002/*"} and suite["m002"].revision == "c0ffee"
    with pytest.raises(KeyError, match="no member 'm009'"):
        suite["m009"]
    with pytest.raises(BundleError, match="load_suite loads all members"):
        load_suite("owner/fm@v1#m001")


def test_fm_predict_pretrained_cli(bundle: Path, tmp_path: Path) -> None:
    comps_file = tmp_path / "comps.txt"
    comps_file.write_text("Cu2 Ni3\n\nMg2 O3\n")
    out = tmp_path / "pred"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["predict", "--pretrained", str(bundle), "--compositions", "Fe2 Al3", "--compositions-file", str(comps_file),
         "--output-dir", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out / "predictions.csv")
    assert list(table["composition"]) == _COMPS and "phase:metal" in table.columns
    assert (out / "seebeck_curve.csv").exists() and (out / "dos_curve.csv").exists()
    info = json.loads((out / "prediction_info.json").read_text())
    # No "#member": every member is combined, with a spread column.
    assert info["members"] == ["m000", "m001", "m002"] and info["combined"] and info["member"] is None
    assert info["license"] == "cc-by-nc-sa-4.0" and info["units"]["dos"] == "shape"
    assert info["yammi_version"] == info["fm_version"]  # the pre-2.1.1 key stays for existing readers
    assert "band_gap:std" in table.columns and "seebeck:std" in pd.read_csv(out / "seebeck_curve.csv").columns

    one = tmp_path / "pred_one"
    result = runner.invoke(main, ["predict", "--pretrained", f"{bundle}#m001", "--compositions", "Fe2 Al3",
                                  "--tasks", "band_gap", "--output-dir", str(one)])  # fmt: skip
    assert result.exit_code == 0, result.output
    info = json.loads((one / "prediction_info.json").read_text())
    assert info["members"] == ["m001"] and info["member"] == "m001" and info["combined"] is None
    assert list(pd.read_csv(one / "predictions.csv").columns) == ["composition", "band_gap"]

    result = runner.invoke(main, ["predict", "--pretrained", str(bundle), "--compositions", "Fe2 Al3",
                                  "--output-dir", str(out), "--checkpoint", "x.pt"])  # fmt: skip
    assert result.exit_code != 0 and "--checkpoint does not apply" in result.output
    result = runner.invoke(main, ["predict", "--pretrained", str(bundle), "--compositions", "Fe2 Al3"])
    assert result.exit_code != 0 and "--output-dir" in result.output
    result = runner.invoke(main, ["predict", "--compositions", "Fe2 Al3"])
    assert result.exit_code != 0 and "--config is required" in result.output
    result = runner.invoke(main, ["predict", "--pretrained", str(bundle), "--compositions", "Fe2 Al3",
                                  "--output-dir", str(out), "--set", "predict.tasks=[\"band_gap\"]", "--seed", "3"])  # fmt: skip
    assert result.exit_code != 0 and "--set, --seed do not apply to --pretrained" in result.output


@pytest.mark.parametrize("bad", ["/tmp/result", "../up", "a/b", "a\\b", "", "_hidden"])
def test_head_names_must_be_safe_file_names(bad: str) -> None:
    from yammi.hub.bundle import HeadSpec

    with pytest.raises(BundleError, match="no path separators"):
        HeadSpec(name=bad, kind="classification", num_classes=2, hidden_dims=(4,))


def test_offline_revision_resolves_from_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compute nodes often have no network: a revision cached on a login node must still load."""
    import huggingface_hub
    from huggingface_hub.errors import OfflineModeIsEnabled

    def offline(*args: object, **kwargs: object) -> None:
        raise OfflineModeIsEnabled("offline mode is enabled")

    seen: dict[str, object] = {}

    def cached(**kwargs: object) -> str:
        seen.update(kwargs)
        return str(tmp_path / "models--owner--fm" / "snapshots" / "c0ffee")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", offline)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached)
    assert pretrained_mod._commit("owner/fm", "v1", None, cache_dir=tmp_path) == "c0ffee"
    assert seen["local_files_only"] is True and seen["revision"] == "v1" and seen["cache_dir"] == tmp_path


def test_offline_and_not_cached_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    def offline(*args: object, **kwargs: object) -> None:
        raise OfflineModeIsEnabled("offline mode is enabled")

    def missing(**kwargs: object) -> str:
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", offline)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(BundleError, match="not in the local cache"):
        pretrained_mod._commit("owner/fm", "v1", None)


@pytest.mark.parametrize(
    ("error", "says"),
    [
        ("GatedRepoError", "is gated: request access on https://huggingface.co/owner/fm"),
        ("RevisionNotFoundError", "has no revision 'v9'"),
        ("RepositoryNotFoundError", "no model repository owner/fm"),
    ],
)
def test_hub_access_errors_say_what_to_do(
    error: str, says: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    import huggingface_hub
    from huggingface_hub import errors

    response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/api/models/owner/fm"))
    exc = getattr(errors, error)("hub said no", response=response)

    def fail(*args: object, **kwargs: object) -> None:
        raise exc

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", fail)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_refs", fail)
    with pytest.raises(BundleError, match=says):
        pretrained_mod._commit("owner/fm", "v9", None)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    with pytest.raises(BundleError, match=says):
        pretrained_mod._snapshot("owner/fm", "v9", ["config.json"], None, tmp_path)


@pytest.mark.parametrize("complete", [True, False])
def test_snapshot_reads_a_complete_cache_without_the_hub(
    complete: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached member loads without a Hub request (and so without progress bars); a partial cache downloads."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    snap = tmp_path / "snap"
    (snap / "members" / "m000").mkdir(parents=True)
    for rel in ("config.json", "manifest.json", "members/m000/member.json"):
        (snap / rel).write_text("{}")
    if complete:
        (snap / "members" / "m000" / "model.safetensors").write_bytes(b"")
    calls: list[bool] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(bool(kwargs.get("local_files_only")))
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    src = pretrained_mod.Source(repo="owner/fm", revision="c0ffee", member="m000")
    assert pretrained_mod._fetch(src, ["m000"], None, tmp_path) == snap
    assert calls == ([True] if complete else [True, False])


def test_suite_treats_an_infinite_member_output_as_out_of_range(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = load_suite(bundle)
    members = list(suite)
    real = [m.predict(_COMPS, ["band_gap"])["band_gap"].to_numpy(float) for m in members]
    original = type(members[0]).predict

    def predict(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        out = original(self, *args, **kwargs)
        if self.member.id == "m000":
            out["band_gap"] = np.inf  # an overflowed inverse
        return out

    monkeypatch.setattr(type(members[0]), "predict", predict)
    with pytest.warns(SaturationWarning, match="left out of the mean"):
        got = suite.predict(_COMPS, ["band_gap"])
    np.testing.assert_allclose(got["band_gap"], np.mean(real[1:], axis=0))
    assert np.isfinite(got["band_gap:std"]).all()
