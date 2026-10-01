# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import report
from report import ReportSettings, collect_results, curve_metrics, normalized, sha256


@pytest.fixture
def cohort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(report, "plot_results", lambda *args: None)
    project = tmp_path / "project"
    fold = project / "data/fold_01"
    fold.mkdir(parents=True)
    temperature = np.linspace(6, 290, 300)
    truth = np.linspace(-2, 100, 300)
    data_path = fold / "data.parquet"
    pd.DataFrame(
        {
            "composition": ["La3 Ni2 O7"],
            "formula": ["La3Ni2O7"],
            "split": ["test"],
            "temperature_K": [temperature],
            "rho_uohm_cm": [truth],
        }
    ).to_parquet(data_path)
    task_path = fold / "tasks_20261001.toml"
    task_path.write_text(
        '[datasets.agis]\npath = "data/fold_01/data.parquet"\n[[tasks]]\nname = "agis_rho_0gpa"\ndataset = "agis"\n'
    )
    units = [
        {"route": route, "fold": 1, "pressure": 0, "checkpoint_index": None if route == "scratch" else 0}
        for route in ("direct", "warm", "scratch")
    ]
    manifest = {
        "settings": {"date": "20261001", "final_epochs": 1000},
        "source_sha256": "reviewed training source",
        "input_sha256": {str(path.relative_to(project)): sha256(path) for path in (data_path, task_path)},
        "units": units,
        "preprocessing": {
            "folds": [
                {
                    "directory": "data/fold_01",
                    "heldout_composition": "La3 Ni2 O7",
                    "scalers": {"0": {"prescale_std": 20.0, "asinh_mean": 0.5, "asinh_std": 0.8}},
                }
            ]
        },
        "selection": {"models": [{"run": "uniformly selected", "sha256": "original checkpoint"}]},
        "pretraining_overlap": "AGIS-label holdout",
    }
    path = project / "manifest.json"
    path.write_text(json.dumps(manifest))
    root = tmp_path / "production"
    root.mkdir()
    identity = {
        "manifest_sha256": sha256(path),
        "source_sha256": manifest["source_sha256"],
        "runtime": {"image_sha256": "fixed"},
    }
    (root / "campaign_identity.json").write_text(json.dumps(identity))
    for unit in units:
        tag = "scratch" if unit["route"] == "scratch" else "c01"
        folder = root / f"{unit['route']}_f01_{tag}_p0"
        folder.mkdir()
        (folder / "DONE").write_text("completed")
        (folder / "campaign_identity.json").write_text(json.dumps(identity))
        (folder / "result.json").write_text(
            json.dumps({**unit, "heldout": "La3 Ni2 O7", "elapsed_seconds": 10, "slurm_job": "123"})
        )
        for label in ["unfrozen"] if unit["route"] == "scratch" else ["frozen", "unfrozen"]:
            fit = folder / label
            (fit / "training/finetune").mkdir(parents=True)
            pd.DataFrame(
                {"composition": ["La3 Ni2 O7"] * 300, "true": truth, "pred": truth + 1, "t": temperature}
            ).to_parquet(fit / "training/finetune/agis_rho_0gpa_pred.parquet")
            (fit / "training/finetune_summary.json").write_text(
                json.dumps({"epochs_run": 1000, "freeze_encoder": label == "frozen"})
            )
            (fit / "training/final_model.pt").write_bytes(b"validated final checkpoint")
            (fit / "initial_model.pt").write_bytes(b"original initial checkpoint")
    return path, root, project


def test_collect_validates_provenance_and_exports_original_unit_metrics(
    cohort: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    path, root, project = cohort
    output = tmp_path / "report"
    frame = collect_results(path, root, project, output, ReportSettings())
    assert len(frame) == 5
    assert np.allclose(frame["rmse_uohm_cm"], 1)
    assert frame["model_sha256"].nunique() == 1
    provenance = json.loads((output / "report_manifest_20261001.json").read_text())
    assert provenance["complete_cohort"] and provenance["n_final_models"] == 5
    assert len(pd.read_parquet(output / "predictions_20261001.parquet")) == 1500
    paired = pd.read_csv(output / "paired_warm_direct_20261001.csv")
    assert len(paired) == 2 and np.allclose(paired["warm_minus_direct"], 0)


def test_missing_units_cannot_be_reported_as_complete(cohort: tuple[Path, Path, Path], tmp_path: Path) -> None:
    path, root, project = cohort
    collect_results(path, root, project, tmp_path / "report", ReportSettings())
    assert len(pd.read_csv(tmp_path / "report/paired_warm_direct_20261001.csv")) == 2
    (root / "warm_f01_c01_p0/DONE").unlink()
    with pytest.raises(ValueError, match="Incomplete agreed cohort"):
        collect_results(path, root, project, tmp_path / "report", ReportSettings())
    collect_results(path, root, project, tmp_path / "report", ReportSettings(allow_partial=True))
    assert not json.loads((tmp_path / "report/report_manifest_20261001.json").read_text())["complete_cohort"]
    assert pd.read_csv(tmp_path / "report/paired_warm_direct_20261001.csv").empty


def recovery_source(cohort: tuple[Path, Path, Path], tmp_path: Path) -> ReportSettings:
    path, root, project = cohort
    manifest = json.loads(path.read_text())
    manifest["source_sha256"] = "reviewed diagnostic-only fix"
    recovery_manifest = project / "recovery.json"
    recovery_manifest.write_text(json.dumps(manifest))
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    identity = json.loads((root / "campaign_identity.json").read_text())
    identity.update(source_sha256=manifest["source_sha256"], manifest_sha256=sha256(recovery_manifest))
    (recovery / "campaign_identity.json").write_text(json.dumps(identity))
    unit = "warm_f01_c01_p0"
    shutil.move(root / unit, recovery / unit)
    (recovery / unit / "campaign_identity.json").write_text(json.dumps(identity))
    return ReportSettings(recovery_manifest=recovery_manifest, recovery_root=recovery)


def test_recovery_is_complete_with_explicit_per_model_source(cohort: tuple[Path, Path, Path], tmp_path: Path) -> None:
    settings = recovery_source(cohort, tmp_path)
    path, root, project = cohort
    output = tmp_path / "combined_report"
    frame = collect_results(path, root, project, output, settings)
    assert len(frame) == 5 and frame["recovered"].sum() == 2
    assert set(frame.loc[frame["route"] == "warm", "campaign_source_sha256"]) == {"reviewed diagnostic-only fix"}
    provenance = json.loads((output / "report_manifest_20261001.json").read_text())
    assert provenance["complete_cohort"]
    assert provenance["recovered_units"] == ["warm_f01_c01_p0"]
    assert provenance["recovery_campaign_identity"]["source_sha256"] == "reviewed diagnostic-only fix"


@pytest.mark.parametrize("change", ["protocol", "runtime", "manifest_hash", "unit_identity", "duplicate"])
def test_recovery_rejects_protocol_or_identity_drift(
    cohort: tuple[Path, Path, Path], tmp_path: Path, change: str
) -> None:
    settings = recovery_source(cohort, tmp_path)
    assert settings.recovery_manifest is not None and settings.recovery_root is not None
    path, root, project = cohort
    recovery = settings.recovery_root
    identity_file = recovery / "campaign_identity.json"
    identity = json.loads(identity_file.read_text())
    if change == "protocol":
        manifest = json.loads(settings.recovery_manifest.read_text())
        manifest["settings"]["final_epochs"] = 2000
        settings.recovery_manifest.write_text(json.dumps(manifest))
        identity["manifest_sha256"] = sha256(settings.recovery_manifest)
    elif change == "runtime":
        identity["runtime"]["image_sha256"] = "different runtime"
    elif change == "manifest_hash":
        identity["manifest_sha256"] = "unbound manifest"
    elif change == "unit_identity":
        (recovery / "warm_f01_c01_p0/campaign_identity.json").write_text("{}")
    else:
        shutil.copytree(recovery / "warm_f01_c01_p0", root / "warm_f01_c01_p0")
    identity_file.write_text(json.dumps(identity))
    with pytest.raises(ValueError):
        collect_results(path, root, project, tmp_path / "report", settings)


def test_recovery_paths_must_be_paired(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both"):
        ReportSettings(recovery_manifest=tmp_path)
    with pytest.raises(ValueError, match="both"):
        ReportSettings(recovery_root=tmp_path)


def test_recovery_cannot_replace_direct_training(cohort: tuple[Path, Path, Path], tmp_path: Path) -> None:
    settings = recovery_source(cohort, tmp_path)
    assert settings.recovery_root is not None
    path, root, project = cohort
    shutil.move(root / "direct_f01_c01_p0", settings.recovery_root / "direct_f01_c01_p0")
    with pytest.raises(ValueError, match="restricted"):
        collect_results(path, root, project, tmp_path / "report", settings)


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("true", 10000.0, "truth differs"),
        ("pred", float("nan"), "Nonfinite"),
        ("t", 0.0, "Temperature grid"),
        ("composition", "another material", "composition mismatch"),
    ],
)
def test_report_rejects_corrupt_predictions(
    cohort: tuple[Path, Path, Path], tmp_path: Path, column: str, value: float | str, match: str
) -> None:
    path, root, project = cohort
    prediction = root / "direct_f01_c01_p0/frozen/training/finetune/agis_rho_0gpa_pred.parquet"
    frame = pd.read_parquet(prediction)
    frame.loc[0, column] = value
    frame.to_parquet(prediction)
    with pytest.raises(ValueError, match=match):
        collect_results(path, root, project, tmp_path / "report", ReportSettings())


def test_report_rejects_unit_runtime_drift(cohort: tuple[Path, Path, Path], tmp_path: Path) -> None:
    path, root, project = cohort
    (root / "direct_f01_c01_p0/campaign_identity.json").write_text("{}")
    with pytest.raises(ValueError, match="Unit identity mismatch"):
        collect_results(path, root, project, tmp_path / "report", ReportSettings())


def test_report_rejects_missing_checkpoints(cohort: tuple[Path, Path, Path], tmp_path: Path) -> None:
    path, root, project = cohort
    (root / "direct_f01_c01_p0/frozen/training/final_model.pt").unlink()
    with pytest.raises(FileNotFoundError):
        collect_results(path, root, project, tmp_path / "report", ReportSettings())


@pytest.mark.parametrize("key,value", [("epochs_run", 999), ("freeze_encoder", False)])
def test_report_rejects_incomplete_fit_budget_or_wrong_freeze_setting(
    cohort: tuple[Path, Path, Path], tmp_path: Path, key: str, value: int | bool
) -> None:
    path, root, project = cohort
    summary = root / "direct_f01_c01_p0/frozen/training/finetune_summary.json"
    state = json.loads(summary.read_text())
    state[key] = value
    summary.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="budget or freeze"):
        collect_results(path, root, project, tmp_path / "report", ReportSettings())


def test_normalized_error_retains_scale_comparability_and_negative_r2() -> None:
    scaler = {"prescale_std": 10.0, "asinh_mean": 0.5, "asinh_std": 0.8}
    temperature = np.linspace(6, 290, 300)
    truth = np.linspace(-2, 5, 300)
    pred = 10 * np.sinh(0.5 + 0.8 * (normalized(truth, scaler) + 2))
    metrics = curve_metrics(truth, pred, temperature, scaler)
    assert metrics["z_rmse"] == pytest.approx(2)
    assert metrics["z_rmse_low_T"] == pytest.approx(2)
    assert metrics["r2"] < 0
    with pytest.raises(ValueError, match="300-point"):
        curve_metrics(truth[:20], pred, temperature, scaler)


def test_agreed_warm_cohorts() -> None:
    with pytest.raises(ValueError, match="first-five or full-ten"):
        ReportSettings(warm_checkpoints=7)
