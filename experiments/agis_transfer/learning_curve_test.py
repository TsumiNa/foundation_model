# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd
import pytest

from learning_curve import aggregate_learning_curve, summarize_learning_curve


def test_learning_curve_weights_materials_after_collapsing_checkpoint_repeats() -> None:
    rows = []
    for composition in range(8):
        for pressure in (0, 10, 20):
            for route, repeats in (("scratch", 1), ("direct", 10)):
                for checkpoint in range(repeats):
                    value = float(composition) + (1 if route == "direct" else 0)
                    rows.append(
                        {
                            "n_train": 3,
                            "composition": str(composition),
                            "pressure": pressure,
                            "fold": composition + 1,
                            "route": route,
                            "setting": "unfrozen",
                            "is_anchor": True,
                            "z_rmse": value,
                            "z_rmse_low_T": value,
                            "relative_rmse": value,
                            "r2": value,
                        }
                    )
    summary, material = aggregate_learning_curve(pd.DataFrame(rows))
    aggregate = summary[(summary.scope == "anchor") & (summary.pressure == "all")].set_index("route")
    assert aggregate.loc["scratch", "z_rmse"] == 3.5
    assert aggregate.loc["direct", "z_rmse"] == 4.5
    assert set(material.groupby(["scope", "route"]).size()) == {24}
    with pytest.raises(ValueError, match="eight material"):
        aggregate_learning_curve(pd.DataFrame(rows).query('composition != "0"'))


@pytest.fixture
def seven_cohorts(tmp_path: Path) -> tuple[Path, Path]:
    reports = tmp_path / "reports"
    for n in range(1, 8):
        folder = reports / f"n{n}"
        folder.mkdir(parents=True)
        rows = []
        for fold in range(1, 9):
            tests = [(fold - 1 + i) % 8 for i in range(8 - n)]
            for pressure in (0, 10, 20):
                for route in ("scratch", "direct", "warm"):
                    for checkpoint in [None] if route == "scratch" else range(10):
                        for setting in ["unfrozen"] if route == "scratch" else ["frozen", "unfrozen"]:
                            for material in tests:
                                value = material / 10 + {"scratch": 1, "direct": 0.75, "warm": 0.5}[route]
                                rows.append(
                                    {
                                        "unit": f"{route}_f{fold}_c{checkpoint}_p{pressure}",
                                        "fold": fold,
                                        "composition": str(material),
                                        "pressure": pressure,
                                        "route": route,
                                        "setting": setting,
                                        "is_anchor": material == fold - 1,
                                        "n_train": n,
                                        "n_test": 8 - n,
                                        "z_rmse": value,
                                        "z_rmse_low_T": value,
                                        "relative_rmse": value,
                                        "r2": 0.5,
                                    }
                                )
        pd.DataFrame(rows).to_parquet(folder / "metrics_20261001.parquet", index=False)
        (folder / "report_manifest_20261001.json").write_text(
            json.dumps(
                {
                    "complete_cohort": True,
                    "n_final_models": 984,
                    "warm_checkpoints": 10,
                    "metric_preprocessing_manifest": {"sha256": "same recorded digest"},
                }
            )
        )
    return reports, reports / "n7"


def test_summarize_exports_paired_material_gains_and_reference_provenance(
    seven_cohorts: tuple[Path, Path], tmp_path: Path
) -> None:
    reports, baseline = seven_cohorts
    out = tmp_path / "analysis"
    summary = summarize_learning_curve(reports, baseline, out, "20261001")
    assert len(summary) == 7 * 5 * 4 * 2
    gains = pd.read_csv(out / "paired_gains_20261001.csv")
    assert len(gains) == 7 * 4 * 2
    assert gains[gains.route == "warm"].gain_z_rmse.mean() == pytest.approx(0.5)
    assert gains[gains.route == "direct"].gain_z_rmse.mean() == pytest.approx(0.25)
    provenance = json.loads((out / "analysis_manifest_20261001.json").read_text())
    assert provenance["n_final_models"] == 6888 and len(provenance["cohorts"]) == 7
    assert provenance["reference_scaler_manifest_sha256"] == "same recorded digest"
    assert len(pd.read_parquet(out / "metrics_all_sizes_20261001.parquet")) == 984 * 28
    for scope in ("anchor", "all_test"):
        assert (out / f"learning_curve_{scope}_20261001.png").stat().st_size > 0


@pytest.mark.parametrize("failure", ["different_reference", "missing_reference", "partial", "duplicates", "wrong_size"])
def test_summarize_rejects_incompatible_cohorts(seven_cohorts: tuple[Path, Path], tmp_path: Path, failure: str) -> None:
    reports, baseline = seven_cohorts
    path = reports / "n2/report_manifest_20261001.json"
    report = json.loads(path.read_text())
    if failure == "different_reference":
        report["metric_preprocessing_manifest"]["sha256"] = "different digest"
    elif failure == "missing_reference":
        report["metric_preprocessing_manifest"] = None
    elif failure == "partial":
        report["complete_cohort"] = False
    else:
        metric_path = path.with_name("metrics_20261001.parquet")
        frame = pd.read_parquet(metric_path)
        if failure == "duplicates":
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        else:
            frame["n_train"] = 3
        frame.to_parquet(metric_path)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        summarize_learning_curve(reports, baseline, tmp_path / "invalid", "20261001")
