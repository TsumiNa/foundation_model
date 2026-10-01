# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Validate a completed AGIS cohort and export equal-compound-weight results."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(kw_only=True)
class ReportSettings:
    warm_checkpoints: int = 5
    allow_partial: bool = False
    recovery_manifest: Path | None = None
    metric_preprocessing_manifest: Path | None = None
    recovery_root: Path | None = None

    def __post_init__(self) -> None:
        if self.warm_checkpoints not in (5, 10):
            raise ValueError("Report the agreed first-five or full-ten warm cohort")
        if (self.recovery_manifest is None) != (self.recovery_root is None):
            raise ValueError("Recovery requires both its manifest and output root")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(values: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    return (np.arcsinh(values / scaler["prescale_std"]) - scaler["asinh_mean"]) / scaler["asinh_std"]


def curve_metrics(truth: np.ndarray, prediction: np.ndarray, temperature: np.ndarray, scaler: dict) -> dict[str, float]:
    if truth.shape != (300,) or prediction.shape != truth.shape or temperature.shape != truth.shape:
        raise ValueError("Metrics require the complete 300-point curve")
    if not np.isfinite(np.stack([truth, prediction, temperature])).all():
        raise ValueError("Metrics require finite curves")
    residual = prediction - truth
    z_error = normalized(prediction, scaler) - normalized(truth, scaler)
    variance_sum = np.square(truth - truth.mean()).sum()
    rms = np.sqrt(np.mean(np.square(truth)))
    return {
        "z_rmse": float(np.sqrt(np.mean(np.square(z_error)))),
        "z_rmse_low_T": float(np.sqrt(np.mean(np.square(z_error[temperature <= 100])))),
        "rmse_uohm_cm": float(np.sqrt(np.mean(np.square(residual)))),
        "mae_uohm_cm": float(np.mean(np.abs(residual))),
        "relative_rmse": float(np.sqrt(np.mean(np.square(residual))) / rms) if rms else float("nan"),
        "r2": 1 - float(np.square(residual).sum() / variance_sum) if variance_sum else float("nan"),
    }


def collect_results(
    manifest_path: Path, root: Path, project: Path, destination: Path, settings: ReportSettings
) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    identity = json.loads((root / "campaign_identity.json").read_text())
    if identity["manifest_sha256"] != sha256(manifest_path) or identity["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Production outputs belong to another manifest or source")
    recovery_identity = None
    if settings.recovery_manifest is not None and settings.recovery_root is not None:
        recovery_manifest = json.loads(settings.recovery_manifest.read_text())
        if {k: v for k, v in recovery_manifest.items() if k != "source_sha256"} != {
            k: v for k, v in manifest.items() if k != "source_sha256"
        }:
            raise ValueError("Recovery changes the experiment protocol or input artifacts")
        recovery_identity = json.loads((settings.recovery_root / "campaign_identity.json").read_text())
        if (
            recovery_identity["manifest_sha256"] != sha256(settings.recovery_manifest)
            or recovery_identity["source_sha256"] != recovery_manifest["source_sha256"]
            or recovery_identity["runtime"] != identity["runtime"]
        ):
            raise ValueError("Recovery manifest, source or runtime identity mismatch")
    for path, expected in manifest["input_sha256"].items():
        if sha256(project / path) != expected:
            raise ValueError(f"Input artifact drift: {path}")
    metric_preprocessing = manifest["preprocessing"]
    learning_curve = metric_preprocessing.get("design") == "balanced_nested_learning_curve"
    n_train = metric_preprocessing.get("n_train", 7)
    metric_provenance = None
    if settings.metric_preprocessing_manifest is not None:
        metric_path = settings.metric_preprocessing_manifest
        metric_preprocessing = json.loads(metric_path.read_text())
        original = manifest["preprocessing"]
        if len(metric_preprocessing["folds"]) != len(original["folds"]):
            raise ValueError("Comparison scaler manifest has different holdout folds")
        for key in ("temperature_min_K", "temperature_max_K", "n_points"):
            if metric_preprocessing["config"][key] != original["config"][key]:
                raise ValueError("Comparison scaler manifest has a different temperature grid")
        for a, b in zip(original["folds"], metric_preprocessing["folds"], strict=True):
            if (
                a["heldout_composition"] != b["heldout_composition"]
                or (not learning_curve and a["fit_compositions"] != b["fit_compositions"])
                or (learning_curve and not set(a["fit_compositions"]).issubset(b["fit_compositions"]))
            ):
                raise ValueError("Comparison scaler manifest has different train/test compositions")
            for pressure, scaler in b["scalers"].items():
                values = [scaler[key] for key in ("prescale_std", "asinh_mean", "asinh_std")]
                if pressure not in a["scalers"] or not np.isfinite(values).all() or values[0] <= 0 or values[2] <= 0:
                    raise ValueError("Invalid comparison scaler parameters")
            if set(a["scalers"]) != set(b["scalers"]):
                raise ValueError("Comparison scaler manifest has different pressures")
        metric_provenance = {"path": str(metric_path), "sha256": sha256(metric_path)}
    if learning_curve and metric_provenance is None:
        raise ValueError("Learning curves require fixed reference scalers for cross-size metrics")
    if (
        learning_curve
        and metric_provenance is not None
        and metric_provenance["sha256"] != manifest["preprocessing"].get("reference_sha256")
    ):
        raise ValueError("Comparison manifest digest differs from the recorded seven-training reference")
    metric_by_composition = {f["heldout_composition"]: f["scalers"] for f in metric_preprocessing["folds"]}
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    missing = []
    recovered_units = []
    for unit in manifest["units"]:
        checkpoint = unit["checkpoint_index"]
        if unit["route"] == "warm" and checkpoint >= settings.warm_checkpoints:
            continue
        tag = "scratch" if checkpoint is None else f"c{checkpoint + 1:02d}"
        name = f"{unit['route']}_f{unit['fold']:02d}_{tag}_p{unit['pressure']}"
        folder = root / name
        unit_identity = identity
        if settings.recovery_root is not None and (settings.recovery_root / name / "DONE").is_file():
            assert recovery_identity is not None
            if (folder / "DONE").is_file():
                raise ValueError(f"Duplicate completed unit in production and recovery: {name}")
            if unit["route"] != "warm":
                raise ValueError("Recovery is restricted to interrupted warm-start routes")
            folder = settings.recovery_root / name
            unit_identity = recovery_identity
            recovered_units.append(name)
        if not (folder / "DONE").is_file():
            missing.append(name)
            continue
        if json.loads((folder / "campaign_identity.json").read_text()) != unit_identity:
            raise ValueError(f"Unit identity mismatch: {name}")
        result = json.loads((folder / "result.json").read_text())
        if any(result[key] != value for key, value in unit.items()):
            raise ValueError(f"Unit metadata mismatch: {name}")
        fold = manifest["preprocessing"]["folds"][unit["fold"] - 1]
        heldout = fold.get("test_compositions", fold["heldout_composition"])
        compositions = [heldout] if isinstance(heldout, str) else heldout
        if result["heldout"] != heldout:
            raise ValueError(f"Heldout metadata mismatch: {name}")
        target = f"agis_rho_{unit['pressure']}gpa"
        tasks = tomllib.loads((project / fold["directory"] / f"tasks_{manifest['settings']['date']}.toml").read_text())
        task = next(task for task in tasks["tasks"] if task["name"] == target)
        data = pd.read_parquet(project / tasks["datasets"][task["dataset"]]["path"])
        gold_rows = data[data["split"] == "test"]
        if len(gold_rows) != len(compositions) or set(gold_rows.composition) != set(compositions):
            raise ValueError(f"Invalid outer holdout dataset: {name}")
        scaler = fold["scalers"][str(unit["pressure"])]
        for label in ["unfrozen"] if unit["route"] == "scratch" else ["frozen", "unfrozen"]:
            fit = folder / label
            frame = pd.read_parquet(fit / "training/finetune" / f"{target}_pred.parquet")
            if len(frame) != 300 * len(compositions) or set(frame["composition"]) != set(compositions):
                raise ValueError(f"Prediction shape or composition mismatch: {name}/{label}")
            if not np.isfinite(frame[["true", "pred", "t"]].to_numpy()).all():
                raise ValueError(f"Nonfinite predictions: {name}/{label}")
            summary = json.loads((fit / "training/finetune_summary.json").read_text())
            if summary["epochs_run"] != manifest["settings"]["final_epochs"] or summary["freeze_encoder"] != (
                label == "frozen"
            ):
                raise ValueError(f"Training budget or freeze setting mismatch: {name}/{label}")
            model = fit / "training/final_model.pt"
            initial = fit / "initial_model.pt"
            metadata = {
                **unit,
                "unit": name,
                "campaign_source_sha256": unit_identity["source_sha256"],
                "recovered": name in recovered_units,
                "setting": label,
                "n_train": n_train,
                "n_test": len(compositions),
                "anchor_composition": fold["heldout_composition"],
                "checkpoint_run": None if checkpoint is None else manifest["selection"]["models"][checkpoint]["run"],
                "source_checkpoint_sha256": None
                if checkpoint is None
                else manifest["selection"]["models"][checkpoint]["sha256"],
                "model_path": str(model),
                "model_sha256": sha256(model),
                "model_bytes": model.stat().st_size,
                "initial_sha256": sha256(initial),
                "prediction_sha256": sha256(fit / "training/finetune" / f"{target}_pred.parquet"),
                "unit_elapsed_seconds": result["elapsed_seconds"],
                "slurm_job": result["slurm_job"],
            }
            for _, gold in gold_rows.iterrows():
                composition = gold["composition"]
                curve = frame[frame.composition == composition]
                truth = np.asarray(gold["rho_uohm_cm"], dtype=float)
                temperature = np.asarray(gold["temperature_K"], dtype=float)
                if len(curve) != 300 or not np.allclose(curve["t"], temperature, rtol=0, atol=2e-5):
                    raise ValueError(f"Temperature grid mismatch: {name}/{label}/{composition}")
                if not np.allclose(
                    normalized(curve["true"].to_numpy(), scaler), normalized(truth, scaler), rtol=1e-5, atol=1e-6
                ):
                    raise ValueError(f"Prediction truth differs from the heldout dataset: {name}/{label}/{composition}")
                metric_scaler = metric_by_composition[composition][str(unit["pressure"])]
                rows.append(
                    {
                        **metadata,
                        "composition": composition,
                        "formula": gold["formula"],
                        "is_anchor": composition == fold["heldout_composition"],
                        **curve_metrics(truth, curve["pred"].to_numpy(), temperature, metric_scaler),
                    }
                )
                curves.append(
                    pd.DataFrame(
                        {
                            "unit": name,
                            "setting": label,
                            "composition": composition,
                            "temperature_K": temperature,
                            "true_uohm_cm": truth,
                            "pred_uohm_cm": curve["pred"].to_numpy(),
                        }
                    )
                )
    if missing and not settings.allow_partial:
        raise ValueError(f"Incomplete agreed cohort: {len(missing)} units missing; first: {missing[0]}")
    if not rows:
        raise ValueError("No completed final models")
    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    date = manifest["settings"]["date"]
    metrics.to_parquet(destination / f"metrics_{date}.parquet", index=False)
    metrics.to_csv(destination / f"metrics_{date}.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(destination / f"predictions_{date}.parquet", index=False)
    # Checkpoint repeats summarize each material first; materials and pressures then receive equal weight.
    matched = metrics[(metrics["route"] == "scratch") | (metrics["checkpoint_index"] < settings.warm_checkpoints)]
    macro = (
        matched.groupby(["route", "setting", "fold", "pressure", "composition"], dropna=False)[
            ["z_rmse", "z_rmse_low_T", "relative_rmse", "r2"]
        ]
        .median()
        .reset_index()
    )
    macro.to_csv(destination / f"per_curve_matched_{date}.csv", index=False)
    aggregate = (
        macro.groupby(["route", "setting"])[["z_rmse", "z_rmse_low_T", "relative_rmse", "r2"]].mean().reset_index()
    )
    aggregate.to_csv(destination / f"aggregate_matched_{date}.csv", index=False)
    all_curves = (
        metrics.groupby(["route", "setting", "fold", "pressure", "composition"], dropna=False)[
            ["z_rmse", "z_rmse_low_T", "relative_rmse", "r2"]
        ]
        .median()
        .reset_index()
    )
    all_curves.to_csv(destination / f"per_curve_all_{date}.csv", index=False)
    all_curves.groupby(["route", "setting"])[
        ["z_rmse", "z_rmse_low_T", "relative_rmse", "r2"]
    ].mean().reset_index().to_csv(destination / f"aggregate_all_{date}.csv", index=False)
    paired = matched[matched["route"].isin(["direct", "warm"])].pivot(
        index=["fold", "pressure", "composition", "checkpoint_index", "setting"], columns="route", values="z_rmse"
    )
    if {"direct", "warm"}.issubset(paired.columns):
        paired = paired.dropna(subset=["direct", "warm"])
        paired["warm_minus_direct"] = paired["warm"] - paired["direct"]
        paired = paired.reset_index()
    else:
        paired = pd.DataFrame(
            columns=[
                "fold",
                "pressure",
                "composition",
                "checkpoint_index",
                "setting",
                "direct",
                "warm",
                "warm_minus_direct",
            ]
        )
    paired.to_csv(destination / f"paired_warm_direct_{date}.csv", index=False)
    (destination / f"report_manifest_{date}.json").write_text(
        json.dumps(
            {
                "complete_cohort": not missing,
                "warm_checkpoints": settings.warm_checkpoints,
                "missing_units": missing,
                "n_final_models": len(metrics.drop_duplicates(["unit", "setting"])),
                "n_evaluated_curves": len(metrics),
                "n_train": n_train,
                "counts": {
                    f"{route}/{setting}": len(group.drop_duplicates(["unit", "setting"]))
                    for (route, setting), group in metrics.groupby(["route", "setting"])
                },
                "campaign_identity": identity,
                "recovery_campaign_identity": recovery_identity,
                "recovered_units": recovered_units,
                "collector_sha256": sha256(Path(__file__)),
                "metric_preprocessing_manifest": metric_provenance,
                "metric_units": {
                    "z_rmse": "comparison fold normalized units"
                    if metric_provenance
                    else "fold training normalized units",
                    "rmse_uohm_cm": "microohm cm",
                    "mae_uohm_cm": "microohm cm",
                },
                "aggregation": "Median over matched checkpoints within each split/material/pressure; equal-weight mean across balanced test curves. Checkpoints are not independent materials.",
                "pretraining_overlap": manifest["pretraining_overlap"],
            },
            indent=2,
        )
    )
    plot_results(metrics, pd.concat(curves, ignore_index=True), destination, settings.warm_checkpoints)
    return metrics


def plot_results(metrics: pd.DataFrame, predictions: pd.DataFrame, destination: Path, warm_checkpoints: int) -> None:
    matched = metrics[(metrics["route"] == "scratch") | (metrics["checkpoint_index"] < warm_checkpoints)]
    # Keep the same eight reference test materials in curve panels at every training size.
    matched = matched[matched.is_anchor]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    groups = [
        ("scratch", "unfrozen"),
        ("direct", "frozen"),
        ("direct", "unfrozen"),
        ("warm", "frozen"),
        ("warm", "unfrozen"),
    ]
    colors = ["#64748b", "#93c5fd", "#2563eb", "#fbbf24", "#d97706"]
    for ax, pressure in zip(axes, (0, 10, 20), strict=True):
        for i, ((route, setting), color) in enumerate(zip(groups, colors, strict=True)):
            subset = matched[
                (matched["route"] == route) & (matched["setting"] == setting) & (matched["pressure"] == pressure)
            ]
            values = subset.groupby("fold")["z_rmse"].median()
            if len(values):
                ax.scatter(np.full(len(values), i), values, color=color, alpha=0.75, s=30)
                ax.plot([i - 0.22, i + 0.22], [values.mean()] * 2, color="black", lw=2)
        ax.set_xticks(range(5), ["Scratch", "Direct\nfrozen", "Direct\nfull", "Warm\nfrozen", "Warm\nfull"], fontsize=9)
        ax.set_title(f"{pressure} GPa")
        ax.set_ylabel("Normalized RMSE (lower is better)")
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(destination / "matched_errors.png", dpi=180)
    plt.close(fig)
    for pressure in (0, 10, 20):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
        for fold, ax in enumerate(axes.flat, 1):
            subset = matched[(matched["fold"] == fold) & (matched["pressure"] == pressure)]
            if subset.empty:
                continue
            composition = subset.iloc[0]["composition"]
            example = predictions[
                (predictions["unit"] == subset.iloc[0]["unit"])
                & (predictions["setting"] == subset.iloc[0]["setting"])
                & (predictions["composition"] == composition)
            ]
            ax.set_yscale("symlog", linthresh=max(1.0, float(np.max(np.abs(example["true_uohm_cm"]))) * 0.001))
            ax.plot(example["temperature_K"], example["true_uohm_cm"], color="black", lw=2, label="Observed")
            for (route, setting), color in zip(groups, colors, strict=True):
                units = subset[(subset["route"] == route) & (subset["setting"] == setting)]["unit"]
                frame = predictions[
                    predictions["unit"].isin(units)
                    & (predictions["setting"] == setting)
                    & (predictions["composition"] == composition)
                ]
                if frame.empty:
                    continue
                pivot = frame.pivot(index="temperature_K", columns="unit", values="pred_uohm_cm")
                ax.plot(pivot.index, pivot.median(axis=1), color=color, label=f"{route} {setting}")
                if len(pivot.columns) > 1:
                    ax.fill_between(
                        pivot.index, pivot.quantile(0.25, axis=1), pivot.quantile(0.75, axis=1), color=color, alpha=0.12
                    )
            ax.set_title(subset.iloc[0]["formula"], fontsize=10)
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Resistivity (μΩ cm)")
            ax.grid(alpha=0.15)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"{pressure} GPa: heldout curves; matched checkpoint median and IQR", fontsize=13)
        fig.savefig(destination / f"curves_{pressure}gpa.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warm-checkpoints", type=int, default=5)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--metric-preprocessing-manifest", type=Path)
    args = parser.parse_args()
    frame = collect_results(
        args.manifest,
        args.root,
        args.project,
        args.output,
        ReportSettings(
            warm_checkpoints=args.warm_checkpoints,
            allow_partial=args.allow_partial,
            recovery_manifest=args.recovery_manifest,
            metric_preprocessing_manifest=args.metric_preprocessing_manifest,
            recovery_root=args.recovery_root,
        ),
    )
    print(f"Collected {len(frame.drop_duplicates(['unit', 'setting']))} validated final models / {len(frame)} curves")


if __name__ == "__main__":
    main()
