# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Plan the six additional AGIS cohorts and summarize the fixed-material learning curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from campaign import CampaignSettings, file_sha256, plan_campaign


def plan_learning_curve(split_manifest: Path, output: Path, base_config: Path, settings: CampaignSettings) -> Path:
    splits = json.loads(split_manifest.read_text())
    if splits["settings"]["seed"] != settings.seed or splits["settings"]["date"] != settings.date:
        raise ValueError("Split and campaign seeds/date differ")
    if output.exists():
        raise FileExistsError("Learning-curve campaign already exists")
    output.mkdir(parents=True)
    manifests = {}
    grids: dict[str, list[str]] = {stage: [] for stage in ("direct", "scratch", "warm_first", "warm_rest")}
    for n_train in range(1, 7):
        folder = output / f"n{n_train}"
        path = plan_campaign(
            base_config, folder, settings, preprocessing_manifest=Path(splits["manifests"][str(n_train)])
        )
        manifests[str(n_train)] = {"path": str(path), "sha256": file_sha256(path)}
        for stage in grids:
            grids[stage].extend(f"{path} {index}\n" for index in (folder / f"{stage}.txt").read_text().splitlines())
    for stage, lines in grids.items():
        (output / f"{stage}.txt").write_text("".join(lines))
    path = output / "learning_curve_manifest.json"
    path.write_text(
        json.dumps(
            {
                "split_manifest": str(split_manifest),
                "split_sha256": file_sha256(split_manifest),
                "manifests": manifests,
                "new_final_models": 5904,
                "reference_preprocessing": splits["manifests"]["7"],
            },
            indent=2,
        )
    )
    return path


def aggregate_learning_curve(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse checkpoints first, then splits, then average the eight material units."""
    scores = ["z_rmse", "z_rmse_low_T", "relative_rmse", "r2"]
    groups = ["n_train", "route", "setting", "composition", "pressure"]
    summaries = []
    materials = []
    for scope in ("anchor", "all_test"):
        selected = metrics[metrics.is_anchor] if scope == "anchor" else metrics
        split = selected.groupby([*groups, "fold"], dropna=False)[scores].median().reset_index()
        per_material = split.groupby(groups, dropna=False)[scores].mean().reset_index()
        per_material["scope"] = scope
        materials.append(per_material)
        for pressure in (None, 0, 10, 20):
            current = per_material if pressure is None else per_material[per_material.pressure == pressure]
            per_compound = current.groupby(["n_train", "route", "setting", "composition"])[scores].mean().reset_index()
            for (n_train, route, setting), group in per_compound.groupby(["n_train", "route", "setting"]):
                if group.composition.nunique() != 8:
                    raise ValueError("Each learning-curve summary requires all eight material units")
                summaries.append(
                    {
                        "scope": scope,
                        "n_train": n_train,
                        "pressure": "all" if pressure is None else str(pressure),
                        "route": route,
                        "setting": setting,
                        "n_materials": len(group),
                        **{s: float(group[s].mean()) for s in scores},
                    }
                )
    return pd.DataFrame(summaries), pd.concat(materials, ignore_index=True)


def summarize_learning_curve(reports: Path, baseline: Path, output: Path, date: str) -> pd.DataFrame:
    frames = []
    provenance = []
    reference_digest = None
    for n_train in range(1, 8):
        folder = baseline if n_train == 7 else reports / f"n{n_train}"
        path = folder / f"metrics_{date}.parquet"
        report_path = folder / f"report_manifest_{date}.json"
        report = json.loads(report_path.read_text())
        if not report["complete_cohort"] or report["n_final_models"] != 984 or report["warm_checkpoints"] != 10:
            raise ValueError("Require the complete 984-model, ten-checkpoint cohort at each size")
        reference = report.get("metric_preprocessing_manifest")
        if not isinstance(reference, dict) or not reference.get("sha256"):
            raise ValueError(
                "Every cohort, including the seven-training baseline, requires recorded reference-scaler provenance"
            )
        if reference_digest is None:
            reference_digest = reference["sha256"]
        elif reference["sha256"] != reference_digest:
            raise ValueError("Learning-curve cohorts use different reference-scaler manifest digests")
        frame = pd.read_parquet(path)
        if n_train == 7:
            frame["is_anchor"] = True
        elif set(frame.n_train) != {n_train} or set(frame.n_test) != {8 - n_train}:
            raise ValueError("Report training/test size differs from the requested cohort")
        frame["n_train"] = n_train
        if len(frame) != 984 * (8 - n_train) or len(frame.drop_duplicates(["unit", "setting"])) != 984:
            raise ValueError("Learning-curve report has duplicate or missing model/material evaluations")
        if frame.composition.nunique() != 8 or set(frame.pressure) != {0, 10, 20}:
            raise ValueError("Learning-curve report requires eight materials at all three pressures")
        frames.append(frame)
        provenance.append(
            {
                "n_train": n_train,
                "metrics_path": str(path),
                "metrics_sha256": file_sha256(path),
                "report_path": str(report_path),
                "report_sha256": file_sha256(report_path),
            }
        )
    metrics = pd.concat(frames, ignore_index=True)
    summary, materials = aggregate_learning_curve(metrics)
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(output / f"metrics_all_sizes_{date}.parquet", index=False)
    summary.to_csv(output / f"summary_{date}.csv", index=False)
    materials.to_csv(output / f"per_material_{date}.csv", index=False)
    paired = materials.pivot(
        index=["scope", "n_train", "composition", "pressure"], columns=["route", "setting"], values="z_rmse"
    )
    scratch = paired[("scratch", "unfrozen")]
    gains = []
    for route in ("direct", "warm"):
        for setting in ("frozen", "unfrozen"):
            difference = scratch - paired[(route, setting)]
            for (scope, n_train), group in difference.groupby(level=["scope", "n_train"]):
                by_material = group.groupby(level="composition").mean().to_numpy()
                rng = np.random.default_rng(20261001)
                boot = rng.choice(by_material, (10000, len(by_material)), replace=True).mean(axis=1)
                gains.append(
                    {
                        "scope": scope,
                        "n_train": n_train,
                        "route": route,
                        "setting": setting,
                        "gain_z_rmse": float(by_material.mean()),
                        "ci_low": float(np.quantile(boot, 0.025)),
                        "ci_high": float(np.quantile(boot, 0.975)),
                        "better_material_pressure_pairs": int((group > 0).sum()),
                        "total_material_pressure_pairs": len(group),
                    }
                )
    gain_frame = pd.DataFrame(gains)
    gain_frame.to_csv(output / f"paired_gains_{date}.csv", index=False)
    styles = [
        ("scratch", "unfrozen", "Scratch", "#64748b"),
        ("direct", "frozen", "Direct frozen", "#93c5fd"),
        ("direct", "unfrozen", "Direct full", "#2563eb"),
        ("warm", "frozen", "Warm frozen", "#fbbf24"),
        ("warm", "unfrozen", "Warm full", "#d97706"),
    ]
    for scope in ("anchor", "all_test"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for pressure, ax in zip((0, 10, 20), axes, strict=True):
            for route, setting, label, color in styles:
                data = summary[
                    (summary.scope == scope)
                    & (summary.pressure == str(pressure))
                    & (summary.route == route)
                    & (summary.setting == setting)
                ].sort_values("n_train")
                ax.plot(data.n_train, data.z_rmse, "o-", label=label, color=color)
            ax.set_xticks(range(1, 8))
            ax.set_xlabel("Training compounds")
            ax.set_ylabel("Fixed-reference normalized RMSE (lower is better)")
            ax.set_title(f"{pressure} GPa")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("Same eight reference holdouts" if scope == "anchor" else "All test compounds, balanced splits")
        fig.savefig(output / f"learning_curve_{scope}_{date}.png", dpi=180)
        fig.savefig(output / f"learning_curve_{scope}_{date}.pdf")
        plt.close(fig)
    (output / f"analysis_manifest_{date}.json").write_text(
        json.dumps(
            {
                "date": date,
                "cohorts": provenance,
                "n_final_models": 6888,
                "new_final_models": 5904,
                "n_independent_materials": 8,
                "reference_scaler_manifest_sha256": reference_digest,
                "metric_scale": "Original seven-train LOCO scaler for each test composition and pressure, used only in scoring",
                "aggregation": "Median over checkpoints per split/material/pressure; mean over balanced splits; equal material/pressure weights",
                "intervals": "Descriptive paired bootstrap of eight material blocks, keeping pressure measurements together; checkpoint repeats are not samples",
                "reference_slide": "DECK_20260915_optimized_v6.pptx page 41",
            },
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--splits", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--base-config", type=Path, default=Path("experiments/agis_transfer/base_pretrained.toml"))
    summary = sub.add_parser("summarize")
    summary.add_argument("--reports", type=Path, required=True)
    summary.add_argument("--baseline", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--date", default="20261001")
    args = parser.parse_args()
    if args.command == "plan":
        print(plan_learning_curve(args.splits, args.output, args.base_config, CampaignSettings()))
    else:
        print(summarize_learning_curve(args.reports, args.baseline, args.output, args.date).to_string(index=False))


if __name__ == "__main__":
    main()
