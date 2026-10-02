"""Collect partial or complete transfer fits without admitting functional/calibration runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect(root: Path, protocol: Path) -> tuple[pd.DataFrame, int]:
    registered = tomllib.loads(protocol.read_text())
    settings, arms = registered["benchmark"], registered["encoders"]
    keys = ["arm", "seed", "target", "fraction", "source_count", "mode"]
    rows = []
    campaign = None
    protocol_sha256 = hashlib.sha256(protocol.read_bytes()).hexdigest()
    for lane in sorted(root.glob("*_s*")):
        if not (lane / "runtime.json").exists():
            continue
        runtime = json.loads((lane / "runtime.json").read_text())
        if runtime["cpu_smoke"] or runtime["functional_smoke"] or runtime["max_epochs_override"] is not None:
            raise ValueError("Scientific collection refuses CPU/smoke/epoch-capped lanes")
        identity = json.loads((lane / "identity.json").read_text())
        if (
            identity["protocol_sha256"] != protocol_sha256
            or identity["manifest_sha256"] != settings["data_manifest_sha256"]
        ):
            raise ValueError("Lane configuration/data hashes differ from the registered campaign")
        if not isinstance(identity["benchmark_revision"], str) or not re.fullmatch(
            r"[0-9a-f]{40}", identity["benchmark_revision"]
        ):
            raise ValueError("Missing benchmark source revision")
        if (
            runtime["benchmark_revision"] != identity["benchmark_revision"]
            or identity["source_count"] != 7
            or identity["cpu_smoke"]
            or identity["max_epochs_override"] is not None
        ):
            raise ValueError("Runtime and scientific lane identity disagree")
        if campaign is not None and identity != campaign:
            raise ValueError("Cannot aggregate different campaign identities")
        campaign = identity
        if runtime["image_revision"] != settings["image_revision"] or runtime["sif_sha256"] != settings["sif_sha256"]:
            raise ValueError("Runtime does not match the registered image")
        for path in sorted((lane / "targets").glob("*/done.json")):
            fit = json.loads(path.read_text())
            if fit["protocol"] != settings:
                raise ValueError("Fit protocol does not match the analysis protocol")
            if fit["arm"] not in arms or fit["seed"] not in settings["seeds"]:
                raise ValueError("Unregistered encoder or seed in completed fit")
            if (
                fit["target"] not in {"dielectric_total", "power_factor"}
                or fit["fraction"] not in settings["fractions"]
            ):
                raise ValueError("Unregistered target or label budget")
            if (fit["mode"], fit["source_count"]) not in {
                ("scratch", 0),
                *[(mode, k) for mode in ("frozen", "full") for k in settings["source_counts"]],
            }:
                raise ValueError("Unregistered transfer mode/source count")
            row = {key: fit[key] for key in keys}
            row.update({key: fit[key] for key in ("epochs_run", "elapsed_seconds", "encoder_parameters")})
            for split in ("val", "test"):
                row.update({f"{split}_{key}": value for key, value in fit["metrics"][split].items()})
            row["path"] = str(path.parent)
            rows.append(row)
    expected = (
        len(arms) * len(settings["seeds"]) * 2 * len(settings["fractions"]) * (1 + 2 * len(settings["source_counts"]))
    )
    frame = pd.DataFrame(rows)
    if not frame.empty and frame.duplicated(keys).any():
        raise ValueError("Duplicate scientific fit identity")
    return frame, expected


def paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    """Negative relative change means transfer improves on the same encoder's scratch fit."""
    pair_keys = ["arm", "seed", "target", "fraction"]
    baselines = frame.loc[frame["mode"].eq("scratch"), pair_keys + ["test_rmse", "test_standardized_rmse"]].rename(
        columns={"test_rmse": "scratch_rmse", "test_standardized_rmse": "scratch_standardized_rmse"}
    )
    transferred = frame.loc[frame["mode"].ne("scratch")].merge(baselines, on=pair_keys, validate="many_to_one")
    valid = transferred["scratch_standardized_rmse"] > 1e-12
    transferred["relative_change_pct"] = np.nan
    transferred.loc[valid, "relative_change_pct"] = 100 * (
        transferred.loc[valid, "test_rmse"] / transferred.loc[valid, "scratch_rmse"] - 1
    )
    mlp = frame.loc[
        frame["arm"].eq("mlp_tuned") & frame["mode"].eq("scratch"), ["seed", "target", "fraction", "test_rmse"]
    ].rename(columns={"test_rmse": "mlp_scratch_rmse"})
    return transferred.merge(mlp, on=["seed", "target", "fraction"], how="left", validate="many_to_one")


def plot_curves(frame: pd.DataFrame, output: Path) -> None:
    for mode in ("frozen", "full"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
        for ax, (target, fraction) in zip(
            axes.ravel(), [(t, f) for t in ("dielectric_total", "power_factor") for f in (0.1, 1.0)]
        ):
            for arm, group in frame.loc[frame["target"].eq(target) & frame["fraction"].eq(fraction)].groupby("arm"):
                selected = group.loc[group["mode"].isin(["scratch", mode])]
                statistics = selected.groupby("source_count")["test_rmse"].agg(["mean", "std", "count"]).sort_index()
                ax.errorbar(
                    statistics.index,
                    statistics["mean"],
                    yerr=statistics["std"].fillna(0),
                    marker="o",
                    capsize=3,
                    label=arm,
                )
            ax.set_title(f"{target}: {int(fraction * 100)}% target training labels", fontsize=16, pad=14)
            ax.set_xlabel("Number of source tasks (0 = target training from scratch)", fontsize=13)
            ax.set_ylabel("Composition-weighted RMSE (dataset-native units)", fontsize=13)
            ax.set_xticks([0, 1, 3, 7])
            ax.tick_params(labelsize=12)
            ax.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=12)
        fig.suptitle(
            f"Transfer versus source task count — {mode} encoder\nMean ± sample SD across paired seeds; partial results are provisional",
            fontsize=18,
        )
        fig.savefig(output / f"source_count_{mode}.png", dpi=180)
        plt.close(fig)


def plot_paired(paired: pd.DataFrame, output: Path) -> None:
    selected = paired.loc[paired["source_count"].eq(7)].copy()
    if selected.empty:
        return
    selected["condition"] = selected["target"] + " / " + (selected["fraction"] * 100).astype(int).astype(str) + "%"
    for mode in ("frozen", "full"):
        group = selected.loc[selected["mode"].eq(mode)]
        if group.empty:
            continue
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
        conditions = sorted(group["condition"].unique())
        arms = sorted(group["arm"].unique())
        for index, arm in enumerate(arms):
            rows = group.loc[group["arm"].eq(arm)]
            x = (
                rows["condition"].map({c: i for i, c in enumerate(conditions)}).to_numpy()
                + (index - (len(arms) - 1) / 2) * 0.065
            )
            ax.scatter(x, rows["relative_change_pct"], label=arm, s=45, alpha=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(conditions)), conditions)
        ax.tick_params(labelsize=13)
        ax.set_ylabel("Relative RMSE change vs paired same-encoder scratch (%)", fontsize=14)
        ax.set_title(
            f"Individual paired transfer effects: {mode}, seven source tasks\nEach point is one seed; values below zero favor transfer",
            fontsize=18,
            pad=18,
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(output / f"paired_effects_{mode}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Scientific artifacts only, excluding calibration/smoke roots"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, expected = collect(args.root, args.protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"completed_target_fits": len(frame), "expected_target_fits": expected, "complete": len(frame) == expected}
    (args.output / "status.json").write_text(json.dumps(status, indent=2))
    if frame.empty:
        print(json.dumps(status))
        return
    frame.to_csv(args.output / "fits.csv", index=False)
    paired = paired_comparisons(frame)
    paired.to_csv(args.output / "paired_transfer.csv", index=False)
    summary = frame.groupby(["arm", "target", "fraction", "source_count", "mode"])[
        ["val_rmse", "test_rmse", "test_standardized_rmse"]
    ].agg(["mean", "std", "count"])
    summary.to_csv(args.output / "summary.csv")
    if not paired.empty:
        paired.groupby(["arm", "target", "fraction", "source_count", "mode"])["relative_change_pct"].agg(
            ["mean", "std", "count"]
        ).to_csv(args.output / "paired_summary.csv")
    # Validation ranks are descriptive for this fixed screen; test scores cannot tune configurations.
    validation = frame.groupby(["arm", "target", "fraction", "source_count", "mode"])["val_rmse"].agg(["mean", "count"])
    validation = validation.reset_index()
    validation["rank_within_target_budget"] = validation.groupby(["target", "fraction"])["mean"].rank(method="min")
    validation.sort_values(["target", "fraction", "rank_within_target_budget"]).to_csv(
        args.output / "validation_ranking.csv", index=False
    )
    plot_curves(frame, args.output)
    plot_paired(paired, args.output)
    notes = [
        "# Transformer transfer screen",
        f"Completed {len(frame)}/{expected} registered target fits. "
        + ("Complete." if status["complete"] else "Partial; all comparisons are provisional."),
        "RMSE averages squared error within each composition, then across compositions, and takes the square root. Curves retain their original 300-point temperature grids.",
        "Relative change (%) = 100 × (transfer RMSE / same-encoder scratch RMSE − 1), paired by target, label budget and seed. Negative values favor transfer. Baselines with standardized RMSE ≤ 10⁻¹² have undefined ratios.",
        "Also compare against mlp_tuned scratch: a transfer gain over a weak architecture's own baseline need not beat the strongest scratch baseline.",
        "Source-count curves include additional accumulated training compute; they do not isolate task diversity or establish a universal scaling law. Power factor has related Seebeck/ZT source tasks; dielectric_total probes a different property family.",
        "Error bars are sample standard deviations across three paired seeds, not predictive uncertainty or confidence intervals. Incomplete seed groups are identified by the count columns.",
        "This is a fixed-hyperparameter first screen, with architecture-specific encoder learning rates. Deployment selection requires validation-only tuning and fresh-seed confirmation.",
        "Per-fit elapsed time includes workflow overhead and may reflect contention. GPU-hours must be obtained from Slurm accounting; summing process times does not measure GPU consumption.",
    ]
    (args.output / "report.md").write_text("\n\n".join(notes) + "\n")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
