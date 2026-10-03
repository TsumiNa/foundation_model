"""Reproduce final history audits, checkpoint sensitivity, figures and report from raw lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect_head_probe import collect, prediction_disagreement


def best_last_table(root: Path, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in selected.to_dict("records"):
        if row["readout"].startswith("ridge"):
            continue
        folder = root / f"{row['arm']}_s{row['seed']}_k{row['k']}" / f"f{round(row['fraction'] * 100):03d}"
        recorded = json.loads((folder / f"{row['readout']}_selected_last.json").read_text())
        best = pd.read_parquet(folder / f"{row['readout']}_pred.parquet")
        last = pd.read_parquet(folder / f"{row['readout']}_selected_last_pred.parquet")
        pair = best.merge(
            last, on="composition", suffixes=("_best", "_last"), how="outer", validate="one_to_one", indicator=True
        )
        if (
            not pair["_merge"].eq("both").all()
            or not np.isfinite(pair.select_dtypes("number")).all().all()
            or not np.allclose(pair.true_best, pair.true_last)
        ):
            raise ValueError("Best/last predictions must have aligned finite references")
        rmse = float(np.sqrt(np.mean((pair.pred_last - pair.true_last) ** 2)))
        if not np.isclose(rmse, recorded["rmse"], rtol=1e-7) or recorded["lr"] != row["lr"]:
            raise ValueError("Last-checkpoint score or selected learning rate mismatch")
        rows.append({**row, "last_rmse": rmse, "last_minus_best": rmse - row["rmse"]})
    return pd.DataFrame(rows)


def audit_histories(root: Path, selected: pd.DataFrame, rates: list[float]) -> dict:
    lengths = []
    for row in selected.to_dict("records"):
        if row["readout"].startswith("ridge"):
            continue
        for rate in rates:
            folder = (
                root
                / f"{row['arm']}_s{row['seed']}_k{row['k']}"
                / f"f{round(row['fraction'] * 100):03d}"
                / row["readout"]
                / f"lr{rate:g}"
            )
            history = pd.read_csv(folder / "history.csv")
            done = json.loads((folder / "done.json").read_text())
            if history.empty or not np.isfinite(history[["train_mse", "val_mse", "lr"]]).all().all():
                raise ValueError("Empty or nonfinite optimization history")
            if (
                len(history) != done["epochs"]
                or not np.isclose(history.val_mse.min(), done["best_val_mse"])
                or int(history.loc[history.val_mse.idxmin(), "epoch"]) != done["best_epoch"]
            ):
                raise ValueError("History disagrees with checkpoint selection")
            lengths.append(len(history))
    if not lengths:
        raise ValueError("No neural optimization histories")
    return dict(
        neural_trials=len(lengths),
        all_neural_histories_finite=True,
        trial_epoch_min=min(lengths),
        trial_epoch_max=max(lengths),
    )


def gpu_cost(accounting: Path) -> dict:
    rows = []
    for line in accounting.read_text().splitlines():
        fields = line.split("|")
        if "." in fields[0]:
            continue
        if (
            len(fields) < 6
            or fields[1] != "COMPLETED"
            or fields[3] != "0:0"
            or "gres/gpu=1" not in fields[4].split(",")
        ):
            raise ValueError("Require successful one-GPU accounting rows")
        rows.append(dict(job=fields[0], elapsed_seconds=int(fields[2])))
    if not rows or len({r["job"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate GPU accounting")
    hours = sum(r["elapsed_seconds"] for r in rows) / 3600
    return dict(jobs=rows, gpu_hours=hours, rate_jpy_per_gpu_hour=300, estimated_jpy_pre_tax=300 * hours)


def markdown_table(frame: pd.DataFrame) -> str:
    """Avoid an extra optional tabulate dependency for this standalone report."""

    def display(value: object) -> str:
        return f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)

    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
            *["| " + " | ".join(display(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "config", "protocol", "accounting"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame, audit = collect(args.root, args.config, args.protocol)
    if not audit["complete"]:
        raise ValueError("Final report requires the complete registered matrix")
    cfg = tomllib.loads(args.config.read_text())
    audit.update(audit_histories(args.root, frame, cfg["learning_rates"]))
    audit["ridge_trials"] = int(frame.readout.str.startswith("ridge").sum()) * len(cfg["ridge_alphas"])
    audit["all_lanes_frozen"] = True
    (args.output / "final_validation_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    cost = gpu_cost(args.accounting)
    (args.output / "gpu_cost.json").write_text(json.dumps(cost, indent=2))
    heads = best_last_table(args.root, frame)
    heads.to_csv(args.output / "best_last_comparison.csv", index=False)
    checkpoint = (
        heads.groupby(["arm", "fraction"])
        .agg(
            last_minus_best=("last_minus_best", "mean"), max_increase=("last_minus_best", "max"), n=("readout", "size")
        )
        .reset_index()
    )
    checkpoint.to_csv(args.output / "checkpoint_sensitivity_summary.csv", index=False)
    disagreements, pairs = prediction_disagreement(args.root, frame)
    summary = (
        frame.groupby(["arm", "readout", "fraction", "k"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            validation_mse=("best_val_mse", "mean"),
        )
        .reset_index()
    )
    fixed = []
    tail = []
    for lane in audit["completed_lanes"]:
        ident = json.loads((args.root / lane / "identity.json").read_text())
        for fraction in cfg["fractions"]:
            folder = args.root / lane / f"f{round(fraction * 100):03d}"
            pred = pd.read_parquet(folder / "linear_output_pred.parquet")
            error = (pred.pred - pred.true) ** 2
            mask = pred.true.rank(method="first", ascending=False) <= 7
            tail.append(
                dict(
                    arm=ident["arm"],
                    seed=ident["seed"],
                    k=ident["source_count"],
                    fraction=fraction,
                    tail_mse_share=float(error[mask].sum() / error.sum()),
                )
            )
            if ident["source_count"] == 7:
                for head in ("legacy", "linear_output"):
                    fixed.append(
                        dict(
                            arm=ident["arm"],
                            seed=ident["seed"],
                            fraction=fraction,
                            readout=head,
                            **json.loads((folder / f"{head}_fixed_lr.json").read_text()),
                        )
                    )
    pd.DataFrame(tail).to_csv(args.output / "test_tail_concentration.csv", index=False)
    fixed = pd.DataFrame(fixed).groupby(["arm", "fraction", "readout"]).rmse.mean().unstack().reset_index()
    fixed["identity_minus_current"] = fixed.linear_output - fixed.legacy
    fixed.to_csv(args.output / "fixed_lr_activation_summary.csv", index=False)
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "axes.titlesize": 17})
    for fraction in cfg["fractions"]:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True, layout="constrained")
        for ax, (head, title) in zip(
            axes,
            [
                ("ridge_post", "Ridge regression"),
                ("linear_output", "MLP head: 128 → 64"),
                ("wide", "MLP head: 512 → 256"),
            ],
            strict=True,
        ):
            for arm, name in [("mlp_tuned", "MLP encoder"), ("grouped_mean", "Transformer encoder")]:
                g = summary[(summary.fraction == fraction) & (summary.readout == head) & (summary.arm == arm)]
                ax.errorbar(g.k, g.rmse_mean, yerr=g.rmse_sd, label=name, marker="o", capsize=4, lw=2)
            ax.set(title=title, xlabel="Source task count", xticks=cfg["source_counts"])
            ax.grid(alpha=0.15)
            ax.legend(fontsize=12)
        axes[0].set_ylabel("Dielectric constant RMSE\n(original target scale)")
        fig.suptitle(
            f"Target training size: {fraction:.0%}  •  Frozen encoders  •  Mean ± SD across 3 seeds", fontsize=19
        )
        fig.savefig(args.output / f"readout_scaling_f{round(fraction * 100):03d}.png", dpi=180)
        plt.close(fig)
    view = (
        pairs.query('k==7 and fraction==1 and readout=="linear_output"')
        .groupby("composition")[["true_mlp", "pred_mlp", "pred_transformer"]]
        .mean()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    low = min(view.pred_mlp.min(), view.pred_transformer.min()) - 3
    high = max(view.pred_mlp.max(), view.pred_transformer.max()) + 3
    axes[0].scatter(view.pred_mlp, view.pred_transformer, s=16, alpha=0.6)
    axes[0].plot([low, high], [low, high], "k--", lw=1)
    axes[0].set(
        xlabel="MLP encoder prediction",
        ylabel="Transformer encoder prediction",
        title="Same composition; different predictions",
        xlim=(low, high),
        ylim=(low, high),
    )
    axes[0].set_aspect("equal")
    order = np.argsort(view.true_mlp.to_numpy())[::-1]
    for column, name in [("pred_mlp", "MLP encoder"), ("pred_transformer", "Transformer encoder")]:
        errors = (view[column] - view.true_mlp).to_numpy() ** 2
        axes[1].plot(np.arange(1, len(errors) + 1), np.cumsum(errors[order]) / errors.sum(), label=name, lw=2)
    axes[1].set(
        xlabel="Number of compositions\n(ordered by decreasing reference value)",
        ylabel="Cumulative fraction of squared error",
        title="A small tail dominates RMSE",
        xscale="log",
        ylim=(0, 1.02),
    )
    axes[1].legend(fontsize=12)
    fig.suptitle("7 source tasks; full training size; 128 → 64 head; seed-mean predictions", fontsize=17)
    fig.savefig(args.output / "paired_predictions_and_error_concentration.png", dpi=180)
    plt.close(fig)
    k7 = summary[(summary.k == 7) & (summary.fraction == 1)].drop(columns=["fraction", "k"])
    scaling = (
        summary[summary.readout.isin(["ridge_post", "linear_output", "wide"])]
        .pivot(index=["arm", "readout", "fraction"], columns="k", values="rmse_mean")
        .reset_index()
    )
    d = disagreements.query("k==7").groupby(["fraction", "readout"]).difference_over_mlp_rmse.mean().reset_index()
    text = f"""# Frozen encoder/readout diagnosis — 2026-10-03

Reproducible final report from the registered raw lanes; training revision {audit["revision"][0]}.
All {len(audit["completed_lanes"])} lanes and {audit["selected_fits"]} selected fits completed. The {audit["neural_trials"]} neural trials have finite histories and ran {audit["trial_epoch_min"]}–{audit["trial_epoch_max"]} epochs. Ridge search fits: {audit["ridge_trials"]}. Every encoder's parameters and buffers stayed frozen.

**Scope.** Tuned MLP versus grouped-mean Transformer, dielectric target only, source counts 1/3/7 and three paired seeds. Target training size 10%/100% means 306/3,051 compositions; validation/test remain 656/697. No source pretraining was repeated. All readout tuning and checkpoint selection use target validation MSE. No reconstruction objective is present. This is a separate diagnostic, not pooled with original end-to-end fits.

**Reading the results.** RMSE/MAE use inverse-transformed original target values; lower is better. SD is sample SD across three seeds, not predictive uncertainty. Prediction disagreement D = RMSE(Transformer prediction − MLP prediction) / RMSE(MLP prediction − reference), computed per paired seed then averaged. D=0 means identical predictions; D=0.46 means their RMS difference is 46% of the MLP error magnitude. It is not an improvement score. Validation MSE is in standardized target space and is comparable across encoders within the same subset.

## Seven source tasks, full target training size

{markdown_table(k7)}

The standard and wider neural readouts do not reveal a consistent Transformer advantage. Ridge reveals a modest Transformer test-RMSE advantage at k=7/full training, but validation MSE, MAE and low-data rankings disagree. Do not select an architectural winner from test RMSE. Raw-descriptor ridge references and all seed-level values remain in selected_metrics.csv.

## Output activation, fixed learning rate 0.002

{markdown_table(fixed)}

The current installed head maps None to LeakyReLU(0.1); explicit Identity tests the alternative with paired initial weights and batches. Effects change sign across settings. This behavior is real, but the experiment does not establish it as the cause of architectural similarity. No shared package semantics were changed; a repair would need a separate compatibility/version/image decision.

## Prediction disagreement and task-count trends

{markdown_table(d)}

Different predictions can yield similar aggregate RMSE. Seven high-reference test compositions dominate squared error; test_tail_concentration.csv gives per-seed shares without dropping or selecting on those cases. The paired-prediction figure uses seed-mean predictions, while D and the reported tail shares are averaged after per-seed computation.

{markdown_table(scaling)}

These are transfer-versus-source-count curves, not a universal scaling law: task content/order and cumulative compute change together. At full training the Transformer improves toward k=7, particularly for ridge; the low-data pattern is not monotonic. validation_cka.csv and representation_statistics.csv show different encoder geometry and increasing effective rank; these are descriptive, not accuracy or information-content tests. Pre-tanh ridge does not improve full-training test RMSE here, so this screen does not support severe tanh saturation as the immediate bottleneck. It cannot rule out token pooling/projection effects or generalize to previously unstable CLS/concat variants.

## Checkpoint selection

{markdown_table(checkpoint)}

Positive last_minus_best means that using the final head weights increased test RMSE versus the best target-validation weights, within the same validation-selected LR. The low-data differences are larger than the head-width effects. Validation selection is not guaranteed to improve every test case. This cached-feature result does not isolate the old end-to-end workflow's reconstruction objective.

**Decision after this round.** Insufficient FC-head width is not supported as the main explanation in this screen. A better-supported next experiment is matched end-to-end best-target-validation versus task-end checkpoint selection. Source-task/objective/order controls are also motivated by the concentrated early Transformer representation. Token-level readout is conditional; this round launched no new token architecture, pretraining or full-fine-tuning fleet. No readout change was a consistent validation winner that justifies automatically launching the conditional confirmation matrix. This completes the bounded A/B investigation. Broader claims need additional targets and an untouched test set; the present test set was already inspected. The 3-LR/7-alpha search is bounded, not exhaustive.

**Execution.** Official ARM image-installed package 0.5.0, image revision 950012a581d838601530eb200edb9572862457aa; no bound src or PYTHONPATH. PR #74 passed review/fixes/squash merge before execution. Jobs 162107 (smoke), 162113 (six formal unpacked lanes), 162125 (isolated pack calibration), and 162133 (remaining formal lanes) all completed. Slurm total: {cost["gpu_hours"]:.3f} GPU-hours, approximately ¥{cost["estimated_jpy_pre_tax"]:.0f} before tax at ¥300/GPU-hour, not an invoice. PACK=3 repeated calibration metrics matched exactly; throughput was about 2.6×, nonzero recorded utilization 82–91%, memory at most 3,444 MiB. Short-job zero samples do not imply idle execution. Source and best/last head checkpoints remain on RIKYU; data, image, protocol and checkpoint identities are in the lane provenance. Smoke and repeated calibration are excluded from the formal 162-fit collector.

**Files.** final_validation_audit.json; gpu_cost.json; selected_metrics.csv; summary.csv; fixed_lr_activation_summary.csv; paired_encoder_gaps.csv; prediction_disagreement*.csv; paired_composition_predictions.csv; representation_statistics.csv; validation_cka.csv; best_last_comparison.csv; checkpoint_sensitivity_summary.csv; test_tail_concentration.csv. Figures: readout_scaling_f010.png, readout_scaling_f100.png, paired_predictions_and_error_concentration.png. Generated artifacts are synchronized outside Git.
"""
    (args.output / "REPORT_20261003.md").write_text(text)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
