"""Reproduce complete-campaign figures and paired inference from audited lane outputs."""

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

from collect import collect, interval

ARMS = ["mlp", "mlp_large", "transformer", "no_attention"]
LABELS = ["MLP", "Large MLP", "Transformer", "No attention"]
COLORS = ["#2775a7", "#88b0ca", "#d56332", "#638849"]
TARGETS = ["Dielectric total", "Bulk modulus", "Shear modulus", "Piezoelectric max"]
MODES = ["ridge", "frozen", "full"]
MODE_LABELS = ["Frozen ridge", "Frozen nonlinear head", "Full fine-tuning"]


def effects(frame: pd.DataFrame, expected_seeds: list[int]) -> pd.DataFrame:
    """Pair each contrast by seed and retain raw-scale and relative differences."""
    rows = []
    for key, group in frame.groupby(["arm", "target", "fraction", "mode"]):
        w = group.pivot(index="seed", columns="condition", values="rmse")
        for baseline in ["random", "real1", "real3", "shuffled7"]:
            if "real7" not in w or baseline not in w:
                raise ValueError("Missing registered contrast")
            q = w[["real7", baseline]].dropna()
            if set(q.index) != set(expected_seeds) or len(q) < 2:
                raise ValueError("Complete paired seeds required")
            if not np.isfinite(q.to_numpy()).all() or (q[baseline] <= 0).any():
                raise ValueError("Finite RMSE and positive denominator required")
            delta = (q.real7 - q[baseline]).to_numpy()
            relative = 100 * (q.real7 / q[baseline] - 1).to_numpy()
            mean, lo, hi = interval(relative)
            rows.append(
                dict(zip(["arm", "target", "fraction", "mode"], key, strict=True))
                | dict(
                    baseline=baseline,
                    n_seeds=len(q),
                    delta_rmse_mean=float(delta.mean()),
                    relative_mean=mean,
                    relative_lo95=lo,
                    relative_hi95=hi,
                    improved_seeds=int((delta < 0).sum()),
                )
            )
    return pd.DataFrame(rows)


def aligned_predictions(paths: list[Path]) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Align compositions across seeds and reject differing ground truth or missing rows."""
    if not paths:
        raise ValueError("Prediction files required")
    truth = None
    samples = []
    for path in paths:
        f = pd.read_parquet(path).set_index("composition").sort_index()
        if f.index.duplicated().any() or not np.isfinite(f[["true", "pred"]].to_numpy()).all():
            raise ValueError("Invalid prediction records")
        if truth is None:
            truth = f.true
        elif not truth.index.equals(f.index) or not np.allclose(truth.to_numpy(), f.true.to_numpy(), rtol=1e-6):
            raise ValueError("Prediction composition or ground-truth mismatch")
        samples.append(f.pred.to_numpy())
    assert truth is not None
    return truth.index, truth.to_numpy(), np.stack(samples)


def save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def figures(frame: pd.DataFrame, paired: pd.DataFrame, root: Path, output: Path, seeds: list[int]) -> None:
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 11})
    for mode, label in zip(MODES, MODE_LABELS, strict=True):
        fig, axes = plt.subplots(4, 2, figsize=(15, 17), layout="constrained")
        for j, target in enumerate(TARGETS):
            for col, fraction in enumerate([0.1, 1.0]):
                ax = axes[j, col]
                for arm, name, color in zip(ARMS, LABELS, COLORS, strict=True):
                    q = frame[
                        (frame.arm == arm)
                        & (frame.target == target)
                        & (frame.fraction == fraction)
                        & (frame["mode"] == mode)
                    ]
                    w = q.pivot(index="seed", columns="condition", values="rmse")
                    v = w[["random", "real1", "real3", "real7"]]
                    ax.errorbar(
                        [0, 1, 3, 7], v.mean(), yerr=v.std(ddof=1), marker="o", color=color, label=name, capsize=3
                    )
                ax.set_title(f"{target} | training size {fraction:.0%}")
                ax.set_xlabel("Source task count")
                ax.set_ylabel("RMSE (original target scale)")
                ax.set_xticks([0, 1, 3, 7])
                ax.grid(alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle(
            f"{label}: source task count at fixed 6,000 source updates\nMean ± seed SD; 0 = random encoder (training from scratch for full fine-tuning)",
            fontsize=18,
        )
        save(fig, output, f"task_count_{mode}.png")

    strata = [(t, f) for t in TARGETS for f in [0.1, 1.0]]
    for baseline, title in [
        ("random", "Seven-source transfer versus no pretraining"),
        ("shuffled7", "Real versus shuffled source labels"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 8), layout="constrained", sharey=True)
        for ax, mode, label in zip(axes, MODES, MODE_LABELS, strict=True):
            for a, (arm, name, color) in enumerate(zip(ARMS, LABELS, COLORS, strict=True)):
                q = (
                    paired[(paired.arm == arm) & (paired["mode"] == mode) & (paired.baseline == baseline)]
                    .set_index(["target", "fraction"])
                    .loc[strata]
                )
                y = np.arange(8) + (a - 1.5) * 0.17
                ax.errorbar(
                    q.relative_mean,
                    y,
                    xerr=np.array([q.relative_mean - q.relative_lo95, q.relative_hi95 - q.relative_mean]),
                    fmt="o",
                    color=color,
                    label=name,
                    capsize=3,
                )
            ax.axvline(0, color="gray", lw=1)
            ax.set_title(label)
            ax.set_xlabel("Paired relative RMSE change (%)")
            ax.grid(axis="x", alpha=0.2)
        axes[0].set_yticks(np.arange(8), [f"{t}\nTraining size {f:.0%}" for t, f in strata])
        axes[0].invert_yaxis()
        axes[-1].legend(loc="best")
        fig.suptitle(
            f"{title}\n100 × (RMSE real7 / RMSE baseline − 1); negative = improvement; paired-seed 95% bootstrap CI",
            fontsize=17,
        )
        save(fig, output, f"paired_vs_{baseline}.png")

    # File selection is driven solely by registered condition/seed, never by test error.
    lanes = {}
    for p in root.glob("case*/identity.json"):
        c = json.loads(p.read_text())["case"]
        lanes[(c["arm"], c["seed"], c["condition"])] = p.parent
    for fraction in [0.1, 1.0]:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), layout="constrained")
        for row, condition in enumerate(["random", "real7"]):
            for col, (arm, name) in enumerate(zip(ARMS, LABELS, strict=True)):
                paths = [
                    lanes[(arm, s, condition)] / f"target0_f{round(fraction * 100):03d}/full_pred.parquet"
                    for s in seeds
                ]
                _, truth, pred = aligned_predictions(paths)
                ax = axes[row, col]
                ax.scatter(truth, pred.mean(axis=0), s=10, alpha=0.5, color=COLORS[col])
                ax.set_xscale("symlog", linthresh=1)
                ax.set_yscale("symlog", linthresh=1)
                ax.set_title(f"{name}\n{'From scratch' if condition == 'random' else 'Seven-source transfer'}")
                ax.set_xlabel("Ground truth")
                ax.set_ylabel("Predicted dielectric constant")
        low = min(min(ax.get_xlim()[0], ax.get_ylim()[0]) for ax in axes.flat)
        high = max(max(ax.get_xlim()[1], ax.get_ylim()[1]) for ax in axes.flat)
        for ax in axes.flat:
            ax.plot([low, high], [low, high], "--", color="gray", lw=1)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle(
            f"Same held-out compositions | training size {fraction:.0%}\nMean prediction across {len(seeds)} seeds; symmetric-log axes, linear within ±1; all points retained",
            fontsize=18,
        )
        save(fig, output, f"dielectric_parity_f{round(fraction * 100):03d}.png")

    for j, target in enumerate(TARGETS):
        bundles = []
        for arm in ARMS:
            bundles.append(
                aligned_predictions([lanes[(arm, s, "real7")] / f"target{j}_f100/full_pred.parquet" for s in seeds])
            )
        index, truth, _ = bundles[0]
        for other_index, other_truth, _ in bundles[1:]:
            if not index.equals(other_index) or not np.allclose(truth, other_truth):
                raise ValueError("Cross-encoder composition mismatch")
        chosen = np.argsort(truth)[np.linspace(0, len(truth) - 1, 6).round().astype(int)]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
        for ax, i in zip(axes.flat, chosen, strict=True):
            for a, (_, _, pred) in enumerate(bundles):
                ax.errorbar(a, pred[:, i].mean(), yerr=pred[:, i].std(ddof=1), fmt="o", color=COLORS[a], capsize=5)
            ax.axhline(truth[i], color="black", linestyle="--", label="Ground truth")
            ax.set_title(str(index[i]))
            ax.set_xticks(range(4), LABELS, rotation=20)
            ax.set_ylabel("Prediction (original target scale)")
            ax.legend()
        fig.suptitle(
            f"{target}: identical compositions, seven-source full fine-tuning, training size 100%\nExamples span ground-truth quantiles; mean ± seed SD (not predictive uncertainty)",
            fontsize=17,
        )
        save(fig, output, f"composition_examples_target{j}.png")


def report(root: Path, config: Path, selection: Path, output: Path) -> dict:
    audit = collect(root, config, selection, output)
    if audit["partial"]:
        raise ValueError("Final report requires the complete registered campaign")
    seeds = tomllib.loads(config.read_text())["study"]["seeds"]
    frame = pd.read_csv(output / "metrics.csv")
    paired = effects(frame, seeds)
    paired.to_csv(output / "paired_relative_effects.csv", index=False)
    figures(frame, paired, root, output, seeds)
    lines = [
        "# Transformer cross-task transfer: complete controlled study",
        "",
        f"Audited {audit['complete_lanes']} lanes and {audit['metrics']} selected endpoints; {len(seeds)} optimization seeds per condition.",
        "",
        "Relative change is computed per paired seed as 100 × (RMSE real7 / RMSE baseline − 1), then averaged. Negative values indicate lower error. Confidence intervals bootstrap paired seeds; they are pointwise, exploratory and unadjusted for multiple endpoints. Seed SD describes training variability, not predictive uncertainty. Absolute RMSE is available in summary.csv; standardized differences and encoder-by-pretraining interactions are in the collector CSVs.",
        "",
        "Joint supervised source training uses the installed encoder components without reconstruction/replay. This isolates task-label information and is not the original continual pretraining workflow. Real1/3/7 use equal total optimizer updates, not equal FLOPs or per-task exposure. Task identity changes with task count. Bulk and shear are related endpoints. The dielectric test set was previously inspected; the study is exploratory.",
        "",
        "Source learning rates were selected with frozen-ridge target validation on two separate pilot seeds. Full fine-tuning searches only 0.1/0.3/1 times that source rate. A null result does not exclude different tokenization, objectives, model size, a wider fine-tuning search or larger budgets. Source accuracy alone does not select an architecture winner.",
        "",
        "## Full fine-tuning: seven-source transfer versus training from scratch",
        "",
        "| Encoder | Target | Training size | Relative RMSE change, mean [95% CI] | Improved seeds |",
        "|---|---|---:|---:|---:|",
    ]
    q = paired[(paired["mode"] == "full") & (paired.baseline == "random")]
    for row in q.itertuples():
        lines.append(
            f"| {row.arm} | {row.target} | {row.fraction:.0%} | {row.relative_mean:.2f}% [{row.relative_lo95:.2f}, {row.relative_hi95:.2f}] | {row.improved_seeds}/{row.n_seeds} |"
        )
    lines += ["", "## Figures", ""]
    for p in sorted(output.glob("*.png")):
        lines.append(f"- [{p.stem}]({p.name})")
    (output / "REPORT_EN_20261003.md").write_text("\n".join(lines) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["root", "config", "selection", "output"]:
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.config, args.selection, args.output), indent=2))
