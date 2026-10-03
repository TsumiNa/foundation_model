"""Direct, composition-aligned dielectric prediction comparisons for the frozen-head study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition

COLORS = ["#35424F", "#2278B5", "#DF781A"]


def paired_summary(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    chosen = frame[(frame.k == 7) & (frame.fraction == fraction) & (frame.readout == "linear_output")].copy()
    numeric = ["true_mlp", "true_transformer", "pred_mlp", "pred_transformer"]
    if (
        chosen.empty
        or chosen[["composition", "seed"]].duplicated().any()
        or not np.isfinite(chosen[numeric]).all().all()
    ):
        raise ValueError("Require unique finite composition/seed predictions")
    if not np.allclose(chosen.true_mlp, chosen.true_transformer, rtol=1e-6, atol=1e-7):
        raise ValueError("Model references disagree")
    if any(set(group.seed) != {0, 1, 2} for _, group in chosen.groupby("composition")):
        raise ValueError("Every composition needs exactly three paired seeds")
    for _, group in chosen.groupby("composition"):
        if not np.allclose(group.true_mlp, group.true_mlp.iloc[0], rtol=1e-6, atol=1e-7):
            raise ValueError("References disagree across seeds")
    summary = chosen.groupby("composition").agg(
        reference=("true_mlp", "mean"),
        mlp=("pred_mlp", "mean"),
        transformer=("pred_transformer", "mean"),
        mlp_sd=("pred_mlp", "std"),
        transformer_sd=("pred_transformer", "std"),
    )
    for seed in range(3):
        row = chosen[chosen.seed == seed].set_index("composition")
        summary[f"mlp_s{seed}"] = row.pred_mlp
        summary[f"transformer_s{seed}"] = row.pred_transformer
    summary["prediction_difference"] = summary.transformer - summary.mlp
    return summary.reset_index().sort_values(["reference", "composition"]).reset_index(drop=True)


def representatives(summary: pd.DataFrame, count: int = 12) -> pd.DataFrame:
    """Select by reference rank only, independent of model predictions and errors."""
    if type(count) is not int or count < 1 or summary.empty:
        raise ValueError("Require nonempty data and a positive integer count")
    ordered = summary.sort_values(["reference", "composition"]).reset_index(drop=True)
    positions = np.rint(np.linspace(0, len(ordered) - 1, min(count, len(ordered)))).astype(int)
    result = ordered.iloc[positions].copy()
    result["reference_rank"] = positions + 1
    result["reference_percentile"] = 100 * positions / max(1, len(ordered) - 1)
    return result


def composition_cards(selected: pd.DataFrame, fraction: float) -> plt.Figure:
    fig, axes = plt.subplots(3, 4, figsize=(17, 10.8))
    for ax, (_, row) in zip(axes.flat, selected.iterrows(), strict=True):
        values = [row.reference, row.mlp, row.transformer]
        ax.bar([0, 1, 2], values, color=COLORS, width=0.54, alpha=0.85, zorder=2)
        points = [row[f"{arm}_s{seed}"] for arm in ["mlp", "transformer"] for seed in range(3)]
        bottom = min(0, *points)
        top = max(*values, *points)
        span = max(top - bottom, 1)
        for x, value in enumerate(values):
            label_height = (
                value if x == 0 else max(value, *[row[f"{['mlp', 'transformer'][x - 1]}_s{s}"] for s in range(3)])
            )
            ax.annotate(
                f"{value:.1f}",
                (x, label_height),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=15,
                fontweight="bold",
            )
        for x, arm in [(1, "mlp"), (2, "transformer")]:
            ax.scatter(
                x + np.array([-0.16, 0, 0.16]),
                [row[f"{arm}_s{s}"] for s in range(3)],
                color="white",
                edgecolor=COLORS[x],
                s=35,
                linewidth=1.2,
                zorder=3,
            )
        formula = Composition(row.composition).formula.replace(" ", "")
        formula = re.sub(r"(\d+(?:\.\d+)?)", r"$_{\1}$", formula)
        ax.set_title(formula, fontsize=18, pad=13)
        ax.set_xticks([0, 1, 2], ["Reference", "MLP", "Transformer"], fontsize=12)
        ax.set_ylim(bottom - 0.12 * span, top + 0.25 * span)
        ax.grid(axis="y", alpha=0.15, zorder=0)
        ax.tick_params(axis="y", labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Same composition: reference vs. two encoder predictions\nTarget training size {fraction:.0%}  •  7 source tasks  •  Frozen encoder + 128 → 64 head",
        fontsize=22,
        y=0.98,
    )
    fig.text(0.02, 0.5, "Dielectric constant (original target scale)", rotation=90, va="center", fontsize=16)
    fig.text(
        0.5,
        0.025,
        "Bars and numbers: mean of 3 seeds; small circles: individual seeds (not prediction intervals).\n12 compositions selected at evenly spaced reference ranks; each panel has its own vertical scale.",
        ha="center",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.13, wspace=0.32, hspace=0.65)
    return fig


def parity_panels(summary: pd.DataFrame, fraction: float, zoom: bool) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    if zoom:
        low, high = -5.0, 70.0
    else:
        values = summary[["reference", "mlp", "transformer"]].to_numpy()
        low = min(-5.0, float(values.min()) - 5)
        high = float(values.max()) * 1.05
    pairs = [
        ("reference", "mlp", "MLP encoder", "#2278B5"),
        ("reference", "transformer", "Transformer encoder", "#DF781A"),
        ("mlp", "transformer", "Direct prediction comparison", "#6B559B"),
    ]
    for ax, (x, y, title, color) in zip(axes, pairs, strict=True):
        ax.scatter(summary[x], summary[y], s=17, alpha=0.6, color=color, rasterized=True)
        ax.plot([low, high], [low, high], "--", color="#666666", lw=1.2)
        ax.set(xlim=(low, high), ylim=(low, high))
        ax.set_aspect("equal")
        ax.set_xlabel("Reference" if x == "reference" else "MLP prediction", fontsize=16)
        ax.set_ylabel("MLP prediction" if y == "mlp" else "Transformer prediction", fontsize=16)
        visible = summary[x].between(low, high) & summary[y].between(low, high)
        ax.set_title(f"{title}\n{visible.sum()} / {len(summary)} compositions in view", fontsize=17, pad=15)
        ax.tick_params(labelsize=13)
        ax.grid(alpha=0.15)
    view = "Zoomed view" if zoom else "Full range"
    fig.suptitle(
        f"{view}: the same test compositions on identical axes\nTraining size {fraction:.0%}  •  7 source tasks  •  Mean prediction across 3 seeds",
        fontsize=22,
        y=0.97,
    )
    fig.text(
        0.5,
        0.035,
        "Dashed line: equality. Left / middle: closer to the line means more accurate.\nRight: distance from the line shows disagreement between models; it does not show which model is better.",
        ha="center",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.73, bottom=0.2, wspace=0.35)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    summaries = {fraction: paired_summary(frame, fraction) for fraction in (1.0, 0.1)}
    if any(len(summary) != 697 for summary in summaries.values()):
        raise ValueError("The completed study requires all 697 test compositions at each training size")
    if set(summaries[1.0].composition) != set(summaries[0.1].composition):
        raise ValueError("Training sizes have different test memberships")
    a = summaries[1.0].set_index("composition").reference
    b = summaries[0.1].set_index("composition").reference.reindex(a.index)
    if not np.allclose(a, b, rtol=1e-6, atol=1e-7):
        raise ValueError("Training sizes have different test references")
    chosen = representatives(summaries[1.0])
    if len(chosen) != 12:
        raise ValueError("The fixed presentation layout requires at least 12 compositions")
    args.output.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output / "Direct_prediction_comparison_20261003.pdf"
    with PdfPages(pdf_path) as pdf:
        for fraction, summary in summaries.items():
            selected = summary.set_index("composition").loc[chosen.composition].reset_index()
            selected["reference_rank"] = chosen.reference_rank.to_numpy()
            selected["reference_percentile"] = chosen.reference_percentile.to_numpy()
            summary.to_csv(args.output / f"all_compositions_f{round(fraction * 100):03d}.csv", index=False)
            selected.to_csv(args.output / f"selected_compositions_f{round(fraction * 100):03d}.csv", index=False)
            figures = [
                ("composition_predictions", composition_cards(selected, fraction)),
                ("parity_zoom", parity_panels(summary, fraction, True)),
                ("parity_full", parity_panels(summary, fraction, False)),
            ]
            for name, fig in figures:
                fig.savefig(args.output / f"{name}_f{round(fraction * 100):03d}.png", dpi=180)
                pdf.savefig(fig)
                plt.close(fig)
    (args.output / "provenance.json").write_text(
        json.dumps(
            dict(
                input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                selection="12 evenly spaced reference ranks, including endpoints; same compositions at both training sizes",
                source_count=7,
                readout="linear_output",
                head_hidden_dims=[128, 64],
                encoder_mode="frozen",
                seeds=[0, 1, 2],
                test_compositions=len(summaries[1.0]),
                fractions=[1.0, 0.1],
            ),
            indent=2,
        )
    )
    (args.output / "README.md").write_text("""# Direct prediction comparison

All figures compare the same test compositions for tuned MLP versus grouped-mean Transformer, with frozen encoders after seven source tasks and a 128→64 head with identity output. These are the completed head-probe results, not new training.

- PDF pages 1–3 use 100% target training size (3,051 training compositions); pages 4–6 use 10% (306). Both use the same 697 test compositions.
- Composition panels show reference values and the mean of three predictions; small circles show individual seed predictions, not prediction intervals. Each panel has a separate original-value axis. Examples are twelve evenly spaced ranks of reference value, including the smallest/largest. No selection uses prediction error or model disagreement.
- The parity pages use identical axes across models. Zoom pages state the count in view; full-range companion pages include all values. On reference-versus-prediction panels the equality line indicates perfect predictions. On the model-versus-model panel it indicates identical predictions, without identifying which is more accurate.
- The figure means are for visual comparison. Reported mean-per-seed RMSE from the study is not the RMSE of these seed-mean predictions. Complete per-composition means/SDs and individual seed values are in the exported CSVs; provenance.json records the input/script hashes and selection rule.
""")
    print(pdf_path)


if __name__ == "__main__":
    main()
