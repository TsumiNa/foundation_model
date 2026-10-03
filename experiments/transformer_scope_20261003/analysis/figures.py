"""Rebuild complete-study interaction and fixed-composition parity figures.

Run with the same --root/--config/--selection/--output arguments as collect.py.
The collector must validate a complete campaign before any figures are emitted.
Examples use the first registered split/seed, 10% training size, full fine-tuning,
and the largest source budget; selection never depends on prediction error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect import TARGETS, collect, expected_cases


@dataclass(frozen=True)
class Panel:
    row: int
    column: int
    arm: str
    condition: str
    steps: int
    frame: pd.DataFrame
    rmse: float


def load_panels(root: Path, cases: list[dict], raw: dict, input_name: str, target: int) -> list[Panel]:
    """Require identical unique held-out compositions/truths across all six panels."""
    cfg = raw["study"]
    panels = []
    reference = None
    for column, architecture in enumerate(["mlp", "transformer", "no_attention"]):
        arm = f"{input_name}_{architecture}"
        for row, condition in enumerate(["random", "real12"]):
            steps = 0 if condition == "random" else max(cfg["source_budgets"])
            matched = [
                i
                for i, case in enumerate(cases)
                if case["arm"] == arm
                and case["condition"] == condition
                and case["steps"] == steps
                and case["split_seed"] == cfg["split_seeds"][0]
                and case["seed"] == cfg["seeds"][0]
            ]
            if len(matched) != 1:
                raise ValueError("Exactly one registered fixed-example case required")
            lane = root / f"case{matched[0]:03d}"
            frame = pd.read_parquet(lane / f"target{target}_f010/full_pred.parquet").sort_values("composition")
            if (
                frame.empty
                or frame.composition.duplicated().any()
                or not np.isfinite(frame[["true", "pred"]]).all().all()
            ):
                raise ValueError("Finite unique prediction compositions required")
            if reference is None:
                reference = frame[["composition", "true"]].reset_index(drop=True)
            elif frame.composition.tolist() != reference.composition.tolist() or not np.array_equal(
                frame.true, reference.true
            ):
                raise ValueError("Prediction panels must use identical compositions and true values")
            done = json.loads((lane / "done.json").read_text())
            metrics = [
                m
                for m in done["metrics"]
                if m["target"] == TARGETS[target] and m["fraction"] == 0.1 and m["mode"] == "full"
            ]
            if len(metrics) != 1 or not np.isclose(
                np.sqrt(np.mean((frame.pred - frame.true) ** 2)), metrics[0]["rmse"], rtol=1e-5
            ):
                raise ValueError("Prediction RMSE does not match the audited endpoint")
            panels.append(Panel(row, column, arm, condition, steps, frame, metrics[0]["rmse"]))
    return panels


def parity(panels: list[Panel], raw: dict, input_name: str, target: int, output: Path) -> None:
    cfg = raw["study"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    lo = min(0.0, min(min(p.frame.true.min(), p.frame.pred.min()) for p in panels))
    hi = max(max(p.frame.true.max(), p.frame.pred.max()) for p in panels) * 1.05
    lo = lo * 1.05 if lo < 0 else -0.03
    for panel in panels:
        ax = axes[panel.row, panel.column]
        ax.scatter(
            panel.frame.true,
            panel.frame.pred,
            s=12,
            alpha=0.45,
            color=["#3264a8", "#bb542e"][panel.row],
            rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], color="#333", ls="--", lw=1)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.text(
            0.04,
            0.95,
            f"RMSE = {panel.rmse:.3g}\nN = {len(panel.frame)}",
            transform=ax.transAxes,
            va="top",
            fontsize=12,
        )
        if panel.row == 0:
            ax.set_title(["MLP", "Transformer", "No attention"][panel.column], pad=12)
        if panel.column == 0:
            ax.set_ylabel(("From scratch" if panel.row == 0 else "12-task pretraining") + "\nPredicted value")
        if panel.row == 1:
            ax.set_xlabel("True value")
    label = "KMD" if input_name == "kmd" else "Composition"
    fig.suptitle(f"{label} input: {TARGETS[target]}", fontsize=23, y=0.98)
    fig.text(
        0.08,
        0.07,
        f"Identical held-out compositions; 10% training size; full fine-tuning; source budget: {max(cfg['source_budgets']):,} updates.\n"
        f"Fixed example: split {cfg['split_seeds'][0]}, seed {cfg['seeds'][0]}. Diagonal: perfect prediction. Original target scale;\n"
        "symmetric-log axes (linear threshold = 1); negative predictions retained. One run illustrates behavior, not statistical evidence.",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.90, bottom=0.19, hspace=0.22, wspace=0.24)
    fig.savefig(output / f"{input_name}_target{target}_same_composition_parity_20261003.png", dpi=150)
    plt.close(fig)


def interactions(output: Path, input_name: str, control: str) -> None:
    data = pd.read_csv(output / "family_interactions.csv")
    data = data[(data.input == input_name) & (data.baseline == control)]
    if len(data) != 12 or data.partial.any() or not np.isfinite(data[["mean", "lo95", "hi95"]]).all().all():
        raise ValueError("Complete registered interaction matrix required")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    colors = ["#1565a7", "#b25b22"]
    for r, mode in enumerate(["ridge", "full"]):
        for c, fraction in enumerate([0.01, 0.1, 1.0]):
            ax = axes[r, c]
            for x, steps in enumerate([6000, 24000]):
                row = data[(data["mode"] == mode) & (data.fraction == fraction) & (data.steps == steps)].iloc[0]
                v = list(json.loads(row.split_means).values())
                ax.scatter(np.linspace(x - 0.12, x + 0.12, len(v)), v, s=32, color=colors[x], alpha=0.5)
                ax.errorbar(
                    x,
                    row["mean"],
                    yerr=[[row["mean"] - row.lo95], [row.hi95 - row["mean"]]],
                    fmt="D",
                    color=colors[x],
                    markersize=7,
                    capsize=6,
                    linewidth=2,
                )
            ax.axhline(0, color="#555", linewidth=1, linestyle="--")
            ax.set_xticks([0, 1], ["6k updates", "24k updates"])
            ax.set_xlim(-0.5, 1.5)
            ax.grid(axis="y", alpha=0.2)
            ax.set_title(f"Training size: {fraction:.0%}", pad=12)
            if c == 0:
                ax.set_ylabel(("Frozen encoder + ridge" if mode == "ridge" else "Full fine-tuning") + "\nInteraction")
    label = "KMD" if input_name == "kmd" else "Composition"
    control_label = "MLP" if control == "mlp" else "capacity-matched no-attention"
    symbol = "MLP" if control == "mlp" else "noatt"
    fig.suptitle(f"{label} input: Transformer versus {control_label}", fontsize=23, y=0.98)
    formula = r"Interaction = $(E_{\mathrm{TF,real12}}-E_{\mathrm{TF,random}})-(E_{\mathrm{CONTROL,real12}}-E_{\mathrm{CONTROL,random}})$"
    fig.text(0.08, 0.145, formula.replace("CONTROL", symbol), fontsize=16)
    fig.text(
        0.08,
        0.105,
        "Negative: larger pretraining gain for Transformer. This does not imply lower absolute prediction error.",
        fontsize=14,
    )
    fig.text(
        0.08,
        0.035,
        "E: family-weighted standardized RMSE. Random: frozen random features (ridge), or training from scratch (full fine-tuning).\n"
        "Diamonds: means; bars: pointwise 95% hierarchical bootstrap intervals; dots: three split means.\n"
        "Three seeds per split; overlapping holdouts from one materials pool. All six arms complete; no multiplicity adjustment.",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.1, right=0.985, top=0.90, bottom=0.25, hspace=0.42, wspace=0.18)
    fig.savefig(output / f"{input_name}_{control}_interaction_final_20261003.png", dpi=160)
    plt.close(fig)


def render(root: Path, config: Path, selection: Path, output: Path) -> None:
    audit = collect(root, config, selection, output)
    if audit["partial"]:
        raise ValueError("Final figures require a complete audited campaign")
    raw = tomllib.loads(config.read_text())
    cases = expected_cases(raw, json.loads(selection.read_text())["learning_rates"])
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15, "mathtext.fontset": "stix"})
    frames = []
    for input_name in ["kmd", "composition"]:
        for target in range(len(TARGETS)):
            panels = load_panels(root, cases, raw, input_name, target)
            parity(panels, raw, input_name, target, output)
            frames.extend(
                p.frame.assign(
                    input=input_name,
                    arm=p.arm,
                    condition=p.condition,
                    steps=p.steps,
                    target=TARGETS[target],
                    training_fraction=0.1,
                    split_seed=raw["study"]["split_seeds"][0],
                    seed=raw["study"]["seeds"][0],
                )
                for p in panels
            )
        for control in ["mlp", "no_attention"]:
            interactions(output, input_name, control)
    pd.concat(frames, ignore_index=True).to_parquet(
        output / "same_composition_predictions_fixed_example_20261003.parquet", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["root", "config", "selection", "output"]:
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.config, args.selection, args.output)
