"""Audit and summarize only the registered full-budget frozen readout study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("CKA requires aligned matrices")
    x, y = x.astype(float) - x.mean(0), y.astype(float) - y.mean(0)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    if denominator == 0:
        raise ValueError("CKA is undefined for constant representations")
    return float(np.linalg.norm(x.T @ y) ** 2 / denominator)


def collect(root: Path, config: Path, protocol: Path) -> tuple[pd.DataFrame, dict]:
    cfg = tomllib.loads(config.read_text())
    base = tomllib.loads(protocol.read_text())["benchmark"]

    def digest(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    rows, revisions, lanes = [], set(), []
    expected = set()
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            for k in cfg["source_counts"]:
                for fraction in cfg["fractions"]:
                    heads = ["linear_output", "wide", "ridge_post", "ridge_pre"] + (["legacy"] if k == 7 else [])
                    if arm == cfg["arms"][0] and k == cfg["source_counts"][0]:
                        heads += ["ridge_descriptor"]
                    expected.update((arm, seed, k, fraction, h) for h in heads)
    for path in sorted(root.glob("*_s*_k*/done.json")):
        done = json.loads(path.read_text())
        identity = json.loads((path.parent / "identity.json").read_text())
        if (
            identity != done["identity"]
            or identity["smoke"]
            or not done["encoder_unchanged"]
            or identity["config_sha256"] != digest(config)
            or identity["protocol_sha256"] != digest(protocol)
            or identity["manifest_sha256"] != base["data_manifest_sha256"]
            or identity["sif_sha256"] != base["sif_sha256"]
        ):
            raise ValueError("Unregistered, smoke or mutated-encoder lane")
        revisions.add(identity["revision"])
        lanes.append(path.parent.name)
        for row in done["selected"]:
            key = (row["arm"], row["seed"], row["k"], row["fraction"], row["readout"])
            if key not in expected or (row["arm"], row["seed"], row["k"]) != (
                identity["arm"],
                identity["seed"],
                identity["source_count"],
            ):
                raise ValueError("Unregistered or mismatched result")
            folder = path.parent / f"f{round(row['fraction'] * 100):03d}"
            pred = pd.read_parquet(folder / f"{row['readout']}_pred.parquet")
            if (
                len(pred) != 697
                or pred.composition.duplicated().any()
                or not np.isfinite(pred[["pred", "true"]]).all().all()
            ):
                raise ValueError("Missing, duplicate or nonfinite test predictions")
            rmse = float(np.sqrt(np.mean((pred.pred - pred.true) ** 2)))
            if not np.isclose(rmse, row["rmse"], rtol=1e-7):
                raise ValueError("Stored metric differs from predictions")
            rows.append({k: v for k, v in row.items() if k != "trials"})
    if len(revisions) > 1:
        raise ValueError("Mixed experiment revisions")
    frame = pd.DataFrame(rows)
    if len(frame) and frame.duplicated(["arm", "seed", "k", "fraction", "readout"]).any():
        raise ValueError("Duplicate results")
    return frame, {
        "completed_lanes": lanes,
        "selected_fits": len(frame),
        "expected_fits": len(expected),
        "complete": len(frame) == len(expected),
        "revision": list(revisions),
    }


def prediction_disagreement(root: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair the same compositions and seeds; disagreement is not an improvement score."""
    rows, compositions = [], []
    for (seed, k, fraction, readout), group in frame.groupby(["seed", "k", "fraction", "readout"]):
        if set(group.arm) != {"mlp_tuned", "grouped_mean"}:
            continue
        frames = [
            pd.read_parquet(root / f"{arm}_s{seed}_k{k}" / f"f{round(fraction * 100):03d}" / f"{readout}_pred.parquet")
            for arm in ("mlp_tuned", "grouped_mean")
        ]
        pair = frames[0].merge(
            frames[1],
            on="composition",
            suffixes=("_mlp", "_transformer"),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not pair["_merge"].eq("both").all() or not np.allclose(pair.true_mlp, pair.true_transformer):
            raise ValueError("Prediction pairs have different compositions or references")
        error = pair.pred_transformer - pair.pred_mlp
        denominator = float(np.sqrt(np.mean((pair.pred_mlp - pair.true_mlp) ** 2)))
        magnitude = float(np.sqrt(np.mean(error**2)))
        keys = {"seed": seed, "k": k, "fraction": fraction, "readout": readout}
        rows.append(
            {
                **keys,
                "prediction_difference_rmse": magnitude,
                "mlp_rmse": denominator,
                "difference_over_mlp_rmse": magnitude / denominator if denominator else None,
                "prediction_correlation": float(pair.pred_mlp.corr(pair.pred_transformer)),
            }
        )
        pair = pair.drop(columns="_merge").assign(**keys, prediction_difference=error)
        compositions.append(pair)
    return pd.DataFrame(rows), pd.concat(compositions, ignore_index=True) if compositions else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "config", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame, audit = collect(args.root, args.config, args.protocol)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    if frame.empty:
        return
    frame.to_csv(args.output / "selected_metrics.csv", index=False)
    summary = (
        frame.groupby(["arm", "readout", "fraction", "k"])
        .agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"), n=("seed", "nunique"))
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    disagreements, compositions = prediction_disagreement(args.root, frame)
    disagreements.to_csv(args.output / "prediction_disagreement.csv", index=False)
    compositions.to_csv(args.output / "paired_composition_predictions.csv", index=False)
    if len(disagreements):
        disagreements.groupby(["k", "fraction", "readout"])[
            ["prediction_difference_rmse", "difference_over_mlp_rmse"]
        ].agg(["mean", "std", "count"]).to_csv(args.output / "prediction_disagreement_summary.csv")
    paired = frame.pivot(index=["seed", "k", "fraction", "readout"], columns="arm", values="rmse").dropna()
    if set(["mlp_tuned", "grouped_mean"]) <= set(paired):
        paired["transformer_minus_mlp"] = paired.grouped_mean - paired.mlp_tuned
        paired.to_csv(args.output / "paired_encoder_gaps.csv")
    fixed = []
    representations = []
    for lane in audit["completed_lanes"]:
        p = args.root / lane
        ident = json.loads((p / "identity.json").read_text())
        for folder in p.glob("f*"):
            fraction = int(folder.name[1:]) / 100
            if ident["source_count"] == 7:
                for head in ("legacy", "linear_output"):
                    fixed.append(
                        {
                            "arm": ident["arm"],
                            "seed": ident["seed"],
                            "fraction": fraction,
                            "readout": head,
                            **json.loads((folder / f"{head}_fixed_lr.json").read_text()),
                        }
                    )
            representations.append(
                {
                    "arm": ident["arm"],
                    "seed": ident["seed"],
                    "k": ident["source_count"],
                    "fraction": fraction,
                    **json.loads((folder / "representation.json").read_text()),
                }
            )
    pd.DataFrame(fixed).to_csv(args.output / "fixed_lr_activation.csv", index=False)
    pd.DataFrame(representations).to_csv(args.output / "representation_statistics.csv", index=False)
    geometry = []
    for seed in sorted(frame.seed.unique()):
        for k in sorted(frame.k.unique()):
            paths = [args.root / f"{a}_s{seed}_k{k}" / "f100/features.npz" for a in ("mlp_tuned", "grouped_mean")]
            if not all(p.parent.parent.name in audit["completed_lanes"] for p in paths):
                continue
            left, right = (np.load(p) for p in paths)
            if not np.array_equal(left["composition"], right["composition"]) or not np.array_equal(
                left["split"], right["split"]
            ):
                raise ValueError("Unaligned CKA samples")
            mask = left["split"] == "val"
            geometry.append(
                {
                    "seed": int(seed),
                    "k": int(k),
                    "pre_cka": linear_cka(left["pre"][mask], right["pre"][mask]),
                    "post_cka": linear_cka(left["post"][mask], right["post"][mask]),
                }
            )
    pd.DataFrame(geometry).to_csv(args.output / "validation_cka.csv", index=False)
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15})
    for fraction in sorted(frame.fraction.unique()):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
        for ax, head in zip(axes, ("ridge_post", "linear_output", "wide"), strict=True):
            for arm in ("mlp_tuned", "grouped_mean"):
                g = summary[(summary.fraction == fraction) & (summary.readout == head) & (summary.arm == arm)]
                ax.errorbar(g.k, g.rmse_mean, yerr=g.rmse_sd, marker="o", capsize=4, label=arm)
            ax.set(title=head, xlabel="Source task count", ylabel="Test RMSE (physical units)", xticks=[1, 3, 7])
            ax.legend()
        fig.suptitle(
            f"Target training size: {fraction:.0%}; mean ± SD over available seeds (partial={not audit['complete']})"
        )
        fig.savefig(args.output / f"rmse_f{round(fraction * 100):03d}.png", dpi=180)
        plt.close(fig)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
