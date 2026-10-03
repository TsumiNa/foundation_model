"""Audit the registered scope matrix and retain split-aware paired interactions."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import tomllib

import numpy as np
import pandas as pd

TARGETS = ["Dielectric total", "Bulk modulus", "Shear modulus", "Piezoelectric max"]
WEIGHTS = dict(zip(TARGETS, [1 / 3, 1 / 6, 1 / 6, 1 / 3], strict=True))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_cases(raw: dict, selection: dict) -> list[dict]:
    s = raw["study"]
    settings = [("random", 0)] + [(c, b) for c in s["source_sets"] for b in s["source_budgets"]]
    return [
        dict(arm=a, seed=seed, split_seed=sp, condition=c, steps=b, lr=selection[a])
        for a in raw["arms"]
        for sp in s["split_seeds"]
        for seed in s["seeds"]
        for c, b in settings
    ]


def paired_summary(values: pd.Series, splits: list[int], seeds: list[int]) -> dict:
    """Resample splits, then paired optimization seeds; do not flatten nine runs into IID data."""
    if not len(values) or not np.isfinite(values).all() or values.index.duplicated().any():
        raise ValueError("Finite unique paired split/seed values required")
    groups = [v.to_numpy() for _, v in values.groupby(level="split_seed")]
    means = values.groupby(level="split_seed").mean()
    complete = set(values.index) == {(sp, seed) for sp in splits for seed in seeds}
    lo = hi = None
    if complete:
        rng = np.random.default_rng(20261003)
        matrix = np.stack(groups)
        split_draw = rng.integers(len(groups), size=(10000, len(groups)))
        seed_draw = rng.integers(matrix.shape[1], size=(10000, len(groups), matrix.shape[1]))
        draws = matrix[split_draw[:, :, None], seed_draw].mean(axis=(1, 2))
        lo, hi = np.quantile(draws, [0.025, 0.975]).tolist()
    return dict(
        mean=float(means.mean()),
        lo95=lo,
        hi95=hi,
        n_splits=len(groups),
        n_pairs=len(values),
        partial=not complete,
        split_means=json.dumps({str(k): float(v) for k, v in means.items()}),
        all_split_means_negative=bool(complete and (means < 0).all()),
    )


def contrasts(frame: pd.DataFrame, raw: dict, output: Path) -> None:
    cfg = raw["study"]
    comparisons = [("real12", base) for base in ["random", *cfg["source_sets"]] if base != "real12"]
    for count in [1, 3]:
        comparisons.extend(combinations([c for c, tasks in cfg["source_sets"].items() if len(tasks) == count], 2))
    effects = []
    interactions = []
    # Random/scratch is shared across budgets. A given scratch result is never counted twice
    # within a contrast; correlations across budget contrasts remain explicit.
    for budget in cfg["source_budgets"]:
        f = frame[(frame.steps == budget) | (frame.condition == "random")]
        for key, g in f.groupby(["arm", "target", "fraction", "mode"]):
            w = g.pivot(index=["split_seed", "seed"], columns="condition", values="standardized_rmse")
            for condition, base in comparisons:
                if not {condition, base} <= set(w.columns):
                    continue
                q = w[[condition, base]].dropna()
                if len(q):
                    effects.append(
                        dict(zip(["arm", "target", "fraction", "mode"], key, strict=True))
                        | dict(
                            steps=budget,
                            baseline=base,
                            condition=condition,
                            **paired_summary(q[condition] - q[base], cfg["split_seeds"], cfg["seeds"]),
                        )
                    )
        for key, g in f.groupby(["input", "target", "fraction", "mode"]):
            w = g.pivot(index=["split_seed", "seed"], columns=["arm", "condition"], values="standardized_rmse")
            for base in ["mlp", "no_attention"]:
                columns = [
                    (f"{key[0]}_transformer", "real12"),
                    (f"{key[0]}_transformer", "random"),
                    (f"{key[0]}_{base}", "real12"),
                    (f"{key[0]}_{base}", "random"),
                ]
                if not all(c in w for c in columns):
                    continue
                q = w[columns].dropna()
                if not len(q):
                    continue
                delta = q[columns[0]] - q[columns[1]] - q[columns[2]] + q[columns[3]]
                for (sp, seed), v in delta.items():
                    interactions.append(
                        dict(zip(["input", "target", "fraction", "mode"], key, strict=True))
                        | dict(steps=budget, baseline=base, split_seed=sp, seed=seed, value=v)
                    )
    pd.DataFrame(effects).to_csv(output / "paired_contrasts.csv", index=False)
    data = pd.DataFrame(interactions)
    data.to_csv(output / "interaction_pairs.csv", index=False)
    if data.empty:
        return
    summaries = []
    for key, g in data.groupby(["input", "target", "fraction", "mode", "steps", "baseline"]):
        summaries.append(
            dict(zip(["input", "target", "fraction", "mode", "steps", "baseline"], key, strict=True))
            | paired_summary(g.set_index(["split_seed", "seed"]).value, cfg["split_seeds"], cfg["seeds"])
        )
    pd.DataFrame(summaries).to_csv(output / "interactions.csv", index=False)
    families = []
    for key, g in data.groupby(["input", "fraction", "mode", "steps", "baseline", "split_seed", "seed"]):
        if set(g.target) != set(TARGETS) or len(g) != 4:
            continue
        families.append(
            dict(zip(["input", "fraction", "mode", "steps", "baseline", "split_seed", "seed"], key, strict=True))
            | dict(value=sum(WEIGHTS[row.target] * row.value for row in g.itertuples()))
        )
    family = pd.DataFrame(families)
    family.to_csv(output / "family_interaction_pairs.csv", index=False)
    if family.empty:
        return
    summaries = []
    for key, g in family.groupby(["input", "fraction", "mode", "steps", "baseline"]):
        summaries.append(
            dict(zip(["input", "fraction", "mode", "steps", "baseline"], key, strict=True))
            | paired_summary(g.set_index(["split_seed", "seed"]).value, cfg["split_seeds"], cfg["seeds"])
        )
    pd.DataFrame(summaries).to_csv(output / "family_interactions.csv", index=False)


def collect(root: Path, config: Path, selection: Path, output: Path) -> dict:
    raw = tomllib.loads(config.read_text())
    chosen = json.loads(selection.read_text())
    cfg = raw["study"]
    if chosen["config_sha256"] != digest(config) or chosen["manifest_sha256"] != cfg["data_manifest_sha256"]:
        raise ValueError("Selection identity mismatch")
    if set(chosen["learning_rates"]) != set(raw["arms"]) or any(
        chosen["learning_rates"][a] not in v["learning_rates"] for a, v in raw["arms"].items()
    ):
        raise ValueError("Unregistered learning rates")
    expected = expected_cases(raw, chosen["learning_rates"])
    script_dir = Path(__file__).parents[1] / "scripts"
    rows = []
    missing = []
    common = None
    n_metrics = 0
    for i, case in enumerate(expected):
        modes = ["ridge", "full"] if case["condition"] in cfg["full_conditions"] else ["ridge"]
        keys = {(t, f, m) for t in TARGETS for f in cfg["fractions"] for m in modes}
        n_metrics += len(keys)
        lane = root / f"case{i:03d}"
        if not (lane / "done.json").exists():
            missing.append(i)
            continue
        identity = json.loads((lane / "identity.json").read_text())
        done = json.loads((lane / "done.json").read_text())
        runtime = json.loads((lane / "run_provenance.json").read_text())["runtime"]
        expected_id = dict(
            case=case,
            phase="formal",
            config=digest(config),
            manifest=cfg["data_manifest_sha256"],
            script=digest(script_dir / "study.py"),
            preparation_script=digest(script_dir / "prepare.py"),
            selection=digest(selection),
            smoke=False,
            cpu_test=False,
            package=cfg["package_version"],
        )
        if (
            any(identity[k] != v for k, v in expected_id.items())
            or done["case"] != case
            or not re.fullmatch(r"[0-9a-f]{40}", identity["revision"] or "")
        ):
            raise ValueError(f"Invalid scientific identity: case {i}")
        if (
            runtime["device"] != "cuda"
            or runtime["sif_sha256"] != cfg["sif_sha256"]
            or runtime["image_revision"] != cfg["image_revision"]
            or "/site-packages/" not in runtime["package_path"]
        ):
            raise ValueError("Invalid image runtime")
        common = common or identity["revision"]
        if common != identity["revision"] or done["source"]["steps"] != case["steps"]:
            raise ValueError("Mixed revision or wrong source budget")
        metrics = done["metrics"]
        if len(metrics) != len(keys) or {(m["target"], m["fraction"], m["mode"]) for m in metrics} != keys:
            raise ValueError("Incomplete endpoint matrix")
        for m in metrics:
            if not np.isfinite([m[k] for k in ["rmse", "mae", "standardized_rmse", "val_mse"]]).all():
                raise ValueError("Nonfinite metrics")
            rows.append(case | dict(input=raw["arms"][case["arm"]]["input"]) | m)
    output.mkdir(parents=True, exist_ok=True)
    audit = dict(
        complete_lanes=len(expected) - len(missing),
        expected_lanes=len(expected),
        metrics=len(rows),
        expected_metrics=n_metrics,
        missing_cases=missing,
        partial=bool(missing),
        revision=common,
    )
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    if rows:
        f = pd.DataFrame(rows)
        f.to_csv(output / "metrics.csv", index=False)
        f.groupby(["arm", "condition", "steps", "target", "fraction", "mode", "split_seed"]).rmse.agg(
            ["mean", "std", "count"]
        ).to_csv(output / "summary_by_split.csv")
        contrasts(f, raw, output)
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["root", "config", "selection", "output"]:
        p.add_argument(f"--{key}", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(collect(a.root, a.config, a.selection, a.output), indent=2))
