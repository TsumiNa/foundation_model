"""Prepare composition-isolated, train-normalized data for the encoder transfer benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, element_features, formula_to_composition

SOURCE_TASKS = ("material_type", "volume", "formation_energy", "efermi", "band_gap", "seebeck", "zt")
TARGET_TASKS = ("dielectric_total", "power_factor")
# Dataset-native raw targets. Previously normalized columns are deliberately not reused.
TASK_COLUMNS = {
    "material_type": ("Material type (label)", None),
    "volume": ("Volume", None),
    "formation_energy": ("Formation energy per atom", None),
    "efermi": ("Efermi", None),
    "band_gap": ("Band gap", None),
    "seebeck": ("Seebeck coefficient", "Seebeck coefficient (T/K)"),
    "zt": ("ZT", "ZT (T/K)"),
    "dielectric_total": ("Dielectric total", None),
    "power_factor": ("Power factor", "Power factor (T/K)"),
}
DATE = "20261002"


@dataclass(kw_only=True)
class PreparationConfig:
    input: Path
    output: Path
    seeds: tuple[int, ...] = (0, 1, 2)
    fractions: tuple[float, ...] = (0.1, 1.0)

    def __post_init__(self) -> None:
        self.input, self.output = Path(self.input), Path(self.output)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if not self.fractions or any(
            isinstance(f, bool) or not isinstance(f, (int, float)) or not np.isfinite(f) or not 0 < f <= 1
            for f in self.fractions
        ):
            raise ValueError("fractions must be finite and in (0, 1]")
        if len({round(f * 100) for f in self.fractions}) != len(self.fractions):
            raise ValueError("fractions must have distinct integer-percent identifiers")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def isolate_compositions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Deduplicate atomic-fraction aliases; any test alias protects the entire composition.

    Keep the first record, as the repository catalog does. Do not merge measurements from
    different phases/records. Split precedence is test > val > train, independently of row order.
    """
    if not frame["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Every record must have an explicit train/val/test split")
    identities: list[str | None] = []
    for formula in frame["composition"]:
        try:
            weights = formula_to_composition(formula)
            if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
                raise ValueError("Composition must be covered by the 94-element vocabulary")
            identities.append("|".join(f"{i}:{w:.12f}" for i, w in enumerate(weights) if w > 0))
        except (ValueError, TypeError, KeyError):
            identities.append(None)
    clean = frame.assign(__identity=identities).dropna(subset=["__identity"]).copy()
    ranks = clean["split"].map({"train": 0, "val": 1, "test": 2})
    precedence = ranks.groupby(clean["__identity"]).transform("max")
    changed = int((ranks != precedence).sum())
    clean["split"] = precedence.map({0: "train", 1: "val", 2: "test"})
    before = len(clean)
    clean = clean.drop_duplicates("__identity", keep="first").reset_index(drop=True)
    audit = {
        "input_records": len(frame),
        "invalid_compositions": len(frame) - before,
        "alias_records_removed": before - len(clean),
        "records_moved_to_stricter_split": changed,
        "unique_compositions": len(clean),
    }
    return clean, audit


def valid_targets(frame: pd.DataFrame, name: str) -> pd.Series:
    column, t_column = TASK_COLUMNS[name]
    if t_column is None:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        return pd.Series(np.isfinite(values), index=frame.index)
    valid = []
    for y, t in zip(frame[column], frame[t_column]):
        if y is None or t is None:
            valid.append(False)
            continue
        y_arr, t_arr = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
        valid.append(
            y_arr.ndim == t_arr.ndim == 1
            and y_arr.size >= 2
            and y_arr.shape == t_arr.shape
            and np.isfinite(y_arr).all()
            and np.isfinite(t_arr).all()
            and (np.diff(t_arr) > 0).all()
        )
    return pd.Series(valid, index=frame.index)


def normalize_target(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit an affine transform on selected TRAIN records only, then transform every split."""
    column, t_column = TASK_COLUMNS[name]
    valid = valid_targets(frame, name)
    train = valid & frame["split"].eq("train")
    if not train.any():
        raise ValueError(f"{name}: no valid training targets")
    values = (
        np.concatenate(frame.loc[train, column].to_list())
        if t_column
        else frame.loc[train, column].to_numpy(dtype=float)
    )
    scaler = StandardScaler().fit(values.reshape(-1, 1))
    out = frame.copy()
    if t_column:
        out[name] = [
            scaler.transform(np.asarray(y, dtype=float).reshape(-1, 1)).ravel() if ok else None
            for y, ok in zip(frame[column], valid)
        ]
        out[f"{name}_t"] = [np.asarray(t, dtype=float) if ok else None for t, ok in zip(frame[t_column], valid)]
    else:
        out[name] = np.nan
        out.loc[valid, name] = scaler.transform(frame.loc[valid, column].to_numpy(dtype=float).reshape(-1, 1)).ravel()
    return out, scaler


def target_subset(frame: pd.DataFrame, name: str, fraction: float, seed: int) -> pd.DataFrame:
    """Drop unselected TRAIN records; retain identical labeled validation/test sets.

    One permutation per target/seed makes low-label budgets nested in the full budget.
    No unlabelled extra compositions enter a low-budget target's reconstruction loss.
    """
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    labeled = frame.loc[valid_targets(frame, name)].copy()
    train_idx = labeled.index[labeled["split"].eq("train")].to_numpy()
    if len(train_idx) < 2:
        raise ValueError(f"{name}: at least two training compositions are needed")
    salt = 0 if name == "dielectric_total" else 1
    order = np.random.default_rng(np.random.SeedSequence([seed, salt])).permutation(train_idx)
    n_keep = min(len(order), max(2, int(np.ceil(fraction * len(order)))))
    return labeled.loc[labeled["split"].ne("train") | labeled.index.isin(order[:n_keep])].copy()


def prepare(config: PreparationConfig) -> dict[str, Any]:
    columns = ["composition", "split", *dict.fromkeys(c for pair in TASK_COLUMNS.values() for c in pair if c)]
    raw = pd.read_parquet(config.input, columns=columns)
    frame, audit = isolate_compositions(raw)
    config.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "date": DATE,
        "input_sha256": sha256_file(config.input),
        "audit": audit,
        "normalization": "StandardScaler fitted on selected training targets only; no smoothing",
        "split_policy": "atomic-fraction identity; test > val > train; first record retained",
        "elements": DEFAULT_ELEMENTS,
        "element_features": list(element_features.columns),
        "n_grids": 8,
        "source_tasks": list(SOURCE_TASKS),
        "target_tasks": list(TARGET_TASKS),
        "targets": [],
    }
    # Source data contains neither global test compositions nor held-out target columns.
    source = frame.loc[frame["split"].ne("test")].copy()
    scalers: dict[str, StandardScaler] = {}
    source["material_type"] = source[TASK_COLUMNS["material_type"][0]].astype(int)
    if not source["material_type"].between(0, 4).all():
        raise ValueError("material_type must use the established five-class label vocabulary")
    for name in SOURCE_TASKS[1:]:
        source, scalers[name] = normalize_target(source, name)
    source_columns = ["composition", "split", *SOURCE_TASKS, "seebeck_t", "zt_t"]
    source_path = config.output / f"source_{DATE}.parquet"
    source[source_columns].to_parquet(source_path, index=False)
    joblib.dump(scalers, config.output / f"source_scalers_{DATE}.joblib")
    manifest["source"] = {
        "file": source_path.name,
        "sha256": sha256_file(source_path),
        "scaler_sha256": sha256_file(config.output / f"source_scalers_{DATE}.joblib"),
        "counts": source["split"].value_counts().to_dict(),
    }
    for name in TARGET_TASKS:
        for seed in config.seeds:
            for fraction in config.fractions:
                subset = target_subset(frame, name, fraction, seed)
                subset, scaler = normalize_target(subset, name)
                stem = f"{name}_f{round(fraction * 100):03d}_s{seed}_{DATE}"
                path = config.output / f"{stem}.parquet"
                output_columns = ["composition", "split", name] + ([f"{name}_t"] if TASK_COLUMNS[name][1] else [])
                subset[output_columns].to_parquet(path, index=False)
                joblib.dump(scaler, config.output / f"{stem}.joblib")
                manifest["targets"].append(
                    {
                        "task": name,
                        "seed": seed,
                        "fraction": fraction,
                        "file": path.name,
                        "scaler": f"{stem}.joblib",
                        "sha256": sha256_file(path),
                        "scaler_sha256": sha256_file(config.output / f"{stem}.joblib"),
                        "counts": subset["split"].value_counts().to_dict(),
                        "mean": float(scaler.mean_[0]),
                        "scale": float(scaler.scale_[0]),
                        "training_compositions": subset.loc[subset["split"].eq("train"), "composition"].tolist(),
                    }
                )
    (config.output / f"manifest_{DATE}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet"))
    parser.add_argument("--output", type=Path, default=Path(f"data/transformer_transfer_{DATE}"))
    args = parser.parse_args()
    result = prepare(PreparationConfig(input=args.input, output=args.output))
    print(
        json.dumps(
            {
                "audit": result["audit"],
                "source": result["source"],
                "targets": [
                    {k: v for k, v in t.items() if k in {"task", "seed", "fraction", "counts"}}
                    for t in result["targets"]
                ],
            },
            indent=2,
        )
    )
