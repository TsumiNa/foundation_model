# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Create balanced, nested AGIS compound splits; fit scalers on each actual training set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from process_agis_data import standardize_fold


@dataclass(kw_only=True)
class LearningCurveSettings:
    date: str = "20261001"
    seed: int = 20261001

    def __post_init__(self) -> None:
        from datetime import datetime

        datetime.strptime(self.date, "%Y%m%d")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def nested_splits(compositions: list[str], seed: int) -> dict[int, list[dict]]:
    """Eight rotations of one seeded permutation, with the original LOCO material as anchor."""
    if len(compositions) != 8 or len(set(compositions)) != 8:
        raise ValueError("Require eight distinct compositions")
    permutation = random.Random(seed).sample(compositions, 8)
    splits: dict[int, list[dict]] = {}
    for n_train in range(1, 8):
        folds = []
        for number, anchor in enumerate(compositions, 1):
            start = permutation.index(anchor) + 1
            order = [permutation[(start + i) % 8] for i in range(7)]
            train = order[:n_train]
            test = [c for c in compositions if c not in train]
            folds.append(
                {"fold": number, "heldout_composition": anchor, "fit_compositions": train, "test_compositions": test}
            )
        assert set(Counter(c for f in folds for c in f["fit_compositions"]).values()) == {n_train}
        splits[n_train] = folds
    return splits


def prepare_learning_curve(reference_path: Path, output_dir: Path, settings: LearningCurveSettings) -> Path:
    reference = json.loads(reference_path.read_text())
    if reference["n_compounds"] != 8 or reference["n_curves"] != 24:
        raise ValueError("Require the complete eight-compound, three-pressure dataset")
    if reference.get("smoothing", {}).get("method", "none") != "none":
        raise ValueError("Learning curve uses the unsmoothed seven-compound baseline")
    curves = pd.read_parquet(reference["curves_path"])
    compositions = [f["heldout_composition"] for f in reference["folds"]]
    if set(curves.composition) != set(compositions):
        raise ValueError("Curve compositions differ from the reference folds")
    grid = np.linspace(6, 290, 300)
    for row in curves.itertuples():
        if not np.array_equal(row.temperature_K, grid) or not np.isfinite(row.rho_uohm_cm).all():
            raise ValueError("Require finite resistivity curves on the common 300-point grid")
    splits = nested_splits(compositions, settings.seed)
    summary_path = output_dir / f"agis_learning_curve_{settings.date}.json"
    folders = [output_dir / f"agis_preprocessing_n{n}_{settings.date}" for n in range(1, 7)]
    if summary_path.exists() or any(p.exists() for p in folders):
        raise FileExistsError("Dated learning-curve outputs already exist; choose another output directory")
    manifests = {"7": str(reference_path)}
    for n_train, folder in zip(range(1, 7), folders, strict=True):
        folder.mkdir(parents=True)
        folds = []
        for split in splits[n_train]:
            fold, scalers = standardize_fold(curves, split["test_compositions"])
            fold_dir = folder / f"fold_{split['fold']:02d}"
            fold_dir.mkdir()
            scaler_path = fold_dir / f"scalers_{settings.date}.joblib"
            joblib.dump(scalers, scaler_path)
            fragments = ["# Nested AGIS split: no held-out material enters any pressure fit.\n"]
            stats = {}
            for pressure in (0, 10, 20):
                data = fold[fold.pressure_GPa == pressure].drop(columns=["raw_temperature_K", "raw_rho_uohm_cm"])
                path = (
                    fold_dir
                    / f"agis_resistivity_n{n_train}_f{split['fold']:02d}_{pressure}gpa_{settings.date}.pd.parquet"
                )
                data.to_parquet(path, index=False)
                name = f"agis_rho_{pressure}gpa"
                key = f"{name}_scaler"
                scaler = scalers[key]
                original = np.concatenate(data.rho_uohm_cm.to_list()).reshape(-1, 1)
                normalized = np.concatenate(data.rho_normalized.to_list()).reshape(-1, 1)
                stats[str(pressure)] = {
                    "n_fit_points": int(scaler["prescale"].n_samples_seen_),
                    "prescale_std": float(scaler["prescale"].scale_[0]),
                    "asinh_mean": float(scaler["standardscaler"].mean_[0]),
                    "asinh_std": float(scaler["standardscaler"].scale_[0]),
                    "max_abs_roundtrip_error_uohm_cm": float(
                        np.max(np.abs(scaler.inverse_transform(normalized) - original))
                    ),
                }
                fragments.append(
                    f"[datasets.{name}]\npath = {json.dumps(str(path))}\n\n"
                    f'[[tasks]]\nname = "{name}"\nkind = "kernel_regression"\ndataset = "{name}"\n'
                    'column = "rho_normalized"\nt_column = "temperature_K"\n'
                    f"[tasks.scaler]\npath = {json.dumps(str(scaler_path))}\nkey = {json.dumps(key)}\n\n"
                )
            (fold_dir / f"tasks_{settings.date}.toml").write_text("".join(fragments))
            folds.append({**split, "directory": str(fold_dir), "scalers": stats})
        manifest = {
            **{
                k: reference[k]
                for k in ("config", "curves_path", "n_compounds", "n_curves", "pressures_GPa", "units", "transform")
            },
            "design": "balanced_nested_learning_curve",
            "n_train": n_train,
            "n_test": 8 - n_train,
            "settings": asdict(settings),
            "folds": folds,
            "reference_path": str(reference_path),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "processor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "holdout": "Every test composition excluded from scaler fitting and every AGIS pressure used for warm-start",
        }
        path = folder / f"manifest_{settings.date}.json"
        path.write_text(json.dumps(manifest, indent=2))
        manifests[str(n_train)] = str(path)
    summary_path.write_text(
        json.dumps({"settings": asdict(settings), "manifests": manifests, "splits": splits}, indent=2)
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path("data/agis_preprocessing_20261001/manifest_20261001.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--date", default="20261001")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    print(
        prepare_learning_curve(args.reference, args.output_dir, LearningCurveSettings(date=args.date, seed=args.seed))
    )


if __name__ == "__main__":
    main()
