# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Prepare AGIS resistivity curves and training-only scalers for compound holdouts.

Run from the repository root with ``uv run python data/scripts/process_agis_data.py``.
The generated task fragments use the existing catalog scaler/inverse-transform interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import savgol_filter
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # type: ignore[import-untyped]

from foundation_model.data.composition_sources import normalize_composition

_PRESSURES_GPA = (0, 10, 20)
_RHO_COLUMNS = {"rho (muOhm cm)", "rho_xx (muOhm cm)"}
_EXPECTED_FORMULAS = (
    "La3Ni2O7",
    "La2.9Sr0.1Ni2O7",
    "La2NdNi2O7",
    "La1.9NdSr0.1Ni2O7",
    "La1.8NdSr0.2Ni2O7",
    "La2EuNi2O7",
    "La1.9EuSr0.1Ni2O7",
    "La1.8EuSr0.2Ni2O7",
)


@dataclass(kw_only=True)
class AGISPreprocessingConfig:
    """A common supported temperature grid; no extrapolation or curve-wise normalization."""

    temperature_min_K: float = 6.0
    temperature_max_K: float = 290.0
    n_points: int = 300
    smooth_window_K: float | None = None
    date_suffix: str = field(default_factory=lambda: datetime.now().astimezone().strftime("%Y%m%d"))

    def __post_init__(self) -> None:
        if not np.isfinite([self.temperature_min_K, self.temperature_max_K]).all():
            raise ValueError("Temperature bounds must be finite.")
        if self.temperature_min_K < 0 or self.temperature_max_K <= self.temperature_min_K:
            raise ValueError("Require 0 <= temperature_min_K < temperature_max_K.")
        if not isinstance(self.n_points, int) or isinstance(self.n_points, bool) or self.n_points < 2:
            raise ValueError("n_points must be an integer >= 2.")
        if not re.fullmatch(r"\d{8}", self.date_suffix):
            raise ValueError("date_suffix must be a YYYYMMDD date.")
        datetime.strptime(self.date_suffix, "%Y%m%d")
        if self.smooth_window_K is not None:
            if not np.isfinite(self.smooth_window_K) or self.smooth_window_K <= 0:
                raise ValueError("smooth_window_K must be finite and positive.")
            if not 3 <= self.smoothing_points <= self.n_points:
                raise ValueError("Smoothing requires an odd window of 3 through n_points samples.")

    @property
    def smoothing_points(self) -> int:
        """Match the notebook's floor-to-samples, next-odd window convention."""
        if self.smooth_window_K is None:
            return 0
        step = (self.temperature_max_K - self.temperature_min_K) / (self.n_points - 1)
        points = int(self.smooth_window_K / step)
        return points + (points % 2 == 0)


def load_agis_curves(input_dir: Path, config: AGISPreprocessingConfig | None = None) -> pd.DataFrame:
    """Read nominal 0/10/20 GPa curves, average identical T values and linearly resample.

    Directory formulas are checked against header formulas, including reordered elements and
    the ``sample name - formula`` header variant. Raw arrays, units and experimental metadata
    remain available for auditing. Nonfinite T/rho pairs are counted and removed; finite negative
    resistivity readings are retained.
    """
    config = config or AGISPreprocessingConfig()
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"AGIS input directory does not exist: {input_dir}")
    grid = np.linspace(config.temperature_min_K, config.temperature_max_K, config.n_points)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for path in sorted(input_dir.glob("*/*.dat")):
        payload = path.read_bytes()
        lines = payload.decode("utf-8-sig").splitlines()
        pressure_match = next(
            (
                re.match(r"^p\s*=\s*([\d.]+)\s+GPa\b", line.strip())
                for line in lines[:15]
                if line.strip().startswith("p")
            ),
            None,
        )
        if pressure_match is None:
            raise ValueError(f"Missing nominal pressure header in {path}")
        nominal_pressure = float(pressure_match.group(1))
        if nominal_pressure not in _PRESSURES_GPA:
            continue
        pressure = int(nominal_pressure)
        header_index = next((i for i, line in enumerate(lines) if line.startswith("T (K)\t")), None)
        if header_index is None:
            raise ValueError(f"Missing T (K) table header in {path}")
        header = dict(line.split(":", 1) for line in lines[:header_index] if ":" in line)
        header = {k.strip(): v.strip() for k, v in header.items()}
        formula = path.parent.name.split(maxsplit=1)[-1]
        composition = normalize_composition(formula)
        header_formula = header.get("sample material", "").rsplit(" - ", 1)[-1].strip()
        if composition is None or normalize_composition(header_formula) != composition:
            raise ValueError(f"Directory/header composition mismatch in {path}: {formula!r}, {header_formula!r}")
        key = (composition, pressure)
        if key in seen:
            raise ValueError(f"Duplicate AGIS curve for composition {composition}, pressure {pressure} GPa")
        seen.add(key)
        data = pd.read_csv(StringIO("\n".join(lines[header_index:])), sep="\t")
        rho_columns = [c for c in data.columns if c in _RHO_COLUMNS]
        if len(rho_columns) != 1:
            raise ValueError(f"Expected one resistivity column in muOhm cm in {path}, got {list(data.columns)}")
        temperature = pd.to_numeric(data["T (K)"], errors="coerce").to_numpy(dtype=float)
        rho = pd.to_numeric(data[rho_columns[0]], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(temperature) & np.isfinite(rho)
        n_dropped = int((~finite).sum())
        if n_dropped:
            logger.warning("Dropping {} nonfinite T/rho pairs from {}", n_dropped, path)
        clean = pd.DataFrame({"temperature": temperature[finite], "rho": rho[finite]})
        unique = clean.groupby("temperature", sort=True).rho.mean()
        if len(unique) < 2:
            raise ValueError(f"Need at least two finite distinct temperatures in {path}")
        t = unique.index.to_numpy(dtype=float)
        if t[0] > grid[0] or t[-1] < grid[-1]:
            raise ValueError(f"{path} does not cover [{grid[0]}, {grid[-1]}] K; observed [{t[0]}, {t[-1]}] K")
        b = pd.to_numeric(data["B (T)"], errors="coerce") if "B (T)" in data else None
        rows.append(
            {
                "formula": formula,
                "composition": composition,
                "pressure_GPa": pressure,
                "temperature_K": grid.tolist(),
                "rho_uohm_cm": np.interp(grid, t, unique.to_numpy(dtype=float)).tolist(),
                "raw_temperature_K": temperature.tolist(),
                "raw_rho_uohm_cm": rho.tolist(),
                "source": path.as_posix(),
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "header_json": json.dumps(header, ensure_ascii=False),
                "n_raw": len(data),
                "n_nonfinite_dropped": n_dropped,
                "n_unique_temperature": len(unique),
                "n_negative_raw": int((rho[finite] < 0).sum()),
                "B_min_T": float(b.min()) if b is not None else None,
                "B_max_T": float(b.max()) if b is not None else None,
            }
        )
    if not rows:
        raise ValueError(f"No AGIS 0/10/20 GPa curves found in {input_dir}")
    curves = pd.DataFrame(rows)
    for compound, group in curves.groupby("composition", sort=False):
        missing = set(_PRESSURES_GPA) - set(group.pressure_GPa)
        if missing:
            raise ValueError(f"Missing pressures {sorted(missing)} GPa for composition {compound}")
    expected = {
        composition for formula in _EXPECTED_FORMULAS if (composition := normalize_composition(formula)) is not None
    }
    observed = set(curves.composition)
    if observed != expected:
        raise ValueError(
            f"Require all eight AGIS compounds; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return curves


def standardize_fold(
    curves: pd.DataFrame, heldout_composition: str, config: AGISPreprocessingConfig | None = None
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Fit one invertible resistivity pipeline per pressure on training compounds only.

    Like the starrydata notebook, fit on concatenated resampled sequence values and restore
    sequence boundaries afterward. Asinh replaces Yeo-Johnson here: the latter has a finite
    inverse domain when its fitted lambda is negative, as it is on these AGIS curves.
    Scaling before asinh makes it independent of the chosen resistivity unit; the final scaler
    centers/standardizes the transformed training values. All arrays remain in float64 on disk.
    """
    if heldout_composition not in set(curves.composition):
        raise ValueError(f"Unknown held-out composition: {heldout_composition}")
    fold = curves.copy(deep=True)
    fold["split"] = np.where(fold.composition == heldout_composition, "test", "train")
    if config is not None and config.smooth_window_K is not None:
        grid = np.linspace(config.temperature_min_K, config.temperature_max_K, config.n_points)
        fold["unsmoothed_rho_uohm_cm"] = fold.rho_uohm_cm.map(lambda values: np.asarray(values).tolist())
        fold["smoothing_applied"] = fold.split == "train"
        for idx in fold.index:
            if not np.array_equal(np.asarray(fold.at[idx, "temperature_K"]), grid):
                raise ValueError("Smoothing requires the configured uniform temperature grid.")
            values = np.asarray(fold.at[idx, "rho_uohm_cm"], dtype=float)
            if values.shape != grid.shape or not np.isfinite(values).all():
                raise ValueError("Smoothing requires one finite resistivity per temperature.")
            if fold.at[idx, "split"] == "train":
                fold.at[idx, "rho_uohm_cm"] = savgol_filter(values, config.smoothing_points, 2, mode="interp").tolist()
    fold["rho_normalized"] = pd.Series([None] * len(fold), index=fold.index, dtype=object)
    scalers: dict[str, Pipeline] = {}
    for pressure in _PRESSURES_GPA:
        selected = fold.pressure_GPa == pressure
        train_cells = fold.loc[selected & (fold.split == "train"), "rho_uohm_cm"]
        if train_cells.empty:
            raise ValueError(f"No training curves for pressure {pressure} GPa")
        values = np.concatenate(train_cells.to_list()).reshape(-1, 1)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite training resistivity values for pressure {pressure} GPa")
        scaler = Pipeline(
            [
                ("prescale", StandardScaler(with_mean=False)),
                ("asinh", FunctionTransformer(np.arcsinh, inverse_func=np.sinh, validate=True)),
                ("standardscaler", StandardScaler()),
            ]
        ).fit(values)
        for idx in fold.index[selected]:
            original = np.asarray(fold.at[idx, "rho_uohm_cm"], dtype=float).reshape(-1, 1)
            normalized = scaler.transform(original)
            restored = scaler.inverse_transform(normalized)
            if not np.isfinite(normalized).all() or not np.isfinite(restored).all():
                raise ValueError(f"Nonfinite resistivity transform for row {idx}")
            if not np.allclose(restored, original, rtol=1e-11, atol=1e-8):
                raise ValueError(f"Resistivity inverse-transform round trip failed for row {idx}")
            fold.at[idx, "rho_normalized"] = normalized.ravel().tolist()
        scalers[f"agis_rho_{pressure}gpa_scaler"] = scaler
    return fold, scalers


def prepare_agis(input_dir: Path, output_dir: Path, config: AGISPreprocessingConfig | None = None) -> Path:
    """Write raw/resampled curves, per-fold task datasets, fitted scalers and a manifest.

    Each pressure owns one composition-keyed file with explicit train/test splits. There is
    no validation compound in this outer split: the held-out compound must never drive early
    stopping or scaler fitting. Refuse existing dated outputs rather than overwrite them.
    """
    config = config or AGISPreprocessingConfig()
    output_dir = Path(output_dir)
    suffix = config.date_suffix
    variant = "_smoothed" if config.smooth_window_K is not None else ""
    curves_path = output_dir / f"agis_resistivity{variant}_{suffix}.pd.parquet"
    campaign_dir = output_dir / f"agis_preprocessing{variant}_{suffix}"
    if curves_path.exists() or campaign_dir.exists():
        raise FileExistsError(
            f"AGIS dated outputs already exist for {suffix} in {output_dir}; choose a new date/directory"
        )
    curves = load_agis_curves(input_dir, config)
    prepared = [standardize_fold(curves, c, config) for c in curves.composition.drop_duplicates()]
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_dir.mkdir()
    curves.to_parquet(curves_path, index=False)
    folds: list[dict[str, object]] = []
    for number, (fold, scalers) in enumerate(prepared, 1):
        fold_dir = campaign_dir / f"fold_{number:02d}"
        fold_dir.mkdir()
        scaler_path = fold_dir / f"scalers_{suffix}.joblib"
        joblib.dump(scalers, scaler_path)
        config_parts = ["# AGIS task fragment. Merge with the model/descriptor/training configuration.\n"]
        stats = {}
        for pressure in _PRESSURES_GPA:
            dataset = fold[fold.pressure_GPa == pressure].copy()
            # Full raw measurement arrays live once in the root parquet rather than in every fold.
            dataset = dataset.drop(columns=["raw_temperature_K", "raw_rho_uohm_cm"])
            data_path = fold_dir / f"agis_resistivity_fold{number:02d}_{pressure}gpa_{suffix}.pd.parquet"
            dataset.to_parquet(data_path, index=False)
            task_name = f"agis_rho_{pressure}gpa"
            scaler_key = f"{task_name}_scaler"
            config_parts.append(
                f"[datasets.{task_name}]\npath = {json.dumps(data_path.as_posix())}\n\n"
                f'[[tasks]]\nname = {json.dumps(task_name)}\nkind = "kernel_regression"\n'
                f'dataset = {json.dumps(task_name)}\ncolumn = "rho_normalized"\nt_column = "temperature_K"\n'
                f"[tasks.scaler]\npath = {json.dumps(scaler_path.as_posix())}\nkey = {json.dumps(scaler_key)}\n\n"
            )
            original = np.concatenate(dataset.rho_uohm_cm.to_list()).reshape(-1, 1)
            normalized = np.concatenate(dataset.rho_normalized.to_list()).reshape(-1, 1)
            scaler = scalers[scaler_key]
            stats[str(pressure)] = {
                "n_fit_points": int(scaler["prescale"].n_samples_seen_),
                "prescale_std": float(scaler["prescale"].scale_[0]),
                "asinh_mean": float(scaler["standardscaler"].mean_[0]),
                "asinh_std": float(scaler["standardscaler"].scale_[0]),
                "max_abs_roundtrip_error_uohm_cm": float(
                    np.max(np.abs(scaler.inverse_transform(normalized) - original))
                ),
            }
        (fold_dir / f"tasks_{suffix}.toml").write_text("".join(config_parts))
        train = fold.loc[fold.split == "train", "composition"].drop_duplicates().to_list()
        heldout = str(fold.loc[fold.split == "test", "composition"].iloc[0])
        folds.append(
            {
                "directory": fold_dir.as_posix(),
                "heldout_composition": heldout,
                "fit_compositions": train,
                "scalers": stats,
            }
        )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "foundation_model_version": version("foundation-model"),
        "numpy_version": version("numpy"),
        "sklearn_version": version("scikit-learn"),
        "config": asdict(config),
        "input_dir": str(input_dir),
        "curves_path": curves_path.as_posix(),
        "n_compounds": int(curves.composition.nunique()),
        "n_curves": len(curves),
        "pressures_GPa": list(_PRESSURES_GPA),
        "units": {"temperature": "K", "resistivity": "muOhm cm"},
        "interpolation": "linear, exact-temperature duplicates averaged; no smoothing or extrapolation",
        "smoothing": {
            "method": "Savitzky-Golay" if config.smooth_window_K is not None else "none",
            "nominal_window_K": config.smooth_window_K,
            "window_points": config.smoothing_points,
            "polynomial_order": 2,
            "mode": "interp",
            "scope": "training compounds only, after interpolation and before fitting scalers; heldout truth unchanged",
            "audit_curves": "root parquet remains unsmoothed; fold files retain unsmoothed_rho_uohm_cm when smoothing",
        },
        "transform": "StandardScaler(with_mean=False) -> asinh -> StandardScaler",
        "inverse_transform": "fitted Pipeline.inverse_transform via tasks.scaler; original muOhm cm",
        "validation": "outer folds contain train/test only; heldout labels cannot select epochs or hyperparameters",
        "folds": folds,
    }
    manifest_path = campaign_dir / f"manifest_{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    logger.info("Prepared {} curves, {} compound holdouts -> {}", len(curves), len(folds), output_dir)
    return manifest_path


def main() -> None:
    """Command-line entry point for the reproducible AGIS preprocessing module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/AGIS"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--smooth-window-K", type=float, default=None, help="Optional training-only SG window in kelvin"
    )
    parser.add_argument(
        "--date",
        default=datetime.now().astimezone().strftime("%Y%m%d"),
        help="Filename suffix YYYYMMDD (default: local date)",
    )
    args = parser.parse_args()
    prepare_agis(
        args.input_dir,
        args.output_dir,
        AGISPreprocessingConfig(date_suffix=args.date, smooth_window_K=args.smooth_window_K),
    )


if __name__ == "__main__":
    main()
