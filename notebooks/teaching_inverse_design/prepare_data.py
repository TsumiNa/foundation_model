# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Build the self-contained teaching dataset for `notebooks/teaching_inverse_design/`.

The teaching notebooks deliberately skip data wrangling: they open ONE parquet file and start
modelling. This script is what produces that file from the four raw sources the research run
used, and it is kept next to the notebooks so the provenance of every column is auditable.

Students do not need to run it — `data/qc_inverse_design_teaching.parquet` is committed.

    uv run python notebooks/teaching_inverse_design/prepare_data.py

Sources (paths relative to the repo root)
-----------------------------------------
qc              data/qc_ac_te_mp_dos_reformat_20250615_enforce_quaternary_test.pd.parquet
                  -> material_type (5 fine labels merged to AC / QC / others), formation_energy
superconductor  data/NEMAD_superconductor_20260425.parquet   -> tc
magnetic        data/NEMAD_magnetic_20260419.parquet         -> magnetization
phonix          data/phonix-db-filtered_20260425.parquet     -> klat

Every source is keyed by the project's canonical composition string
(`data.composition_sources.normalize_composition`) and outer-joined on that key, so one row =
one composition and a missing property is simply NaN (the model masks those out per task).
"""

from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from loguru import logger

from foundation_model.data.composition_sources import normalize_composition

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).resolve().parent / "data" / "qc_inverse_design_teaching.parquet"

QC_PATH = REPO_ROOT / "data/qc_ac_te_mp_dos_reformat_20250615_enforce_quaternary_test.pd.parquet"
QC_PREPROCESSING = REPO_ROOT / "data/preprocessing_objects_20250615.pkl.z"
SUPERCONDUCTOR_PATH = REPO_ROOT / "data/NEMAD_superconductor_20260425.parquet"
MAGNETIC_PATH = REPO_ROOT / "data/NEMAD_magnetic_20260419.parquet"
PHONIX_PATH = REPO_ROOT / "data/phonix-db-filtered_20260425.parquet"

#: The qc set has 48 631 "others" against 367 AC/QC rows. Keeping every minority row and
#: sampling the majority down keeps the teaching file small and the class ratio workable
#: without touching the signal the inverse-design objective actually cares about.
N_OTHERS = 12_000

#: The 5 fine material-type labels collapse to 3: AC = {DAC, IAC}, QC = {DQC, IQC}, others.
#: Index order matches `MATERIAL_TYPE_CLASSES` below (0 = AC, 1 = QC, 2 = others).
MATERIAL_TYPE_MERGE = {0: 0, 2: 0, 1: 1, 3: 1, 4: 2}
MATERIAL_TYPE_CLASSES = ("AC", "QC", "others")

#: Raw (non-qc) targets span orders of magnitude — Tc from 0.01 K to 200 K, klat over four
#: decades. log1p compresses that, a train-only z-score centres it, and the clip tames the tail
#: so no single outlier dominates the loss. The qc columns arrive already normalised.
RAW_TARGET_CLIP = 5.0

RANDOM_SEED = 42


def _load_qc(rng: np.random.Generator) -> pd.DataFrame:
    """qc rows: every AC/QC row plus a random sample of `others`."""
    df = pd.read_parquet(
        QC_PATH,
        columns=["composition", "Material type (label)", "Formation energy per atom (normalized)", "split"],
    )
    if QC_PREPROCESSING.exists():
        dropped = joblib.load(QC_PREPROCESSING).get("dropped_idx", [])
        df = df.loc[~df.index.isin(dropped)]

    labels = df["Material type (label)"]
    minority = df[labels != 4]  # DAC / DQC / IAC / IQC
    others = df[labels == 4]
    if len(others) > N_OTHERS:
        others = others.iloc[rng.choice(len(others), size=N_OTHERS, replace=False)]
    df = pd.concat([minority, others])
    logger.info(f"qc: {len(minority)} AC/QC + {len(others)} others = {len(df)} rows")

    out = pd.DataFrame(index=df.index)
    out["__key__"] = [normalize_composition(v) for v in df["composition"]]
    out["material_type"] = labels.map(MATERIAL_TYPE_MERGE)
    out["formation_energy"] = df["Formation energy per atom (normalized)"].astype(float)
    out["split"] = df["split"].astype(str)
    return out.dropna(subset=["__key__"]).drop_duplicates(subset="__key__", keep="first").set_index("__key__")


def _load_raw(path: Path, column: str, name: str) -> pd.DataFrame:
    """One raw source: canonical key + the single target column, still in physical units."""
    df = pd.read_parquet(path, columns=["composition", column])
    out = pd.DataFrame(index=df.index)
    out["__key__"] = [normalize_composition(v) for v in df["composition"]]
    out[name] = df[column].astype(float)
    out = out.dropna(subset=["__key__"]).drop_duplicates(subset="__key__", keep="first").set_index("__key__")
    logger.info(f"{name}: {out[name].notna().sum()} labelled rows from {path.name}")
    return out


def _normalise(values: pd.Series, is_train: pd.Series) -> pd.Series:
    """log1p → train-only z-score → clip. Stats come from train rows so val/test cannot leak in."""
    v = np.log1p(values.clip(lower=0.0))
    ref = v[is_train & v.notna()]
    ref = ref if len(ref) else v.dropna()
    mean, std = float(ref.mean()), float(ref.std(ddof=0)) or 1.0
    return ((v - mean) / std).clip(-RAW_TARGET_CLIP, RAW_TARGET_CLIP)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    qc = _load_qc(rng)
    tc = _load_raw(SUPERCONDUCTOR_PATH, "Transition temperature[K]", "tc")
    mag = _load_raw(MAGNETIC_PATH, "Magnetization[A·m²/mol]", "magnetization")
    klat = _load_raw(PHONIX_PATH, "klat[W/mK]", "klat")

    # Outer join on the canonical composition key: one row per composition, NaN where a source
    # has nothing to say about it. The model masks NaN targets per task, so no row is wasted.
    merged = qc.join([tc, mag, klat], how="outer")

    # Split: qc rows keep the project's published split; everything else gets a random 70/15/15
    # assigned once here, so the notebooks (and every student) see the identical partition.
    missing = merged["split"].isna()
    merged.loc[missing, "split"] = rng.choice(["train", "val", "test"], size=int(missing.sum()), p=[0.7, 0.15, 0.15])
    is_train = merged["split"] == "train"

    for name in ("tc", "magnetization", "klat"):
        merged[name] = _normalise(merged[name], is_train)

    # Drop compositions no task can learn from: a row that is NaN in all five targets only
    # inflates the file and the descriptor cache. (The magnetic source contributes ~20k rows but
    # labels magnetization for only ~1.6k of them.)
    targets = ["material_type", "formation_energy", "magnetization", "tc", "klat"]
    before = len(merged)
    merged = merged.dropna(subset=targets, how="all")
    logger.info(f"dropped {before - len(merged)} compositions with no label on any task")

    merged.index.name = "composition"
    merged = merged.reset_index()
    merged["material_type_name"] = merged["material_type"].map({i: n for i, n in enumerate(MATERIAL_TYPE_CLASSES)})
    merged = merged[
        [
            "composition",
            "split",
            "material_type",
            "material_type_name",
            "formation_energy",
            "magnetization",
            "tc",
            "klat",
        ]
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)
    logger.info(f"wrote {OUT_PATH} — {len(merged)} rows, {merged.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory")
    for col in ("material_type", "formation_energy", "magnetization", "tc", "klat"):
        logger.info(f"  {col:18s} {merged[col].notna().sum():6d} labelled")
    logger.info(f"  split: {merged['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
