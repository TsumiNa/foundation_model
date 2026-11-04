#!/usr/bin/env python3
"""Record the target transforms behind the NEMAD and phonix-db `_norm` tables, and prove them.

Why. `NEMAD_magnetic_20260419_norm`, `NEMAD_superconductor_20260425_norm` and
`phonix-db-filtered_20260425_norm` (7 training tasks) were written without saving their transforms or
the code that produced them, so their predictions could not be mapped back to physical units. The
transforms were recovered on 2026-11-04 (docs/refactor/pretrained-release/00-overview.md):

  1. one row per composition: the first row of the raw table;
  2. log1p of the value clipped below at 0;
  3. z-score with the mean and population std (ddof=0) of the training split (`split` column of the
     `_norm` table);
  4. phonix-db kp / klat only: clip at +5.

This script rebuilds each transform from the raw table and the `_norm` table's split, applies it to
every row, and checks that the result equals the `_norm` column to within 1e-12. Only when every table
passes does it write the transforms (foundation_model.data.targets, plain JSON) beside the tables. The tables themselves
are not changed.

Outputs (new files; nothing existing is overwritten):
  data/NEMAD_magnetic_20260419_transforms.json
  data/NEMAD_superconductor_20260425_transforms.json
  data/phonix-db-filtered_20260425_transforms.json

    uv run python data/scripts/record_nemad_phonix_transforms_20261104.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from foundation_model.data.targets import ClipStep, Log1pStep, StandardScalerStep, TargetTransform

DATA = Path(__file__).resolve().parent.parent
TOL = 1e-12

# table stem -> [(raw column, normalised column, clip upper bound or None)]
TABLES: dict[str, list[tuple[str, str, float | None]]] = {
    "NEMAD_magnetic_20260419": [
        ("Magnetic moment[μB/f.u.]", "Magnetic moment (normalized)", None),
        ("Magnetization[A·m²/mol]", "Magnetization (normalized)", None),
        ("Curie temperature[K]", "Curie temperature (normalized)", None),
        ("Neel temperature[K]", "Neel temperature (normalized)", None),
    ],
    "NEMAD_superconductor_20260425": [
        ("Transition temperature[K]", "Transition temperature (normalized)", None),
    ],
    "phonix-db-filtered_20260425": [
        ("kp[W/mK]", "kp (normalized)", 5.0),
        ("klat[W/mK]", "klat (normalized)", 5.0),
    ],
}


def recover(raw: pd.DataFrame, norm: pd.DataFrame, raw_col: str, norm_col: str, clip_upper: float | None) -> dict:
    first = raw.drop_duplicates("composition", keep="first")[["composition", raw_col]]
    merged = norm[["composition", "split", norm_col]].merge(first, on="composition", how="left", validate="one_to_one")
    target = merged[norm_col].to_numpy(float)
    labelled = np.isfinite(target)
    x = merged[raw_col].to_numpy(float)
    if np.isnan(x[labelled]).any():
        raise ValueError(f"{norm_col}: labelled rows without a raw value")

    log1p = Log1pStep(clip_lower=0.0)
    z = log1p.forward(x)
    train = labelled & (merged["split"] == "train").to_numpy()
    steps = [log1p, StandardScalerStep(mean=float(np.mean(z[train])), scale=float(np.std(z[train])))]
    if clip_upper is not None:
        steps.append(ClipStep(upper=clip_upper))
    transform = TargetTransform(tuple(steps))

    err = float(np.max(np.abs(transform.transform(x[labelled]) - target[labelled])))
    if err > TOL:
        raise ValueError(f"{norm_col}: recovered transform differs from the _norm table by {err:.3e} (> {TOL})")
    print(f"  {norm_col:40s} rows={int(labelled.sum()):6d} max|diff|={err:.1e}")
    return {
        "raw_column": raw_col,
        "normalized_column": norm_col,
        "n_train": int(train.sum()),
        "max_abs_error": err,
        "saturates": transform.saturates,
        "transform": transform.to_dict(),
    }


def main() -> None:
    outputs = {stem: DATA / f"{stem}_transforms.json" for stem in TABLES}
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing:
        raise FileExistsError(f"{existing} exist; this script never overwrites")
    # Validate every table first; write only when all of them reproduce, so a failure leaves nothing behind.
    records: dict[str, dict] = {}
    for stem, columns in TABLES.items():
        raw = pd.read_parquet(DATA / f"{stem}.parquet")
        norm = pd.read_parquet(DATA / f"{stem}_norm.parquet")
        print(stem)
        records[stem] = {
            "source": f"{stem}.parquet",
            "normalized": f"{stem}_norm.parquet",
            "recipe": "first row per composition -> log1p(max(x, 0)) -> train z-score (ddof=0)"
            + (" -> clip at +5" if any(c is not None for *_, c in columns) else ""),
            "script": "data/scripts/record_nemad_phonix_transforms_20261104.py",
            "tasks": {norm_col: recover(raw, norm, raw_col, norm_col, clip) for raw_col, norm_col, clip in columns},
        }
    for stem, record in records.items():
        outputs[stem].write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {outputs[stem].relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
