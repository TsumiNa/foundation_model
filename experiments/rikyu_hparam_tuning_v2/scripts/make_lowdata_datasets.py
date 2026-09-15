#!/usr/bin/env python3
"""Low-data variants of the 2026-09-11 dataset: keep a fraction of each task's TRAINING labels.

For every (fraction, seed) one new parquet is written; the source file is never modified. For each
task column listed below, training-split rows outside a seeded subset have their label set to NaN
(= missing, the way every task already treats a missing value); val and test rows are untouched, so
every run at every fraction is scored on the same test rows as the full-data baselines. Classification
tasks are subsampled per class (at least one row per class kept); regression tasks at random.

    python scripts/make_lowdata_datasets.py --src data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet \\
        --out data/lowdata --fractions 0.05,0.10,0.25,0.50 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = {  # task name -> label column
    "material_type": "Material type (label)",
    "band_gap": "Band gap (normalized)", "density_atomic": "Density atomic (normalized)",
    "magnetization_per_volume": "Total magnetization per volume (normalized)", "magnetization_per_fu": "Total magnetization per formula unit (normalized)",
    "reaction_energy": "Equilibrium reaction energy per atom (normalized)", "cbm": "CBM (normalized)", "vbm": "VBM (normalized)",
    "bulk_modulus": "Bulk modulus (normalized)", "shear_modulus": "Shear modulus (normalized)", "poisson_ratio": "Poisson ratio (normalized)",
    "universal_anisotropy": "Universal anisotropy (normalized)", "refractive_index": "Refractive index (normalized)", "piezoelectric_max": "Piezoelectric max (normalized)",
    "magnetic_ordering": "Magnetic ordering (label)", "is_metal": "Is metal (label)", "is_gap_direct": "Is gap direct (label)", "space_group": "Space group (task label)",
}
CLASSIFICATION = {"material_type", "magnetic_ordering", "is_metal", "is_gap_direct", "space_group"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fractions", default="0.05,0.10,0.25,0.50"); ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.src)
    train = (df["split"] == "train").to_numpy()
    manifest = {"source": str(a.src), "files": []}
    for frac in [float(x) for x in a.fractions.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            rng = np.random.default_rng(1000 * seed + int(round(frac * 100)))
            out = df.copy(); kept = {}
            for task, col in TASKS.items():
                has = train & out[col].notna().to_numpy()
                idx = np.flatnonzero(has)
                if task in CLASSIFICATION:
                    keep = []
                    for c in np.unique(out[col].to_numpy()[idx]):
                        ci = idx[out[col].to_numpy()[idx] == c]
                        k = max(1, int(round(frac * len(ci))))
                        keep.extend(rng.choice(ci, size=k, replace=False))
                    keep = np.array(sorted(keep))
                else:
                    keep = np.sort(rng.choice(idx, size=max(1, int(round(frac * len(idx)))), replace=False))
                drop = np.setdiff1d(idx, keep)
                out.iloc[drop, out.columns.get_loc(col)] = np.nan
                kept[task] = {"train_before": int(len(idx)), "train_kept": int(len(keep))}
            name = f"qc_20260912_f{int(round(frac * 100)):02d}_s{seed}.parquet"
            out.to_parquet(a.out / name)
            manifest["files"].append({"file": name, "fraction": frac, "seed": seed, "kept": kept})
            print(name, {t: v["train_kept"] for t, v in kept.items() if t in ("material_type", "band_gap", "piezoelectric_max")}, flush=True)
    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest['files'])} files in {a.out}")


if __name__ == "__main__":
    main()
