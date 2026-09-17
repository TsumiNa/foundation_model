#!/usr/bin/env python3
"""Low-data variants of the 2026-09-11 dataset: keep a fraction, or a fixed number, of each task's TRAINING labels.

For every (fraction | count, seed) one new parquet is written; the source file is never modified. For each
task column listed below, training-split rows outside a seeded subset have their label set to NaN
(= missing, the way every task already treats a missing value); val and test rows are untouched, so
every run at every size is scored on the same test rows as the full-data baselines. Classification
tasks are subsampled per class in proportion (at least one row per class kept); regression tasks at random.

Fractions (the 2026-09-16 study; 5 % of a 20k-row task and 5 % of a 1k-row task are not comparable):
    python scripts/make_lowdata_datasets.py --src data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet \\
        --out data/lowdata --fractions 0.05,0.10,0.25,0.50 --seeds 0,1,2

Fixed counts (the 2026-09-17 study; the same number of labelled rows for every task, the share of the
task's labels is reported beside it). A task with fewer labelled training rows than the count keeps
them all, and the manifest says so:
    python scripts/make_lowdata_datasets.py --src data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet \\
        --out data/lowdata_n --counts 100,300,1000,3000,10000 --seeds 0,1,2
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


def subsample(out, col, idx, task, want, rng):
    """Indices to keep: `want` rows (per class in proportion for classification, at least one per class)."""
    if want >= len(idx):
        return idx
    if task in CLASSIFICATION:
        labels = out[col].to_numpy()[idx]; keep = []
        for c in np.unique(labels):
            ci = idx[labels == c]
            k = max(1, int(round(want * len(ci) / len(idx))))
            keep.extend(rng.choice(ci, size=min(k, len(ci)), replace=False))
        return np.array(sorted(keep))
    return np.sort(rng.choice(idx, size=max(1, want), replace=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fractions", help="e.g. 0.05,0.10,0.25,0.50"); g.add_argument("--counts", help="e.g. 100,300,1000,3000,10000")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.src)
    train = (df["split"] == "train").to_numpy()
    sizes = [("fraction", float(x)) for x in a.fractions.split(",")] if a.fractions else [("count", int(x)) for x in a.counts.split(",")]
    manifest = {"source": str(a.src), "mode": sizes[0][0], "files": []}
    for kind, size in sizes:
        for seed in [int(x) for x in a.seeds.split(",")]:
            rng = np.random.default_rng(1000 * seed + (int(round(size * 100)) if kind == "fraction" else 100_000 + size))
            out = df.copy(); kept = {}
            for task, col in TASKS.items():
                idx = np.flatnonzero(train & out[col].notna().to_numpy())
                want = int(round(size * len(idx))) if kind == "fraction" else size
                keep = subsample(out, col, idx, task, want, rng)
                out.iloc[np.setdiff1d(idx, keep), out.columns.get_loc(col)] = np.nan
                kept[task] = {"train_before": int(len(idx)), "train_kept": int(len(keep)), "all_kept": bool(len(keep) == len(idx))}
            name = f"qc_20260912_f{int(round(size * 100)):02d}_s{seed}.parquet" if kind == "fraction" else f"qc_20260912_n{size:05d}_s{seed}.parquet"
            out.to_parquet(a.out / name)
            entry = {"file": name, "seed": seed, "kept": kept}; entry[kind] = size
            manifest["files"].append(entry)
            print(name, {t: v["train_kept"] for t, v in kept.items() if t in ("material_type", "band_gap", "piezoelectric_max")}, flush=True)
    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest['files'])} files in {a.out}")


if __name__ == "__main__":
    main()
