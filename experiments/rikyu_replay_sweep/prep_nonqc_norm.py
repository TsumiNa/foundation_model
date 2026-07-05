#!/usr/bin/env python3
"""Pre-normalize the 7 non-qc regression targets exactly like the deleted continual_rehearsal_full.py
so the generic `fm pretrain` CLI (which trains on the column as-is) can consume them.

For each raw column:  v = log1p(clip(x, >=0));  z = (v - mean_train)/std_train;  clip(z, +-5).
Stats use the TRAIN rows of a deterministic seed-42 split written into a `split` column (the CLI
honors an existing split column, so normalization and training use the same train set — no leak).
Missing values stay NaN and are masked by the datamodule.

Writes small derived parquets holding only: composition, <col> (normalized)..., split.
"""
import numpy as np
import pandas as pd
from pathlib import Path

CLIP = 5.0
SEED = 42
DATA = Path("data")

SPECS = {
    "NEMAD_magnetic_20260419": {
        "Magnetic moment[μB/f.u.]": "Magnetic moment (normalized)",
        "Magnetization[A·m²/mol]": "Magnetization (normalized)",
        "Curie temperature[K]": "Curie temperature (normalized)",
        "Neel temperature[K]": "Neel temperature (normalized)",
    },
    "NEMAD_superconductor_20260425": {
        "Transition temperature[K]": "Transition temperature (normalized)",
    },
    "phonix-db-filtered_20260425": {
        "kp[W/mK]": "kp (normalized)",
        "klat[W/mK]": "klat (normalized)",
    },
}


def main():
    for stem, colmap in SPECS.items():
        src = DATA / f"{stem}.parquet"
        df = pd.read_parquet(src)
        # dedup by composition (keep first), mirroring the old key-based dedup
        df = df.dropna(subset=["composition"]).drop_duplicates(subset="composition", keep="first").reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        split = rng.choice(["train", "val", "test"], size=len(df), p=[0.7, 0.15, 0.15])
        out = pd.DataFrame({"composition": df["composition"].to_numpy(), "split": split})
        is_train = split == "train"
        for raw, norm in colmap.items():
            v = np.log1p(df[raw].astype(float).clip(lower=0.0))   # pandas Series; NaN preserved
            ref = v[is_train] if is_train.any() else v
            mean = float(ref.mean())                # Series.mean() skips NaN
            std = float(ref.std(ddof=0)) or 1.0
            out[norm] = ((v - mean) / std).clip(-CLIP, CLIP).to_numpy()
            n = int(np.isfinite(out[norm]).sum())
            print(f"{stem}: {raw!r} -> {norm!r}  (mean={mean:.4f} std={std:.4f} non-null={n})")
        dst = DATA / f"{stem}_norm.parquet"
        out.to_parquet(dst, index=False)
        print(f"  wrote {dst}  rows={len(out)} cols={list(out.columns)}\n")


if __name__ == "__main__":
    main()
