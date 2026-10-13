#!/usr/bin/env python3
"""Per-task label distribution of every composition-keyed dataset — results/label_distribution.csv

For each scalar `(normalized)` column: rows with a value, skewness, share beyond |z| > 4, share of the most frequent
value (a spike such as exact zeros), min and max.

    uv run python experiments/data_audit_20261013/analysis/label_distribution.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/data_audit_20261013/results/label_distribution.csv"
FILES = {
    "qc": "data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet",
    "nemad_mag": "data/NEMAD_magnetic_20260419_norm.parquet",
    "nemad_sc": "data/NEMAD_superconductor_20260425_norm.parquet",
    "phonix": "data/phonix-db-filtered_20260425_norm.parquet",
    "hea": "data/KKR_CPA_HEA_20261011_norm.parquet",
    "qmof": "data/QMOF_20261011_norm.parquet",
    "cathub": "data/CatalysisHub_MamunHighT2019_20261012_norm.parquet",
}


def main() -> None:
    rows = []
    for name, rel in FILES.items():
        d = pd.read_parquet(ROOT / rel)
        for col in [c for c in d.columns if c.endswith("(normalized)")]:
            v = d[col].dropna()
            if v.empty or isinstance(v.iloc[0], (list, np.ndarray)):
                continue  # curve tasks are profiled in the notebook, not here
            v = v.astype(float)
            rows.append(
                {
                    "dataset": name,
                    "task": col.replace(" (normalized)", ""),
                    "n": len(v),
                    "skew": float(skew(v)),
                    "share_abs_z_gt_4": float((v.abs() > 4).mean()),
                    "mode_share": float(v.round(6).value_counts(normalize=True).iloc[0]),
                    "min": float(v.min()),
                    "max": float(v.max()),
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{len(rows)} scalar tasks -> {OUT}")


if __name__ == "__main__":
    main()
