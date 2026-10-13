#!/usr/bin/env python3
"""Summarize the single-task QMOF runs: R2 (normalized scale) and MAE in physical units per task.

uv run python experiments/qmof_learnability/analysis/summarize.py
"""

import json
import statistics as st
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/qmof_learnability/artifacts/single"
PRE = joblib.load(ROOT / "data/QMOF_preprocessing_20261011.pkl.z")


def main() -> None:
    rows = []
    for task in ["qmof_band_gap_pbe", "qmof_energy_per_atom_pbe", "qmof_cbm_pbe", "qmof_vbm_pbe"]:
        r2, mae = [], []
        for pred in sorted(RUNS.glob(f"{task}_s*/training/step01_{task}/{task}_pred.parquet")):
            d = pd.read_parquet(pred)
            scaler = PRE[f"{task}_scaler"]
            true = scaler.inverse_transform(d[["true"]].to_numpy()).ravel()
            est = scaler.inverse_transform(d[["pred"]].to_numpy()).ravel()
            mae.append(float(np.mean(np.abs(true - est))))
            r2.append(json.loads(pred.with_name(f"{task}_metrics.json").read_text())["r2"])
        if r2:
            rows.append(
                {
                    "task": task,
                    "n_seeds": len(r2),
                    "r2_mean": st.fmean(r2),
                    "r2_sd": st.stdev(r2) if len(r2) > 1 else 0.0,
                    "mae_physical": st.fmean(mae),
                }
            )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
