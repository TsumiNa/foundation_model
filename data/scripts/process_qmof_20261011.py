#!/usr/bin/env python3
"""QMOF (quantum-chemical properties of metal-organic frameworks) -> composition-keyed task table.

Source. Rosen, A. S. et al. Machine learning the quantum-chemical properties of metal–organic
frameworks for accelerated materials discovery. Matter 4, 1578–1597 (2021). Data: figshare
10.6084/m9.figshare.13147324, version 18 (2025-11-15), `qmof_database.zip` (CC BY 4.0, downloaded
2026-10-11, sha256 97d23c0b4f9e5a30888e53dc16222b90443ad7167c3284d2258615d9f44eceef), kept in
data/QMOF/; this script reads the tabulated `qmof_database/qmof.csv` inside it.

Policy (decided 2026-10-11).
  composition the reduced composition of `info.formula`, written as a formula string through
              `convert_composition`. The model reads atomic fractions, so two MOFs with the same
              reduced formula are the same input.
  duplicates  1,545 reduced formulas are shared by more than one MOF (3,604 rows); for each, keep the
              MOF with the lowest PBE energy per atom (outputs.pbe.energy_total / info.natoms) and take
              every task value from that one MOF, so a row never mixes structures.
  tasks       QMOF band gap PBE [eV]      every MOF (PBE-D3(BJ) relaxed, PBE single point)
              QMOF CBM PBE [eV], QMOF VBM PBE [eV]
              (the HSE06 band gap is left out: only about half the MOFs have it, and a second band-gap
              column from another functional would sit beside the PBE one under a similar name)
              QMOF energy per atom PBE [eV/atom]
              Task names carry the "QMOF" prefix: same functional family as the Materials Project
              columns, different code settings and chemical space, so they are separate tasks.
  scaling     per column StandardScaler -> PowerTransformer(yeo-johnson) on the non-null values,
              `<name> (normalized)` (process_ac_qe_te_data.ipynb convention).
  split       70 / 15 / 15 over compositions, train_test_split(random_state=42).

Outputs (new files; nothing existing is overwritten):
  data/QMOF_20261011.parquet, data/QMOF_20261011_norm.parquet, data/QMOF_preprocessing_20261011.pkl.z

    uv run python data/scripts/process_qmof_20261011.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

DATA = Path(__file__).resolve().parent.parent
SRC = DATA / "QMOF" / "qmof_database" / "qmof.csv"
OUT_RAW = DATA / "QMOF_20261011.parquet"
OUT_NORM = DATA / "QMOF_20261011_norm.parquet"
OUT_PRE = DATA / "QMOF_preprocessing_20261011.pkl.z"

TASKS = {  # source column -> (task name, unit)
    "outputs.pbe.bandgap": ("QMOF band gap PBE", "eV"),
    "outputs.pbe.cbm": ("QMOF CBM PBE", "eV"),
    "outputs.pbe.vbm": ("QMOF VBM PBE", "eV"),
    "energy_per_atom": ("QMOF energy per atom PBE", "eV/atom"),
}


def convert_composition(comp) -> str | None:
    """Same contract as data/scripts/phonix-db.ipynb: a pymatgen formula string, or None."""
    try:
        c = Composition(comp)
        if len(c) == 0:
            return None
        c.weight
        return c.formula
    except Exception:
        return None


def main() -> None:
    for out in (OUT_RAW, OUT_NORM, OUT_PRE):
        if out.exists():
            raise SystemExit(f"{out} exists — datasets are never overwritten; bump the date instead.")
    q = pd.read_csv(SRC, low_memory=False)
    q["energy_per_atom"] = q["outputs.pbe.energy_total"] / q["info.natoms"]
    q["composition"] = q["info.formula"].map(lambda f: convert_composition(Composition(f).reduced_composition))
    q = q.dropna(subset=["composition", "energy_per_atom", "outputs.pbe.bandgap"])
    n_rows, n_shared = len(q), int((q["composition"].value_counts() > 1).sum())
    kept = q.sort_values(["composition", "energy_per_atom", "qmof_id"]).drop_duplicates("composition", keep="first")

    raw = kept[["composition", "qmof_id"]].rename(columns={"qmof_id": "QMOF id"}).reset_index(drop=True)
    for src_col, (name, unit) in TASKS.items():
        raw[f"{name}[{unit}]"] = kept[src_col].to_numpy(np.float64)

    norm = raw[["composition"]].copy()
    pre: dict = {}
    for src_col, (name, unit) in TASKS.items():
        col = f"{name}[{unit}]"
        idx = raw[col].notna()
        pipe = Pipeline(
            [("standardscaler", StandardScaler()), ("powertransformer", PowerTransformer(method="yeo-johnson"))]
        )
        norm[f"{name} (normalized)"] = np.nan
        norm.loc[idx, f"{name} (normalized)"] = pipe.fit_transform(raw.loc[idx, [col]].to_numpy(np.float64)).ravel()
        pre[f"{'_'.join(name.lower().split())}_scaler"] = pipe

    ids = norm.index.to_numpy()
    tr, tmp = train_test_split(ids, test_size=0.3, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, random_state=42)
    split = pd.Series(index=norm.index, dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    norm.insert(1, "split", split)

    raw.to_parquet(OUT_RAW, index=False)
    norm.to_parquet(OUT_NORM, index=False)
    joblib.dump(pre, OUT_PRE)
    print(
        f"{n_rows:,} MOFs, {n_shared:,} reduced formulas shared -> {len(raw):,} compositions (lowest PBE energy per atom kept)"
    )
    print(norm["split"].value_counts().to_dict())
    for c in [c for c in norm.columns if c not in ("composition", "split")]:
        print(f"  {c:40s} {int(norm[c].notna().sum()):6,d} rows")


if __name__ == "__main__":
    main()
