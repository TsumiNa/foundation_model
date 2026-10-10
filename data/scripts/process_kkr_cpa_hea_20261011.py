#!/usr/bin/env python3
"""KKR-CPA equiatomic quaternary high-entropy alloys -> composition-keyed task table.

Source. Fukushima, T., Akai, H., Chikyow, T. & Kino, H. Automatic exhaustive calculations of large
material space by Korringa-Kohn-Rostoker coherent potential approximation method applied to
equiatomic quaternary high entropy alloys. Phys. Rev. Mater. 6, 023802 (2022). Data: Zenodo
10.5281/zenodo.5606502, `result_gather-2021-10.26.csv` (CC BY 4.0, downloaded 2026-10-11, sha256
695691dee3cc65ddd698fd23208eaf31cf3f9427ba48c8b807c3bfb78c11d3a6), kept in data/KKR_CPA_HEA/.

What is in the file. One row per (quaternary, structure): 38 elements given by atomic number, bcc and
fcc for every equiatomic quaternary, a convergence flag, the lattice constant and volume, four local
moments, the total moment, the magnetic transition temperature TC(K), the residual resistivity, and
the all-electron total energy TE(Ry).

Policy (decided 2026-10-11).
  rows        converged (`conv`) only; one composition = the equiatomic quaternary, written as a
              formula string through `convert_composition` like every other dataset.
  tasks       each property is a separate task per structure: N tasks = 2 structures x 3 properties
                Total magnetic moment bcc/fcc [μB]  — absolute value: 141 of 146,314 rows carry a
                    negative total moment, a sign convention of the calculation, not a physical quantity
                Curie temperature bcc/fcc [K]       — the magnetic transition temperature as computed
                Residual resistivity bcc/fcc [μΩ cm]
              plus one classification task, Structure preference = the structure with the lower TE
              ("bcc" / "fcc"), from compositions converged in both structures.
  not tasks   TE(Ry) itself (all-electron absolute energy, set by the core electrons; no same-method
              elemental references in the data to form a mixing energy); volume and lattice constant
              (identical per-atom volume for bcc and fcc in 99.97 % of compositions, a_fcc / a_bcc =
              2^(1/3): an input of the calculation, not a result); the four site moments (ordered by
              element, not a composition-level quantity).
  outliers    4 compositions with |E_fcc - E_bcc| > 0.68 eV/atom (every other composition is within
              ±0.587 eV/atom) are left out of Structure preference; their other tasks are kept.
  units       energies are reported in eV/atom (TE is per atom: the four-element average of the
              all-electron atomic energies; 1 Ry = 13.605693 eV), not Ry.
  scaling     per regression column: StandardScaler -> PowerTransformer(yeo-johnson) on the non-null
              values, `<name> (normalized)`; label column via LabelEncoder, `<name> (label)` — the
              conventions of process_ac_qe_te_data.ipynb / rebuild_mp_gga_20260912.py.
  split       70 / 15 / 15 train / val / test over compositions, train_test_split(random_state=42).

Outputs (new files; nothing existing is overwritten):
  data/KKR_CPA_HEA_20261011.parquet        composition + raw task columns (units in the name)
  data/KKR_CPA_HEA_20261011_norm.parquet   composition, split, `(normalized)` / `(label)` columns
  data/KKR_CPA_HEA_preprocessing_20261011.pkl.z   the fitted scalers and the label encoder

    uv run python data/scripts/process_kkr_cpa_hea_20261011.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

DATA = Path(__file__).resolve().parent.parent
SRC = DATA / "KKR_CPA_HEA" / "result_gather-2021-10.26.csv"
OUT_RAW = DATA / "KKR_CPA_HEA_20261011.parquet"
OUT_NORM = DATA / "KKR_CPA_HEA_20261011_norm.parquet"
OUT_PRE = DATA / "KKR_CPA_HEA_preprocessing_20261011.pkl.z"
RY_TO_EV = 13.605693
OUTLIER_EV = 0.68  # = 0.05 Ry

PROPS = {  # source column -> task name stem (unit in brackets)
    "total moment": ("Total magnetic moment", "μB"),
    "TC(K)": ("Curie temperature", "K"),
    "resistivity(micro ohm cm)": ("Residual resistivity", "μΩ cm"),
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
    src = pd.read_csv(SRC)
    src = src[src["conv"].astype(bool)].copy()
    els = src[["element1", "element2", "element3", "element4"]].to_numpy()
    src["composition"] = [convert_composition({Element.from_Z(int(z)).symbol: 1 for z in row}) for row in els]
    if src["composition"].isna().any():
        raise SystemExit("unparseable composition")

    raw = pd.DataFrame(index=pd.Index(sorted(src["composition"].unique()), name=None))
    for poly in ("bcc", "fcc"):
        part = src[src["polytyp"] == poly].drop_duplicates("composition").set_index("composition")
        for col, (name, unit) in PROPS.items():
            values = part[col].abs() if col == "total moment" else part[col]
            raw[f"{name} {poly}[{unit}]"] = values.reindex(raw.index)
    te = src.pivot_table(index="composition", columns="polytyp", values="TE(Ry)")
    delta = ((te["fcc"] - te["bcc"]) * RY_TO_EV).reindex(raw.index)  # eV/atom
    pref = pd.Series(np.where(delta < 0, "fcc", "bcc"), index=raw.index, dtype=object)
    pref[delta.isna() | (delta.abs() > OUTLIER_EV)] = None
    raw["Structure preference"] = pref
    raw["Energy fcc - bcc[eV/atom]"] = delta  # kept for provenance; not a task
    raw.index.name = "composition"
    raw = raw.reset_index()

    norm = raw[["composition"]].copy()
    pre: dict = {}
    task_cols = [c for c in raw.columns if c not in ("composition", "Structure preference", "Energy fcc - bcc[eV/atom]")]
    for col in task_cols:
        name = col.split("[")[0]
        idx = raw[col].notna()
        pipe = Pipeline([("standardscaler", StandardScaler()), ("powertransformer", PowerTransformer(method="yeo-johnson"))])
        norm[f"{name} (normalized)"] = np.nan
        norm.loc[idx, f"{name} (normalized)"] = pipe.fit_transform(raw.loc[idx, [col]].to_numpy(np.float64)).ravel()
        pre[f"{'_'.join(name.lower().split())}_scaler"] = pipe
    idx = raw["Structure preference"].notna()
    le = LabelEncoder()
    norm["Structure preference (label)"] = np.nan
    norm.loc[idx, "Structure preference (label)"] = le.fit_transform(raw.loc[idx, "Structure preference"])
    pre["structure_preference_label_encoder"] = le

    ids = norm.index.to_numpy()
    tr, tmp = train_test_split(ids, test_size=0.3, random_state=42)
    va, te_ = train_test_split(tmp, test_size=0.5, random_state=42)
    split = pd.Series(index=norm.index, dtype=object)
    split[tr], split[va], split[te_] = "train", "val", "test"
    norm.insert(1, "split", split)

    raw.to_parquet(OUT_RAW, index=False)
    norm.to_parquet(OUT_NORM, index=False)
    joblib.dump(pre, OUT_PRE)

    print(f"{len(raw):,} compositions -> {OUT_RAW.name}, {OUT_NORM.name}")
    print(norm["split"].value_counts().to_dict())
    for c in [c for c in norm.columns if c not in ("composition", "split")]:
        print(f"  {c:48s} {int(norm[c].notna().sum()):6,d} rows")
    print("  Structure preference classes:", dict(zip(le.classes_, np.bincount(norm['Structure preference (label)'].dropna().astype(int)))))


if __name__ == "__main__":
    main()
