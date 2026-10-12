#!/usr/bin/env python3
"""Catalysis-Hub MamunHighT2019 adsorption energies -> composition-keyed task table.

Source. Mamun, O., Winther, K. T., Boes, J. R. & Bligaard, T. High-throughput calculations of
catalytic properties of bimetallic alloy surfaces. Sci. Data 6, 76 (2019). Data: Catalysis-Hub
publication MamunHighT2019 (https://www.catalysis-hub.org/publications/MamunHighT2019), pulled through
https://api-catalysis-hub.slac.stanford.edu/graphql by fetch_cathub_mamun2019.py on 2026-10-11/12
into data/CatalysisHub_MamunHighT2019/pages/ (88,587 reactions). Quantum ESPRESSO 5.1, BEEF-vdW;
A1 and L1_2 (A3B) slabs cut along (111), L1_0 (AB) along (101) — so the facet follows from the
stoichiometry, and the bulk composition is a complete input.

Policy (decided 2026-10-11/12).
  reactions   only reactions whose product is ONE adsorbed species (e.g. "0.5H2(g) + * -> H*",
              "H2O(g) - H2(g) + * -> O*"). The multi-adsorbate dissociations in the publication
              (e.g. "CH2CH2(g) + 6* -> 2C* + 4H*") are linear combinations of these and are dropped.
  composition `surfaceComposition` (the bulk stoichiometry: Ag, MnRu, Hg3La, ...) as a formula string
              through `convert_composition`; `chemicalComposition` (the slab's atom count) is not used.
  sites       every adsorption site of an adsorbate on a surface is a separate reaction; the task
              value is the lowest reaction energy over sites (the most stable site), and that site
              is recorded beside it.
  tasks       one task per adsorbed species: "<X>* adsorption energy[eV]", referenced to the gas-phase
              molecules of its equation (recorded in the source equation column).
  scaling     per column StandardScaler -> PowerTransformer(yeo-johnson) on the non-null values,
              `<name> (normalized)`.
  split       70 / 15 / 15 over compositions, train_test_split(random_state=42).

Outputs (new files; nothing existing is overwritten):
  data/CatalysisHub_MamunHighT2019_20261012.parquet, ..._norm.parquet,
  data/CatalysisHub_MamunHighT2019_preprocessing_20261012.pkl.z

    uv run python data/scripts/process_cathub_mamun2019_20261012.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

DATA = Path(__file__).resolve().parent.parent
PAGES = DATA / "CatalysisHub_MamunHighT2019" / "pages"
OUT_RAW = DATA / "CatalysisHub_MamunHighT2019_20261012.parquet"
OUT_NORM = DATA / "CatalysisHub_MamunHighT2019_20261012_norm.parquet"
OUT_PRE = DATA / "CatalysisHub_MamunHighT2019_preprocessing_20261012.pkl.z"
EXPECTED = 88587


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


def single_adsorbate(products: str) -> str | None:
    """'{"Hstar": 1}' -> 'H*'; None when the product is not exactly one adsorbed species."""
    p = {k: v for k, v in json.loads(products).items() if k != "star"}
    if len(p) != 1:
        return None
    (name, count), = p.items()
    return name[: -len("star")] + "*" if name.endswith("star") and count == 1 else None


def main() -> None:
    for out in (OUT_RAW, OUT_NORM, OUT_PRE):
        if out.exists():
            raise SystemExit(f"{out} exists — datasets are never overwritten; bump the date instead.")
    pages = sorted(PAGES.glob("page_*.json"))
    rows = [e["node"] for f in pages for e in json.loads(f.read_text())["edges"]]
    last = json.loads(pages[-1].read_text())
    if last["pageInfo"]["hasNextPage"] or len(rows) != EXPECTED:
        raise SystemExit(f"fetch incomplete: {len(rows)} of {EXPECTED} reactions — rerun fetch_cathub_mamun2019.py")
    d = pd.DataFrame(rows).drop_duplicates("id")
    d["adsorbate"] = d["products"].map(single_adsorbate)
    single = d.dropna(subset=["adsorbate", "reactionEnergy"]).copy()
    single["composition"] = single["surfaceComposition"].map(convert_composition)
    single = single.dropna(subset=["composition"])
    best = single.sort_values(["composition", "adsorbate", "reactionEnergy"]).drop_duplicates(["composition", "adsorbate"])

    adsorbates = best["adsorbate"].value_counts().index.tolist()  # most covered first
    raw = pd.DataFrame(index=pd.Index(sorted(best["composition"].unique())))
    for a in adsorbates:
        part = best[best["adsorbate"] == a].set_index("composition")
        raw[f"{a} adsorption energy[eV]"] = part["reactionEnergy"]
        raw[f"{a} site"] = part["sites"].map(lambda s: next(iter(json.loads(s).values())) if s else None)
        raw[f"{a} equation"] = part["Equation"]
    raw.index.name = "composition"
    raw = raw.reset_index()

    norm = raw[["composition"]].copy()
    pre: dict = {}
    for a in adsorbates:
        col = f"{a} adsorption energy[eV]"; idx = raw[col].notna()
        pipe = Pipeline([("standardscaler", StandardScaler()), ("powertransformer", PowerTransformer(method="yeo-johnson"))])
        norm[f"{a} adsorption energy (normalized)"] = np.nan
        norm.loc[idx, f"{a} adsorption energy (normalized)"] = pipe.fit_transform(raw.loc[idx, [col]].to_numpy(np.float64)).ravel()
        pre[f"{a.replace('*', '_star').lower()}_adsorption_energy_scaler"] = pipe
    ids = norm.index.to_numpy()
    tr, tmp = train_test_split(ids, test_size=0.3, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, random_state=42)
    split = pd.Series(index=norm.index, dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    norm.insert(1, "split", split)

    raw.to_parquet(OUT_RAW, index=False)
    norm.to_parquet(OUT_NORM, index=False)
    joblib.dump(pre, OUT_PRE)
    print(f"{len(d):,} reactions, {len(single):,} single-adsorbate, {len(raw):,} compositions")
    print(norm["split"].value_counts().to_dict())
    for a in adsorbates:
        v = raw[f"{a} adsorption energy[eV]"].dropna()
        eq = raw[f"{a} equation"].dropna().value_counts().index[0]
        print(f"  {a:6s} {len(v):5,d} compositions | E {v.min():+.2f} … {v.median():+.2f} … {v.max():+.2f} eV | {eq}")


if __name__ == "__main__":
    main()
