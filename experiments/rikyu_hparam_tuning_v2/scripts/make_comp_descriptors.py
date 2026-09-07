#!/usr/bin/env python3
"""Composition descriptors for the descriptor-contrast experiment — data/desc_comp_{classic,nosum}.parquet

KMD, the campaign's descriptor, is built from atomic FRACTIONS: Fe2O3 and Fe4O6 are the same input,
so nothing about cell scale reaches the model, and volume / final_energy / dos_density are the
tasks whose labels depend on it. This script builds the classic composition descriptor (XenonPy's
"classic" preset, recomputed here from the same 58-property element table KMD uses, because the
container has no xenonpy) in two variants, keyed by the pipeline's own canonical composition
string, which is NOT reduced (Fe2O3 != Fe4O6):

  classic  weighted sum + weighted average + weighted variance + max + min   (5 x 58 = 290 columns)
  nosum    the same without the weighted-sum block                            (4 x 58 = 232 columns)

The weighted sum uses the raw element amounts of the cell formula, so it is the one block that
carries cell scale; the other four are scale-free. If classic beats KMD and nosum does not, the
descriptor's scale-blindness is what caps these tasks. Every column is z-scored over all qc
compositions (KMD scales internally; the precomputed path does not).

    uv run python scripts/make_comp_descriptors.py            # writes both files under ../../data/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from foundation_model.data.composition_sources import normalize_composition
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS, element_features

ROOT = Path(__file__).resolve().parents[3]           # the project root, where data/ lives
SRC = ROOT / "data" / "qc_ac_te_mp_dos_reformat_20260515.pd.parquet"

E = element_features.copy()
if list(E.index[:3]) != DEFAULT_ELEMENTS[:3]:
    E.index = DEFAULT_ELEMENTS
E = E.loc[DEFAULT_ELEMENTS].astype(float)
props = list(E.columns)
P = E.values                                          # 94 x 58
elem_ix = {e: i for i, e in enumerate(DEFAULT_ELEMENTS)}

comps = pd.read_parquet(SRC, columns=["composition"])["composition"].dropna().unique()
print(f"  {len(comps)} unique composition strings in qc")

keys, rows, dropped = [], [], 0
for s in comps:
    key = normalize_composition(s)
    if key is None:
        dropped += 1
        continue
    amounts = Composition(key).get_el_amt_dict()
    if any(el not in elem_ix for el in amounts):
        dropped += 1                                  # elements outside the 94-element table
        continue
    w = np.zeros(len(DEFAULT_ELEMENTS))
    for el, a in amounts.items():
        w[elem_ix[el]] = a
    total = w.sum()
    if total <= 0:
        dropped += 1
        continue
    f = w / total
    present = w > 0
    s_ = w @ P                                        # weighted SUM: raw amounts -> carries cell scale
    a_ = f @ P                                        # weighted average
    v_ = f @ (P - a_) ** 2                            # weighted variance
    mx = P[present].max(axis=0)
    mn = P[present].min(axis=0)
    keys.append(key)
    rows.append(np.concatenate([s_, a_, v_, mx, mn]))
print(f"  featurised {len(keys)}, dropped {dropped}")

cols = [f"{blk}:{p}" for blk in ("sum", "ave", "var", "max", "min") for p in props]
X = pd.DataFrame(np.vstack(rows), index=pd.Index(keys, name="composition"), columns=cols)
X = X[~X.index.duplicated(keep="first")]
mean, sd = X.mean(), X.std(ddof=0)
keep = sd > 0
X = ((X.loc[:, keep] - mean[keep]) / sd[keep]).astype(np.float32)
print(f"  z-scored; dropped {int((~keep).sum())} constant column(s); {X.shape[1]} columns remain")

out_c = ROOT / "data" / "desc_comp_classic.parquet"
out_n = ROOT / "data" / "desc_comp_nosum.parquet"
X.reset_index().to_parquet(out_c, index=False)
X.loc[:, [c for c in X.columns if not c.startswith("sum:")]].reset_index().to_parquet(out_n, index=False)
print(f"  wrote {out_c} {X.shape}  and  {out_n} ({X.shape[0]}, {sum(not c.startswith('sum:') for c in X.columns)})")
# a scale check: the sum block must separate Fe2O3 from Fe4O6 while the other blocks do not
if "Fe2 O3" in X.index and "Fe4 O6" in X.index:
    d = (X.loc["Fe2 O3"] - X.loc["Fe4 O6"]).abs()
    print(f"  Fe2O3 vs Fe4O6: |diff| sum-block {d[[c for c in X.columns if c.startswith('sum:')]].max():.3f}, "
          f"other blocks {d[[c for c in X.columns if not c.startswith('sum:')]].max():.3g}")
