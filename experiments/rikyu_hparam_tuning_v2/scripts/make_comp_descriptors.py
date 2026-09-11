#!/usr/bin/env python3
"""Composition-keyed descriptor tables for the descriptor-contrast experiment.

The descriptors are NOT computed here. They are the ones data/data/scripts/calculate_compositional_desc.ipynb
produced: XenonPy `Compositions(featurizers="classic")` on the qc dataset (weighted sum / average /
variance / max / min over 58 element properties, 290 columns), then StandardScaler -> PowerTransformer
(yeo-johnson) fitted on all rows, stored as data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet
indexed by material id. This script only re-keys that table by the pipeline's canonical composition
string (fm's `[descriptor] kind = "precomputed"` looks rows up by composition) and writes two variants:

  classic  all 290 columns
  nosum    the 232 columns left after dropping the weighted-sum block, the one block that carries
           cell scale (XenonPy's weighted sum uses the raw cell amounts: O4F8 -> sum:atomic_number 104)

Compositions shared by several ids collapse to one row (keep-first), as the DataModule does.

    uv run python experiments/rikyu_hparam_tuning_v2/scripts/make_comp_descriptors.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from foundation_model.data.composition_sources import normalize_composition

ROOT = Path(__file__).resolve().parents[3]
DESC = ROOT / "data" / "qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet"
DATA = ROOT / "data" / "qc_ac_te_mp_dos_reformat_20260515.pd.parquet"

desc = pd.read_parquet(DESC)                                            # index: material id
comp = pd.read_parquet(DATA, columns=["composition"])["composition"]    # index: material id
keys = pd.Series({i: normalize_composition(c) for i, c in comp.items()})
print(f"  notebook descriptor: {desc.shape}, ids matched to the current dataset: {desc.index.isin(keys.index).sum()}")

out = desc.copy()
out.index = pd.Index([keys.get(i) for i in desc.index], name="composition")
out = out[out.index.notna()]
dup = out.index.duplicated(keep="first")
out = out[~dup].astype(np.float32)
nosum = [c for c in out.columns if not c.startswith("sum:")]
out.reset_index().to_parquet(ROOT / "data" / "desc_xenonpy_classic_trans.parquet", index=False)
out.loc[:, nosum].reset_index().to_parquet(ROOT / "data" / "desc_xenonpy_nosum_trans.parquet", index=False)
print(f"  wrote desc_xenonpy_classic_trans.parquet {out.shape} and desc_xenonpy_nosum_trans.parquet "
      f"({out.shape[0]}, {len(nosum)}); {int(dup.sum())} duplicate composition keys collapsed")
