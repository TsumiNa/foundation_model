"""Create a dated, composition-isolated scalar-task matrix with fixed target subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from foundation_model.utils.kmd_plus import KMD, element_features, formula_to_composition

SOURCE = [
    "Formation energy per atom",
    "Band gap",
    "Efermi",
    "Volume",
    "Density",
    "Final energy per atom",
    "Total magnetization",
]
TARGET = ["Dielectric total", "Bulk modulus", "Shear modulus", "Piezoelectric max"]
DATE = "20261003"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def isolate(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Explicit train/val/test split required")
    keys = []
    for c in frame.composition:
        try:
            w = formula_to_composition(c)
            if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
                raise ValueError("Invalid composition")
            keys.append("|".join(f"{i}:{v:.12f}" for i, v in enumerate(w) if v > 0))
        except (ValueError, TypeError, KeyError):
            keys.append(None)
    f = frame.assign(identity=keys).dropna(subset=["identity"]).copy()
    rank = f.split.map({"train": 0, "val": 1, "test": 2})
    f["split"] = rank.groupby(f.identity).transform("max").map({0: "train", 1: "val", 2: "test"})
    return f.drop_duplicates("identity", keep="first").reset_index(drop=True)


def prepare(input_path: Path, output: Path) -> dict:
    f = isolate(pd.read_parquet(input_path, columns=["composition", "split", *SOURCE, *TARGET]))
    for name in SOURCE + TARGET:
        f[name] = pd.to_numeric(f[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if any((f.split.eq(s) & f[name].notna()).sum() < 10 for s in ["train", "val", "test"]):
            raise ValueError(f"{name}: insufficient partition coverage")
    weights = np.stack([formula_to_composition(c) for c in f.composition])
    x = KMD(element_features.to_numpy(), n_grids=8).transform(weights).astype("float32")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite descriptors")
    # Fixed nested training subsets, independent of architecture and optimization seed.
    for j, name in enumerate(TARGET):
        indices = np.flatnonzero(f.split.eq("train") & f[name].notna())
        selected = np.random.default_rng(20261003 + j).permutation(indices)[: int(np.ceil(len(indices) * 0.1))]
        f[f"target{j}_low_train"] = f.index.isin(selected)
    output.mkdir(parents=True, exist_ok=True)
    fp = output / f"matrix_{DATE}.parquet"
    xp = output / f"descriptors_{DATE}.npz"
    f.to_parquet(fp, index=False)
    np.savez_compressed(xp, x=x)
    result = {
        "date": DATE,
        "input_sha256": digest(input_path),
        "source_tasks": SOURCE,
        "target_tasks": TARGET,
        "files": {fp.name: digest(fp), xp.name: digest(xp)},
        "rows": len(f),
        "input_dim": x.shape[1],
        "counts": {
            n: {s: int((f.split.eq(s) & f[n].notna()).sum()) for s in ["train", "val", "test"]} for n in SOURCE + TARGET
        },
        "policy": "Atomic-fraction split isolation; original first composition retained; targets remain raw until train-only normalization in worker.",
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.input, a.output), indent=2))
