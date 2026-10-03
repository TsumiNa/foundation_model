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
    "Energy above hull",
    "Equilibrium reaction energy per atom",
    "Density atomic",
    "CBM",
    "VBM",
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


def partitions(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Hash composition identity, independent of labels and row order; nested training subsets."""
    f = frame.copy()
    values = np.array([int(hashlib.sha256(f"{seed}:{k}".encode()).hexdigest()[:12], 16) / 16**12 for k in f.identity])
    f["split"] = np.where(values < 0.7, "train", np.where(values < 0.85, "val", "test"))
    for j, name in enumerate(TARGET):
        idx = f.index[f.split.eq("train") & f[name].notna()].to_numpy()
        order = sorted(
            idx, key=lambda i: hashlib.sha256(f"subset:{seed}:{j}:{f.loc[i, 'identity']}".encode()).hexdigest()
        )
        for fraction in [0.01, 0.1, 1.0]:
            selected = order[: max(2, int(np.ceil(len(order) * fraction)))]
            f[f"target{j}_f{round(fraction * 100):03d}_train"] = f.index.isin(selected)
    return f


def prepare(input_path: Path, output: Path) -> dict:
    f = isolate(pd.read_parquet(input_path, columns=["composition", "split", *SOURCE, *TARGET]))
    for name in SOURCE + TARGET:
        f[name] = pd.to_numeric(f[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    weights = np.stack([formula_to_composition(c) for c in f.composition]).astype("float32")
    x = KMD(element_features.to_numpy(), n_grids=8).transform(weights).astype("float32")
    if not np.isfinite(x).all() or weights.shape[1] != 94:
        raise ValueError("Require finite KMD and 94-element fractions")
    output.mkdir(parents=True, exist_ok=True)
    xp = output / f"descriptors_{DATE}.npz"
    np.savez_compressed(xp, kmd=x, composition=weights)
    files = {xp.name: digest(xp)}
    counts = {}
    for seed in [20261011, 20261012, 20261013]:
        split_frame = partitions(f, seed)
        counts[str(seed)] = {
            n: {s: int((split_frame.split.eq(s) & split_frame[n].notna()).sum()) for s in ["train", "val", "test"]}
            for n in SOURCE + TARGET
        }
        if any(v < 10 for c in counts[str(seed)].values() for v in c.values()):
            raise ValueError("Insufficient task partition coverage")
        fp = output / f"matrix_split{seed}_{DATE}.parquet"
        split_frame.to_parquet(fp, index=False)
        files[fp.name] = digest(fp)
    result = dict(
        date=DATE,
        input_sha256=digest(input_path),
        source_tasks=SOURCE,
        target_tasks=TARGET,
        files=files,
        rows=len(f),
        counts=counts,
        policy="First composition record after alias collapse; three label-independent composition-hash splits; 94-element fractions and KMD from the same records; nested 1/10/100 percent training sizes; repeated holdout is not external unseen data.",
    )
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.input, a.output), indent=2))
