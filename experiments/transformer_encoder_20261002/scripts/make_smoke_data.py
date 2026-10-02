"""Small real-data fixture for functional smoke only; never used for scientific results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from benchmark import DATE, SOURCE_TASKS, TARGET_TASKS, file_hash


def make_smoke(source_dir: Path, output: Path) -> None:
    manifest = json.loads((source_dir / f"manifest_{DATE}.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(source_dir / manifest["source"]["file"])
    indices = set()
    for name in SOURCE_TASKS:
        for split, count in (("train", 32), ("val", 8)):
            indices.update(source.index[source[name].notna() & source["split"].eq(split)][:count])
    for split, count in (("train", 320), ("val", 80)):
        selected = source.loc[sorted(indices)]
        needed = count - int(selected["split"].eq(split).sum())
        indices.update(source.index[source["split"].eq(split) & ~source.index.isin(indices)][:needed])
    source = source.loc[sorted(indices)]
    source.to_parquet(output / manifest["source"]["file"], index=False)
    manifest["source"]["sha256"] = file_hash(output / manifest["source"]["file"])
    manifest["source"]["counts"] = source["split"].value_counts().to_dict()
    shutil.copy2(source_dir / f"source_scalers_{DATE}.joblib", output)
    for target in TARGET_TASKS:
        for seed in (0, 1, 2):
            entries = [t for t in manifest["targets"] if t["task"] == target and t["seed"] == seed]
            low = pd.read_parquet(source_dir / next(t["file"] for t in entries if t["fraction"] == 0.1))
            low_train = set(low.loc[low["split"].eq("train"), "composition"].iloc[:32])
            for entry in entries:
                frame = pd.read_parquet(source_dir / entry["file"])
                train = frame.loc[frame["split"].eq("train")]
                extra = train.loc[~train["composition"].isin(low_train), "composition"].iloc[:288]
                keep = low_train | (set(extra) if entry["fraction"] == 1 else set())
                subset = pd.concat(
                    [
                        train.loc[train["composition"].isin(keep)],
                        frame.loc[frame["split"].eq("val")].iloc[:40],
                        frame.loc[frame["split"].eq("test")].iloc[:40],
                    ]
                )
                subset.to_parquet(output / entry["file"], index=False)
                shutil.copy2(source_dir / entry["scaler"], output)
                entry["counts"] = subset["split"].value_counts().to_dict()
                entry["sha256"] = file_hash(output / entry["file"])
                entry["training_compositions"] = subset.loc[subset["split"].eq("train"), "composition"].tolist()
    manifest["functional_smoke"] = True
    manifest["note"] = "Small subsets reuse actual training-fitted scalers; this is a functional fixture only."
    (output / f"manifest_{DATE}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_smoke(args.data_dir, args.output)
