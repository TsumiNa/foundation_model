"""Create dated KMD-compatible task tables and a shared composition-group split manifest.

Raw sidecars are copied unchanged for provenance, not used as training tables.
No descriptors, labels or normalization parameters are invented or refitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from foundation_model.data.composition_sources import (
    canonical_key,
    composition_group_key,
    normalize_composition,
    resolve_splits,
)
from foundation_model.utils.kmd_plus import formula_to_composition

DATE = "20261016"
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_dataset_folder(source: Path, destination: Path) -> dict:
    """Preserve every supported record; change only split assignments in task tables."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    index_path = source / "task_index.csv"
    index = pd.read_csv(index_path)
    tables: dict[str, pd.DataFrame] = {}
    frames: dict[str, pd.DataFrame] = {}
    row_maps: dict[str, pd.DataFrame] = {}
    rejected: list[dict] = []
    checked: dict[str, tuple[str, str, str | None]] = {}
    source_hashes = {index_path.name: sha256(index_path)}
    for filename in sorted(index.file.unique()):
        path = source / filename
        source_hashes[filename] = sha256(path)
        df = pd.read_parquet(path)
        if not {"composition", "split"}.issubset(df):
            raise ValueError(f"Missing composition/split in {filename}")
        rows = []
        keep = []
        for pos, (original_index, row) in enumerate(df.iterrows()):
            raw = row.composition
            if isinstance(raw, dict):
                # Existing notebook representation uses None for absent elements.
                raw = {k: v for k, v in raw.items() if v is not None}
            cache_key = str(raw)
            if cache_key not in checked:
                try:
                    w = formula_to_composition(raw)
                    if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1.0, atol=1e-12, rtol=0):
                        raise ValueError("Invalid complete atomic-fraction vector")
                    key = canonical_key(raw, normalize_composition)
                    checked[cache_key] = (key, composition_group_key(key), None)
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    checked[cache_key] = (str(raw), "", str(exc))
            key, group, error = checked[cache_key]
            record = dict(
                source_file=filename,
                row_position=pos,
                source_row_id=str(row.get("id", original_index)),
                composition_key=key,
                composition_group=group,
                original_split=row.split,
            )
            if error is not None:
                rejected.append({**record, "reason": error})
                continue
            keep.append(pos)
            rows.append(record)
        table = df.iloc[keep].copy()
        mapping = pd.DataFrame(
            rows,
            columns=[
                "source_file",
                "row_position",
                "source_row_id",
                "composition_key",
                "composition_group",
                "original_split",
            ],
        )
        frames[filename] = pd.DataFrame({"split": mapping.original_split.to_numpy()}, index=mapping.composition_key)
        tables[filename] = table
        row_maps[filename] = mapping
    keys = sorted({k for frame in frames.values() for k in frame.index})
    if not keys:
        raise ValueError("No supported composition records in source training tables")
    resolved = resolve_splits(frames, keys, {name: "split" for name in frames})
    groups = {k: composition_group_key(k) for k in keys}
    group_splits = {groups[k]: str(v) for k, v in resolved.items()}
    destination.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    all_rows = []
    for filename, table in tables.items():
        mapping = row_maps[filename]
        mapping["split"] = mapping.composition_group.map(group_splits)
        table["split"] = mapping.split.to_numpy()
        original = pd.read_parquet(source / filename).iloc[mapping.row_position.to_numpy()]
        pd.testing.assert_frame_equal(table.drop(columns="split"), original.drop(columns="split"))
        tables[filename] = table
        table.to_parquet(destination / filename, index=True)
        output_hashes[filename] = sha256(destination / filename)
        all_rows.append(mapping)
    records = pd.concat(all_rows, ignore_index=True)
    if records.groupby("composition_group").split.nunique().max() != 1:
        raise AssertionError("Cross-split composition group in output")
    manifest = (
        records.groupby("composition_group", sort=True)
        .agg(split=("split", "first"), n_records=("split", "size"))
        .reset_index()
    )
    manifest_name = f"split_manifest_{DATE}.csv"
    manifest.to_csv(destination / manifest_name, index=False)
    records.to_parquet(destination / f"row_manifest_{DATE}.parquet", index=False)
    records[records.original_split != records.split].to_csv(destination / f"split_changes_{DATE}.csv", index=False)
    pd.DataFrame(rejected, columns=[*records.columns.drop("split"), "reason"]).to_csv(
        destination / f"rejected_rows_{DATE}.csv", index=False
    )
    # Archive sidecars byte for byte; never include them in the training split pool.
    for name in sorted(set(index.raw_file.dropna()) - set(tables)):
        source_hashes[name] = sha256(source / name)
        shutil.copy2(source / name, destination / name)
        output_hashes[name] = sha256(destination / name)
    for i, task in index.iterrows():
        d = tables[task.file]
        labeled = d[task.column].notna()
        index.loc[i, "n_total"] = int(labeled.sum())
        for split in ["train", "val", "test"]:
            index.loc[i, f"n_{split}"] = int((labeled & d.split.eq(split)).sum())
    index.to_csv(destination / "task_index.csv", index=False)
    for p in sorted(destination.iterdir()):
        if p.is_file():
            output_hashes[p.name] = sha256(p)
    result = dict(
        date=DATE,
        package_version=tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"],
        script_sha256=sha256(Path(__file__)),
        source_hashes=source_hashes,
        output_hashes=output_hashes,
        tasks=len(index),
        training_files=len(tables),
        retained_records=len(records),
        rejected_records=len(rejected),
        composition_groups=len(manifest),
        changed_split_records=int((records.original_split != records.split).sum()),
        policy="Full atomic fractions at 12 decimals; test > val > train; record/formula/label values retained; raw sidecars excluded from training; archived normalization unchanged.",
        cross_split_groups=0,
        split_manifest=manifest_name,
    )
    (destination / f"provenance_{DATE}.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "README.md").write_text(
        f"# KMD-compatible task datasets ({DATE})\n\n"
        f"{len(index)} tasks; {len(tables)} training tables; {len(records):,} retained records; {len(rejected)} rejected table records.\n\n"
        "Original formulas, labels, curves and normalization values are unchanged. Equivalent atomic-fraction compositions share one global split, including records with different absolute stoichiometry. Records are not averaged or merged.\n\n"
        f'Set `[data] split_manifest = "data/{destination.name}/{manifest_name}"` in source and downstream catalogs. Use the same manifest for every stage. The row manifest maps every retained source row; split_changes and rejected_rows explain exclusions/movements. Source and output hashes are in provenance.\n\n'
        "Files ending `_raw.parquet` are unchanged provenance sidecars, NOT repaired training tables. Train-only target normalization remains a separate unresolved issue. Existing checkpoints may have prior exposure to held-out composition groups; this dataset does not retroactively remove it. No training was run.\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/fm_task_datasets_20261013")
    parser.add_argument("--output", type=Path, default=ROOT / f"data/fm_task_datasets_{DATE}")
    args = parser.parse_args()
    print(json.dumps(repair_dataset_folder(args.source, args.output), indent=2))
