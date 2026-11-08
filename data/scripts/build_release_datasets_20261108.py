"""Derive the release-training task folder from the post-#99 one.

Input: ``data/fm_task_datasets_20261016/`` (composition-group splits, PR #99). Output, a new folder that
is never overwritten: ``data/fm_task_datasets_20261108_release/``. Two changes, nothing else:

1. **Licence exclusion.** Rows of Materials Project origin whose licence tag cannot be read are dropped
   (``docs/licensing/data_sources.md``, open item 1). They are the ``mp_properties`` rows with no
   ``MP id``; the same legacy ``id`` is dropped from every other table that carries it (the DOS curves
   and the ``others`` rows of ``material_type``).
2. **Class names for the binary MP heads.** ``is_metal`` and ``is_gap_direct`` were encoded as
   ``int(bool)`` without a ``LabelEncoder``, and ``yammi export`` needs class names. A ``LabelEncoder``
   fitted on ``[False, True]`` reproduces that encoding; the script checks it on every labelled row and
   writes it to ``label_encoders_20261108.pkl.z``.

Splits, values, curves and normalisation are copied unchanged, and the split manifest is copied as is,
so the release run uses the same composition-group assignment as every other post-#99 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATE = "20261108"
ROOT = Path(__file__).resolve().parents[2]
MP_TABLE = "mp_properties.parquet"
BINARY_LABELS = {
    "is_metal": ("Is metal", "Is metal (label)"),
    "is_gap_direct": ("Is gap direct", "Is gap direct (label)"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unresolved_mp_ids(mp: pd.DataFrame) -> set[str]:
    """Legacy ids of Materials Project rows whose current MP id could not be resolved."""
    return set(mp.loc[mp["MP id"].isna(), "id"].astype(str))


def binary_label_encoders(mp: pd.DataFrame) -> dict[str, LabelEncoder]:
    """``LabelEncoder`` per binary head, checked against the stored integer labels."""
    encoders = {}
    for task, (raw, label) in BINARY_LABELS.items():
        rows = mp[label].notna()
        enc = LabelEncoder().fit([False, True])
        encoded = enc.transform(mp.loc[rows, raw].astype(bool))
        if not (encoded == mp.loc[rows, label].to_numpy()).all():
            raise ValueError(f"{task}: the stored labels are not LabelEncoder([False, True]) codes of {raw!r}.")
        encoders[f"{task}_label_encoder"] = enc
    return encoders


def build(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    index = pd.read_csv(source / "task_index.csv")
    mp = pd.read_parquet(source / MP_TABLE)
    excluded = unresolved_mp_ids(mp)
    encoders = binary_label_encoders(mp)

    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(f"Remove the stale staging folder {staging} first")
    staging.mkdir(parents=True)
    dropped: dict[str, int] = {}
    removed = []
    training_files = set(index.file)
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        if path.name in training_files:
            table = pd.read_parquet(path)
            hit = table["id"].astype(str).isin(excluded) if "id" in table else pd.Series(False, index=table.index)
            dropped[path.name] = int(hit.sum())
            if hit.any():
                removed.append(table.loc[hit, ["id", "composition", "split"]].assign(source_file=path.name))
            table.loc[~hit].to_parquet(staging / path.name, index=True)
        elif path.name in {"README.md", "task_index.csv"} or path.name.startswith("provenance_"):
            continue  # rewritten below
        else:
            shutil.copy2(path, staging / path.name)

    for i, task in index.iterrows():
        d = pd.read_parquet(staging / task.file, columns=[task.column, "split"])
        labeled = d[task.column].notna()
        index.loc[i, "n_total"] = int(labeled.sum())
        for split in ("train", "val", "test"):
            index.loc[i, f"n_{split}"] = int((labeled & d.split.eq(split)).sum())
    index.to_csv(staging / "task_index.csv", index=False)
    joblib.dump(encoders, staging / f"label_encoders_{DATE}.pkl.z")
    excluded_rows = (
        pd.concat(removed, ignore_index=True)
        if removed
        else pd.DataFrame(columns=["id", "composition", "split", "source_file"])
    )
    excluded_rows.to_csv(staging / f"licence_excluded_rows_{DATE}.csv", index=False)

    result = dict(
        date=DATE,
        source=source.name,
        script_sha256=sha256(Path(__file__)),
        source_provenance=json.loads(next(source.glob("provenance_*.json")).read_text()).get("date"),
        excluded_ids=len(excluded),
        dropped_rows=dropped,
        label_encoders=sorted(encoders),
        split_manifest=next(p.name for p in staging.glob("split_manifest_*.csv")),
        output_hashes={p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()},
    )
    (staging / f"provenance_{DATE}.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = ", ".join(f"{k} {v}" for k, v in dropped.items() if v)
    (staging / "README.md").write_text(
        f"# Release-training task datasets ({DATE})\n\n"
        f"`{source.name}` minus the Materials Project rows whose licence tag could not be read "
        f"({len(excluded)} legacy ids; rows dropped: {rows}). The ids are listed in "
        f"`licence_excluded_rows_{DATE}.csv`; see `docs/licensing/data_sources.md`, open item 1. The licence "
        "audit counted 30 `material_type` rows for these ids; matching by id finds 31, and all are dropped.\n\n"
        f"`label_encoders_{DATE}.pkl.z` holds `LabelEncoder([False, True])` for `is_metal` and `is_gap_direct`, "
        "checked against the stored integer labels, so `yammi export` can name their classes.\n\n"
        "Splits, values, curves and normalisation are unchanged from the source folder, and the split manifest "
        "is the same file.\n"
    )
    staging.rename(destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "data/fm_task_datasets_20261016")
    parser.add_argument("--output", type=Path, default=ROOT / f"data/fm_task_datasets_{DATE}_release")
    args = parser.parse_args()
    out = build(args.source, args.output)
    print(json.dumps({k: v for k, v in out.items() if k != "output_hashes"}, indent=2))
