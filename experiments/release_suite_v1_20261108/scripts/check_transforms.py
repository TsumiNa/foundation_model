"""Gate before training: every task's configured transform must reproduce its training column.

    .venv/bin/python experiments/release_suite_v1_20261108/scripts/check_transforms.py experiments/release_suite_v1_20261108/configs/export_v1.toml

Two checks per task, both must pass (exit 1 otherwise):

1. **Export path.** The head spec ``fm export`` would write is built from the catalog: scaler or label
   encoder loads and converts to plain data, units are set, class counts match.
2. **Data.** The plain-data transform applied to the raw values reproduces the normalised column:
   - regression / curves in the same table: ``transform(raw) == normalised`` (max abs error <= 1e-9);
   - classification: ``classes[label] == str(raw)`` on every labelled row;
   - shape heads: ``raw / max|raw| == normalised`` per row;
   - NEMAD / phonix-db, whose raw values sit in a separate table keyed by first row per composition:
     the error recorded when the transform was recovered (``*_transforms.json``, PR #104).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from foundation_model.cli.main import load_raw_config
from foundation_model.data.targets import ClassLabels, normalize_max_abs
from foundation_model.hub.export import _head_spec
from foundation_model.workflows._sections import build_model_section
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

TOL = 1e-9
CATALOG_KEYS = ("data", "descriptor", "datasets", "tasks")


def as_rows(col: pd.Series) -> list[np.ndarray]:
    return [np.asarray(v, dtype=float) for v in col]


def main(path: str) -> int:
    raw_cfg = load_raw_config(path, [])
    catalog = TaskCatalog(build_task_catalog_config({k: raw_cfg[k] for k in CATALOG_KEYS if k in raw_cfg}))
    model_cfg = build_model_section(raw_cfg.get("model", {}))
    data_dir = Path(raw_cfg["datasets"][next(iter(raw_cfg["datasets"]))]["path"]).parent
    index = pd.read_csv(data_dir / "task_index.csv").set_index("task")
    tables: dict[str, pd.DataFrame] = {}
    failures = []
    for task in raw_cfg["tasks"]:
        name = task["name"]
        row = index.loc[name]
        try:
            spec = _head_spec(catalog, model_cfg, name)
        except Exception as exc:  # report every task, not just the first failure
            failures.append(name)
            print(f"FAIL {name:26s} export: {exc}")
            continue
        file = data_dir / row.file
        df = tables.setdefault(row.file, pd.read_parquet(file))
        norm_col = task["column"]
        if row.raw_file != row.file:
            record = json.loads(Path(task["scaler"]["path"]).read_text())["tasks"][task["scaler"]["key"]]
            err, how = record["max_abs_error"], "recorded at recovery (PR #104)"
        elif spec.kind == "classification":
            labeled = df[norm_col].notna()
            assert isinstance(spec.classes, ClassLabels)
            names = np.asarray(spec.classes.classes)[df.loc[labeled, norm_col].astype(int).to_numpy()]
            raw = df.loc[labeled, row.raw_column].astype(str).to_numpy()
            if name in ("is_metal", "is_gap_direct"):
                raw = df.loc[labeled, row.raw_column].astype(bool).astype(str).to_numpy()
            err, how = float((names != raw).sum()), f"mismatched labels of {labeled.sum()}"
        elif spec.output_scale == "shape":
            ok = df[norm_col].notna()
            err = max(
                float(np.max(np.abs(normalize_max_abs(r) - n)))
                for r, n in zip(as_rows(df.loc[ok, row.raw_column]), as_rows(df.loc[ok, norm_col]))
            )
            how = f"per-row max-abs over {ok.sum()} curves"
        else:
            ok = df[norm_col].notna()
            transform = spec.transform
            assert transform is not None
            if spec.kind == "kernel_regression":
                pairs = zip(as_rows(df.loc[ok, row.raw_column]), as_rows(df.loc[ok, norm_col]))
                err = max(float(np.nanmax(np.abs(transform.transform(r) - n))) for r, n in pairs)
                how = f"elementwise over {ok.sum()} curves"
            else:
                r = df.loc[ok, row.raw_column].to_numpy(dtype=float)
                err = float(np.max(np.abs(transform.transform(r) - df.loc[ok, norm_col].to_numpy(dtype=float))))
                how = f"over {ok.sum()} rows"
        status = "PASS" if err <= TOL else "FAIL"
        if status == "FAIL":
            failures.append(name)
        unit = spec.unit if spec.kind != "kernel_regression" else f"{spec.unit} vs {spec.t_unit}"
        print(f"{status} {name:26s} {spec.kind:17s} err={err:.2e} {how}; unit={unit}")
    print(f"\n{len(raw_cfg['tasks']) - len(failures)}/{len(raw_cfg['tasks'])} tasks pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
