#!/usr/bin/env python3
"""Aggregate summary/space_group_study.json: per arm, mean ± sd over seeds of every test metric, for
the last-epoch weights (what the pipeline reports) and the best-validation-loss weights.

    uv run python analysis/space_group_study_table.py summary/space_group_study.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from space_group_study import ARMS


def agg(vals):
    return (statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> None:
    src = Path(sys.argv[1])
    runs = json.loads(src.read_text())["runs"]
    by = {}
    for r in runs:
        by.setdefault(r["arm"], []).append(r)
    rows = []
    print(f"{'arm':26s} {'desc':7s} {'scale':7s} {'model':5s} {'loss':8s} {'recipe':6s} {'split':7s} {'n':>1s} {'ep':>5s} {'best':>5s} | "
          f"{'LAST acc':>9s} {'F1':>6s} {'top5':>6s} | {'BEST acc':>9s} {'F1':>6s} {'top5':>6s} {'top10':>6s} {'top30':>6s}")
    for arm in ARMS:
        rs = by.get(arm)
        if not rs:
            continue
        f = rs[0]["factors"]
        ep = agg([r["epochs_run"] for r in rs]); be = agg([r["best_epoch"] for r in rs])
        la = {k: agg([r["last"][k] for r in rs]) for k in rs[0]["last"]}
        bb = {k: agg([r["best"][k] for r in rs]) for k in rs[0]["best"]}
        rows.append({"arm": arm, "factors": f, "n": len(rs), "epochs": ep, "best_epoch": be,
                     "last": {k: {"mean": v[0], "sd": v[1]} for k, v in la.items()},
                     "best": {k: {"mean": v[0], "sd": v[1]} for k, v in bb.items()}})
        print(f"{arm:26s} {f['descriptor']:7s} {f['scaling']:7s} {f['model']:5s} {f['loss']:8s} {f['recipe']:6s} {f.get('split','dataset'):7s} {len(rs)} {ep[0]:5.0f} {be[0]:5.0f} | "
              f"{la['accuracy'][0]:.4f}±{la['accuracy'][1]:.3f} {la['macro_f1'][0]:.3f} {la['top5'][0]:.3f} | "
              f"{bb['accuracy'][0]:.4f}±{bb['accuracy'][1]:.3f} {bb['macro_f1'][0]:.3f} {bb['top5'][0]:.3f} {bb['top10'][0]:.3f} {bb['top30'][0]:.3f}")
    out = src.with_name("space_group_study_table.json")
    out.write_text(json.dumps({"arms": rows}, indent=1) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
