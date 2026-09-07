#!/usr/bin/env python3
"""Descriptor contrast on the extensive properties, single task — summary/descriptor.json

volume, final_energy and dos_density sit at R2 0.62-0.77 alone and are the tasks warm-start loses
on. KMD is built from atomic FRACTIONS, so cell scale never reaches the model, and these are the
labels that depend on it. Three single-task arms, same recipe (probe6 + the adopted values, 5 seeds):

  kmd      the campaign's descriptor (stage_single, stA / stB — the adopted ceilings)
  classic  composition descriptor: weighted sum + average + variance + max + min (stage_desc, stDc)
  nosum    the same without the weighted-sum block, the one block that carries scale (stDn)

If classic beats kmd and nosum does not, scale-blindness is what caps these tasks. If both beat
kmd, it is the descriptor family; if neither does, the ceiling is elsewhere.

    python analysis/descriptor.py --desc <outroot>/stage_desc --ceilings summary/ceilings_adopted_v2.json \\
        -o summary/descriptor.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from common import N_TRAIN, final_metrics
from ft import diff

TASKS = ["volume", "final_energy", "dos_density"]
ARMS = {"classic": "stDc", "nosum": "stDn"}


def collect(root: Path, prefix: str, task: str) -> tuple[list[float], list[int]]:
    vals, epochs = [], []
    for run in sorted(root.glob(f"{prefix}_{task}_s*")):
        if not (run / "DONE").exists():
            continue
        m, _ = final_metrics(run)
        v = m.get(task, {}).get("r2")
        if v is not None:
            vals.append(float(v))
            logs = sorted(run.glob("**/metrics.csv"))
            if logs:
                import csv
                ep = [int(float(r["epoch"])) for r in csv.DictReader(open(logs[0])) if r.get("epoch") not in (None, "")]
                epochs.append(max(ep) + 1 if ep else 0)
    return vals, epochs


def stats(vals):
    if not vals:
        return None
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals), "values": vals}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desc", type=Path, required=True, help="<outroot>/stage_desc")
    ap.add_argument("--ceilings", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    ceil = json.loads(args.ceilings.read_text())

    rows = []
    for task in TASKS:
        base = ceil[task]
        row = {"task": task, "n_train": N_TRAIN[task],
               "kmd": {"mean": base["mean"], "sd": base["sd"], "n": base["n"]}}
        arms = {}
        for key, prefix in ARMS.items():
            vals, epochs = collect(args.desc, prefix, task)
            arms[key] = vals
            row[key] = stats(vals)
            row[f"{key}_epochs"] = statistics.median(epochs) if epochs else None
            row[f"{key}_vs_kmd"] = diff(vals, base["mean"], base["sd"], base["n"])
        if arms["classic"] and len(arms["nosum"]) > 1:
            row["classic_vs_nosum"] = diff(arms["classic"], statistics.fmean(arms["nosum"]),
                                           statistics.stdev(arms["nosum"]), len(arms["nosum"]))
        else:
            row["classic_vs_nosum"] = None
        rows.append(row)

    out = {"question": "is the ~0.7 R2 ceiling on the extensive properties the descriptor's scale-blindness?",
           "arms": {"kmd": "KMD on atomic fractions (adopted ceilings)",
                    "classic": "composition descriptor, sum + average + variance + max + min",
                    "nosum": "composition descriptor without the weighted-sum block"},
           "per_task": rows,
           "notes": ["Single task, 5 seeds per arm, probe6 recipe with the adopted values; early stopping on, "
                     "150-epoch cap — the same recipe as the KMD baselines.",
                     "Descriptor tables from scripts/make_comp_descriptors.py, z-scored over all qc compositions."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    def pc(d):
        if not d:
            return f"{'-':>9s}"
        star = "*" if d["matters"] else ("·" if d["separated"] else " ")
        return f"{d['relative_pct']:+7.1f}%{star}"

    def v(s):
        return f"{s['mean']:6.4f}" if s else f"{'-':>6s}"

    print(f"{'task':14s} {'N':>6s} {'kmd':>6s} | {'classic':>7s} {'vs kmd':>9s} | {'nosum':>6s} {'vs kmd':>9s} | {'classic-nosum':>13s} | epochs c/n")
    for r in rows:
        print(f"{r['task']:14s} {r['n_train']:6d} {v(r['kmd'])} | {v(r['classic']):>7s} {pc(r['classic_vs_kmd'])} | "
              f"{v(r['nosum']):>6s} {pc(r['nosum_vs_kmd'])} | {pc(r['classic_vs_nosum']):>13s} | "
              f"{r['classic_epochs'] or '-'} / {r['nosum_epochs'] or '-'}")
    print("\n  R2, last-epoch weights; vs = relative % with both arms' SE; * separated and |delta| >= 0.01   · separated only")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
