#!/usr/bin/env python3
"""Single-task baselines on the 2026-09-12 dataset — summary/baselines_mp2026.json

The nine Materials Project tasks whose labels changed in the rebuild and the sixteen properties added
with it, each trained alone with the stage_single recipe (probe6_mp2026.toml + the adopted values,
KMD, five seeds, early stopping). Per task: the primary metric (R² for regression, macro-F1 and
accuracy for classification) as mean ± sd over seeds, MAE, the test-row count, the 2026-05-15 ceiling
where the task existed before, and — for the page's scatter plots — the test predictions of one seed
(regression: a 2,000-row sample of (true, pred); classification: the confusion matrix).

Runs on RIKYU (stdlib only for the metrics; the prediction parquets are read with pandas, so run the
--preds step where pandas exists, or pass --no-preds and pull them separately).

    PYTHONPATH=analysis python3 analysis/baselines_mp2026.py --runs <outroot>/stage_single_mp2026 \\
        --ceilings summary/ceilings_adopted_v2.json -o summary/baselines_mp2026.json [--preds]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

UPDATED = ["final_energy", "formation_energy", "volume", "density", "total_magnetization", "efermi",
           "dielectric_total", "dielectric_ionic", "dielectric_electronic"]
NEW = ["band_gap", "density_atomic", "magnetization_per_volume", "magnetization_per_fu", "reaction_energy",
       "cbm", "vbm", "bulk_modulus", "shear_modulus", "poisson_ratio", "universal_anisotropy",
       "refractive_index", "piezoelectric_max", "magnetic_ordering", "is_metal", "is_gap_direct", "space_group"]
CLASSIFICATION = {"magnetic_ordering": ["AFM", "FM", "FiM", "NM"], "is_metal": ["no", "yes"], "is_gap_direct": ["no", "yes"], "space_group": None}
SPACE_GROUP_CLASSES = Path(__file__).resolve().parents[1] / "summary" / "space_group_classes.json"
TOP_K = 12   # classes shown individually in the space-group confusion matrix; the rest fold into "other"
SEED_FOR_PREDS = 2025


def stats(vals):
    if not vals:
        return None
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals), "values": vals}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--ceilings", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--preds", action="store_true", help="also read seed-2025 test predictions (needs pandas)")
    args = ap.parse_args()
    ceil = json.loads(args.ceilings.read_text())

    rows = []
    for task in UPDATED + NEW:
        kind = "classification" if task in CLASSIFICATION else "regression"
        per_seed = []
        for run in sorted(args.runs.glob(f"stN_{task}_s*")):
            if not (run / "DONE").exists():
                continue
            steps = sorted(run.glob(f"training/step*_{task}"))
            if not steps:
                continue
            mfile = steps[-1] / f"{task}_metrics.json"
            if not mfile.exists():
                continue
            mtr = json.loads(mfile.read_text())
            per_seed.append({"run": run.name, **{k: mtr.get(k) for k in ("r2", "mae", "accuracy", "macro_f1", "samples", "primary")}})
        if not per_seed:
            rows.append({"task": task, "kind": kind, "group": "updated" if task in UPDATED else "new", "n_seeds": 0})
            continue
        row = {"task": task, "kind": kind, "group": "updated" if task in UPDATED else "new", "n_seeds": len(per_seed),
               "n_test": per_seed[0]["samples"], "per_seed": per_seed}
        if kind == "regression":
            row["r2"] = stats([s["r2"] for s in per_seed if s["r2"] is not None])
            row["mae"] = stats([s["mae"] for s in per_seed if s["mae"] is not None])
            old = ceil.get(task)
            row["old_ceiling"] = {"mean": old["mean"], "sd": old["sd"], "n": old["n"]} if old else None
        else:
            row["macro_f1"] = stats([s["macro_f1"] for s in per_seed if s["macro_f1"] is not None])
            row["accuracy"] = stats([s["accuracy"] for s in per_seed if s["accuracy"] is not None])
            if task == "space_group":
                sg = json.loads(SPACE_GROUP_CLASSES.read_text())
                row["classes"] = sg["classes"]; row["class_counts"] = sg["counts"]
            else:
                row["classes"] = CLASSIFICATION[task]
        if args.preds:
            import numpy as np
            import pandas as pd
            run = args.runs / f"stN_{task}_s{SEED_FOR_PREDS}"
            steps = sorted(run.glob(f"training/step*_{task}"))
            pf = steps[-1] / f"{task}_pred.parquet" if steps else None
            if pf and pf.exists():
                df = pd.read_parquet(pf)
                if kind == "regression":
                    t, p = df["true"].to_numpy(float), df["pred"].to_numpy(float)
                    rng = np.random.default_rng(0)
                    idx = rng.choice(len(t), min(2000, len(t)), replace=False)
                    row["scatter"] = {"true": [round(float(x), 4) for x in t[idx]], "pred": [round(float(x), 4) for x in p[idx]],
                                      "n": int(len(t)), "r2_seed": float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())}
                else:
                    t, p = df["true"].to_numpy(int), df["pred"].to_numpy(int)
                    classes = row["classes"]; k = len(classes)
                    if task == "space_group":
                        # per-class precision / recall / F1 over all classes, and a confusion over the TOP_K most
                        # common classes with everything else folded into "other"
                        per = []
                        for c in range(k):
                            tp = int(((t == c) & (p == c)).sum()); fn = int(((t == c) & (p != c)).sum()); fp = int(((t != c) & (p == c)).sum())
                            rec = tp / (tp + fn) if tp + fn else None; prec = tp / (tp + fp) if tp + fp else None
                            f1 = (2 * prec * rec / (prec + rec)) if prec and rec else 0.0
                            per.append({"class": classes[c], "n_test": tp + fn, "recall": rec, "precision": prec, "f1": f1})
                        row["per_class"] = per
                        order = sorted(range(k), key=lambda c: -row["class_counts"][classes[c]])[:TOP_K]
                        fold = lambda x: order.index(x) if x in order else TOP_K
                        tf, pf = [fold(x) for x in t], [fold(x) for x in p]
                        cm = [[sum(1 for a, b in zip(tf, pf) if a == i and b == j) for j in range(TOP_K + 1)] for i in range(TOP_K + 1)]
                        row["confusion_top"] = {"matrix": cm, "labels": [classes[c] for c in order] + ["other"], "n": int(len(t))}
                    else:
                        cm = [[int(((t == i) & (p == j)).sum()) for j in range(k)] for i in range(k)]
                        row["confusion"] = {"matrix": cm, "n": int(len(t))}
        rows.append(row)

    out = {"question": "single-task baselines on the 2026-09-12 dataset: the nine relabelled tasks and the sixteen added properties",
           "recipe": "probe6_mp2026.toml + adopted values via --set, KMD, five seeds, early stopping, last-epoch weights",
           "per_task": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")

    print(f"{'task':26s} {'kind':14s} {'seeds':>5s} {'n_test':>6s} {'metric':>16s} {'MAE':>8s} {'old ceiling':>12s}")
    for r in rows:
        if r["n_seeds"] == 0:
            print(f"{r['task']:26s} {r['kind']:14s} {'0':>5s}")
            continue
        if r["kind"] == "regression":
            old = r["old_ceiling"]
            print(f"{r['task']:26s} {r['kind']:14s} {r['n_seeds']:5d} {r['n_test']:6d} {r['r2']['mean']:8.4f} ± {r['r2']['sd']:.4f} {r['mae']['mean']:8.4f} {(f'{old['mean']:.4f}' if old else '-'):>12s}")
        else:
            print(f"{r['task']:26s} {r['kind']:14s} {r['n_seeds']:5d} {r['n_test']:6d} F1 {r['macro_f1']['mean']:.4f} ± {r['macro_f1']['sd']:.4f}  acc {r['accuracy']['mean']:.4f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
