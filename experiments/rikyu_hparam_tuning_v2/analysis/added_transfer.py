#!/usr/bin/env python3
"""Transfer to the 17 added Materials Project tasks — summary/added_transfer.json

Warm-start (stage_ft ftA_<task>_e0..e4: encoder from five 24-task library checkpoints that never saw
the task, fresh head, encoder + head trained, class weights off) against the from-scratch baselines
on the same 2026-09-11 dataset (baselines_mp2026.json for regression; the class-weights-off runs for
the classification heads). Per task: mean ± sd of both arms, Δ, relative %, 2×SE of the difference
(both arms' SE), verdict (separated at 2×SE and |Δ| ≥ 0.01), epochs run.

Runs on RIKYU (stdlib only):
    python3 analysis/added_transfer.py --runs <outroot>/stage_ft --summary summary -o summary/added_transfer.json
Then pull the JSON.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ADDED = ["band_gap", "density_atomic", "magnetization_per_volume", "magnetization_per_fu", "reaction_energy", "cbm", "vbm", "bulk_modulus",
         "shear_modulus", "poisson_ratio", "universal_anisotropy", "refractive_index", "piezoelectric_max", "magnetic_ordering", "is_metal",
         "is_gap_direct", "space_group"]
CLF = {"magnetic_ordering", "is_metal", "is_gap_direct", "space_group"}


def stats(v):
    return {"mean": st.fmean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "n": len(v), "values": v}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True); ap.add_argument("--summary", type=Path, required=True); ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    base = {r["task"]: r for r in json.loads((a.summary / "baselines_mp2026.json").read_text())["per_task"]}
    cw = json.loads((a.summary / "classification_weights.json").read_text())["tasks"]
    sg = json.loads((a.summary / "space_group_confirm.json").read_text())["arms"]["plain_kmd"]["runs"]
    rows = []
    for t in ADDED:
        metric = "macro_f1" if t in CLF else "r2"
        if t == "space_group":
            alone = [r["macro_f1"] for r in sg]; alone_acc = [r["accuracy"] for r in sg]
        elif t in CLF:
            alone = [r["macro_f1"] for r in cw[t]["arms"]["none"]["runs"]]; alone_acc = [r["accuracy"] for r in cw[t]["arms"]["none"]["runs"]]
        else:
            alone = [s["r2"] for s in base[t]["per_seed"]]; alone_acc = None
        warm, warm_acc, epochs = [], [], []
        for k in range(5):
            d = a.runs / f"ftA_{t}_e{k}"
            m = d / "training" / "finetune" / f"{t}_metrics.json"
            if not (d / "DONE").exists() or not m.exists():
                continue
            mt = json.loads(m.read_text()); warm.append(mt[metric]); warm_acc.append(mt.get("accuracy"))
            s = d / "training" / "finetune_summary.json"
            epochs.append(json.loads(s.read_text()).get("epochs_run") if s.exists() else None)
        row = {"task": t, "kind": "classification" if t in CLF else "regression", "metric": metric, "alone": stats(alone), "warm": stats(warm) if warm else None,
               "epochs": [e for e in epochs if e is not None]}
        if alone_acc:
            row["alone_accuracy"] = st.fmean(alone_acc)
        if warm and any(x is not None for x in warm_acc):
            row["warm_accuracy"] = st.fmean(x for x in warm_acc if x is not None)
        if warm:
            d = row["warm"]["mean"] - row["alone"]["mean"]
            se = ((row["warm"]["sd"] ** 2 / len(warm)) + (row["alone"]["sd"] ** 2 / len(alone))) ** 0.5
            row["delta"] = d; row["relative_pct"] = d / abs(row["alone"]["mean"]) * 100 if row["alone"]["mean"] else None
            row["se_of_difference"] = se; row["separated"] = abs(d) > 2 * se; row["practically_significant"] = abs(d) >= 0.01
            row["verdict"] = ("better" if d > 0 else "worse") if row["separated"] and row["practically_significant"] else "unresolved"
        rows.append(row)
    counts = {"better": sum(r.get("verdict") == "better" for r in rows), "worse": sum(r.get("verdict") == "worse" for r in rows), "unresolved": sum(r.get("verdict") == "unresolved" for r in rows)}
    out = {"question": "warm-start from the 24-task library encoders (never saw the task) vs training alone, 17 added MP tasks, 2026-09-11 dataset, class weights off",
           "per_task": rows, "counts": counts}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out) + "\n")
    print(f"{'task':26s} {'metric':8s} {'alone (5)':>16s} {'warm-start (n)':>18s} {'Δ':>8s} {'rel %':>7s} {'2×SE':>7s} verdict  epochs")
    for r in rows:
        if not r["warm"]:
            print(f"{r['task']:26s} {r['metric']:8s} {r['alone']['mean']:8.4f} ± {r['alone']['sd']:.4f}   (no runs yet)"); continue
        print(f"{r['task']:26s} {r['metric']:8s} {r['alone']['mean']:8.4f} ± {r['alone']['sd']:.4f} {r['warm']['mean']:8.4f} ± {r['warm']['sd']:.4f} ({r['warm']['n']}) {r['delta']:+8.4f} {r['relative_pct']:+7.1f} {2 * r['se_of_difference']:7.4f} {r['verdict']:10s} {st.fmean(r['epochs']) if r['epochs'] else float('nan'):.0f}")
    print("counts:", counts)


if __name__ == "__main__":
    main()
