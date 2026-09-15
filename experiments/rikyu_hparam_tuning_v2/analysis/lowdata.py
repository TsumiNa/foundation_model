#!/usr/bin/env python3
"""Low-data learning curves — summary/lowdata.json

For every task, fraction of training labels kept (5 / 10 / 25 / 50 %, plus the full-data point from
the existing runs) and arm (alone = stL_*, warm-start = ftL_*), the per-seed primary metric and the
paired difference on the same subsample seed. material_type additionally gets the 3-class macro-F1
(QC / AC / others) computed from the prediction files, because its 5-class macro-F1 rides on a 1-row
and a 3-row test class.

Runs on RIKYU (stdlib):
    python3 analysis/lowdata.py --single <outroot>/stage_single_mp2026 --ft <outroot>/stage_ft --summary summary -o summary/lowdata.json
Then pull the JSON (and, for material_type, the prediction parquets listed in it).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

TASKS = ["material_type", "band_gap", "density_atomic", "magnetization_per_volume", "magnetization_per_fu", "reaction_energy", "cbm", "vbm",
         "bulk_modulus", "shear_modulus", "poisson_ratio", "universal_anisotropy", "refractive_index", "piezoelectric_max", "magnetic_ordering",
         "is_metal", "is_gap_direct", "space_group"]
CLF = {"material_type", "magnetic_ordering", "is_metal", "is_gap_direct", "space_group"}
FRACS = {"material_type": [10, 25, 50]}
DEFAULT_FRACS = [5, 10, 25, 50]


def metric_file(run: Path, task: str, mode: str):
    if mode == "alone":
        steps = sorted(run.glob(f"training/step*_{task}"))
        return (steps[-1] / f"{task}_metrics.json") if steps else None
    return run / "training" / "finetune" / f"{task}_metrics.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", type=Path, required=True); ap.add_argument("--ft", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True); ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    base = {r["task"]: r for r in json.loads((a.summary / "baselines_mp2026.json").read_text())["per_task"]}
    cw = json.loads((a.summary / "classification_weights.json").read_text())["tasks"]
    sg = json.loads((a.summary / "space_group_confirm.json").read_text())["arms"]["plain_kmd"]["runs"]
    mt = json.loads((a.summary / "material_type_weights.json").read_text())["arms"]["none"]["runs"]
    added = {r["task"]: r for r in json.loads((a.summary / "added_transfer.json").read_text())["per_task"]}
    mtw = json.loads((a.summary / "material_type_warmstart_none.json").read_text())["arms"]
    out = {"tasks": {}}
    for task in TASKS:
        key = "macro_f1" if task in CLF else "r2"
        # full-data points from the existing runs
        if task == "material_type":
            full_alone = [r["macro_f1"] for r in mt]; full_warm = [r["after"]["macro_f1"] for r in mtw["warm_unseen"]]
        elif task == "space_group":
            full_alone = [r["macro_f1"] for r in sg]; full_warm = added[task]["warm"]["values"] if added[task].get("warm") else []
        elif task in CLF:
            full_alone = [r["macro_f1"] for r in cw[task]["arms"]["none"]["runs"]]; full_warm = added[task]["warm"]["values"] if added[task].get("warm") else []
        else:
            full_alone = [s["r2"] for s in base[task]["per_seed"]]; full_warm = added[task]["warm"]["values"] if added[task].get("warm") else []
        points = {100: {"alone": full_alone, "warm": full_warm, "pred": {}}}
        for f in FRACS.get(task, DEFAULT_FRACS):
            pt = {"alone": [], "warm": [], "pred": {"alone": [], "warm": []}, "epochs": {"alone": [], "warm": []}}
            for s in range(3):
                for mode, root, prefix in (("alone", a.single, "stL"), ("warm", a.ft, "ftL")):
                    run = root / f"{prefix}_{task}_f{f:02d}_s{s}"
                    mf = metric_file(run, task, mode)
                    if not (run / "DONE").exists() or mf is None or not mf.exists():
                        pt[mode].append(None); continue
                    m = json.loads(mf.read_text()); pt[mode].append(m[key])
                    pt["pred"][mode].append(str(mf.with_name(f"{task}_pred.parquet")))
                    if mode == "alone":
                        c = list(run.glob("logs/step*/version_0/metrics.csv")); pt["epochs"][mode].append(sum(1 for line in open(c[0]) if ",," not in line[:1]) if c else None)
                    else:
                        sm = run / "training" / "finetune_summary.json"; pt["epochs"][mode].append(json.loads(sm.read_text()).get("epochs_run") if sm.exists() else None)
            points[f] = pt
        out["tasks"][task] = {"metric": key, "points": points}
    a.out.write_text(json.dumps(out) + "\n")
    print(f"{'task':26s} " + " ".join(f"{'f' + str(f) + ' alone→warm':>20s}" for f in DEFAULT_FRACS) + f" {'full alone→warm':>20s}")
    for task, d in out["tasks"].items():
        cells = []
        for f in DEFAULT_FRACS:
            pt = d["points"].get(f)
            if not pt:
                cells.append(f"{'—':>20s}"); continue
            al = [x for x in pt["alone"] if x is not None]; wm = [x for x in pt["warm"] if x is not None]
            cells.append(f"{(st.fmean(al) if al else float('nan')):.3f}→{(st.fmean(wm) if wm else float('nan')):.3f} ({len(al)}/{len(wm)})".rjust(20))
        fa = d["points"][100]["alone"]; fw = d["points"][100]["warm"]
        print(f"{task:26s} " + " ".join(cells) + f" {st.fmean(fa):.3f}→{(st.fmean(fw) if fw else float('nan')):.3f}".rjust(21))


if __name__ == "__main__":
    main()
