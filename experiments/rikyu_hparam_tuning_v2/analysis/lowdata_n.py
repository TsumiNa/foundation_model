#!/usr/bin/env python3
"""Low-data learning curves at fixed label counts — summary/lowdata_n.json

For every task in summary/lowdata_n_plan.json, every count (100 / 300 / 1,000 / 3,000 / 10,000 labelled
training rows) and arm (alone = stLn_*, warm-start = ftLn_*), the per-seed primary metric, the rows
actually kept (from the data/lowdata_n MANIFEST: classification keeps at least one row per class, so
space_group at "100" has ~190) and the share of the task's labelled training rows. The full-data point
comes from the existing runs, as in lowdata.py.

Runs on RIKYU (stdlib):
    python3 analysis/lowdata_n.py --single <outroot>/stage_single_mp2026 --ft <outroot>/stage_ft \\
        --summary summary --manifest data/lowdata_n/MANIFEST.json -o summary/lowdata_n.json
Then pull the JSON (and, for material_type, the prediction parquets listed in it for the 3-class F1).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

CLF = {"material_type", "magnetic_ordering", "is_metal", "is_gap_direct", "space_group"}


def metric_file(run: Path, task: str, mode: str):
    if mode == "alone":
        steps = sorted(run.glob(f"training/step*_{task}"))
        return (steps[-1] / f"{task}_metrics.json") if steps else None
    return run / "training" / "finetune" / f"{task}_metrics.json"


def full_data_points(summary: Path):
    base = {r["task"]: r for r in json.loads((summary / "baselines_mp2026.json").read_text())["per_task"]}
    cw = json.loads((summary / "classification_weights.json").read_text())["tasks"]
    sg = json.loads((summary / "space_group_confirm.json").read_text())["arms"]["plain_kmd"]["runs"]
    mt = json.loads((summary / "material_type_weights.json").read_text())["arms"]["none"]["runs"]
    added = {r["task"]: r for r in json.loads((summary / "added_transfer.json").read_text())["per_task"]}
    mtw = json.loads((summary / "material_type_warmstart_none.json").read_text())["arms"]

    def point(task):
        if task == "material_type":
            return [r["macro_f1"] for r in mt], [r["after"]["macro_f1"] for r in mtw["warm_unseen"]]
        warm = added[task]["warm"]["values"] if added.get(task, {}).get("warm") else []
        if task == "space_group":
            return [r["macro_f1"] for r in sg], warm
        if task in CLF:
            return [r["macro_f1"] for r in cw[task]["arms"]["none"]["runs"]], warm
        return [s["r2"] for s in base[task]["per_seed"]], warm
    return point


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", type=Path, required=True); ap.add_argument("--ft", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True); ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads((a.summary / "lowdata_n_plan.json").read_text()); man = json.loads(a.manifest.read_text())
    kept = {}
    for f in man["files"]:
        for t, v in f["kept"].items():
            kept.setdefault(t, {}).setdefault(f["count"], []).append(v["train_kept"])
    full = full_data_points(a.summary)
    out = {"mode": "count", "seeds": plan["seeds"], "tasks": {}}
    for task, counts in plan["plan"].items():
        key = "macro_f1" if task in CLF else "r2"; n_all = plan["train_rows"][task]
        fa, fw = full(task)
        points = {"full": {"rows": n_all, "share": 1.0, "alone": fa, "warm": fw, "pred": {}}}
        for n in counts:
            pt = {"rows": st.fmean(kept[task][n]), "rows_per_seed": kept[task][n], "share": st.fmean(kept[task][n]) / n_all,
                  "alone": [], "warm": [], "pred": {"alone": [], "warm": []}, "epochs": {"alone": [], "warm": []}}
            for s in plan["seeds"]:
                for mode, root, prefix in (("alone", a.single, "stLn"), ("warm", a.ft, "ftLn")):
                    run = root / f"{prefix}_{task}_n{n:05d}_s{s}"
                    mf = metric_file(run, task, mode)
                    if not (run / "DONE").exists() or mf is None or not mf.exists():
                        pt[mode].append(None); continue
                    m = json.loads(mf.read_text()); pt[mode].append(m[key])
                    pt["pred"][mode].append(str(mf.with_name(f"{task}_pred.parquet")))
                    if mode == "alone":
                        c = list(run.glob("logs/step*/version_0/metrics.csv")); pt["epochs"][mode].append(sum(1 for line in open(c[0]) if ",," not in line[:1]) if c else None)
                    else:
                        sm = run / "training" / "finetune_summary.json"; pt["epochs"][mode].append(json.loads(sm.read_text()).get("epochs_run") if sm.exists() else None)
            points[str(n)] = pt
        out["tasks"][task] = {"metric": key, "train_rows": n_all, "points": points}
    a.out.write_text(json.dumps(out) + "\n")
    print(f"{'task':26s} " + " ".join(f"{n:>21d}" for n in plan["counts"]) + f" {'full':>21s}")
    for task, d in out["tasks"].items():
        cells = []
        for n in plan["counts"]:
            pt = d["points"].get(str(n))
            if not pt:
                cells.append(f"{'—':>21s}"); continue
            al = [x for x in pt["alone"] if x is not None]; wm = [x for x in pt["warm"] if x is not None]
            cells.append(f"{(st.fmean(al) if al else float('nan')):.3f}→{(st.fmean(wm) if wm else float('nan')):.3f} ({len(al)}/{len(wm)})".rjust(21))
        fa = d["points"]["full"]["alone"]; fw = d["points"]["full"]["warm"]
        print(f"{task:26s} " + " ".join(cells) + f" {st.fmean(fa):.3f}→{(st.fmean(fw) if fw else float('nan')):.3f}".rjust(22))


if __name__ == "__main__":
    main()
