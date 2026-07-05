#!/usr/bin/env python3
"""Compare replay-sweep runs: final primary metric per task, vs each other and the reference.

Usage:
    python compare_sweep.py [RESULTS_DIR]

RESULTS_DIR holds one long-format metrics_table.csv per run, named mt_<tag>.csv
(tags: base0p05, 0p10, 0p15, 0p20, n100, n200, n500, n1000, n1500, n2000, n2500).
Rsync each run's
<output_dir>/training/metrics_table.csv into RESULTS_DIR/mt_<tag>.csv first, e.g.:

    rsync -avz rikyu-login:/home/ea0094/projects/foundation_model/artifacts/replay_sweep/replay_n100_rikyu/training/metrics_table.csv results/mt_n100.csv

Long format columns: step,new_task,epochs_run,task,r2,mae,samples,primary,points,accuracy,macro_f1
final metric per task = the row at the max step; primary = R2 (reg/kr) or accuracy (clf).
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results"
# committed reference metrics (artifacts/ itself is gitignored, so we ship a copy)
REF = HERE.parent / "reference/metrics_table.csv"

ORDER = ["density","efermi","final_energy","total_magnetization","volume","dielectric_total",
         "dielectric_ionic","dielectric_electronic","magnetization","curie","neel","kp",
         "magnetic_susceptibility","zt","power_factor","thermal_conductivity","electrical_resistivity",
         "dos_density","seebeck","formation_energy","magnetic_moment","tc","klat","material_type"]
REG_LIKE = [t for t in ORDER if t != "material_type"]
TAG_ORDER = ["base0p05","0p10","0p15","0p20","n100","n200","n500","n1000","n1500","n2000","n2500"]
TAG_LABEL = {"base0p05":"0.05*","0p10":"0.10","0p15":"0.15","0p20":"0.20",
             "n100":"100","n200":"200","n500":"500","n1000":"1000",
             "n1500":"1500","n2000":"2000","n2500":"2500"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_run_final(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {}
    max_step = max(int(r["step"]) for r in rows)
    return {r["task"]: fnum(r["primary"]) for r in rows if int(r["step"]) == max_step}


def load_ref_final(path):
    return {r["task"]: fnum(r["final"]) for r in csv.DictReader(open(path))}


ref = load_ref_final(REF) if REF.exists() else {}
runs = {t: load_run_final(RESULTS / f"mt_{t}.csv") for t in TAG_ORDER if (RESULTS / f"mt_{t}.csv").exists()}
cols = [t for t in TAG_ORDER if t in runs]


def fmt(v):
    return "   -   " if v is None else f"{v:7.3f}"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


hdr = f"{'task':22} {'ref':>7}" + "".join(f" {TAG_LABEL[t]:>7}" for t in cols)
print(hdr); print("-" * len(hdr))
for task in ORDER:
    print(f"{task:22} {fmt(ref.get(task))}" + "".join(f" {fmt(runs[t].get(task))}" for t in cols))
print("-" * len(hdr))
print(f"{'MEAN reg/kr R2':22} {fmt(mean([ref.get(t) for t in REG_LIKE]))}" +
      "".join(f" {fmt(mean([runs[t].get(x) for x in REG_LIKE]))}" for t in cols))
print(f"{'material_type acc':22} {fmt(ref.get('material_type'))}" +
      "".join(f" {fmt(runs[t].get('material_type'))}" for t in cols))
print("\n* 0.05 baseline also carries per_task=0.10 on the fixed tail; sweep runs are uniform.")
