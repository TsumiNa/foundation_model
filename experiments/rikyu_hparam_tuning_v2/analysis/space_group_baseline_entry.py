#!/usr/bin/env python3
"""The adopted space-group baseline in the per-task schema of baselines_mp2026.json — summary/space_group_baseline_adopted.json

Built from the stW single-task runs (5 seeds; summary/space_group_confirm.json) and the seed-2025
prediction file (scratchpad sgpred/stW.parquet, pulled from RIKYU stage_single_mp2026/stW_space_group_s2025).
The labels page swaps this entry in for the original stN entry, so baselines_mp2026.json is left as recorded.

    uv run python analysis/space_group_baseline_entry.py --pred <sgpred/stW.parquet>
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
S = HERE / "summary"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True); ap.add_argument("-o", "--out", type=Path, default=S / "space_group_baseline_adopted.json")
    a = ap.parse_args()
    base = next(r for r in json.loads((S / "baselines_mp2026.json").read_text())["per_task"] if r["task"] == "space_group")
    runs = json.loads((S / "space_group_confirm.json").read_text())["arms"]["plain_kmd"]["runs"]
    classes = base["classes"]; counts = base["class_counts"]
    df = pd.read_parquet(a.pred); t = df["true"].to_numpy(int); p = df["pred"].to_numpy(int)
    assert len(df) == base["n_test"], (len(df), base["n_test"])
    per_class = []
    for i, c in enumerate(classes):
        tp = int(((t == i) & (p == i)).sum()); nt = int((t == i).sum()); npred = int((p == i).sum())
        rec = tp / nt if nt else 0.0; prec = tp / npred if npred else 0.0
        per_class.append({"class": c, "n_test": nt, "recall": rec, "precision": prec, "f1": (2 * rec * prec / (rec + prec)) if (rec + prec) else 0.0})
    top = sorted(range(len(classes)), key=lambda i: -counts[classes[i]])[:12]
    fold = {i: k for k, i in enumerate(top)}; tf = np.array([fold.get(x, 12) for x in t]); pf = np.array([fold.get(x, 12) for x in p])
    matrix = [[int(((tf == i) & (pf == j)).sum()) for j in range(13)] for i in range(13)]
    mf = [r["macro_f1"] for r in runs]; acc = [r["accuracy"] for r in runs]
    entry = dict(base)
    entry.update({
        "source": "stW_space_group_s2025..s2029 (stage_single_mp2026); seed 2025 for per_class and confusion_top",
        "per_seed": [{"run": r["run"], "r2": None, "mae": None, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "samples": base["n_test"], "primary": r["accuracy"]} for r in runs],
        "macro_f1": {"mean": st.fmean(mf), "sd": st.stdev(mf), "n": len(mf), "values": mf},
        "accuracy": {"mean": st.fmean(acc), "sd": st.stdev(acc), "n": len(acc), "values": acc},
        "per_class": per_class,
        "confusion_top": {"matrix": matrix, "labels": [classes[i] for i in top] + ["other"], "n": int(len(df))},
    })
    a.out.write_text(json.dumps(entry) + "\n")
    print(f"macro-F1 {entry['macro_f1']['mean']:.4f} ± {entry['macro_f1']['sd']:.4f}; accuracy {entry['accuracy']['mean']:.4f} ± {entry['accuracy']['sd']:.4f}; seed-2025 accuracy {(t == p).mean():.4f}")
    print("recall of the twelve largest groups (seed 2025):")
    for i in top:
        pc = per_class[i]; print(f"  {classes[i]:10s} n_all {counts[classes[i]]:5d}  n_test {pc['n_test']:4d}  recall {pc['recall']:.2f}  precision {pc['precision']:.2f}  f1 {pc['f1']:.2f}")
    print("groups with F1 > 0:", sum(1 for c in per_class if c["f1"] > 0), "of", len(per_class))
    print(a.out)


if __name__ == "__main__":
    main()
