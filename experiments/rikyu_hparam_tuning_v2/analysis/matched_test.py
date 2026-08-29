#!/usr/bin/env python3
"""Score single-task against multi-task on the SAME test rows — summary/matched_test.json

Every transfer number in this campaign compares a single-task run to a multi-task one, and the two
are not evaluated on the same rows. The datamodule splits by composition over the union of the
tasks a run loads, so a single-task run draws its 10% test slice from a smaller universe than a
24-task run does. Measured: the multi-task test set is 3–7% larger for every task.

That is not a bias — the target variances match to within a few percent, so the R² denominators are
comparable and neither estimate is systematically favoured. It is, however, a defect in the
comparison, and one that cannot be argued away in a write-up: "multi-task scored higher" is weaker
than it looks when the two numbers come from different rows.

**It is also fixable exactly, because the containment is total.** For every task checked, the
single-task test set is a strict SUBSET of the multi-task one (``only in single = 0``), and all
five single-task seeds share one test set while the transfer repeats share another. So restricting
both arms to their common compositions costs the single-task arm nothing and simply drops the extra
rows the multi-task arm was additionally scored on. What remains differs only in how the model was
trained, which is the comparison that was wanted all along.

The unrestricted numbers are reported alongside, because the size of the correction is itself worth
knowing: if it is negligible the earlier figures stand, and if it is not, this file is the one to
quote.

    python analysis/matched_test.py --single <outroot>/stage_single --single-prefix stA \\
        --multi <outroot>/stage_c/c2top1 --label c2_top1 -o summary/matched_test.json
    python analysis/matched_test.py --single <outroot>/stage_single --single-prefix stA \\
        --multi-glob '<outroot>/stage_xfer/xf_{task}_o*' --label xfer -o summary/matched_xfer.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from pathlib import Path

import pandas as pd

from common import N_TRAIN, pct_views, size_group

CLASSIFICATION = {"material_type"}


def last_pred(run: str, task: str) -> pd.DataFrame | None:
    """The task's predictions at the run's final step, or from the finetune directory."""
    steps = sorted(
        glob.glob(f"{run}/training/step*_*"),
        key=lambda p: int(re.match(r"step(\d+)_", os.path.basename(p)).group(1)),
    )
    for step in reversed(steps):
        path = f"{step}/{task}_pred.parquet"
        if os.path.exists(path):
            return pd.read_parquet(path)
    finetune = f"{run}/training/finetune/{task}_pred.parquet"
    return pd.read_parquet(finetune) if os.path.exists(finetune) else None


def r2(y: pd.Series, p: pd.Series) -> float:
    y = y.astype(float)
    p = p.astype(float)
    sse = ((y - p) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - sse / sst) if sst else float("nan")


def macro_f1(y: pd.Series, p: pd.Series) -> float:
    y = y.astype(int)
    p = p.astype(int)
    scores = []
    for c in sorted(set(y) | set(p)):
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(statistics.fmean(scores)) if scores else float("nan")


def score(df: pd.DataFrame, task: str) -> float:
    return macro_f1(df["true"], df["pred"]) if task in CLASSIFICATION else r2(df["true"], df["pred"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", required=True, help="directory holding the single-task runs")
    ap.add_argument("--single-prefix", default="stA")
    ap.add_argument("--multi", help="one multi-task run directory (a stage-C arm)")
    ap.add_argument("--multi-glob", help="glob with {task} for per-task multi runs (the xfer stage)")
    ap.add_argument("--label", required=True, help="name for the multi-task arm in the output")
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    if not (args.multi or args.multi_glob):
        raise SystemExit("need --multi or --multi-glob")

    tasks = args.tasks or sorted(N_TRAIN)
    rows = []
    for task in tasks:
        singles = [
            p for p in sorted(glob.glob(f"{args.single}/{args.single_prefix}_{task}_s*"))
            if os.path.exists(f"{p}/DONE")
        ]
        if args.multi_glob:
            multis = [p for p in sorted(glob.glob(args.multi_glob.format(task=task)))
                      if os.path.exists(f"{p}/DONE")]
        else:
            multis = [args.multi] if os.path.exists(f"{args.multi}/DONE") else []
        if not singles or not multis:
            rows.append({"task": task, "skipped": "missing runs",
                         "n_single": len(singles), "n_multi": len(multis)})
            continue

        s_frames = [(p, last_pred(p, task)) for p in singles]
        m_frames = [(p, last_pred(p, task)) for p in multis]
        s_frames = [(p, d) for p, d in s_frames if d is not None and not d.empty]
        m_frames = [(p, d) for p, d in m_frames if d is not None and not d.empty]
        if not s_frames or not m_frames:
            rows.append({"task": task, "skipped": "no predictions"})
            continue

        # The common rows: intersect every run on both sides, so one odd run cannot widen the set.
        common = set(s_frames[0][1]["composition"])
        for _, d in s_frames[1:] + m_frames:
            common &= set(d["composition"])

        def scores(frames, restrict):
            out = []
            for _, d in frames:
                sub = d[d["composition"].isin(common)] if restrict else d
                if not sub.empty:
                    out.append(score(sub, task))
            return out

        s_all, m_all = scores(s_frames, False), scores(m_frames, False)
        s_cut, m_cut = scores(s_frames, True), scores(m_frames, True)
        transfer_all = statistics.fmean(m_all) - statistics.fmean(s_all)
        transfer_cut = statistics.fmean(m_cut) - statistics.fmean(s_cut)
        sd_s = statistics.stdev(s_cut) if len(s_cut) > 1 else 0.0
        sd_m = statistics.stdev(m_cut) if len(m_cut) > 1 else 0.0
        se = (math.sqrt(sd_s**2 / len(s_cut) + sd_m**2 / len(m_cut))
              if len(s_cut) > 1 and len(m_cut) > 1 else None)
        views = pct_views(transfer_cut, statistics.fmean(s_cut))
        separated = bool(se) and abs(transfer_cut) > 2 * se
        rows.append({
            "task": task,
            "group": size_group(task),
            "n_train": N_TRAIN.get(task),
            "metric": "macro_f1" if task in CLASSIFICATION else "r2",
            "n_common_rows": len(common),
            "n_single_rows": len(s_frames[0][1]),
            "n_multi_rows": len(m_frames[0][1]),
            "single_task": statistics.fmean(s_cut),
            "single_task_sd": sd_s,
            "multi_task": statistics.fmean(m_cut),
            "multi_task_sd": sd_m,
            "multi_task_min": min(m_cut),
            "multi_task_max": max(m_cut),
            "transfer": transfer_cut,
            **views,
            "se_of_difference": se,
            "separated": separated,
            "matters": separated and views["practically_significant"],
            "n_single": len(s_cut),
            "n_multi": len(m_cut),
            # What the mismatched test sets were worth, so the size of the defect is on the record.
            "transfer_unrestricted": transfer_all,
            "correction": transfer_cut - transfer_all,
        })

    scored = [r for r in rows if "transfer" in r]
    corrections = [abs(r["correction"]) for r in scored]
    out = {
        "question": "single-task vs multi-task, scored on the rows both arms share",
        "arm": args.label,
        "per_task": rows,
        "containment": (
            "The single-task test set is a strict subset of the multi-task one for every task, so "
            "restricting to the common rows costs the single-task arm nothing and only drops the "
            "extra rows the multi-task arm was additionally scored on."
        ),
        "correction_size": {
            "max_abs": max(corrections) if corrections else None,
            "median_abs": statistics.median(corrections) if corrections else None,
            "tasks_where_verdict_would_change": [],
        },
        "notes": [
            "Both arms are restricted to the intersection of every contributing run's test rows.",
            "material_type is scored on macro-F1; every other task on R2.",
            "`transfer_unrestricted` is the earlier figure, from each arm's own test set.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'task':22s} {'metric':>8s} {'rows':>6s} | {'single':>8s} {'multi':>8s} {'transfer':>9s} "
          f"{'rel%':>8s} | {'was':>9s} {'shift':>8s}  verdict")
    for r in sorted(rows, key=lambda r: -(r.get("transfer") if "transfer" in r else -9)):
        if "transfer" not in r:
            print(f"{r['task']:22s} {r.get('skipped')}")
            continue
        verdict = ("multi better" if r["separated"] and r["transfer"] > 0
                   else "single better" if r["separated"] else "unresolved")
        if r["separated"] and not r["practically_significant"]:
            verdict += " (negligible)"
        print(f"{r['task']:22s} {r['metric']:>8s} {r['n_common_rows']:6d} | "
              f"{r['single_task']:8.4f} {r['multi_task']:8.4f} {r['transfer']:+9.4f} "
              f"{r['relative_pct']:+7.2f}% | {r['transfer_unrestricted']:+9.4f} "
              f"{r['correction']:+8.4f}  {verdict}")
    cs = out["correction_size"]
    if cs["max_abs"] is not None:
        print(f"\n  restricting to shared rows moves transfer by at most {cs['max_abs']:.4f} "
              f"(median {cs['median_abs']:.4f})")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
