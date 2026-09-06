#!/usr/bin/env python3
"""Replace the single-task ceiling of tasks whose original runs hit the epoch cap.

An audit of the 120 single-task baselines found power_factor capped at 150 epochs in 5 of 5 seeds
and still improving, and seebeck capped in 4 of 5. Their stA_ ceilings are therefore biased low.
The stB_ reruns differ only in training.max_epochs (150 -> 400), so early stopping decides.

This writes a NEW file rather than editing ceilings_adopted.json in place: the original is what
every published number was computed against, and the diff between the two files is itself a result.

    python analysis/patch_ceilings.py --runs <outroot>/stage_single --tasks power_factor seebeck \\
        --base summary/ceilings_adopted.json -o summary/ceilings_adopted_v2.json
"""
import argparse, glob, json, math, os, statistics
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--runs", required=True)
ap.add_argument("--tasks", nargs="+", required=True)
ap.add_argument("--prefix", default="stB")
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("-o", "--out", type=Path, required=True)
args = ap.parse_args()

ceil = json.loads(args.base.read_text())
ceil.setdefault("_patched", {})
for task in args.tasks:
    vals = []
    for run in sorted(glob.glob(f"{args.runs}/{args.prefix}_{task}_s*")):
        if not os.path.exists(f"{run}/DONE"):
            continue
        m = f"{run}/training/step01_{task}/{task}_metrics.json"
        if not os.path.exists(m):
            continue
        d = json.load(open(m))
        v = d.get("macro_f1") if "macro_f1" in d else d.get("r2")
        if v is not None:
            vals.append(float(v))
    if len(vals) < 2:
        raise SystemExit(f"{task}: only {len(vals)} finished {args.prefix} run(s); refusing to patch on that")
    old = ceil[task]
    new = {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals), "n": len(vals),
           "sem": statistics.stdev(vals) / math.sqrt(len(vals))}
    ceil[task] = new
    ceil["_patched"][task] = {"from": old, "to": new, "delta": new["mean"] - old["mean"],
                              "reason": "original single-task runs hit the 150-epoch cap; rerun at 400"}
    print(f"  {task:14s} {old['mean']:.4f} -> {new['mean']:.4f}  (delta {new['mean']-old['mean']:+.4f}, n={len(vals)})")
args.out.write_text(json.dumps(ceil, indent=2) + "\n")
print(f"  wrote {args.out}")
