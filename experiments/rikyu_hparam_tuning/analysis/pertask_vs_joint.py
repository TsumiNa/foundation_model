#!/usr/bin/env python3
"""Compare per-task head tuning against joint (shared-head) tuning on the same probe.

Stage B tunes each task's head in isolation, which is what makes 24 tasks affordable and
explicable — but isolation is an assumption, and this is where it is priced. Three arms, all
measured on one multi-task probe so nothing else differs:

    mt_base      untuned shared head              (a grid point of the B-mt sweep)
    mt_joint     best shared head, tuned jointly  (a grid point of the B-mt sweep)
    mt_pertask   each task's own stage-B winner   (a generated per-task-override config)

The table it prints is the campaign's honest statement about its own method: per task, what each
arm scores, and whether tuning heads in isolation gained or lost relative to tuning them together.

    python .../pertask_vs_joint.py bmt.csv --joint bmtreg_H128-64_L0p002 \\
        --base bmtreg_H64_L0p005 --pertask mt_pertask_reg
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

LOWER_IS_BETTER = {"mae"}


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="collected CSV covering all three arms")
    ap.add_argument("--base", required=True, help="runid of the untuned shared-head arm")
    ap.add_argument("--joint", required=True, help="runid of the best jointly-tuned shared head")
    ap.add_argument("--pertask", required=True, help="runid of the per-task-winners arm")
    ap.add_argument("--metric", default="r2")
    args = ap.parse_args()

    # value[runid][task]
    value: dict[str, dict[str, float]] = defaultdict(dict)
    for r in csv.DictReader(open(args.csv)):
        v = fnum(r.get(args.metric))
        if v is not None:
            value[r["runid"]][r["task"]] = v

    arms = [("untuned", args.base), ("joint", args.joint), ("per-task", args.pertask)]
    for label, runid in arms:
        if runid not in value:
            raise SystemExit(f"{label} arm {runid!r} not in {args.csv} (have: {sorted(value)[:8]}...)")

    tasks = sorted(set(value[args.base]) & set(value[args.joint]) & set(value[args.pertask]))
    if not tasks:
        raise SystemExit("the three arms share no task")

    better = (lambda a, b: a < b) if args.metric in LOWER_IS_BETTER else (lambda a, b: a > b)
    sign = -1.0 if args.metric in LOWER_IS_BETTER else 1.0

    print(f"# {args.csv.name} — per-task vs joint head tuning, metric={args.metric}")
    print(f"#   untuned  {args.base}\n#   joint    {args.joint}\n#   per-task {args.pertask}")
    head = f"{'task':26s}  {'untuned':>9s}  {'joint':>9s}  {'per-task':>9s}  {'pt−joint':>9s}  verdict"
    print(head)
    print("-" * len(head))

    deltas = []
    for task in tasks:
        base, joint, pertask = value[args.base][task], value[args.joint][task], value[args.pertask][task]
        d = sign * (pertask - joint)
        deltas.append(d)
        verdict = "per-task wins" if better(pertask, joint) else ("tie" if pertask == joint else "joint wins")
        print(f"{task:26s}  {base:9.4f}  {joint:9.4f}  {pertask:9.4f}  {d:+9.4f}  {verdict}")

    print("-" * len(head))
    means = [statistics.fmean(value[r][t] for t in tasks) for _, r in arms]
    print(f"{'mean':26s}  {means[0]:9.4f}  {means[1]:9.4f}  {means[2]:9.4f}  {statistics.fmean(deltas):+9.4f}")
    wins = sum(1 for d in deltas if d > 0)
    print(f"\nper-task beats joint on {wins}/{len(tasks)} tasks; "
          f"mean advantage {statistics.fmean(deltas):+.4f} {args.metric}")
    print("Both arms are tuned; the untuned column is the reference for how much either bought.")


if __name__ == "__main__":
    main()
