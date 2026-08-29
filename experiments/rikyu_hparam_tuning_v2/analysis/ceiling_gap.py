#!/usr/bin/env python3
"""Per-task: does a 24-task arm actually beat training that task alone? — summary/ceiling_gap.json

``stage_c.py`` reports deficits averaged over a size group, which is the right summary but hides
the only thing a group mean can be made of: a group of two tasks whose ceilings have very different
seed spread. v1's consolidated arm reads -0.019 on the small group, i.e. PAST the single-task
ceiling; that group is magnetization (ceiling seed sd 0.024) and magnetic_moment (sd 0.007), so
whether -0.019 means anything depends entirely on which of the two produced it.

This unpacks the group means into per-task gaps and tests each one.

WHAT THE TEST ASSUMES, AND WHY IT IS OPTIMISTIC
-----------------------------------------------
A stage-C arm is ONE seed — a 20-hour 24-task run cannot be repeated five times — while the ceiling
is five. So the arm's own seed spread is not measured, and the standard error of the difference
cannot be computed honestly. Assuming the arm's per-task spread resembles the single-task run's
(sigma_arm ~ sigma_ceiling = sigma) gives

    SE = sigma * sqrt(1/1 + 1/n_ceiling)   ->   1.10 * sigma at n=5

and a gap is called separated only beyond 2*SE. That assumption is the weak point and it errs
toward CLAIMING separation: if multi-task training is noisier per task than single-task training,
which is the direction one would expect from 24 tasks sharing an encoder, the real SE is larger and
some "separated" calls here would not survive. Read a result from this script as a hypothesis that
the ordering experiment (analysis/transfer.py, three orderings per task) is there to confirm.

    python analysis/ceiling_gap.py --arm "c_tuned_cons=<dir>" \\
        --ceilings summary/ceilings_adopted.json -o summary/ceiling_gap.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from common import N_TRAIN, pct_views, size_group
from stage_c import ACCURACY_TASKS, EXCLUDED, read_arm


def gaps_for_arm(metrics: dict[str, dict], ceilings: dict[str, dict]) -> list[dict]:
    rows = []
    for task, stats in sorted(ceilings.items()):
        if task in ACCURACY_TASKS or task in EXCLUDED:
            continue
        got = metrics.get(task, {}).get("r2")
        if got is None:
            continue
        sigma, n = stats["sd"], stats["n"]
        # sigma_arm is unmeasured (one seed); assumed equal to the ceiling's. See module docstring.
        se = sigma * math.sqrt(1.0 + 1.0 / n) if sigma > 0 and n else 0.0
        gap = float(got) - stats["mean"]
        separated = bool(se) and abs(gap) > 2 * se
        views = pct_views(gap, stats["mean"])
        rows.append({
            "task": task,
            "group": size_group(task),
            "n_train": N_TRAIN.get(task),
            "arm_r2": round(float(got), 4),
            "ceiling_mean": round(stats["mean"], 4),
            "ceiling_sd": round(sigma, 4),
            "gap": round(gap, 4),
            **views,
            "matters": separated and views["practically_significant"],
            "se_of_difference": round(se, 4) if se else None,
            "separated": separated,
            "verdict": (
                "beats single-task" if se and gap > 2 * se
                else "below single-task" if se and gap < -2 * se
                else "unresolved"
            ),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--ceilings", type=Path, required=True, help="summary/ceilings_adopted.json")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    ceilings = json.loads(args.ceilings.read_text())
    out_arms, missing = [], []
    for spec in args.arm:
        label, _, path = spec.partition("=")
        try:
            metrics, _ = read_arm(Path(path))
        except FileNotFoundError as exc:
            missing.append(f"{label}: {exc}")
            continue
        rows = gaps_for_arm(metrics, ceilings)
        beats = [r["task"] for r in rows if r["verdict"] == "beats single-task"]
        below = [r["task"] for r in rows if r["verdict"] == "below single-task"]
        matters = [r["task"] for r in rows if r["matters"]]
        out_arms.append({
            "label": label,
            "dir": path,
            "per_task": rows,
            "beats_single_task": beats,
            "below_single_task": below,
            "unresolved": [r["task"] for r in rows if r["verdict"] == "unresolved"],
            "separated_and_practically_significant": matters,
            "mean_gap": round(statistics.fmean([r["gap"] for r in rows]), 4) if rows else None,
            "mean_relative_pct": round(statistics.fmean(
                [r["relative_pct"] for r in rows if r["relative_pct"] is not None]), 3) if rows else None,
        })

    out = {
        "question": "per task, does the 24-task arm beat the same-regime single-task ceiling?",
        "arms": out_arms,
        "missing_arms": missing,
        "assumption": (
            "One seed per arm, so the arm's per-task seed spread is assumed equal to the ceiling's "
            "(SE = sigma*sqrt(1+1/n)). This errs toward claiming separation and is a hypothesis "
            "for the ordering experiment to confirm, not a result on its own."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")

    for arm in out_arms:
        print(f"\n=== {arm['label']}   mean gap {arm['mean_gap']:+.4f}  "
              f"(mean relative {arm['mean_relative_pct']:+.2f}%)")
        print(f"{'task':24s} {'grp':6s} {'arm':>8s} {'ceiling':>8s} {'gap':>8s} {'rel%':>8s} "
              f"{'2SE':>7s}  verdict")
        for r in sorted(arm["per_task"], key=lambda r: r["gap"]):
            se2 = 2 * (r["se_of_difference"] or 0.0)
            verdict = r["verdict"]
            if r["separated"] and not r["practically_significant"]:
                verdict += " (negligible)"
            rel = f"{r['relative_pct']:+.2f}%" if r["relative_pct"] is not None else "-"
            print(f"{r['task']:24s} {r['group']:6s} {r['arm_r2']:8.4f} {r['ceiling_mean']:8.4f} "
                  f"{r['gap']:+8.4f} {rel:>8s} {se2:7.4f}  {verdict}")
        print(f"  beats single-task: {arm['beats_single_task'] or 'none'}")
        print(f"  below single-task: {arm['below_single_task'] or 'none'}")
        print(f"  separated AND |gap| >= 0.01: "
              f"{arm['separated_and_practically_significant'] or 'none'}")
    if missing:
        print("\n  MISSING ARMS:", "; ".join(missing))
    print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
