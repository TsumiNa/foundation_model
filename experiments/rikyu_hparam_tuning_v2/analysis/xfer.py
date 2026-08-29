#!/usr/bin/env python3
"""Transfer at deployment scale, per task — summary/transfer_xfer.json

Each of the 24 tasks was trained as the LAST step of a 24-task sequence whose other 23 tasks were
shuffled independently, and that was repeated ``n`` times per task.

**The unit of analysis is the task, not the campaign.** Tasks here differ by two orders of
magnitude in labels (851 to 33 556) and by a factor of six in single-task ceiling (0.18 to 0.99);
whether a task benefits from sharing an encoder is a property OF THAT TASK. So this reports, for
every task separately: n, mean, sd, min and max over its repeats, the same for its single-task
baseline, and the difference between them.

Nothing is pooled into a campaign-level headline. An average over 24 tasks that transfer
differently is a number with no referent — the action it would inform (train this task jointly, or
alone) is taken per task.

``n = 3`` is the deliberate starting point: the smallest count that yields a mean and a spread at
all, chosen to get a first per-task answer cheaply. It is expected to grow to 5 or 10 for whichever
tasks the first pass shows are worth it, so nothing here assumes n = 3 —
``scripts/make_grids.py xfer --orders N --append-to`` extends the set without disturbing runs that
already exist, and this script reports whatever n it finds, per task.

The repeats vary the shuffle AND the seed together, so a task's spread across them covers ordering
effects and seed noise jointly. Its single-task baseline's spread (five seeds, no ordering to vary)
is printed beside it as the reference: a task whose repeat spread is much wider than its seed
spread is one where ordering plausibly matters, and at n = 3 that is a flag for more repeats rather
than a conclusion.

    python analysis/xfer.py --runs <outroot>/stage_xfer --ceilings summary/ceilings_adopted.json \\
        -o summary/transfer_xfer.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from common import N_TRAIN, final_metrics, fnum, pct_views, size_group

RUNID = re.compile(r"^xf_(?P<task>.+)_o(?P<order>\d+)$")


def collect(root: Path) -> dict[str, dict[int, float]]:
    """{task: {order_index: final metric}} — R2, or macro_f1 for the one classification task."""
    out: dict[str, dict[int, float]] = {}
    for run in sorted(root.glob("xf_*")):
        m = RUNID.match(run.name)
        if not m or not (run / "DONE").exists():
            continue
        task = m.group("task")
        try:
            metrics, _ = final_metrics(run)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        value = fnum(metrics.get(task, {}).get("r2"))
        if value is None:
            value = fnum(metrics.get(task, {}).get("macro_f1"))
        if value is not None:
            out.setdefault(task, {})[int(m.group("order"))] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--ceilings", type=Path, required=True, help="summary/ceilings_adopted.json")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    got = collect(args.runs)
    single = json.loads(args.ceilings.read_text())

    rows: list[dict] = []
    for task in sorted(set(got) | set(single)):
        orders = got.get(task, {})
        base = single.get(task)
        values = [orders[k] for k in sorted(orders)]
        if not values or not base:
            rows.append({"task": task, "n_orders": len(values),
                         "skipped": "no runs" if not values else "no single-task baseline"})
            continue
        m_multi = statistics.fmean(values)
        sd_multi = statistics.stdev(values) if len(values) > 1 else 0.0
        sd_single, n_single = base["sd"], base["n"]
        se = math.sqrt(sd_multi**2 / len(values) + sd_single**2 / n_single) if len(values) > 1 else None
        transfer = m_multi - base["mean"]
        # Kept per task, never averaged across tasks: it says whether THIS task's spread over
        # shuffled orderings looks like its own seed noise.
        ratio = (sd_multi / sd_single) if sd_single > 0 and len(values) > 1 else None
        separated = bool(se) and abs(transfer) > 2 * se
        views = pct_views(transfer, base["mean"])
        rows.append({
            "task": task,
            "group": size_group(task),
            "n_train": N_TRAIN.get(task),
            "single_task_r2": base["mean"],
            "multi_task_r2": m_multi,
            "transfer": transfer,
            **views,
            "matters": separated and views["practically_significant"],
            "se_of_difference": se,
            "separated": separated,
            "n_repeats": len(values),
            "multi_task_sd": sd_multi,
            "multi_task_min": min(values),
            "multi_task_max": max(values),
            "multi_task_range": max(values) - min(values),
            "single_task_sd": sd_single,
            "single_task_n": n_single,
            "single_task_min": base.get("min"),
            "single_task_max": base.get("max"),
            "repeat_to_seed_sd_ratio": ratio,
            # At n=3 a spread is barely an estimate; flag for more repeats, do not conclude.
            "spread_wider_than_seed_noise": bool(ratio is not None and ratio > 2.0),
            "per_repeat_r2": {str(k): orders[k] for k in sorted(orders)},
        })

    scored = [r for r in rows if "transfer" in r]
    helped = [r["task"] for r in scored if r["separated"] and r["transfer"] > 0]
    hurt = [r["task"] for r in scored if r["separated"] and r["transfer"] < 0]
    matters = [r["task"] for r in scored if r["matters"]]
    # Deliberately NOT a pooled ordering statistic. An earlier version reported the median of
    # sd(repeats)/sd(seeds) across all 24 tasks as a single "does order matter" number; that
    # averages over tasks whose difficulty and data scale differ by orders of magnitude, so it
    # answers a question nobody asks. The per-task ratio lives in each row instead, and the only
    # thing collected here is WHICH tasks look worth more repeats.
    wide = [r["task"] for r in scored if r.get("spread_wider_than_seed_noise")]

    out = {
        "question": "at 24 tasks, does arriving last beat training alone — and does the order matter?",
        "per_task": rows,
        # Lists, not averages: these enumerate which tasks landed where. There is deliberately no
        # mean transfer, no mean relative percent and no by-size-group mean — averaging over tasks
        # that differ by two orders of magnitude in labels and six-fold in ceiling produces a
        # number with no referent, and the decision it would inform is taken per task anyway.
        "summary": {
            "tasks_helped": helped,
            "tasks_hurt": hurt,
            "tasks_unresolved": [r["task"] for r in scored if not r["separated"]],
            "tasks_that_matter": matters,
            "resolved_but_negligible": [r["task"] for r in scored
                                        if r["separated"] and not r["practically_significant"]],
        },
        "tasks_worth_more_repeats": {
            "tasks": wide,
            "criterion": "sd across repeats > 2x the task's own single-task seed sd",
            "note": (
                "At n=3 a standard deviation is barely an estimate, so this is a flag for "
                "extending that task's repeat count (make_grids.py xfer --orders N --append-to), "
                "not a finding about task ordering."
            ),
        },
        "notes": [
            "The task under test is always the LAST step; the other 23 are shuffled independently "
            "per repeat, so the three repeats differ in order AND seed.",
            "Single-task baselines are the same-régime ceilings: adopted configuration, five "
            "seeds, differing from a campaign run only in pretrain.task_sequence.",
            "material_type is scored on macro_f1 (it has no R2); its transfer is on that scale and "
            "is not comparable to the regression tasks' numbers.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'task':22s} {'grp':6s} {'N':>7s} | {'multi mean':>10s} {'sd':>7s} {'min':>7s} "
          f"{'max':>7s} {'n':>2s} | {'single':>7s} {'sd':>7s} | {'transfer':>9s} {'rel%':>8s} "
          f"{'2SE':>7s}  verdict")
    for r in sorted(rows, key=lambda r: -(r.get("transfer") if "transfer" in r else -9)):
        if "transfer" not in r:
            print(f"{r['task']:22s} {'':6s} {'':>7s} | {r['skipped']} (n_repeats={r['n_repeats']})")
            continue
        verdict = ("multi-task better" if r["separated"] and r["transfer"] > 0
                   else "single-task better" if r["separated"] else "unresolved")
        if r["separated"] and not r["practically_significant"]:
            verdict += " (negligible)"
        if r.get("spread_wider_than_seed_noise"):
            verdict += "  [wide spread: more repeats]"
        print(f"{r['task']:22s} {r['group']:6s} {r['n_train']:7d} | "
              f"{r['multi_task_r2']:10.4f} {r['multi_task_sd']:7.4f} {r['multi_task_min']:7.4f} "
              f"{r['multi_task_max']:7.4f} {r['n_repeats']:2d} | "
              f"{r['single_task_r2']:7.4f} {r['single_task_sd']:7.4f} | "
              f"{r['transfer']:+9.4f} {r['relative_pct']:+7.2f}% "
              f"{2 * (r['se_of_difference'] or 0):7.4f}  {verdict}")

    s = out["summary"]
    print(f"\n  multi-task better: {s['tasks_helped'] or 'none'}")
    print(f"  single-task better: {s['tasks_hurt'] or 'none'}")
    print(f"  unresolved at this repeat count: {len(s['tasks_unresolved'])} task(s)")
    w = out["tasks_worth_more_repeats"]["tasks"]
    print(f"  spread wider than the task's own seed noise -> worth more repeats: {w or 'none'}")
    print("  (no campaign-level average is reported: transfer is a per-task property)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
