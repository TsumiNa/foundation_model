#!/usr/bin/env python3
"""Transfer at deployment scale, and whether task ORDER matters — summary/transfer_xfer.json

Each of the 24 tasks was trained as the LAST step of a 24-task sequence whose other 23 tasks were
shuffled independently, three times. Two questions come out of that one set of runs.

**1. Does arriving last, after an encoder shaped by 23 other tasks, beat training alone?**
The probe answered this for six tasks (analysis/transfer.py). This answers it for all 24, in the
régime that ships. Each task's three runs are compared against its same-régime single-task
baseline (five seeds, adopted configuration, differing only in ``pretrain.task_sequence``).

**2. Does the order of the preceding 23 matter?**
The three runs per task differ in BOTH the shuffle and the seed, so their spread contains ordering
effects plus seed noise. If ordering contributes nothing, that spread should look like the seed
spread already measured on the single-task runs.

Per task that comparison is hopeless — three points against five estimate a variance ratio with
enormous error bars, and an F-test on n=3 would be theatre. So it is POOLED: the ratio
sd(orders) / sd(seeds) is computed for every task and the distribution across all 24 is reported.
Ordering that mattered would push that distribution above 1 systematically; twenty-four weak
estimates of the same quantity make a usable one, where any single task's would not.

The comparison is one-sided in interpretation but not in construction: a median ratio well BELOW 1
would be just as suspicious (it would mean the multi-task runs are implausibly stable) and is
reported rather than rounded to "no effect".

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

from common import N_TRAIN, final_metrics, fnum, size_group

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

    rows, ratios = [], []
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
        ratio = (sd_multi / sd_single) if sd_single > 0 and len(values) > 1 else None
        if ratio is not None:
            ratios.append(ratio)
        rows.append({
            "task": task,
            "group": size_group(task),
            "n_train": N_TRAIN.get(task),
            "single_task_r2": base["mean"],
            "multi_task_r2": m_multi,
            "transfer": transfer,
            "se_of_difference": se,
            "separated": bool(se) and abs(transfer) > 2 * se,
            "n_orders": len(values),
            "sd_across_orders": sd_multi,
            "sd_across_seeds_single": sd_single,
            "order_to_seed_sd_ratio": ratio,
            "per_order_r2": {str(k): orders[k] for k in sorted(orders)},
        })

    scored = [r for r in rows if "transfer" in r]
    helped = [r["task"] for r in scored if r["separated"] and r["transfer"] > 0]
    hurt = [r["task"] for r in scored if r["separated"] and r["transfer"] < 0]
    ordering = {
        "n_tasks_with_ratio": len(ratios),
        "median_ratio": statistics.median(ratios) if ratios else None,
        "mean_ratio": statistics.fmean(ratios) if ratios else None,
        "tasks_above_1": sum(1 for r in ratios if r > 1),
        "interpretation": (
            "sd across three shuffled orders divided by sd across five seeds of the same "
            "single-task configuration. A median near 1 means shuffling the preceding 23 tasks "
            "adds nothing beyond seed noise. Pooled across tasks because n=3 per task cannot "
            "estimate a variance ratio on its own."
        ),
    }

    out = {
        "question": "at 24 tasks, does arriving last beat training alone — and does the order matter?",
        "per_task": rows,
        "summary": {
            "tasks_helped": helped,
            "tasks_hurt": hurt,
            "tasks_unresolved": [r["task"] for r in scored if not r["separated"]],
            "mean_transfer": statistics.fmean([r["transfer"] for r in scored]) if scored else None,
            "by_group": {
                g: statistics.fmean([r["transfer"] for r in scored if r["group"] == g])
                for g in ("big", "mid", "small")
                if any(r["group"] == g for r in scored)
            },
        },
        "ordering": ordering,
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

    print(f"{'task':24s} {'grp':6s} {'N':>7s} {'single':>8s} {'multi':>8s} {'transfer':>9s} "
          f"{'2SE':>7s} {'sd_ord/sd_seed':>15s}  verdict")
    for r in sorted(rows, key=lambda r: -(r.get("transfer") or -9)):
        if "transfer" not in r:
            print(f"{r['task']:24s} {'':6s} {'':>7s} {'':>8s} {'':>8s} {'':>9s} {'':>7s} {'':>15s}  "
                  f"{r['skipped']} (n_orders={r['n_orders']})")
            continue
        verdict = ("multi-task better" if r["separated"] and r["transfer"] > 0
                   else "single-task better" if r["separated"] else "unresolved")
        ratio = r["order_to_seed_sd_ratio"]
        print(f"{r['task']:24s} {r['group']:6s} {r['n_train']:7d} {r['single_task_r2']:8.4f} "
              f"{r['multi_task_r2']:8.4f} {r['transfer']:+9.4f} {2 * (r['se_of_difference'] or 0):7.4f} "
              f"{(f'{ratio:.2f}' if ratio is not None else '-'):>15s}  {verdict}")

    s = out["summary"]
    print(f"\n  helped: {s['tasks_helped'] or 'none'}")
    print(f"  hurt:   {s['tasks_hurt'] or 'none'}")
    print("  mean transfer by group: " +
          "  ".join(f"{g} {v:+.4f}" for g, v in s["by_group"].items()))
    o = ordering
    if o["median_ratio"] is not None:
        print(f"\n  DOES ORDER MATTER? sd(orders)/sd(seeds) over {o['n_tasks_with_ratio']} tasks: "
              f"median {o['median_ratio']:.2f}, mean {o['mean_ratio']:.2f}, "
              f"{o['tasks_above_1']} of {o['n_tasks_with_ratio']} above 1")
        print("  (a median near 1 means shuffling the preceding 23 adds nothing beyond seed noise)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
