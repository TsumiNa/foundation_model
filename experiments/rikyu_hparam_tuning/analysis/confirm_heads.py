#!/usr/bin/env python3
"""Confirm each per-task head winner against its own seed band, and emit the deployable set.

Stage B picks each task's head on one seed. Stage A already showed what a single seed is worth
here: its leader scored +23.9% at seed 2025 and +15.9%/+15.5% at 2026/2027. Stage B4 re-runs every
task's winner AND its untuned baseline at two further seeds, so each per-task claim gets a range.

**Rule, fixed before B4 was read, and the same principle stage A adopted** (do not pay for what
you cannot measure):

    A task keeps its tuned head only if (mean winner − mean baseline) exceeds that task's own
    seed band. Otherwise it reverts to the untuned default.

Consistency matters more than squeezing the last point: stage A preferred the simplest
configuration among ties, and this is that principle applied per task. Two outputs are written —
the confirmed set (used by stage C) and the raw point-estimate set (what the grid produced) — so
the report can state exactly how much of the grid's output survived.

    python .../confirm_heads.py b4.csv --winners ../results/head_winners.json \\
        -o ../results/head_winners_confirmed.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

LOWER_IS_BETTER = {"mae"}
RUNID = re.compile(r"^b4_(win|base)_(.+)_s(\d+)$")


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="collected B4 runs")
    ap.add_argument("--winners", type=Path, required=True, help="head_winners.json from pick_heads.py")
    ap.add_argument("--stage-b", type=Path, help="collected stage-B CSV, to add the seed-2025 point")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    winners = json.loads(args.winners.read_text())

    # arm_values[task]["win"|"base"] = [metric at each seed]
    arm_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"win": [], "base": []})
    for r in csv.DictReader(open(args.csv)):
        m = RUNID.match(r["runid"])
        if not m:
            continue
        arm, task, _seed = m.groups()
        w = winners.get(task)
        if not w:
            continue
        v = fnum(r.get(w["metric"]))
        if v is not None and r["task"] == task:
            arm_values[task][arm].append(v)

    # The seed-2025 points already exist in the stage-B grid: the winner run itself and the
    # baseline grid point. Include them so each arm has three seeds rather than two.
    if args.stage_b and args.stage_b.exists():
        by_run: dict[str, dict[str, str]] = {}
        for r in csv.DictReader(open(args.stage_b)):
            by_run.setdefault(r["runid"], r)
        for task, w in winners.items():
            v = fnum(by_run.get(w["runid"], {}).get(w["metric"]))
            if v is not None:
                arm_values[task]["win"].append(v)
            if w.get("baseline_value") is not None:
                arm_values[task]["base"].append(w["baseline_value"])

    confirmed: dict[str, dict] = {}
    report: list[str] = []
    kept = 0
    for task in sorted(winners):
        w = winners[task]
        wins, bases = arm_values[task]["win"], arm_values[task]["base"]
        if len(wins) < 2 or len(bases) < 2:
            report.append(f"{task:26s}  INCOMPLETE — win {len(wins)} seeds, base {len(bases)} seeds")
            continue
        sign = -1.0 if w["metric"] in LOWER_IS_BETTER else 1.0
        mw, mb = statistics.fmean(wins), statistics.fmean(bases)
        band = max(max(wins) - min(wins), max(bases) - min(bases))
        gain = sign * (mw - mb)
        survives = gain > band
        entry = dict(w)
        entry |= {
            "seeds_win": len(wins), "seeds_base": len(bases),
            "mean_win": mw, "mean_base": mb, "band": band, "confirmed_gain": gain,
            "confirmed": survives,
        }
        if not survives:
            entry["override"] = {}  # revert to the untuned default
        else:
            kept += 1
        confirmed[task] = entry
        mark = "KEEP" if survives else "revert"
        report.append(
            f"{task:26s}  {w['metric']:8s}  win {mw:8.4f}  base {mb:8.4f}  "
            f"gain {gain:+8.4f}  band {band:7.4f}  -> {mark}"
        )

    print(f"# {args.csv.name} — per-task confirmation against each task's own seed band")
    print("\n".join(report))
    print(f"\n{kept}/{len(confirmed)} tasks keep a tuned head; the rest revert to the untuned default.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(confirmed, indent=2, sort_keys=True) + "\n")
    print(f"{args.output}")


if __name__ == "__main__":
    main()
