#!/usr/bin/env python3
"""Score stage a4 — does the LR schedule earn its place? — and emit summary/stage_a4.json.

Stage A' produced two findings that point the same way. `encoder_lr` gets monotonically worse as
it rises, and short scheduler `patience` — cut the LR early and often — is one of only two axes
whose effect clears the seed band. Read together they suggest the model simply wants a low
effective learning rate quickly, in which case the schedule may be an elaborate route to
somewhere a constant low LR reaches directly.

Stage A' could not answer that: its `encoder_lr` floor was 1e-3, so below that is unmeasured, and
every point it ran had the scheduler on. a4 crosses scheduler on/off with an `encoder_lr` axis
extended a decade lower, everything else held identical, so the two questions are answered by the
same runs and "schedule vs none" is measured AT each LR rather than confounded with it.

READING THE RESULT
------------------
The comparison that matters is PAIRED and per-LR, not a single overall verdict. A schedule that
helps at 5e-3 and does nothing at 2e-4 is a real and useful finding — it would say the schedule's
job is rescuing a too-high starting LR, which is a different claim from "the schedule helps".

Two limits are reported alongside every number, because neither is visible in the data:

  * ``[training.scheduler]`` governs all four parameter groups, so ``enabled = false`` also freezes
    the head / KR / AE learning rates. A loss for the flat arm does not say WHICH group needed
    annealing; there is no per-group switch to decompose it with.
  * every task step builds a fresh Trainer, so the optimizer — and the LR — is rebuilt at the
    configured value at each of the six steps. Annealing never carries across the sequence. That
    is why `patience` mattered so much in stage A' and it bounds how much any schedule can do
    here: within a 45-75 epoch step, patience 24 gets to act about once.

    python analysis/stage_a4.py --runs <outroot>/stage_a -o summary/stage_a4.json \\
        --stage0 summary/stage0.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from common import (
    PROBE6,
    band,
    group_by_config,
    load_runs,
    relative_score,
    seeds_needed,
)

# a4sched_E0p0005 / a4flat_E0p0001
LABEL = re.compile(r"^a4(sched|flat)_E([0-9p\-e.]+)$")

# The floor stage A' could not search below. Anything under this is new ground.
PREVIOUS_LR_FLOOR = 1e-3


def parse_label(label: str) -> tuple[str, float] | None:
    m = LABEL.match(label)
    if not m:
        return None
    return m.group(1), float(m.group(2).replace("p", "."))


def welch_se(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    return math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--stage0", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=PROBE6)
    args = ap.parse_args()

    stage0 = json.loads(args.stage0.read_text())
    reference = stage0["reference"]
    seed_band = stage0["arms"]["s0_base"]["score"]
    metric_by_task = stage0["probe"]["metric_by_task"]

    runs = load_runs(args.runs, "a4*", args.tasks)
    if not runs:
        raise SystemExit(f"no complete a4 runs under {args.runs}")
    configs = group_by_config(runs)

    cells: dict[str, dict[str, dict]] = {"sched": {}, "flat": {}}
    for label, per_seed in configs.items():
        parsed = parse_label(label)
        if parsed is None:
            continue
        arm, lr = parsed
        scores = [
            s for s in (relative_score(m, reference, metric_by_task, args.tasks)
                        for m in per_seed.values()) if s is not None
        ]
        if scores:
            cells[arm][f"{lr:g}"] = {"lr": lr, "label": label, **band(scores)}

    lrs = sorted({c["lr"] for arm in cells.values() for c in arm.values()})
    paired = []
    for lr in lrs:
        key = f"{lr:g}"
        s, f = cells["sched"].get(key), cells["flat"].get(key)
        if not (s and f):
            continue
        delta = s["mean"] - f["mean"]
        se = welch_se(s["values"], f["values"])
        paired.append({
            "encoder_lr": lr,
            "below_previous_floor": lr < PREVIOUS_LR_FLOOR,
            "scheduled_mean": s["mean"],
            "flat_mean": f["mean"],
            "delta_schedule_minus_flat": delta,
            "se_of_difference": se,
            "separated": bool(se) and abs(delta) > 2 * se,
            "verdict": (
                "schedule helps" if se and delta > 2 * se
                else "schedule HURTS" if se and delta < -2 * se
                else "no measurable difference"
            ),
        })

    def best(arm: str) -> dict | None:
        entries = list(cells[arm].values())
        return max(entries, key=lambda c: c["mean"]) if entries else None

    best_sched, best_flat = best("sched"), best("flat")
    head_to_head = None
    if best_sched and best_flat:
        d = best_sched["mean"] - best_flat["mean"]
        se = welch_se(best_sched["values"], best_flat["values"])
        head_to_head = {
            "best_scheduled": {"lr": best_sched["lr"], "mean": best_sched["mean"]},
            "best_flat": {"lr": best_flat["lr"], "mean": best_flat["mean"]},
            "delta": d,
            "se_of_difference": se,
            "separated": bool(se) and abs(d) > 2 * se,
            "verdict": (
                "the schedule earns its place" if se and d > 2 * se
                else "a constant LR is BETTER" if se and d < -2 * se
                else "a constant LR matches the schedule — the schedule is not earning its place"
            ),
        }

    # Did extending the axis downward find anything? Compare the best below the old floor with the
    # best at or above it, within each arm.
    extension = {}
    for arm in ("sched", "flat"):
        below = [c for c in cells[arm].values() if c["lr"] < PREVIOUS_LR_FLOOR]
        at_above = [c for c in cells[arm].values() if c["lr"] >= PREVIOUS_LR_FLOOR]
        if below and at_above:
            b, a = max(below, key=lambda c: c["mean"]), max(at_above, key=lambda c: c["mean"])
            se = welch_se(b["values"], a["values"])
            extension[arm] = {
                "best_below_floor": {"lr": b["lr"], "mean": b["mean"]},
                "best_at_or_above_floor": {"lr": a["lr"], "mean": a["mean"]},
                "delta": b["mean"] - a["mean"],
                "se_of_difference": se,
                "worth_extending_further": bool(se) and (b["mean"] - a["mean"]) > 2 * se,
            }

    sigma = statistics.fmean([c["sigma"] for arm in cells.values() for c in arm.values() if c["n"] > 1])
    out = {
        "stage": "a4",
        "question": "does the LR schedule earn its place, and is the optimum below the searched floor?",
        "n_runs": len(runs),
        "seed_band_from_stage0": seed_band,
        "cells": cells,
        "paired_by_lr": paired,
        "head_to_head": head_to_head,
        "downward_extension": extension,
        "pooled_sigma": sigma,
        "seeds_to_resolve_1pct": seeds_needed(sigma, 0.01),
        "caveats": [
            "[training.scheduler] governs all four optimizer groups, so enabled=false also freezes "
            "the head / KR / AE learning rates. A loss for the flat arm does not identify WHICH "
            "group needed annealing — there is no per-group switch.",
            "Every task step builds a fresh Trainer, so the optimizer and its LR are rebuilt at the "
            "configured value at each of the six steps; annealing never carries across the "
            "sequence. Within a 45-75 epoch step, patience 24 acts about once.",
            "Five seeds per cell. Differences smaller than roughly 2 SE of the difference are not "
            "resolved and are reported as such rather than ordered.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"stage a4 — {len(runs)} runs, pooled sigma {sigma:.3%}")
    print(f"  {'encoder_lr':>11s} {'scheduled':>11s} {'flat':>11s} {'sched-flat':>11s} {'2SE':>8s}  verdict")
    for p in paired:
        mark = "*" if p["below_previous_floor"] else " "
        se = p["se_of_difference"] or 0.0
        print(f"  {p['encoder_lr']:11g}{mark}{p['scheduled_mean']:+11.3%} {p['flat_mean']:+11.3%} "
              f"{p['delta_schedule_minus_flat']:+11.3%} {2 * se:8.3%}  {p['verdict']}")
    print("  (* = below stage A's 1e-3 floor, never searched before)")
    if head_to_head:
        h = head_to_head
        print(f"\n  best scheduled: lr={h['best_scheduled']['lr']:g} {h['best_scheduled']['mean']:+.3%}")
        print(f"  best flat     : lr={h['best_flat']['lr']:g} {h['best_flat']['mean']:+.3%}")
        print(f"  -> {h['verdict']}")
    for arm, e in extension.items():
        print(f"  {arm}: best below floor {e['best_below_floor']['mean']:+.3%} "
              f"vs best above {e['best_at_or_above_floor']['mean']:+.3%} "
              f"-> extend further: {e['worth_extending_further']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
