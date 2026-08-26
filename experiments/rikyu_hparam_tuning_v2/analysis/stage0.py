#!/usr/bin/env python3
"""Score stage 0 — the anchor — and emit summary/stage0.json.

Stage 0 answers one question and measures two more.

THE QUESTION. v1's whole campaign ran under a broken LR-scheduler cadence, and fixing that
(PR #45) raised the baseline on its own. So a v2 number that is better than v1's is not evidence
that v2 TUNED better — the fix alone would produce that. The untuned configuration, measured on
the new image, is the point that separates the two effects:

    v1 c_base       untuned + broken scheduler
    v2 stage-0 base untuned + fixed scheduler   <- here: what the upgrade alone bought
    v2 stage-C' top tuned   + fixed scheduler   <- what tuning bought ON TOP

Skipping this point is the specific mistake v1 made, and it is why v1 cannot say how much of its
own headline number was its tuning.

THE TWO MEASUREMENTS. The rest of the campaign is sized from numbers that are, until this stage
runs, guesses:

  * per-run wall clock on probe6. The estimate (0.8h) was extrapolated from a 3-task probe with no
    kernel-regression heads, and KR steps dominate the wall clock — so the extrapolation could be
    wrong in the expensive direction.
  * the seed band. The 8.48% everyone quotes belongs to probe3. probe6 exists partly to SHRINK it
    by averaging over more tasks, so reusing probe3's band would misstate every margin that
    follows, and the '25 seeds' figure derived from it would be either wasteful or insufficient.

    python analysis/stage0.py --runs <outroot>/stage0 -o summary/stage0.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from common import (
    PROBE6,
    band,
    exceeds_band,
    fnum,
    group_by_config,
    load_runs,
    pick_metric_per_task,
    relative_score,
    seeds_needed,
    size_group,
    task_values,
)

ARMS = {
    "s0_base": "untuned defaults on 0.3.2 — THE anchor",
    "s0_v1enc": "v1's adopted encoder (latent 384, encoder_lr 1e-3) on 0.3.2",
}

# Differences the campaign will need to resolve, for the seed-count table.
RESOLUTION_TARGETS = [0.05, 0.03, 0.02, 0.015, 0.01]


def timing(outroot: Path) -> dict:
    """Per-run wall clock, from the worker's own log rather than from an estimate."""
    tsv = outroot / "_timing.tsv"
    if not tsv.exists():
        return {}
    secs = []
    for line in tsv.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2] == "0":
            v = fnum(parts[1])
            if v is not None:
                secs.append(v)
    if not secs:
        return {}
    hours = [s / 3600 for s in secs]
    return {
        "n_runs": len(hours),
        "mean_hours": statistics.fmean(hours),
        "min_hours": min(hours),
        "max_hours": max(hours),
        "total_gpu_hours": sum(hours),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="the stage0 output root")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=PROBE6)
    args = ap.parse_args()

    runs = load_runs(args.runs, "s0_*", args.tasks)
    if not runs:
        raise SystemExit(f"no complete runs under {args.runs}")
    configs = group_by_config(runs)
    metric_by_task = pick_metric_per_task(configs, args.tasks)

    # The reference is the untuned arm's per-task MEAN over seeds. Referencing a single seed would
    # fold that seed's luck into every margin in the campaign.
    if "s0_base" not in configs:
        raise SystemExit("the untuned anchor arm s0_base is missing — nothing can be referenced")
    reference = {
        task: statistics.fmean(task_values(configs["s0_base"], task, metric_by_task[task]))
        for task in args.tasks
    }

    arms: dict[str, dict] = {}
    for label, per_seed in sorted(configs.items()):
        scores = {}
        for seed, seed_metrics in sorted(per_seed.items()):
            s = relative_score(seed_metrics, reference, metric_by_task, args.tasks)
            if s is not None:
                scores[seed] = s
        per_task = {}
        for task in args.tasks:
            metric = metric_by_task[task]
            values = task_values(per_seed, task, metric)
            per_task[task] = {
                "metric": metric,
                "group": size_group(task),
                "mean": statistics.fmean(values) if values else None,
                "n": len(values),
                "spread": (max(values) - min(values)) if values else None,
                # Both metrics are carried whatever the scoring axis is: reporting only the one
                # that happens to look good is the disclosure failure PLAN §6.2 forbids.
                "r2_mean": (
                    statistics.fmean(task_values(per_seed, task, "r2"))
                    if task_values(per_seed, task, "r2") else None
                ),
                "mae_mean": (
                    statistics.fmean(task_values(per_seed, task, "mae"))
                    if task_values(per_seed, task, "mae") else None
                ),
            }
        arms[label] = {
            "label": ARMS.get(label, label),
            "n_seeds": len(scores),
            "score": band(list(scores.values())),
            "per_seed": scores,
            "per_task": per_task,
        }

    base_band = arms["s0_base"]["score"]
    comparisons = []
    for label, arm in arms.items():
        if label == "s0_base":
            continue
        delta = arm["score"]["mean"] - base_band["mean"]
        outside, multiple = exceeds_band(delta, base_band)
        comparisons.append({
            "from": "s0_base",
            "to": label,
            "delta_score": delta,
            "band_width": base_band.get("range"),
            "vs_band": multiple,
            "exceeds_band": outside,
            "verdict": "outside the seed band" if outside else "INSIDE the seed band — not a result",
        })

    sigma = base_band.get("sigma") or 0.0
    calibration = {
        "wallclock": timing(args.runs),
        "sigma_per_run": sigma,
        "band_range_at_n": {str(base_band["n"]): base_band.get("range")},
        "seeds_needed_to_resolve": {f"{t:.3f}": seeds_needed(sigma, t) for t in RESOLUTION_TARGETS},
        "v1_probe3_band_for_reference": 0.0848,
        "note": (
            "sigma is measured on probe6 and supersedes v1's probe3 band for every v2 margin. "
            "seeds_needed is what sizes the A'/B' finals; PLAN's 25 is an inherited estimate."
        ),
    }

    out = {
        "stage": "stage0",
        "probe": {
            "tasks": args.tasks,
            "metric_by_task": metric_by_task,
            "groups": {t: size_group(t) for t in args.tasks},
        },
        "reference": reference,
        "arms": arms,
        "comparisons": comparisons,
        "calibration": calibration,
        "runs_used": sorted(runs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")

    # --- human-readable echo -------------------------------------------------------------
    print(f"stage 0 — {len(runs)} runs, probe6")
    print(f"  metric per task: {metric_by_task}")
    for label, arm in arms.items():
        s = arm["score"]
        print(f"  {label:12s} n={arm['n_seeds']:2d}  mean {s['mean']:+.4%}  "
              f"range {s['range']:.4%}  sigma {s['sigma']:.4%}")
    for c in comparisons:
        print(f"  {c['from']} -> {c['to']}: {c['delta_score']:+.4%} "
              f"({c['vs_band']:+.2f}x band) — {c['verdict']}")
    w = calibration["wallclock"]
    if w:
        print(f"  wall clock: mean {w['mean_hours']:.2f}h  max {w['max_hours']:.2f}h  "
              f"({w['n_runs']} runs, {w['total_gpu_hours']:.1f} GPU-h)")
    print(f"  sigma {sigma:.4%} -> seeds needed: {calibration['seeds_needed_to_resolve']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
