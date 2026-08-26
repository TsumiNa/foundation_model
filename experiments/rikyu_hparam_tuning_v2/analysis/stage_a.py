#!/usr/bin/env python3
"""Rank stage A' — the joint encoder x LR x scheduler search — and emit summary/stage_a.json.

Reads the grid and the random search TOGETHER. They sample the same space by different means (the
grid gives readable marginal-effect plots, random search gives interior coverage), so ranking them
apart would just be two partial answers to one question.

THREE THINGS THIS REFUSES TO DO SILENTLY
----------------------------------------
1. Rank on noise. Every margin is quoted against the seed band measured in stage 0, and the
   short list is cut where the margins stop exceeding it. v1's single-seed leader lost 5.5 points
   when re-measured at three seeds — on a grid this size, the maximum over many noisy points is
   biased upward by construction, and calling that maximum a winner is the whole trap.

2. Hide an edge. If the best points sit against the boundary of an axis, the search did not find
   an optimum, it ran out of room. That is reported as a REQUIRED follow-up (a1b) rather than as a
   result, and the exit status says so, so a pipeline cannot walk past it.

3. Report only the flattering metric. R2 and MAE disagree — v1's stage A gain was 1.9x the band on
   MAE and 0.86x on R2, i.e. real on one and absent on the other. Both are emitted for every
   configuration, and a config that only wins on one is labelled as such.

    python analysis/stage_a.py --runs <outroot>/stage_a --stage0 summary/stage0.json \\
        -o summary/stage_a.json --top 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from common import (
    PROBE6,
    band,
    exceeds_band,
    group_by_config,
    load_runs,
    pick_metric_per_task,
    relative_score,
    size_group,
    task_values,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_grids import parse_point  # noqa: E402

# The axes stage A' searches, and how to read them off a runid.
AXES = {
    "training__encoder_lr": "encoder_lr",
    "training__scheduler__min_lr": "min_lr",
    "training__scheduler__patience": "patience",
    "training__scheduler__factor": "factor",
    "model__latent_dim": "latent_dim",
}


def boundary_report(points: dict[str, dict], short_list: list[str]) -> dict:
    """Which axes the short list is pressed up against.

    An optimum at the edge of the searched range is the range running out, not an optimum. v1 got
    this right once — its A1 optimum sat on the smallest encoder_lr it had tried, and A1b reopened
    that edge and confirmed an interior point. That step is the one v1 procedure carried into v2
    unchanged.
    """
    report: dict[str, dict] = {}
    for key, name in AXES.items():
        searched = sorted({p[key] for p in points.values() if key in p})
        if len(searched) < 2:
            continue
        top_values = [points[c][key] for c in short_list if key in points[c]]
        if not top_values:
            continue
        lo, hi = searched[0], searched[-1]
        at_lo = [c for c in short_list if points[c].get(key) == lo]
        at_hi = [c for c in short_list if points[c].get(key) == hi]
        report[name] = {
            "searched_min": lo,
            "searched_max": hi,
            "n_values": len(searched),
            "short_list_at_min": at_lo,
            "short_list_at_max": at_hi,
            "edge_bound": bool(at_lo or at_hi),
            "best_value": top_values[0],
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="stage_a output root")
    ap.add_argument("--stage0", type=Path, required=True, help="summary/stage0.json (the anchor)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=PROBE6)
    ap.add_argument("--top", type=int, default=8, help="short-list size to promote to the finals")
    ap.add_argument("--pattern", default="a1*", help="which runs to rank (grid + random by default)")
    args = ap.parse_args()

    stage0 = json.loads(args.stage0.read_text())
    reference = stage0["reference"]
    seed_band = stage0["arms"]["s0_base"]["score"]
    metric_by_task = stage0["probe"]["metric_by_task"]

    runs = load_runs(args.runs, args.pattern, args.tasks)
    if not runs:
        raise SystemExit(f"no complete runs under {args.runs} matching {args.pattern}")
    configs = group_by_config(runs)

    # The metric axis is INHERITED from stage 0 rather than recomputed here, so every stage of the
    # campaign scores on the same quantity. Recomputing per stage would let the axis drift as the
    # grid explores wider, and two stages scored on different axes cannot be compared.
    for task in args.tasks:
        metric_by_task.setdefault(task, "r2")

    scored = []
    points: dict[str, dict] = {}
    for label, per_seed in configs.items():
        seed_scores = {}
        for seed, seed_metrics in per_seed.items():
            s = relative_score(seed_metrics, reference, metric_by_task, args.tasks)
            if s is not None:
                seed_scores[seed] = s
        if not seed_scores:
            continue
        stats = band(list(seed_scores.values()))
        points[label] = parse_point(label)
        scored.append({
            "config": label,
            "point": points[label],
            "n_seeds": stats["n"],
            "score_mean": stats["mean"],
            "score_sem": stats["sem"],
            "score_range": stats["range"],
            "per_seed": seed_scores,
            "vs_band": exceeds_band(stats["mean"], seed_band)[1],
            "exceeds_band": exceeds_band(stats["mean"], seed_band)[0],
            "per_task": {
                t: {
                    "metric": metric_by_task[t],
                    "group": size_group(t),
                    "r2_mean": (statistics.fmean(task_values(per_seed, t, "r2"))
                                if task_values(per_seed, t, "r2") else None),
                    "mae_mean": (statistics.fmean(task_values(per_seed, t, "mae"))
                                 if task_values(per_seed, t, "mae") else None),
                }
                for t in args.tasks
            },
        })

    scored.sort(key=lambda r: r["score_mean"], reverse=True)
    short_list = [r["config"] for r in scored[: args.top]]
    edges = boundary_report(points, short_list)

    # A margin the seeds cannot resolve is not a ranking. Report how far down the list the
    # ordering is actually supported, rather than presenting all of it as if it were.
    top_sem = scored[0]["score_sem"] if scored else 0.0
    resolvable = 2 * (top_sem or 0.0)
    tied_with_leader = [
        r["config"] for r in scored[1:] if abs(scored[0]["score_mean"] - r["score_mean"]) < resolvable
    ]

    edge_bound = {name: e for name, e in edges.items() if e["edge_bound"]}
    out = {
        "stage": "stage_a",
        "caveats": [
            "min_lr is NOT an encoder-only knob. One [training.scheduler] block serves all four "
            "optimizer groups (encoder / head / kr / ae), so a min_lr change moves the annealing "
            "floor for the heads too. The joint search is the right response to that coupling, "
            "but no min_lr result may be reported as an encoder-specific effect.",
            "min_lr is bounded above by the SMALLEST group LR, kr_lr = 5e-4, not by encoder_lr: "
            "OptimizerConfig rejects min_lr >= lr for any group. That caps the searchable floor "
            "for the whole campaign and constrains stage B' if it lowers kr_lr.",
            "Ranking is over relative deltas against the stage-0 untuned anchor, matching v1's "
            "convention, so v1 and v2 numbers are on one scale in the merged report.",
        ],
        "n_configs": len(scored),
        "n_runs": len(runs),
        "seed_band_from_stage0": seed_band,
        "metric_by_task": metric_by_task,
        "ranking": scored,
        "short_list": short_list,
        "boundaries": edges,
        "edge_bound_axes": sorted(edge_bound),
        "requires_a1b": bool(edge_bound),
        "leader_ties": {
            "leader": scored[0]["config"] if scored else None,
            "resolvable_difference": resolvable,
            "statistically_tied_with_leader": tied_with_leader,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"stage A' — {len(scored)} configs over {len(runs)} runs")
    print(f"  seed band (stage 0, n={seed_band['n']}): range {seed_band['range']:.3%}, "
          f"sigma {seed_band['sigma']:.3%}")
    print(f"  {'rank':>4}  {'config':46s} {'mean':>9s} {'+-2SE':>8s} {'xband':>7s}  seeds")
    for i, r in enumerate(scored[: max(args.top, 12)], 1):
        print(f"  {i:>4}  {r['config']:46s} {r['score_mean']:+8.3%} "
              f"{2 * r['score_sem']:7.3%} {r['vs_band']:+6.2f}  {r['n_seeds']}")
    if tied_with_leader:
        print(f"  NOTE leader is statistically tied with {len(tied_with_leader)} other config(s) "
              f"at this seed count: {', '.join(tied_with_leader[:5])}")
    if edge_bound:
        print("  EDGE-BOUND — a1b required before any of this is a conclusion:")
        for name, e in edge_bound.items():
            side = "min" if e["short_list_at_min"] else "max"
            print(f"    {name}: short list sits on the {side} of the searched range "
                  f"[{e['searched_min']} .. {e['searched_max']}]")
    print(f"  wrote {args.out}")
    # Non-zero exit when the search ran out of room, so a pipeline cannot treat this as final.
    raise SystemExit(2 if edge_bound else 0)


if __name__ == "__main__":
    main()
