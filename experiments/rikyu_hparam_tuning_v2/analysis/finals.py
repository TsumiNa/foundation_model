#!/usr/bin/env python3
"""Score a finals stage (A'3 / B'3) at high seed count and emit its summary JSON.

The finals exist to answer one question the grid cannot: **is the ordering real?**

v1 could not answer it and said so. Its top three were 1.5-1.8% apart while three seeds resolve
only ~5.8%, so it reported a three-way tie — the honest reading, and the reason PLAN §7.1 spends
surplus compute on seeds rather than on more grid points. Ranking N noisy points by their maximum
biases that maximum upward; adding points makes it worse, adding seeds makes it better.

So this script does NOT just sort by mean. It reports, for every pair:

  * the difference, and the seeds' ability to resolve a difference that size;
  * whether the pair is separated or merely ordered-by-noise;
  * both R2 and MAE, because v1's stage-A gain was real on one and absent on the other, and
    picking whichever agrees with the ranking is the disclosure failure PLAN §6.2 forbids.

A finals result that cannot separate its top configurations is a VALID and reportable outcome. It
means the configurations are equivalent at this budget, and the adoption decision should fall to a
secondary criterion (wall clock, or the simpler architecture) stated openly rather than to a
decimal place that is noise.

    python analysis/finals.py --runs <outroot>/stage_a --pattern 'a3_*' \\
        --stage0 summary/stage0.json -o summary/finals_a.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

from common import (
    PROBE6,
    band,
    exceeds_band,
    group_by_config,
    load_runs,
    relative_score,
    seeds_needed,
    size_group,
    task_values,
)


def welch(a: list[float], b: list[float]) -> dict:
    """Welch's t statistic and degrees of freedom for two seed samples.

    Welch rather than Student because the arms are not assumed to share a variance: a
    configuration can be both better AND more stable, and pooling would hide the second half.
    No p-value is computed — with 25 seeds the effect size against the band is the honest
    summary, and a p-value here would invite reading significance into a ranking that the
    band already says is a tie.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"t": None, "df": None, "se": None}
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": None, "df": None, "se": 0.0}
    t = (statistics.fmean(a) - statistics.fmean(b)) / se
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    return {"t": t, "df": num / den if den else None, "se": se}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--pattern", required=True, help="e.g. 'a3_*'")
    ap.add_argument("--stage0", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=PROBE6)
    ap.add_argument("--stage", default="finals")
    args = ap.parse_args()

    stage0 = json.loads(args.stage0.read_text())
    reference = stage0["reference"]
    seed_band = stage0["arms"]["s0_base"]["score"]
    metric_by_task = stage0["probe"]["metric_by_task"]

    runs = load_runs(args.runs, args.pattern, args.tasks)
    if not runs:
        raise SystemExit(f"no complete runs under {args.runs} matching {args.pattern}")
    configs = group_by_config(runs)

    arms: dict[str, dict] = {}
    for label, per_seed in sorted(configs.items()):
        scores = {}
        for seed, seed_metrics in sorted(per_seed.items()):
            s = relative_score(seed_metrics, reference, metric_by_task, args.tasks)
            if s is not None:
                scores[seed] = s
        if not scores:
            continue
        arms[label] = {
            "n_seeds": len(scores),
            "score": band(list(scores.values())),
            "per_seed": scores,
            "per_task": {
                t: {
                    "metric": metric_by_task[t],
                    "group": size_group(t),
                    "r2_mean": (statistics.fmean(task_values(per_seed, t, "r2"))
                                if task_values(per_seed, t, "r2") else None),
                    "mae_mean": (statistics.fmean(task_values(per_seed, t, "mae"))
                                 if task_values(per_seed, t, "mae") else None),
                    "r2_sigma": (statistics.stdev(task_values(per_seed, t, "r2"))
                                 if len(task_values(per_seed, t, "r2")) > 1 else None),
                }
                for t in args.tasks
            },
        }

    order = sorted(arms, key=lambda k: arms[k]["score"]["mean"], reverse=True)

    pairwise = []
    for a, b in itertools.combinations(order, 2):
        sa = list(arms[a]["per_seed"].values())
        sb = list(arms[b]["per_seed"].values())
        diff = arms[a]["score"]["mean"] - arms[b]["score"]["mean"]
        w = welch(sa, sb)
        # Separated when the gap clears two standard errors of the DIFFERENCE — the quantity that
        # actually governs whether the ordering survives another draw of seeds.
        separated = bool(w["se"]) and abs(diff) > 2 * w["se"]
        pairwise.append({
            "better": a,
            "worse": b,
            "delta": diff,
            "se_of_difference": w["se"],
            "resolvable_at_this_n": 2 * w["se"] if w["se"] else None,
            "t": w["t"],
            "separated": separated,
            "verdict": "ordering is supported" if separated else "TIED — ordering is not supported",
        })

    leader = order[0] if order else None
    tied_with_leader = [
        p["worse"] for p in pairwise if p["better"] == leader and not p["separated"]
    ]

    # What it would have taken to separate the pairs that came out tied.
    sigma_pool = statistics.fmean([arms[a]["score"]["sigma"] for a in order]) if order else 0.0
    unresolved = [p for p in pairwise if not p["separated"] and p["delta"]]
    seeds_for_ties = {
        f"{p['better']} vs {p['worse']}": seeds_needed(sigma_pool, abs(p["delta"]))
        for p in unresolved
    }

    out = {
        "stage": args.stage,
        "pattern": args.pattern,
        "n_runs": len(runs),
        "seed_band_from_stage0": seed_band,
        "metric_by_task": metric_by_task,
        "ranking": order,
        "arms": arms,
        "pairwise": pairwise,
        "leader": leader,
        "statistically_tied_with_leader": tied_with_leader,
        "ranking_is_fully_resolved": not tied_with_leader,
        "seeds_that_would_resolve_the_ties": seeds_for_ties,
        "vs_anchor": [
            {
                "arm": a,
                "delta_vs_untuned": arms[a]["score"]["mean"],
                "vs_band": exceeds_band(arms[a]["score"]["mean"], seed_band)[1],
                "exceeds_band": exceeds_band(arms[a]["score"]["mean"], seed_band)[0],
            }
            for a in order
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{args.stage} — {len(arms)} arms over {len(runs)} runs")
    for i, a in enumerate(order, 1):
        s = arms[a]["score"]
        print(f"  {i}. {a:44s} {s['mean']:+8.3%}  +-2SE {2 * s['sem']:7.3%}  "
              f"sigma {s['sigma']:6.3%}  n={s['n']}")
    print("  pairwise:")
    for p in pairwise:
        mark = "OK  " if p["separated"] else "TIE "
        print(f"    {mark} {p['better']} > {p['worse']}: {p['delta']:+.3%} "
              f"(resolvable {p['resolvable_at_this_n']:.3%})" if p["resolvable_at_this_n"]
              else f"    {mark} {p['better']} > {p['worse']}")
    if tied_with_leader:
        print(f"  LEADER NOT SEPARATED from: {', '.join(tied_with_leader)}")
        print(f"    seeds that would separate them: {seeds_for_ties}")
    else:
        print(f"  ranking fully resolved; leader = {leader}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
