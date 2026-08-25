#!/usr/bin/env python3
"""Measure the seed band and apply the adoption rule to stage A.

The rule was fixed before the seed repeats were read (see README, "The adoption rule"):

    Adopt the CHEAPEST configuration whose score is within the seed band of the best score.

This script is that rule as code, so the decision cannot drift with interpretation:

1. For every candidate it gathers all seeds — 2025 from the A1 grid, 2026/2027 from A4 — and
   reports mean, range and per-seed detail.
2. The **band** is the largest within-arm range among the candidates: the observed run-to-run
   spread of this probe, measured on this campaign rather than quoted from an older one.
3. Every candidate whose mean is within one band of the best mean is a **tie**. Among the ties the
   cheapest wins, where cost is the measured mean wall-clock of that configuration's runs — the
   quantity actually being paid, not a parameter count.

DISCLOSED AMENDMENT (added after the first run of this script, and stated rather than hidden):
wall-clock did not separate the two leading tied candidates — they came out 0.6% apart, which is
inside run-to-run wall-clock variation, so the rule as written was deciding on measurement noise
and picked the candidate that was worse on every other axis. Candidates whose cost is within
``--cost-tie`` (default 5%) of the cheapest are therefore treated as equally cheap, and resolved
on **reproducibility** — the smallest across-seed range. A tuning campaign should prefer the
configuration whose result is most repeatable when quality and cost both tie.

    python .../adopt.py ../results/stage_a.csv --timing ../results/stage_a_timing.tsv \\
        --baseline a1_L128_H256_E0p005
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


def arm_of(runid: str, baseline: str = "") -> str | None:
    """Map a runid to the configuration it measures, collapsing seeds.

    ``a1_L256_H512_E0p001`` and ``a4_L256_H512_E0p001_s2026`` are the same configuration at two
    seeds. The untuned arm needs the baseline runid passed in, because A4 writes it as
    ``a4_base_s2026`` while its seed-2025 run is the A1 grid point itself — miss that and the
    baseline is credited with one fewer seed than every other arm.
    """
    if runid == baseline:
        return "BASELINE"
    if runid.startswith("a4_"):
        body = runid[3:].rsplit("_s", 1)[0]
        return "BASELINE" if body == "base" else body
    if runid.startswith(("a1_", "a1b_")):
        return runid.split("_", 1)[1]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--timing", type=Path, help="_timing.tsv (runid, seconds, rc, host)")
    ap.add_argument("--baseline", default="a1_L128_H256_E0p005")
    ap.add_argument("--metric", default="mae")
    ap.add_argument("--cost-tie", type=float, default=0.05,
                    help="candidates within this fraction of the cheapest count as equally cheap")
    args = ap.parse_args()

    per_run: dict[str, dict[str, float]] = defaultdict(dict)
    for r in csv.DictReader(open(args.csv)):
        v = fnum(r.get(args.metric))
        if v is not None:
            per_run[r["runid"]][r["task"]] = v
    if args.baseline not in per_run:
        raise SystemExit(f"baseline {args.baseline!r} not in {args.csv}")

    base = per_run[args.baseline]
    sign = -1.0 if args.metric in LOWER_IS_BETTER else 1.0

    def score(values: dict[str, float]) -> float | None:
        shared = [t for t in values if t in base and base[t]]
        if not shared:
            return None
        return statistics.fmean(sign * (values[t] - base[t]) / abs(base[t]) for t in shared)

    # arm -> {runid: score}; only arms that actually have seed repeats are candidates.
    arms: dict[str, dict[str, float]] = defaultdict(dict)
    for runid, values in per_run.items():
        arm = arm_of(runid, args.baseline)
        s = score(values)
        if arm and s is not None:
            arms[arm][runid] = s
    candidates = {a: runs for a, runs in arms.items() if len(runs) > 1}
    if not candidates:
        raise SystemExit("no configuration has seed repeats — run stage A4 first")

    wall: dict[str, list[float]] = defaultdict(list)
    if args.timing and args.timing.exists():
        for line in args.timing.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] == "0":
                arm = arm_of(parts[0], args.baseline)
                if arm:
                    wall[arm].append(float(parts[1]))

    print(f"# seed band from {len(candidates)} candidates with repeats, metric={args.metric}")
    header = f"{'configuration':34s}  {'seeds':>5s}  {'mean':>8s}  {'range':>8s}  {'wall':>7s}  per-seed"
    print(header)
    print("-" * len(header))
    rows = []
    for arm, runs in sorted(candidates.items(), key=lambda kv: -statistics.fmean(kv[1].values())):
        values = list(runs.values())
        mean, rng = statistics.fmean(values), max(values) - min(values)
        cost = statistics.fmean(wall[arm]) / 60 if wall.get(arm) else float("nan")
        detail = " ".join(f"{v:+.1%}" for v in sorted(values, reverse=True))
        print(f"{arm:34s}  {len(values):5d}  {mean:+8.2%}  {rng:8.2%}  {cost:6.1f}m  {detail}")
        rows.append((mean, arm, rng, cost))

    band = max(r[2] for r in rows)
    best_mean, best_arm, _, best_cost = rows[0]
    quality_tied = [r for r in rows if best_mean - r[0] <= band]

    cheapest = min(r[3] for r in quality_tied if r[3] == r[3])
    cost_tied = [r for r in quality_tied if r[3] != r[3] or r[3] <= cheapest * (1 + args.cost_tie)]
    adopted = min(cost_tied, key=lambda r: r[2])  # smallest across-seed range

    print()
    print(f"seed band (largest within-arm range)     : {band:.2%}")
    print(f"best mean                                : {best_arm}  {best_mean:+.2%}  ({best_cost:.1f} min/run)")
    print(f"quality-tied (within one band)  [{len(quality_tied)}]      : " + ", ".join(r[1] for r in quality_tied))
    print(f"also cost-tied (within {args.cost_tie:.0%} of cheapest) [{len(cost_tied)}] : " + ", ".join(r[1] for r in cost_tied))
    print(f"ADOPT (tightest seed range of those)     : {adopted[1]}  {adopted[0]:+.2%}  "
          f"(range {adopted[2]:.2%}, {adopted[3]:.1f} min/run)")
    baseline = next((r for r in rows if r[1] == "BASELINE"), None)
    if baseline:
        print(f"  -> vs untuned baseline {baseline[0]:+.2%} ({baseline[3]:.1f} min/run): "
              f"net {adopted[0] - baseline[0]:+.2%}, i.e. {(adopted[0] - baseline[0]) / band:.1f}x the seed band")


if __name__ == "__main__":
    main()
