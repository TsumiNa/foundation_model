#!/usr/bin/env python3
"""How a task's outcome depends on how many tasks preceded it — summary/position.json

The transfer stage placed one task last in each run, and analysis/xfer.py reads exactly that: the
task under test, at position 24. But a run has 24 positions, and every step directory records the
metrics of every task trained so far. So a task that is last in its own runs also sits at some
random earlier position in every OTHER run, and its score there was written to disk at the time.

Fixing the viewpoint on one task and sweeping the runs therefore yields, for free, the curve this
campaign never budgeted for: **the task's performance as a function of how many tasks came before
it**, with several independent orderings at each position.

TWO READINGS, AND THE GAP BETWEEN THEM IS THE INTERESTING PART
--------------------------------------------------------------
Each position gives two numbers:

  * ``at_train`` — the score in the step directory where the task was trained. What the shared
    encoder, shaped by the k-1 tasks before it, was worth to this task at that moment.
  * ``at_end`` — the same task's score in the FINAL step directory, after every later task has
    trained and replay has been rehearsing it. What survived.

``at_end - at_train`` is retention: positive means later training helped it further, negative means
replay did not fully hold it. A task read only at position 24 cannot show either, because there is
no "later" for it.

Both are compared against the same-régime single-task baseline, so "does multi-task help" is
answered per position rather than once.

CAVEATS THAT LIMIT WHAT THIS CAN CLAIM
--------------------------------------
* Position is confounded with WHICH tasks preceded it. The orderings are random, so across enough
  runs the identity of the predecessors averages out — but at ~3-6 samples per position it averages
  out weakly, and a per-position mean carries that noise.
* The test rows here are each run's own, not the intersection used by matched_test.py. Within this
  file every arm is a multi-task run scored on the same universe, so the position comparison is
  internally consistent; the single-task baseline is the one measured on a smaller universe, and
  that offset applies equally to every position.

    python analysis/position.py --runs <outroot>/stage_xfer \\
        --ceilings summary/ceilings_adopted.json -o summary/position.json
"""

from __future__ import annotations

import argparse
import json
import re
import math
import statistics
from collections import defaultdict
from pathlib import Path

from common import N_TRAIN, pct_views, size_group

STEP = re.compile(r"step(\d+)_(.+)$")


def run_steps(run: Path) -> dict[int, tuple[str, Path]]:
    """{position: (task trained there, its step directory)} for one run."""
    out: dict[int, tuple[str, Path]] = {}
    for d in run.glob("training/step*_*"):
        m = STEP.match(d.name)
        if m:
            out[int(m.group(1))] = (m.group(2), d)
    return out


def metric(step_dir: Path, task: str) -> float | None:
    path = step_dir / f"{task}_metrics.json"
    if not path.exists():
        return None
    data = json.load(open(path))
    value = data.get("r2")
    if value is None:
        value = data.get("macro_f1")
    return float(value) if value is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--ceilings", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--min-samples", type=int, default=2,
                    help="positions with fewer samples are recorded but flagged")
    args = ap.parse_args()

    single = json.loads(args.ceilings.read_text())
    runs = [d for d in sorted(args.runs.iterdir()) if d.is_dir() and (d / "DONE").exists()]

    # task -> position -> list of (at_train, at_end)
    seen: dict[str, dict[int, list[tuple[float | None, float | None]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        steps = run_steps(run)
        if not steps:
            continue
        last_dir = steps[max(steps)][1]
        for pos, (task, step_dir) in steps.items():
            seen[task][pos].append((metric(step_dir, task), metric(last_dir, task)))

    rows = []
    for task in sorted(seen):
        base = single.get(task)
        by_pos = []
        for pos in sorted(seen[task]):
            pairs = seen[task][pos]
            at_train = [a for a, _ in pairs if a is not None]
            at_end = [b for _, b in pairs if b is not None]
            if not at_train:
                continue
            entry = {
                "position": pos,
                "n": len(at_train),
                "at_train_mean": statistics.fmean(at_train),
                "at_train_sd": statistics.stdev(at_train) if len(at_train) > 1 else 0.0,
                "at_end_mean": statistics.fmean(at_end) if at_end else None,
                "at_end_sd": statistics.stdev(at_end) if len(at_end) > 1 else 0.0,
                "thin": len(at_train) < args.min_samples,
            }
            if base:
                entry["vs_single_at_train"] = entry["at_train_mean"] - base["mean"]
                if entry["at_end_mean"] is not None:
                    entry["vs_single_at_end"] = entry["at_end_mean"] - base["mean"]
                    entry["retention"] = entry["at_end_mean"] - entry["at_train_mean"]
            by_pos.append(entry)
        if not by_pos:
            continue
        # Group the raw observations, not the per-position means: a mean of means throws away the
        # sample count each position carries and cannot give a standard error.
        early_raw = [a for pos in seen[task] if pos <= 8 for a, _ in seen[task][pos] if a is not None]
        late_raw = [a for pos in seen[task] if pos >= 17 for a, _ in seen[task][pos] if a is not None]
        ret_raw = [b - a for pos in seen[task] for a, b in seen[task][pos]
                   if a is not None and b is not None]

        def against_single(sample):
            """Group mean vs the single-task baseline, with both arms' uncertainty."""
            if not sample or not base:
                return None
            m = statistics.fmean(sample)
            sd = statistics.stdev(sample) if len(sample) > 1 else 0.0
            se = (math.sqrt(sd**2 / len(sample) + base["sd"] ** 2 / base["n"])
                  if len(sample) > 1 and base["n"] > 1 else None)
            d = m - base["mean"]
            views = pct_views(d, base["mean"])
            sep = bool(se) and abs(d) > 2 * se
            return {"n": len(sample), "mean": m, "sd": sd, "delta": d,
                    "relative_pct": views["relative_pct"], "se_of_difference": se,
                    "separated": sep, "practically_significant": views["practically_significant"],
                    "matters": sep and views["practically_significant"]}

        early_s, late_s = against_single(early_raw), against_single(late_raw)
        # Early vs late is the position effect itself, and it needs no baseline at all.
        pos_effect = None
        if len(early_raw) > 1 and len(late_raw) > 1:
            me, ml = statistics.fmean(early_raw), statistics.fmean(late_raw)
            se = math.sqrt(statistics.stdev(early_raw) ** 2 / len(early_raw)
                           + statistics.stdev(late_raw) ** 2 / len(late_raw))
            d = ml - me
            views = pct_views(d, me)
            sep = bool(se) and abs(d) > 2 * se
            pos_effect = {"delta_late_minus_early": d, "relative_pct": views["relative_pct"],
                          "se_of_difference": se, "separated": sep,
                          "practically_significant": views["practically_significant"],
                          "matters": sep and views["practically_significant"],
                          "n_early": len(early_raw), "n_late": len(late_raw)}
        retention = None
        if len(ret_raw) > 1:
            m = statistics.fmean(ret_raw)
            se = statistics.stdev(ret_raw) / math.sqrt(len(ret_raw))
            retention = {"mean": m, "sd": statistics.stdev(ret_raw), "n": len(ret_raw),
                         "se": se, "separated_from_zero": abs(m) > 2 * se,
                         "relative_pct": (m / base["mean"] * 100) if base and base["mean"] else None}

        rows.append({
            "task": task,
            "group": size_group(task),
            "n_train": N_TRAIN.get(task),
            "single_task": base["mean"] if base else None,
            "single_task_sd": base["sd"] if base else None,
            "positions_covered": len(by_pos),
            "total_observations": sum(e["n"] for e in by_pos),
            "by_position": by_pos,
            "early_1_8": early_s,
            "late_17_24": late_s,
            "position_effect": pos_effect,
            "retention": retention,
        })

    out = {
        "question": "for each task, how does its outcome depend on how many tasks preceded it?",
        "per_task": rows,
        "reading": {
            "at_train": "score in the step where the task was trained — what the encoder shaped by "
                        "the preceding tasks was worth to it at that moment",
            "at_end": "the same task's score in the final step directory — what survived after every "
                      "later task trained and replay rehearsed it",
            "retention": "at_end - at_train; positive means later training helped it further",
        },
        "caveats": [
            "Position is confounded with WHICH tasks preceded it; random orderings average that out "
            "only weakly at a few samples per position.",
            "Scores are each run's own test rows, not the intersection used by matched_test.py. The "
            "comparison ACROSS positions is internally consistent; the single-task baseline carries "
            "a fixed offset that applies equally to every position.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    def cell(d):
        """Absolute and relative together — a delta alone does not say whether it is a lot."""
        if not d:
            return f"{'-':>19s}"
        star = "*" if d.get("matters") else (" " if not d.get("separated") else "·")
        return f"{d['delta']:+8.4f} {d['relative_pct']:+7.2f}%{star}"

    print(f"{'task':22s} {'N':>6s} {'single':>7s} | {'early (pos 1-8)':>19s} | "
          f"{'late (pos 17-24)':>19s} | {'late - early':>19s} | {'retention':>17s}")
    for r in sorted(rows, key=lambda r: -(r["n_train"] or 0)):
        pe = r["position_effect"]
        pe_cell = (f"{pe['delta_late_minus_early']:+8.4f} {pe['relative_pct']:+7.2f}%"
                   + ("*" if pe.get("matters") else ("·" if pe["separated"] else " "))) if pe else f"{'-':>19s}"
        ret = r["retention"]
        ret_cell = (f"{ret['mean']:+8.4f} {ret['relative_pct']:+6.2f}%"
                    + ("*" if ret["separated_from_zero"] else " ")) if ret else f"{'-':>17s}"
        print(f"{r['task']:22s} {r['n_train'] or 0:6d} "
              f"{r['single_task'] if r['single_task'] is None else round(r['single_task'], 4):>7} | "
              f"{cell(r['early_1_8'])} | {cell(r['late_17_24'])} | {pe_cell} | {ret_cell}")

    print("\n  * = separated AND |delta| >= 0.01   · = separated but below the practical threshold")
    print("  early / late: the task's score at that band of positions minus its single-task baseline")
    print("  late - early: the position effect itself, needing no baseline")
    print("  retention: final score minus the score right after the task trained")

    def count(key, cond):
        return sum(1 for r in rows if r.get(key) and cond(r[key]))
    print(f"\n  tasks where multi-task MATTERS at early positions: "
          f"{count('early_1_8', lambda d: d['matters'] and d['delta'] > 0)} better, "
          f"{count('early_1_8', lambda d: d['matters'] and d['delta'] < 0)} worse")
    print(f"  tasks where multi-task MATTERS at late positions:  "
          f"{count('late_17_24', lambda d: d['matters'] and d['delta'] > 0)} better, "
          f"{count('late_17_24', lambda d: d['matters'] and d['delta'] < 0)} worse")
    print(f"  tasks with a resolvable POSITION effect (late vs early): "
          f"{count('position_effect', lambda d: d['matters'])} of {len(rows)}")
    print(f"  tasks whose score still IMPROVES after they finish training: "
          f"{count('retention', lambda d: d['separated_from_zero'] and d['mean'] > 0)} of {len(rows)}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
