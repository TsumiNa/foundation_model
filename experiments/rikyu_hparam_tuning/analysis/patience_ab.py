#!/usr/bin/env python3
"""Score the patience A/B and emit results/patience_ab.json.

The question: did #45 — stepping ReduceLROnPlateau once per EPOCH instead of once per BATCH —
change training performance?

Three arms, three seeds each, on the probe3 3-task sequence:

    pab_old_*   image 0.2.1 + probe3.toml           per-BATCH cadence
    pab_asis_*  image 0.3.2 + probe3.toml           per-EPOCH cadence, SAME weight decays as 0.2.1
    pab_new_*   image 0.3.2 + patience_ab_new.toml  per-EPOCH cadence, weight decays deliberately
                                                    altered (encoder 1e-2->1e-3, head 1e-5->1e-3)

    old  -> asis   THE ANSWER: cadence changed, everything else held equal
    asis -> new    a weight-decay sensitivity probe, free with the runs already done
    old  -> new    both changes at once; reported only to show they do not interact

A CORRECTION IS BAKED INTO THESE LABELS. This experiment was designed believing #42 had changed
the weight-decay defaults (global 1e-3 -> encoder 1e-2 / head 1e-5), so pab_new "pinned them back"
to 1e-3. That belief was wrong: 0.2.1 hardcoded encoder=1e-2 (_engine.py:80) and head=1e-5
(_HEAD_WEIGHT_DECAY), and #42 lifted exactly those values into named config fields without
changing any of them. Verified field by field against the pre-#42 tree.

So pab_new does not restore 0.2.1 — it *introduces* a deviation, and pab_ASIS is the arm that
holds weight decay equal to 0.2.1. The clean cadence measurement is therefore old -> asis, not
old -> new. The conclusion is unchanged and slightly stronger; only which comparison carries it
has moved.

SEEDS ARE NOT OPTIONAL HERE. Stage A measured a seed band of 8.48% on this same probe, wide
enough to have overturned its single-seed leader. Every arm is repeated at 3 seeds and every
delta is reported against the band rather than as a bare number.

The mechanism is not re-derived here — `test_scheduler_steps_once_per_epoch_not_per_batch` already
asserts the per-epoch cadence, and no LR column is written to metrics.csv. This measures OUTCOME.

    python analysis/patience_ab.py --runs <outroot> -o results/patience_ab.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

PROBE_TASKS = ["formation_energy", "tc", "magnetization"]
ARMS = {
    "old": "0.2.1 · per-batch cadence",
    "asis": "0.3.2 · per-epoch cadence, weight decay identical to 0.2.1",
    "new": "0.3.2 · per-epoch cadence, weight decay altered (enc 1e-3, head 1e-3)",
}


def final_metrics(run_dir: Path) -> tuple[dict[str, dict], int]:
    """Every task's metrics at the LAST completed step, from the authoritative step JSONs.

    Same source as analysis/stage_c.py, and for the same reason: a --resume'd run writes a
    PARTIAL metrics_table.csv (only the resuming process's in-memory records) while the per-step
    JSONs are always complete.
    """
    training = run_dir / "training"
    steps: dict[int, Path] = {}
    for d in training.glob("step*_*"):
        m = re.match(r"step(\d+)_", d.name)
        if m:
            steps[int(m.group(1))] = d
    if not steps:
        raise SystemExit(f"{run_dir}: no step directories under {training}")
    last = max(steps)
    out = {}
    for jf in sorted(steps[last].glob("*_metrics.json")):
        out[jf.name[: -len("_metrics.json")]] = json.load(open(jf))
    return out, last


def mean_r2(metrics: dict[str, dict]) -> float | None:
    """Mean R² over the probe's three tasks; None if any is missing (never silently partial)."""
    values = []
    for task in PROBE_TASKS:
        entry = metrics.get(task)
        if entry is None or entry.get("r2") is None:
            return None
        values.append(float(entry["r2"]))
    return statistics.fmean(values)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True, help="the patience_ab output root")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    per_arm: dict[str, dict[int, float]] = {a: {} for a in ARMS}
    per_arm_tasks: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    missing: list[str] = []

    for arm in ARMS:
        for run in sorted(args.runs.glob(f"pab_{arm}_s*")):
            seed = int(run.name.rsplit("_s", 1)[1])
            if not (run / "DONE").exists():
                missing.append(f"{run.name} (no DONE marker)")
                continue
            metrics, _last = final_metrics(run)
            score = mean_r2(metrics)
            if score is None:
                missing.append(f"{run.name} (a probe task has no r2)")
                continue
            per_arm[arm][seed] = score
            per_arm_tasks[arm][seed] = {t: float(metrics[t]["r2"]) for t in PROBE_TASKS}

    summary = {}
    for arm, label in ARMS.items():
        seeds = per_arm[arm]
        if not seeds:
            continue
        vals = list(seeds.values())
        summary[arm] = {
            "label": label,
            "n_seeds": len(vals),
            "mean_r2": statistics.fmean(vals),
            # Band = spread across seeds. With 3 seeds a range is more honest than a stdev.
            "band": max(vals) - min(vals),
            "per_seed": seeds,
            "per_seed_tasks": per_arm_tasks[arm],
        }

    def delta(a: str, b: str) -> dict | None:
        if a not in summary or b not in summary:
            return None
        d = summary[b]["mean_r2"] - summary[a]["mean_r2"]
        # Compare against the WIDER of the two arms' bands: a delta smaller than the noise of
        # either arm is not a result.
        band = max(summary[a]["band"], summary[b]["band"])
        return {
            "delta_mean_r2": d,
            "band": band,
            "ratio": (d / band) if band else None,
            "verdict": "exceeds seed band" if band and abs(d) > band else "within seed band",
        }

    payload = {
        "probe_tasks": PROBE_TASKS,
        "arms": summary,
        "comparisons": {
            "PATIENCE EFFECT (old -> asis)": delta("old", "asis"),
            "weight-decay sensitivity (asis -> new)": delta("asis", "new"),
            "both changes together (old -> new)": delta("old", "new"),
        },
        "incomplete_runs": missing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"{'arm':<6}{'seeds':>6}{'mean R²':>10}{'band':>9}   {'description'}")
    for arm in ARMS:
        if arm not in summary:
            print(f"{arm:<6}{'-':>6}{'(no completed runs)':>10}")
            continue
        s = summary[arm]
        print(f"{arm:<6}{s['n_seeds']:>6}{s['mean_r2']:>10.4f}{s['band']:>9.4f}   {s['label']}")
    print()
    for name, cmp in payload["comparisons"].items():
        if cmp is None:
            print(f"{name:<40} (arm missing)")
            continue
        print(
            f"{name:<40} {cmp['delta_mean_r2']:+.4f}  band {cmp['band']:.4f}  ({cmp['ratio']:+.2f}x)  {cmp['verdict']}"
        )
    if missing:
        print(f"\nINCOMPLETE ({len(missing)}): " + ", ".join(missing))
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
