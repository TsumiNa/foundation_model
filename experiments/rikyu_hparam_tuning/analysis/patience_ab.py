#!/usr/bin/env python3
"""Score the patience A/B and emit results/patience_ab.json.

The question: did #45 — stepping ReduceLROnPlateau once per EPOCH instead of once per BATCH —
change training performance?

Three arms, three seeds each, all on the probe3 3-task sequence:

    pab_old_*   image 0.2.1 + probe3.toml           per-BATCH cadence, one global weight_decay 1e-3
    pab_new_*   image 0.3.2 + patience_ab_new.toml  per-EPOCH cadence, weight decays PINNED to
                                                    0.2.1's values
    pab_asis_*  image 0.3.2 + probe3.toml           per-EPOCH cadence AND #42's new per-group
                                                    weight-decay defaults

    old  -> new   the patience effect, isolated (this is the question)
    new  -> asis  the weight-decay side effect that #42 introduced
    old  -> asis  what simply switching images does, both effects together

Why the pinning matters: #42 replaced one global ``weight_decay = 1e-3`` with per-group defaults
(encoder 1e-2, head 1e-5, ae 1e-3). Comparing 0.2.1 to 0.3.2 on an unmodified probe3.toml would
move the encoder's decay 10x and the heads' 100x at the same time as the cadence, and the result
would be unattributable. pab_new pins all four back; pab_asis deliberately does not, so the side
effect is measured instead of hidden.

SEEDS ARE NOT OPTIONAL HERE. Stage A measured a seed band of 8.48% on this same probe, which was
large enough to overturn its single-seed leader. A one-seed A/B on a 3-task probe cannot
distinguish a real effect from seed noise, so every arm is repeated at 3 seeds and every delta is
reported against the band rather than as a bare number.

The mechanism is NOT re-derived here — it is already pinned down by
`test_scheduler_steps_once_per_epoch_not_per_batch`, which asserts the scheduler is stepped once
per epoch rather than once per batch. No LR column is written to metrics.csv, so this script
measures the OUTCOME and leaves the mechanism to that test.

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
    "new": "0.3.2 · per-epoch cadence, weight decay pinned to 0.2.1",
    "asis": "0.3.2 · per-epoch cadence + #42 weight-decay defaults",
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
            "patience effect (old -> new)": delta("old", "new"),
            "weight-decay side effect (new -> asis)": delta("new", "asis"),
            "switching images as-is (old -> asis)": delta("old", "asis"),
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
