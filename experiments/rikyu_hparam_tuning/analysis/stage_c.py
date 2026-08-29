#!/usr/bin/env python3
"""Score the stage-C arms and emit results/stage_c.json.

Reads each arm straight from its per-step `training/stepNN_<task>/<task>_metrics.json` files
rather than from `metrics_table.csv`: a `--resume`d pretrain writes the table from the resuming
process's in-memory records only, so a run recovered from a walltime kill has a PARTIAL table
while the step JSONs are always complete (same caveat handled in
`experiments/replay_epoch_sweep/analysis/rebuild_metrics_from_stepdirs.py`). Stage C's runs are
long enough that at least one resume is likely, so this path is the default, not a fallback.

Reports what the replay campaign reports, so the two are comparable: mean R² over the 23
regression/kernel tasks (material_type is an accuracy task and is listed separately), plus the
deficit to each task's single-task ceiling averaged within the big/mid/small size groups.

    python .../stage_c.py --arm "untuned=<dir>" --arm "tuned=<dir>" \\
        --arm "untuned + consolidation=<dir>" -o ../results/stage_c.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

# Training-split label counts (from the replay campaign's analysis).
N_TRAIN = {
    "density": 23678, "efermi": 23668, "final_energy": 23678, "total_magnetization": 23678,
    "volume": 23678, "dielectric_total": 3124, "dielectric_ionic": 3124,
    "dielectric_electronic": 3124, "magnetization": 1160, "curie": 6272, "neel": 3466,
    "kp": 3875, "zt": 3445, "power_factor": 3638, "thermal_conductivity": 4272,
    "electrical_resistivity": 5051, "dos_density": 7009, "seebeck": 8072,
    "formation_energy": 23180, "magnetic_moment": 851, "tc": 7207, "klat": 3863,
}  # fmt: skip

# Single-task ceilings (H200, untuned architecture) — the deficit reference the replay campaign
# used. Kept for comparability with REPORT_20260809, not as a target.
CEILING = {
    "formation_energy": 0.9950, "density": 0.9882, "efermi": 0.9140,
    "dielectric_electronic": 0.8626, "tc": 0.7985, "curie": 0.7657, "magnetization": 0.7462,
    "total_magnetization": 0.7204, "klat": 0.6959, "final_energy": 0.6872, "kp": 0.6745,
    "neel": 0.6703, "dielectric_total": 0.6694, "thermal_conductivity": 0.6618, "zt": 0.6532,
    "magnetic_moment": 0.6408, "power_factor": 0.6336, "dielectric_ionic": 0.6078,
    "seebeck": 0.6026, "dos_density": 0.5999, "volume": 0.5685,
    "electrical_resistivity": 0.1622, "magnetic_susceptibility": 0.1238,
}  # fmt: skip

GROUPS = {
    "big": [t for t, n in N_TRAIN.items() if n >= 20000],
    "mid": [t for t, n in N_TRAIN.items() if 3000 <= n < 20000],
    "small": [t for t, n in N_TRAIN.items() if n < 3000],
}
ACCURACY_TASKS = {"material_type"}


def final_metrics(run_dir: Path) -> tuple[dict[str, dict], int]:
    """Every task's metrics at the LAST completed step, read from the authoritative step JSONs."""
    training = run_dir / "training"
    if not training.is_dir():
        # `fm finetune` writes a summary instead of step directories.
        summary = run_dir / "training" / "finetune_summary.json"
        raise SystemExit(f"{run_dir}: no training/ directory (looked for {summary})")

    steps: dict[int, Path] = {}
    for d in training.glob("step*_*"):
        m = re.match(r"step(\d+)_", d.name)
        if m:
            steps[int(m.group(1))] = d
    if steps:
        last = max(steps)
        out = {}
        for jf in sorted(steps[last].glob("*_metrics.json")):
            out[jf.name[: -len("_metrics.json")]] = json.load(open(jf))
        return out, last

    summary_path = training / "finetune_summary.json"
    if summary_path.exists():
        payload = json.load(open(summary_path))
        return payload.get("metrics_after") or payload.get("metrics_before") or {}, -1
    raise SystemExit(f"{run_dir}: neither step directories nor finetune_summary.json")


def score(metrics: dict[str, dict]) -> dict:
    r2 = {t: m["r2"] for t, m in metrics.items() if t not in ACCURACY_TASKS and m.get("r2") is not None}
    result = {
        "n_tasks": len(r2),
        "mean_r2": statistics.fmean(r2.values()) if r2 else float("nan"),
        "per_task": {t: round(v, 4) for t, v in sorted(r2.items())},
    }
    for group, tasks in GROUPS.items():
        deficits = [CEILING[t] - r2[t] for t in tasks if t in r2 and t in CEILING]
        result[group] = statistics.fmean(deficits) if deficits else float("nan")
    for task in ACCURACY_TASKS:
        if task in metrics:
            result[task] = {k: metrics[task].get(k) for k in ("accuracy", "macro_f1")}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        label, _, path = spec.partition("=")
        metrics, last_step = final_metrics(Path(path))
        entry = {"label": label, "dir": path, "last_step": last_step} | score(metrics)
        arms.append(entry)

    width = max(len(a["label"]) for a in arms)
    print(f"{'arm':{width}s}  {'tasks':>5s}  {'mean R²':>8s}  {'big':>7s}  {'mid':>7s}  {'small':>7s}  material_type")
    for a in arms:
        clf = a.get("material_type") or {}
        clf_text = f"acc {clf['accuracy']:.3f} / F1 {clf['macro_f1']:.3f}" if clf.get("accuracy") else "-"
        print(f"{a['label']:{width}s}  {a['n_tasks']:5d}  {a['mean_r2']:8.4f}  "
              f"{a['big']:7.4f}  {a['mid']:7.4f}  {a['small']:7.4f}  {clf_text}")

    notes = [
        "deficit = single-task ceiling (H200, untuned arch) − final R², averaged in the group.",
        "groups: big ≥20k (6 tasks) · mid 3k–8.1k (14) · small ≤1.2k (2); material_type excluded.",
        "read from per-step metrics JSONs, which stay complete across a --resume.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"arms": arms, "notes": notes}, indent=2) + "\n")
    print(f"\n{args.output}")


if __name__ == "__main__":
    main()
