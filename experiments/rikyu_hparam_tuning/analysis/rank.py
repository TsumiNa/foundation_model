#!/usr/bin/env python3
"""Rank a stage's grid points against the in-campaign untuned control.

A grid point's quality is its improvement over the SAME probe at the untuned baseline
architecture — never against the historical single-task ceilings, which were measured on other
hardware with a different container. The ceilings are printed as headroom context only.

Every grid point is scored by the MEAN of its per-task deltas, so a config has to win across the
probe's tasks rather than exploit one. Stage A's probe is a 3-task sequence (one task per size
group); stage B's grids span 2-3 tasks per head family.

    python .../rank.py stage_a.csv --metric mae --baseline a1_L128_H256_E0p005
    python .../rank.py stage_a.csv --metric r2 --score absolute --knobs model.latent_dim
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

# Single-task ceilings from experiments/rikyu_replay_sweep/results/warm_restart.csv (H200,
# untuned architecture). Context for how much headroom a probe task has — never a target.
CEILING = {
    "formation_energy": 0.9950, "density": 0.9882, "material_type": 0.9840, "efermi": 0.9140,
    "dielectric_electronic": 0.8626, "tc": 0.7985, "curie": 0.7657, "magnetization": 0.7462,
    "total_magnetization": 0.7204, "klat": 0.6959, "final_energy": 0.6872, "kp": 0.6745,
    "neel": 0.6703, "dielectric_total": 0.6694, "thermal_conductivity": 0.6618, "zt": 0.6532,
    "magnetic_moment": 0.6408, "power_factor": 0.6336, "dielectric_ionic": 0.6078,
    "seebeck": 0.6026, "dos_density": 0.5999, "volume": 0.5685,
    "electrical_resistivity": 0.1622, "magnetic_susceptibility": 0.1238,
}  # fmt: skip

# Metrics where a LOWER value is better.
LOWER_IS_BETTER = {"mae"}


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load(path: Path) -> list[dict]:
    """Every row of the collected CSV — collect.py already restricts to the final step.

    The stage-A probe is a 3-task sequence, so its final step carries one row per task (the
    newest plus the replayed older ones); all of them are the measurement.
    """
    return list(csv.DictReader(open(path)))


def config_key(runid: str, tasks: list[str]) -> str:
    """Strip a trailing probe-task suffix so the same config across tasks groups together."""
    for task in sorted(tasks, key=len, reverse=True):
        if runid.endswith(f"_{task}"):
            return runid[: -len(task) - 1]
    return runid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--metric", default="mae", help="mae | r2 | macro_f1 | accuracy")
    ap.add_argument("--baseline", required=True, help="runid (or config key) of the untuned control")
    ap.add_argument("--knobs", nargs="*", default=[], help="resolved-config columns to show")
    ap.add_argument("--top", type=int, default=0, help="print only the top N (0 = all)")
    ap.add_argument(
        "--score",
        choices=["auto", "absolute", "relative"],
        default="auto",
        help="how per-task deltas are combined; auto = relative when the probe spans >1 task",
    )
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        raise SystemExit(f"{args.csv}: no usable rows")
    tasks = sorted({r["task"] for r in rows})

    # metric[config][task] = value
    metric: dict[str, dict[str, float]] = defaultdict(dict)
    knobs: dict[str, dict[str, str]] = {}
    epochs: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        value = fnum(r.get(args.metric))
        if value is None:
            continue
        key = config_key(r["runid"], tasks)
        metric[key][r["task"]] = value
        knobs.setdefault(key, {k: r.get(k, "") for k in args.knobs})
        if fnum(r.get("epochs_run")) is not None:
            epochs[key].append(int(float(r["epochs_run"])))

    base_key = config_key(args.baseline, tasks)
    if base_key not in metric:
        raise SystemExit(f"baseline {args.baseline!r} (key {base_key!r}) not in {sorted(metric)}")
    base = metric[base_key]

    # Absolute deltas cannot be averaged across tasks whose metric lives on different scales
    # (formation_energy MAE ~0.06 vs a small task's ~0.4 — the big task would contribute almost
    # nothing to the mean). Relative improvement is scale-free, so it is the default whenever a
    # grid point spans more than one probe task.
    sign = -1.0 if args.metric in LOWER_IS_BETTER else 1.0
    relative = args.score == "relative" or (args.score == "auto" and len(tasks) > 1)

    def delta(value: float, ref: float) -> float:
        if relative:
            return sign * (value - ref) / abs(ref) if ref else 0.0
        return sign * (value - ref)

    scored = []
    for key, per_task in metric.items():
        shared = [t for t in per_task if t in base]
        if not shared:
            continue
        deltas = {t: delta(per_task[t], base[t]) for t in shared}
        scored.append((statistics.fmean(deltas.values()), key, per_task, deltas, shared))
    scored.sort(reverse=True)

    unit = "rel" if relative else "abs"
    print(f"# {args.csv.name} — metric={args.metric} [{unit}] vs {base_key}, {len(scored)} configs")
    labels = [f"{t} (ceiling R2 {CEILING[t]:.3f})" if t in CEILING else t for t in tasks]
    print(f"# probe tasks: {', '.join(labels)}")
    header = ["rank", "config", f"mean Δ{args.metric}({unit})"] + [f"{t}" for t in tasks] + ["epochs"] + args.knobs
    widths = [4, max(len(k) for _, k, *_ in scored) + 1, 16] + [max(18, len(t) + 1) for t in tasks] + [7]
    widths += [max(10, len(k) + 1) for k in args.knobs]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))

    for i, (mean_delta, key, per_task, deltas, _) in enumerate(scored, 1):
        if args.top and i > args.top:
            break
        cells = [str(i), key, (f"{mean_delta:+7.2%}" if relative else f"{mean_delta:+.4f}")]
        for t in tasks:
            if t not in deltas:
                cells.append("-")
            else:
                d = f"{deltas[t]:+.1%}" if relative else f"{deltas[t]:+.3f}"
                cells.append(f"{per_task[t]:.4f}({d})")
        cells.append(str(max(epochs[key]) if epochs[key] else "-"))
        cells += [str(knobs[key].get(k, "")) for k in args.knobs]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    print()
    print(f"baseline {base_key}: " + ", ".join(f"{t}={base[t]:.4f}" for t in sorted(base)))


if __name__ == "__main__":
    main()
