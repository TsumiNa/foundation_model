#!/usr/bin/env python3
"""Rank a stage's grid points against the in-campaign untuned control.

Every stage-A/B run is a single-task probe, so a grid point's quality is read as its improvement
over the SAME probe run at the untuned baseline architecture — not against the historical
single-task ceilings, which were measured on different hardware with a different container. The
historical ceiling is printed as context only.

A grid point that spans several probe tasks (stage A5, all of stage B) is scored by the mean of
its per-task deltas, so a config has to win across tasks rather than exploit one.

    python .../rank.py <collected.csv> --metric mae --baseline a1_L128_H256_E0p005
    python .../rank.py <collected.csv> --metric r2 --group-suffix-tasks formation_energy volume
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
    rows = []
    for r in csv.DictReader(open(path)):
        # A single-task probe evaluates exactly one head; guard against stray rows.
        if r.get("new_task") and r["task"] != r["new_task"]:
            continue
        rows.append(r)
    return rows


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

    sign = -1.0 if args.metric in LOWER_IS_BETTER else 1.0
    scored = []
    for key, per_task in metric.items():
        shared = [t for t in per_task if t in base]
        if not shared:
            continue
        deltas = {t: sign * (per_task[t] - base[t]) for t in shared}
        scored.append((statistics.fmean(deltas.values()), key, per_task, deltas, shared))
    scored.sort(reverse=True)

    print(f"# {args.csv.name} — metric={args.metric} (baseline {base_key}), {len(scored)} configs")
    labels = [f"{t} (ceiling R2 {CEILING[t]:.3f})" if t in CEILING else t for t in tasks]
    print(f"# probe tasks: {', '.join(labels)}")
    header = ["rank", "config", f"mean Δ{args.metric}"] + [f"{t}" for t in tasks] + ["epochs"] + args.knobs
    widths = [4, max(len(k) for _, k, *_ in scored) + 1, 13] + [max(11, len(t) + 1) for t in tasks] + [7]
    widths += [max(10, len(k) + 1) for k in args.knobs]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))

    for i, (mean_delta, key, per_task, deltas, _) in enumerate(scored, 1):
        if args.top and i > args.top:
            break
        cells = [str(i), key, f"{mean_delta:+.4f}"]
        for t in tasks:
            cells.append(f"{per_task[t]:.4f}({deltas[t]:+.3f})" if t in deltas else "-")
        cells.append(str(max(epochs[key]) if epochs[key] else "-"))
        cells += [str(knobs[key].get(k, "")) for k in args.knobs]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    print()
    print(f"baseline {base_key}: " + ", ".join(f"{t}={base[t]:.4f}" for t in sorted(base)))


if __name__ == "__main__":
    main()
