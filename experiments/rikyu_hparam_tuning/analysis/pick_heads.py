#!/usr/bin/env python3
"""Pick each task's own head configuration from the stage-B grids.

Stage B is 24 independent optimisations: for every task, the grid of head configurations is
ranked *within that task* and its winner recorded. The output JSON is exactly the set of
`[[tasks]]` overrides the final config needs, so nothing is retyped by hand between the grid and
stage C.

    python .../pick_heads.py stage_b.csv -o ../results/head_winners.json

Metric choice is per task and automatic, because the catalog's tasks do not share a regime:

* classification -> ``macro_f1`` (``material_type`` measured accuracy 0.989 / macro-F1 0.551, so
  accuracy carries no signal);
* regression / kernel-regression -> ``r2``, unless the grid's whole R² spread for that task is
  below ``--r2-floor`` (default 0.005), i.e. the task is saturated or degenerate and R² cannot
  separate the configurations. Those tasks fall back to ``mae``.

The rule applied is printed per task, never silently.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# Baseline (untuned) head configuration, = configs/single_task.toml.
BASE_REG = {"hidden_dims": [64], "lr": 5e-3}
BASE_KR = {"x_hidden_dims": [128, 64], "n_kernel": 15, "lr": 5e-4}

LOWER_IS_BETTER = {"mae"}

# Only these runid prefixes are per-task head grids. The stage-B output directory also holds the
# B-mt control (multi-task probes) and the B4 seed repeats, whose rows carry the SAME task names —
# without this filter a task's winner can be picked from a different probe entirely, which is not
# a comparison at all. (This happened: magnetization's winner was once taken from a bmtreg run.)
PER_TASK_PREFIXES = ("breg_", "bkr_", "bclf_")


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_list(text: str) -> list[int]:
    """Resolved-config columns arrive as a python-repr list string."""
    return [int(v) for v in text.strip("[]").replace(" ", "").split(",") if v]


def kind_of(runid: str) -> str:
    return runid.split("_", 1)[0]  # breg | bkr | bclf


def overrides_for(kind: str, row: dict) -> dict:
    """The `[[tasks]]` keys that carry this run's head configuration."""
    if kind == "bkr":
        return {
            "x_hidden_dims": parse_list(row["model.kr_x_hidden_dims"]),
            "n_kernel": int(float(row["model.n_kernel"])),
            "lr": float(row["training.kr_lr"]),
        }
    return {
        "hidden_dims": parse_list(row["model.head_hidden_dims"]),
        "lr": float(row["training.head_lr"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--r2-floor", type=float, default=0.005, help="R² spread below which a task falls back to MAE")
    ap.add_argument("--top", type=int, default=3, help="grid points to show per task")
    args = ap.parse_args()

    # rows[task][runid] = row
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    skipped = 0
    for r in csv.DictReader(open(args.csv)):
        if not r["runid"].startswith(PER_TASK_PREFIXES):
            skipped += 1
            continue
        rows[r["task"]][r["runid"]] = r
    if not rows:
        raise SystemExit(f"{args.csv}: no per-task rows (prefixes {PER_TASK_PREFIXES})")
    if skipped:
        print(f"# ignored {skipped} row(s) from other probes (B-mt control / B4 repeats)")

    winners: dict[str, dict] = {}
    report: list[str] = []

    for task in sorted(rows):
        entries = rows[task]
        kind = kind_of(next(iter(entries)))

        if kind == "bclf":
            metric, why = "macro_f1", "classification"
        else:
            r2s = [v for v in (fnum(r.get("r2")) for r in entries.values()) if v is not None]
            spread = (max(r2s) - min(r2s)) if r2s else 0.0
            if r2s and spread >= args.r2_floor:
                metric, why = "r2", f"R² spread {spread:.4f}"
            else:
                metric, why = "mae", f"R² spread {spread:.4f} < {args.r2_floor} — saturated, using MAE"

        scored = [
            (fnum(r.get(metric)), runid, r) for runid, r in entries.items() if fnum(r.get(metric)) is not None
        ]
        if not scored:
            report.append(f"{task:26s}  SKIPPED — no usable {metric}")
            continue
        scored.sort(key=lambda t: t[0], reverse=metric not in LOWER_IS_BETTER)

        best_value, best_runid, best_row = scored[0]
        base = BASE_KR if kind == "bkr" else BASE_REG
        chosen = overrides_for(kind, best_row)
        # Drop keys that already equal the baseline: the final config stays as small as possible
        # and the diff shows only what tuning actually changed.
        override = {k: v for k, v in chosen.items() if v != base.get(k)}
        winners[task] = {
            "kind": kind,
            "metric": metric,
            "value": best_value,
            "runid": best_runid,
            "override": override,
            "full": chosen,
        }

        base_runid = next(
            (runid for runid, r in entries.items() if overrides_for(kind, r) == base), None
        )
        base_value = fnum(entries[base_runid].get(metric)) if base_runid else None
        if base_value is not None:
            sign = -1 if metric in LOWER_IS_BETTER else 1
            delta = sign * (best_value - base_value)
            rel = delta / abs(base_value) if base_value else 0.0
            winners[task]["baseline_value"] = base_value
            winners[task]["gain"] = delta
            gain = f"{delta:+.4f} ({rel:+.1%})"
        else:
            gain = "no baseline point"

        report.append(
            f"{task:26s}  {metric:8s}  best {best_value:.4f}  vs base {gain:22s}  "
            f"{best_runid.rsplit('_' + task, 1)[0]}   [{why}]"
        )
        for value, runid, _ in scored[1 : args.top]:
            report.append(f"{'':26s}  {'':8s}       {value:.4f}  {'':22s}  {runid.rsplit('_' + task, 1)[0]}")

    print(f"# {args.csv.name} — per-task head winners ({len(winners)} tasks)")
    print("\n".join(report))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(winners, indent=2, sort_keys=True) + "\n")
    changed = sum(1 for w in winners.values() if w["override"])
    print(f"\n{args.output}  ({changed}/{len(winners)} tasks differ from the untuned head)")


if __name__ == "__main__":
    main()
