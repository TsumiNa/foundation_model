#!/usr/bin/env python3
"""Warm-start fine-tune from the transfer checkpoints, per task — summary/ft.json

The transfer stage said 18 of 24 tasks do worse when they arrive last in a continual sequence. But
at that step the task shares the encoder with replay from 23 other tasks and owns about 1% of the
gradient if it is small. This stage takes the SAME final checkpoint and trains the SAME task again
with replay removed, in two arms:

  ftz  frozen encoder, head only    is the shared representation itself good for X?
  ftf  encoder + head               warm-start fine-tune, the model library's intended use

Three per-task differences answer three different questions, and none is pooled across tasks:

  ftf - single   does warm-starting beat training alone?        (the library's use case)
  ftz - single   is the representation alone enough?            (representation quality)
  ftf - xfer     how much did replay at step 24 cost?           (the dilution mechanism)

Every difference carries the SE of BOTH arms. material_type is scored on macro-F1 (its `primary` is
accuracy, which sits at 0.99 in every arm and cannot move); everything else on R2.

    python analysis/ft.py --runs <outroot>/stage_ft --ceilings summary/ceilings_adopted.json \\
        --xfer summary/matched_xfer.json -o summary/ft.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from common import N_TRAIN, pct_views, size_group

ARMS = {"ftz": "frozen encoder, head only", "ftf": "encoder + head (warm-start)"}


def metric_of(path: Path) -> float | None:
    """macro-F1 for the classification task, R2 otherwise — never `primary`."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    v = d.get("macro_f1") if "macro_f1" in d else d.get("r2")
    return float(v) if v is not None else None


def collect(runs: Path, arm: str, task: str) -> list[float]:
    out = []
    for run in sorted(runs.glob(f"{arm}_{task}_o*")):
        if not (run / "DONE").exists():
            continue
        v = metric_of(run / "training" / "finetune" / f"{task}_metrics.json")
        if v is not None:
            out.append(v)
    return out


def diff(a: list[float], b_mean: float, b_sd: float, b_n: int) -> dict | None:
    """a's mean minus b, with both arms' uncertainty; None if a is empty."""
    if not a:
        return None
    m = statistics.fmean(a)
    sd = statistics.stdev(a) if len(a) > 1 else 0.0
    se = math.sqrt(sd ** 2 / len(a) + (b_sd ** 2 / b_n if b_n > 1 else 0.0)) if len(a) > 1 else None
    d = m - b_mean
    views = pct_views(d, b_mean)
    sep = bool(se) and abs(d) > 2 * se
    return {"n": len(a), "mean": m, "sd": sd, "delta": d, "relative_pct": views["relative_pct"],
            "se_of_difference": se, "separated": sep,
            "practically_significant": views["practically_significant"],
            "matters": sep and views["practically_significant"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--ceilings", type=Path, required=True)
    ap.add_argument("--xfer", type=Path, required=True, help="summary/matched_xfer.json")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    single = json.loads(args.ceilings.read_text())
    xfer = {r["task"]: r for r in json.loads(args.xfer.read_text())["per_task"] if "multi_task" in r}

    rows = []
    for task in sorted(N_TRAIN, key=lambda t: -N_TRAIN[t]):
        base = single.get(task)
        if not base:
            continue
        ftz, ftf = collect(args.runs, "ftz", task), collect(args.runs, "ftf", task)
        xr = xfer.get(task)
        # xfer's per-task spread is in transfer_xfer.json; matched_xfer carries only the mean.
        # Against a single number the SE is one-sided, which is the honest reading of that column.
        ftf_vs_xfer = diff(ftf, xr["multi_task"], 0.0, 1) if (ftf and xr) else None
        ftf_vs_ftz = None
        if ftf and len(ftz) > 1:
            ftf_vs_ftz = diff(ftf, statistics.fmean(ftz), statistics.stdev(ftz), len(ftz))
        xfer_vs_single = None
        if xr:
            views = pct_views(xr["transfer"], base["mean"])
            xfer_vs_single = {"delta": xr["transfer"], "relative_pct": xr["relative_pct"],
                              "separated": bool(xr["separated"]),
                              "practically_significant": views["practically_significant"],
                              "matters": bool(xr["separated"]) and views["practically_significant"]}
        rows.append({
            "task": task, "group": size_group(task), "n_train": N_TRAIN[task],
            "metric": "macro_f1" if task == "material_type" else "r2",
            "single_task": base["mean"], "single_task_sd": base["sd"], "single_task_n": base["n"],
            "xfer_with_replay": xr["multi_task"] if xr else None,
            "ftz": {"mean": statistics.fmean(ftz), "sd": statistics.stdev(ftz) if len(ftz) > 1 else 0.0,
                    "n": len(ftz)} if ftz else None,
            "ftf": {"mean": statistics.fmean(ftf), "sd": statistics.stdev(ftf) if len(ftf) > 1 else 0.0,
                    "n": len(ftf)} if ftf else None,
            "xfer_vs_single": xfer_vs_single,
            "ftz_vs_single": diff(ftz, base["mean"], base["sd"], base["n"]),
            "ftf_vs_single": diff(ftf, base["mean"], base["sd"], base["n"]),
            "ftf_vs_xfer": ftf_vs_xfer,
            "ftf_vs_ftz": ftf_vs_ftz,
        })

    def verdict(d):
        if not d:
            return "-"
        if not d["separated"]:
            return "unresolved"
        return ("better" if d["delta"] > 0 else "worse") + ("" if d["practically_significant"] else " (negligible)")

    out = {
        "question": "with replay removed, does the transfer checkpoint beat training the task alone?",
        "arms": ARMS,
        "per_task": rows,
        "counts": {
            key: {v: sum(1 for r in rows if verdict(r[key]) == v)
                  for v in ("better", "worse", "unresolved", "better (negligible)", "worse (negligible)")}
            for key in ("xfer_vs_single", "ftz_vs_single", "ftf_vs_single", "ftf_vs_xfer")
        },
        "notes": [
            "fm finetune loads only the target task, so the test split is the single-task universe and "
            "no row matching is needed against the single-task baseline.",
            "ftf_vs_xfer compares against matched_xfer's per-task mean only, so its SE counts one arm.",
            "material_type is scored on macro-F1; its accuracy sits at 0.99 in every arm.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    def pc(d):
        """relative % against single-task, with the significance marker; blank when unavailable."""
        if not d:
            return f"{'-':>8s}"
        star = "*" if d["matters"] else ("·" if d["separated"] else " ")
        return f"{d['relative_pct']:+6.1f}%{star}"

    def v(x):
        return f"{x:6.4f}" if x is not None else f"{'-':>6s}"

    print(f"{'task':22s} {'N':>6s} {'single':>6s} | {'xfer':>6s} {'vs':>8s} | "
          f"{'frozen':>6s} {'vs':>8s} | {'warm':>6s} {'vs':>8s}")
    for r in rows:
        print(f"{r['task']:22s} {r['n_train']:6d} {v(r['single_task'])} | "
              f"{v(r['xfer_with_replay'])} {pc(r['xfer_vs_single'])} | "
              f"{v(r['ftz']['mean'] if r['ftz'] else None)} {pc(r['ftz_vs_single'])} | "
              f"{v(r['ftf']['mean'] if r['ftf'] else None)} {pc(r['ftf_vs_single'])}")
    print("\n  vs = relative change against the single-task baseline")
    print("  * = separated AND |delta| >= 0.01    · = separated but below the practical threshold")
    for key, label in (("xfer_vs_single", "xfer (replay, placed last) vs alone"),
                       ("ftz_vs_single", "frozen encoder vs alone"),
                       ("ftf_vs_single", "warm-start vs alone"),
                       ("ftf_vs_xfer", "warm-start vs the replay step")):
        c = out["counts"][key]
        print(f"  {label:38s} better {c['better']:2d}  worse {c['worse']:2d}  unresolved {c['unresolved']:2d}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
