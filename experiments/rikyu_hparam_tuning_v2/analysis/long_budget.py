#!/usr/bin/env python3
"""Is warm-start's residual loss on the extensive properties undertraining, optimisation, or the
representation? — summary/long_budget.json

Warm-start fine-tuning ends up training ONE task with no replay, so it has no obvious reason to
trail a model trained on that task alone. The arms scored here, all on final_energy, volume and
dos_density, against the early-stopped arms already scored (ftf in summary/ft.json, stA/stB in the
adopted ceilings):

  ftfl   warm-start, 500 epochs, early stopping OFF        (10 orderings)   the budget check
  stL    alone, 500 epochs, early stopping OFF             (5 seeds)        its same-budget control
  stC    alone, 500 epochs, early stopping OFF, LR schedule never fires (5 seeds)
         -> can a fresh model reach warm-start's training loss if its learning rate is not decayed?
  ftflr  warm-start, encoder LR 10x lower (2e-4), early stopping ON, 400-epoch cap (10 orderings)
         -> is the overfitting controllable by slowing the encoder rather than by budget?

Every arm reports LAST-epoch weights, so each run's per-epoch log is read too: the epoch of the
lowest validation loss, how far the final validation loss sits above it, and the final training
loss. Those three numbers are what tell "trained longer" from "overfit" and "fits the training
set better" from "fits it worse".

    python analysis/long_budget.py --ft <outroot>/stage_ft --single <outroot>/stage_single \\
        --ceilings summary/ceilings_adopted_v2.json --ftjson summary/ft.json -o summary/long_budget.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from common import N_TRAIN, final_metrics
from ft import diff, metric_of

TASKS = ["final_energy", "volume", "dos_density"]
ARMS = {
    # key: (stage, run-id prefix, kind, description)
    "warm_500": ("ft", "ftfl", "finetune", "warm-start, 500 epochs, early stopping off"),
    "alone_500": ("single", "stL", "pretrain", "alone, 500 epochs, early stopping off"),
    "alone_500_constlr": ("single", "stC", "pretrain", "alone, 500 epochs, early stopping off, LR never decayed"),
    "warm_lowlr": ("ft", "ftflr", "finetune", "warm-start, encoder LR 2e-4, early stopping on, 400-epoch cap"),
}


def curve(run: Path, task: str) -> dict | None:
    """Best-vs-last validation loss and the final training loss, from the run's per-epoch CSV log."""
    logs = sorted(run.glob("**/metrics.csv"))
    if not logs:
        return None
    rows = list(csv.DictReader(open(logs[0])))
    if not rows:
        return None
    cols = rows[0].keys()
    vcol = next((c for c in ("val_final_loss", "val_loss") if c in cols), None) or \
        next((c for c in cols if "val" in c and "loss" in c), None)
    tcol = f"train_{task}_raw_loss_epoch" if f"train_{task}_raw_loss_epoch" in cols else None
    vraw = f"val_{task}_raw_loss" if f"val_{task}_raw_loss" in cols else None
    if not vcol:
        return None
    # Lightning writes several rows per epoch; keep the last value logged for each epoch.
    val: dict[int, float] = {}
    train: dict[int, float] = {}
    vr: dict[int, float] = {}
    for i, r in enumerate(rows):
        e = int(float(r["epoch"])) if r.get("epoch") not in (None, "") else i
        if r.get(vcol) not in (None, ""):
            val[e] = float(r[vcol])
        if tcol and r.get(tcol) not in (None, ""):
            train[e] = float(r[tcol])
        if vraw and r.get(vraw) not in (None, ""):
            vr[e] = float(r[vraw])
    if not val:
        return None
    epochs = sorted(val)
    best_e = min(epochs, key=lambda e: val[e])
    last_e = epochs[-1]
    return {"epochs_logged": len(epochs), "best_epoch": best_e, "best_val": val[best_e],
            "last_val": val[last_e], "epochs_past_best": last_e - best_e,
            "last_over_best_pct": (val[last_e] / val[best_e] - 1) * 100 if val[best_e] else None,
            "final_train_loss": train[max(train)] if train else None,
            "final_val_raw_loss": vr[max(vr)] if vr else None}


def stats(vals: list[float]) -> dict | None:
    if not vals:
        return None
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals), "values": vals}


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def curve_summary(curves: list[dict | None]) -> dict:
    return {k: med([c and c.get(k) for c in curves])
            for k in ("best_epoch", "epochs_past_best", "last_over_best_pct", "final_train_loss",
                      "final_val_raw_loss", "epochs_logged")}


def collect(stage_root: Path, prefix: str, kind: str, task: str) -> tuple[list[float], list[dict | None]]:
    vals, curves = [], []
    glob = f"{prefix}_{task}_o*" if kind == "finetune" else f"{prefix}_{task}_s*"
    for run in sorted(stage_root.glob(glob)):
        if not (run / "DONE").exists():
            continue
        if kind == "finetune":
            v = metric_of(run / "training" / "finetune" / f"{task}_metrics.json")
        else:
            m, _ = final_metrics(run)
            v = m.get(task, {}).get("r2")
            v = float(v) if v is not None else None
        if v is not None:
            vals.append(v)
            curves.append(curve(run, task))
    return vals, curves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ft", type=Path, required=True, help="<outroot>/stage_ft")
    ap.add_argument("--single", type=Path, required=True, help="<outroot>/stage_single")
    ap.add_argument("--ceilings", type=Path, required=True, help="summary/ceilings_adopted_v2.json")
    ap.add_argument("--ftjson", type=Path, required=True, help="summary/ft.json")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    ceil = json.loads(args.ceilings.read_text())
    ft = {r["task"]: r for r in json.loads(args.ftjson.read_text())["per_task"]}
    roots = {"ft": args.ft, "single": args.single}

    rows = []
    for task in TASKS:
        base = ceil[task]                      # alone, early-stopped (stA / stB)
        warm150 = ft[task]["ftf"]              # warm-start, early-stopped, 150-epoch cap
        row = {"task": task, "n_train": N_TRAIN[task],
               "alone_early_stop": {"mean": base["mean"], "sd": base["sd"], "n": base["n"]},
               "warm_early_stop": warm150}
        arms: dict[str, list[float]] = {}
        for key, (stage, prefix, kind, _) in ARMS.items():
            vals, curves = collect(roots[stage], prefix, kind, task)
            arms[key] = vals
            row[key] = stats(vals)
            row[f"{key}_curve"] = curve_summary(curves) if curves else None

        def against(a_key: str, b: dict | None) -> dict | None:
            if not arms.get(a_key) or not b:
                return None
            return diff(arms[a_key], b["mean"], b["sd"], b["n"])

        row.update({
            # the budget check
            "warm_500_vs_alone_early_stop": against("warm_500", row["alone_early_stop"]),
            "warm_500_vs_alone_500": against("warm_500", row["alone_500"]),
            "warm_500_vs_warm_early_stop": against("warm_500", warm150),
            "alone_500_vs_alone_early_stop": against("alone_500", row["alone_early_stop"]),
            # optimisation: does a never-decayed LR let a fresh model fit like warm-start?
            "alone_500_constlr_vs_alone_500": against("alone_500_constlr", row["alone_500"]),
            "alone_500_constlr_vs_alone_early_stop": against("alone_500_constlr", row["alone_early_stop"]),
            # regularisation: does a slow encoder close the gap?
            "warm_lowlr_vs_alone_early_stop": against("warm_lowlr", row["alone_early_stop"]),
            "warm_lowlr_vs_warm_early_stop": against("warm_lowlr", warm150),
        })
        rows.append(row)

    out = {"question": "is warm-start's residual loss on the extensive properties undertraining, "
                       "optimisation, or the representation?",
           "arms": {k: v[3] for k, v in ARMS.items()},
           "per_task": rows,
           "notes": ["Every arm reports last-epoch weights, like every other arm in the campaign.",
                     "*_curve fields are medians over runs of per-epoch log quantities: best_epoch = epoch of "
                     "the lowest validation loss; epochs_past_best = last epoch minus that; last_over_best_pct = "
                     "how far the final validation loss sits above the minimum; final_train_loss and "
                     "final_val_raw_loss = the task's own raw loss at the last epoch."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    def pc(d):
        if not d:
            return f"{'-':>9s}"
        star = "*" if d["matters"] else ("·" if d["separated"] else " ")
        return f"{d['relative_pct']:+7.1f}%{star}"

    def v(s):
        return f"{s['mean']:6.4f}" if s else f"{'-':>6s}"

    def tl(c):
        return f"{c['final_train_loss']:.4f}" if c and c.get("final_train_loss") is not None else "-"

    print("== R2, last-epoch weights ==")
    print(f"{'task':14s} {'alone-es':>8s} {'alone-500':>9s} {'const-lr':>9s} | {'warm-es':>7s} {'warm-500':>8s} "
          f"{'vs alone-500':>12s} | {'warm-lowlr':>10s} {'vs alone-es':>11s} {'vs warm-es':>10s}")
    for r in rows:
        print(f"{r['task']:14s} {v(r['alone_early_stop']):>8s} {v(r['alone_500']):>9s} {v(r['alone_500_constlr']):>9s} | "
              f"{v(r['warm_early_stop']):>7s} {v(r['warm_500']):>8s} {pc(r['warm_500_vs_alone_500']):>12s} | "
              f"{v(r['warm_lowlr']):>10s} {pc(r['warm_lowlr_vs_alone_early_stop']):>11s} {pc(r['warm_lowlr_vs_warm_early_stop']):>10s}")
    print("\n== per-epoch logs, medians: best epoch / epochs past best / final val over best % / final train loss ==")
    for r in rows:
        parts = []
        for key in ARMS:
            c = r.get(f"{key}_curve")
            if c:
                parts.append(f"{key}: {c['best_epoch']:.0f} / {c['epochs_past_best']:.0f} / "
                             f"{c['last_over_best_pct']:+.1f}% / {tl(c)}")
        print(f"{r['task']:14s} " + "   ".join(parts))
    print("\n  es = early-stopped arm already scored; vs = relative % with both arms' SE; "
          "* separated and |delta| >= 0.01   · separated only")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
