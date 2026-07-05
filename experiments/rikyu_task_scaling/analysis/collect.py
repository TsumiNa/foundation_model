#!/usr/bin/env python3
"""Collect task-scaling results from a local mirror of artifacts/task_scaling into two CSVs.

Usage:
    python collect.py MIRROR_DIR [OUT_DIR]

MIRROR_DIR holds the rsync'd run outputs (checkpoints excluded):
    pre_o{0,1,2}/training/metrics_table.csv          (pretraining — realized task order)
    ws_o{0,1,2}/k{01..21}/training/metrics_table.csv (warm-start evals)
    ws_o{0,1,2}/k{01..21}/inverse/fe_down_diel_up/{summary,results}.json
    ft_o{0,1,2}/k{01..21}/training/finetune_summary.json (+ inverse/ like ws)
    scratch_s{2025,2026,2027}/training/metrics_table.csv (k=0 baseline)

Outputs (written to OUT_DIR, default ../results):
    scaling.csv — mode,ord,k,task_added_at_k,target,r2,mae,samples
                  (mode ws/ft; scratch rows use mode=scratch, ord=seed, k=0)
    inverse.csv — mode,ord,k,path,objective_mean,objective_std,seed_objective_mean,
                  <task>_after_mean per target, elapsed_s
"""

import csv
import json
import sys
from pathlib import Path

TARGETS = ["dielectric_total", "dielectric_ionic", "dielectric_electronic"]
SCENARIO = "fe_down_diel_up"
INV_TASKS = ["formation_energy", *TARGETS]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def final_rows(mt_path: Path) -> dict[str, dict]:
    """Final-step metrics per task from a long-format metrics_table.csv."""
    rows = list(csv.DictReader(open(mt_path)))
    if not rows:
        return {}
    mx = max(int(r["step"]) for r in rows)
    return {r["task"]: r for r in rows if int(r["step"]) == mx}


def added_task_at(pre_mt: Path, k: int) -> str:
    """The task introduced at pretraining step k (from the run's own metrics table)."""
    for r in csv.DictReader(open(pre_mt)):
        if int(r["step"]) == k and r["new_task"] == r["task"]:
            return r["task"]
    return ""


def collect(mirror: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scaling: list[dict] = []
    inverse: list[dict] = []

    def add_training_rows(mode: str, replay_n: int, ord_id: str, k: int, per_task: dict[str, dict], added: str) -> None:
        for t in TARGETS:
            r = per_task.get(t)
            if r is None:
                continue
            scaling.append(
                {
                    "mode": mode,
                    "replay_n": replay_n,
                    "ord": ord_id,
                    "k": k,
                    "task_added_at_k": added,
                    "target": t,
                    "r2": fnum(r.get("r2")),
                    "mae": fnum(r.get("mae")),
                    "samples": r.get("samples"),
                }
            )

    def add_inverse_rows(mode: str, replay_n: int, ord_id: str, k: int, inv_dir: Path) -> None:
        summary_p = inv_dir / SCENARIO / "summary.json"
        results_p = inv_dir / SCENARIO / "results.json"
        if not summary_p.exists():
            return
        summary = json.loads(summary_p.read_text())
        seed_obj = None
        if results_p.exists():
            payload = json.loads(results_p.read_text())
            objs = payload.get("seed_predictions", {}).get("objective", [])
            seed_obj = sum(objs) / len(objs) if objs else None
        for row in summary:
            rec = {
                "mode": mode,
                "replay_n": replay_n,
                "ord": ord_id,
                "k": k,
                "path": row["path"],
                "objective_mean": row.get("objective_mean"),
                "objective_std": row.get("objective_std"),
                "seed_objective_mean": None if seed_obj is None else round(seed_obj, 4),
                "elapsed_s": row.get("elapsed_s"),
            }
            for t in INV_TASKS:
                rec[f"{t}_after_mean"] = row.get(f"{t}_after_mean")
            inverse.append(rec)

    for replay_n, tag in ((1000, ""), (1500, "_n1500")):
        for o in range(3):
            pre_mt = mirror / f"pre{tag}_o{o}/training/metrics_table.csv"
            for k in range(1, 22):
                kk = f"k{k:02d}"
                added = added_task_at(pre_mt, k) if pre_mt.exists() else ""
                ws = mirror / f"ws{tag}_o{o}/{kk}/training/metrics_table.csv"
                if ws.exists():
                    add_training_rows("ws", replay_n, str(o), k, final_rows(ws), added)
                ft = mirror / f"ft{tag}_o{o}/{kk}/training/finetune_summary.json"
                if ft.exists():
                    after = json.loads(ft.read_text()).get("metrics_after", {})
                    add_training_rows("ft", replay_n, str(o), k, after, added)
                for mode in ("ws", "ft"):
                    add_inverse_rows(mode, replay_n, str(o), k, mirror / f"{mode}{tag}_o{o}/{kk}/inverse")

        for seed in (2025, 2026, 2027):
            mt = mirror / f"scratch{tag}_s{seed}/training/metrics_table.csv"
            if mt.exists():
                add_training_rows("scratch", replay_n, str(seed), 0, final_rows(mt), "")

    for name, rows in (("scaling.csv", scaling), ("inverse.csv", inverse)):
        if not rows:
            print(f"{name}: no rows found — skipped")
            continue
        path = out_dir / name
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    mirror = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "results"
    collect(mirror, out)
