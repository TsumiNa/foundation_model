#!/usr/bin/env python3
"""Grids for the fixed-count low-data curves — configs/grid_lowalone_n.txt, configs/grid_lowwarm_n.txt

Same recipe as the fraction study (stages lowalone / lowwarm) but every task gets the same number of
labelled training rows: 100 / 300 / 1,000 / 3,000 / 10,000, three subsample seeds, on the label-masked
files in data/lowdata_n/ (scripts/make_lowdata_datasets.py --counts). A count is used for a task only
when it is at most 80 % of the task's labelled training rows; above that the full-data run already
covers it. Alone = single-task from scratch (stLn_*); warm-start = fine-tune from a library encoder that
never saw the task, fresh head, encoder trained (ftLn_*; material_type from the stage_xu encoders, the
added tasks from _ckpt/density/o0–o2, as in the fraction study).

    uv run python scripts/make_grid_lowdata_n.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
COUNTS = [100, 300, 1000, 3000, 10000]
SEEDS = [0, 1, 2]
MAX_SHARE = 0.8
ALONE = ("--set model.latent_dim=384 --set training.encoder_lr=0.002 --set training.scheduler.min_lr=1e-05 --set training.scheduler.patience=5 "
         "--set 'pretrain.task_sequence=[\"{task}\"]' --set 'datasets.qc.path=\"data/lowdata_n/{file}\"' --seed {seed}")
WARM = "--checkpoint {ckpt} --set 'finetune.tasks=[\"{task}\"]' --set 'datasets.qc.path=\"data/lowdata_n/{file}\"'"


def main():
    man = json.loads((HERE / "summary" / "lowdata_manifest.json").read_text())
    total = {t: v["train_before"] for t, v in man["files"][0]["kept"].items()}
    alone, warm, plan = [], [], {}
    for task, n_all in total.items():
        plan[task] = [n for n in COUNTS if n <= MAX_SHARE * n_all]
        for n in plan[task]:
            for s in SEEDS:
                f = f"qc_20260912_n{n:05d}_s{s}.parquet"
                alone.append(f"stLn_{task}_n{n:05d}_s{s}\t" + ALONE.format(task=task, file=f, seed=2025 + s))
                ckpt = f"/out/_ckpt_u/material_type/o{s}.pt" if task == "material_type" else f"/out/_ckpt/density/o{s}.pt"
                warm.append(f"ftLn_{task}_n{n:05d}_s{s}\t" + WARM.format(ckpt=ckpt, task=task, file=f))
    (HERE / "configs" / "grid_lowalone_n.txt").write_text("\n".join(alone) + "\n")
    (HERE / "configs" / "grid_lowwarm_n.txt").write_text("\n".join(warm) + "\n")
    (HERE / "summary" / "lowdata_n_plan.json").write_text(json.dumps({"counts": COUNTS, "max_share": MAX_SHARE, "seeds": SEEDS, "train_rows": total, "plan": plan}, indent=1) + "\n")
    print(f"{len(alone)} alone + {len(warm)} warm-start runs; {sum(len(v) for v in plan.values())} task-points")
    for t, v in plan.items():
        print(f"  {t:26s} {total[t]:6,d} rows  counts {v}")


if __name__ == "__main__":
    main()
