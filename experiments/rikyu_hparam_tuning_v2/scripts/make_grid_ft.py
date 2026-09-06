#!/usr/bin/env python3
"""Grids for the warm-start fine-tune stage: one row per (task, ordering), two arms.

Each transfer run xf_<task>_o<k> ended on <task> with replay pulling the encoder toward the other 23
tasks. This stage takes that final checkpoint and trains <task> again with NO replay, so that
"the encoder is bad for X" and "X was drowned by replay at its own step" can be told apart.

Two arms, one config each (freeze_encoder is a bool and bools do not travel through --set safely):
  ftz  frozen encoder, head only   -> is the shared representation itself good for X?
  ftf  encoder + head              -> warm-start fine-tune, the model library's intended use

    python scripts/make_grid_ft.py            # grid_ftz.txt, grid_ftf.txt (240 rows each)
    python scripts/make_grid_ft.py --smoke    # grid_ftzs.txt, grid_ftfs.txt (2 rows each)
"""
import argparse, sys
sys.path.insert(0, "analysis")
from common import N_TRAIN

ap = argparse.ArgumentParser()
ap.add_argument("--orders", type=int, default=10)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

tasks = sorted(N_TRAIN)
if args.smoke:
    # one small and one large task, one ordering: enough to prove the wiring end to end
    pairs = [("magnetic_moment", 0), ("material_type", 0)]
else:
    pairs = [(t, k) for t in tasks for k in range(args.orders)]

for arm in ("ftz", "ftf"):
    name = f"grid_{arm}{'s' if args.smoke else ''}.txt"
    with open(f"configs/{name}", "w") as fh:
        for task, k in pairs:
            # /out is the stage root inside the container; _ckpt is a copy of the model library
            fh.write(f"{arm}_{task}_o{k}\t--checkpoint /out/_ckpt/{task}/o{k}.pt "
                     f"--set 'finetune.tasks=[\"{task}\"]'\n")
    print(f"  wrote configs/{name}: {len(pairs)} rows")
