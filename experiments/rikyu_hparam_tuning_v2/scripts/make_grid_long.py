#!/usr/bin/env python3
"""Grids for the long-budget check on the resolvable warm-start losers.

Question: is warm-start's residual loss on final_energy, volume and dos_density just undertraining?
Fine-tuning ends up training ONE task with no replay, so it has no obvious reason to trail a model
trained on that task alone -- unless an encoder pulled toward what 24 tasks share needs more epochs
to specialise back than early stopping (patience 24) allowed. Both arms get the same 500-epoch,
no-early-stop budget so that the budget itself is not the confound:

  ftfl     warm-start from the transfer-stage checkpoints, ft_full_long.toml  (10 orderings/task)
  singlel  training alone, probe6_long.toml + the adopted values via --set    (5 seeds/task)

    python scripts/make_grid_long.py    # grid_ftfl.txt (30 rows), grid_singlel.txt (15 rows)
"""
TASKS = ["final_energy", "volume", "dos_density"]
SEEDS = [2025, 2026, 2027, 2028, 2029]
# identical to the stage_single / singlex rows apart from the epoch budget
ADOPTED = ("--set model.latent_dim=384 --set training.encoder_lr=0.002 "
           "--set training.scheduler.min_lr=1e-05 --set training.scheduler.patience=5")

with open("configs/grid_ftfl.txt", "w") as fh:
    for t in TASKS:
        for k in range(10):
            fh.write(f"ftfl_{t}_o{k}\t--checkpoint /out/_ckpt/{t}/o{k}.pt --set 'finetune.tasks=[\"{t}\"]'\n")
with open("configs/grid_singlel.txt", "w") as fh:
    for t in TASKS:
        for s in SEEDS:
            fh.write(f"stL_{t}_s{s}\t{ADOPTED} --set 'pretrain.task_sequence=[\"{t}\"]' --seed {s} "
                     f"--set training.max_epochs=500\n")
print("  wrote configs/grid_ftfl.txt: 30 rows, configs/grid_singlel.txt: 15 rows")

# ---- follow-ups to the long-budget result (2026-09-08) ----------------------------------------
# The 500-epoch check found warm-start fitting the training set BETTER than training alone (lower
# train loss) while validating worse. Two cheap arms separate "optimisation" from "representation":
#   singlec  alone, 500 epochs, early stopping off, LR schedule off (plateau patience never fires):
#            can a fresh model reach warm-start's training loss if its learning rate is not decayed?
#   ftflr    warm-start with the encoder learning rate 10x lower (2e-4), early stopping ON, 400-epoch
#            cap: is the overfitting controllable by slowing the encoder rather than by budget?
with open("configs/grid_singlec.txt", "w") as fh:
    for t in TASKS:
        for s in SEEDS:
            fh.write(f"stC_{t}_s{s}\t{ADOPTED} --set 'pretrain.task_sequence=[\"{t}\"]' --seed {s} "
                     f"--set training.max_epochs=500 --set training.scheduler.patience=100000\n")
with open("configs/grid_ftflr.txt", "w") as fh:
    for t in TASKS:
        for k in range(10):
            fh.write(f"ftflr_{t}_o{k}\t--checkpoint /out/_ckpt/{t}/o{k}.pt --set 'finetune.tasks=[\"{t}\"]' "
                     f"--set training.encoder_lr=0.0002 --set finetune.epochs=400\n")
print("  wrote configs/grid_singlec.txt: 15 rows, configs/grid_ftflr.txt: 30 rows")
