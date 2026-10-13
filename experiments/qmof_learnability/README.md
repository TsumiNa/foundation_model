# QMOF learnability check (2026-10-11)

**Question.** Can the composition-only model learn the QMOF properties (data/QMOF_20261011_norm.parquet,
built by `data/scripts/process_qmof_20261011.py`) when each task is trained alone?

**Design.** The campaign-v2 single-task recipe (KMD descriptor; latent_dim 384, encoder_lr 2e-3, scheduler
min_lr 1e-5 / patience 5, early stopping on the validation loss), one task per run, seeds 2025–2027, the
dataset's own 70 / 15 / 15 split (2,747 test compositions). Config: `configs/single_qmof.toml`.

**Run and summarize**

```bash
bash experiments/qmof_learnability/scripts/run_single.sh          # 4 tasks x 3 seeds, CPU, ~1 min per run
uv run python experiments/qmof_learnability/analysis/summarize.py
```

Runs land in `artifacts/single/<task>_s<seed>/` (untracked).

**Outcome (2026-10-11, 3 seeds each).** R² on the normalized scale; MAE in physical units.

| task | R² | MAE |
|---|---|---|
| qmof_energy_per_atom_pbe | 0.980 ± 0.001 | 0.038 eV/atom |
| qmof_band_gap_pbe | 0.659 ± 0.003 | 0.48 eV |
| qmof_vbm_pbe | 0.585 ± 0.048 | 0.74 eV |
| qmof_cbm_pbe | 0.558 ± 0.069 | 0.85 eV |

The energy per atom follows almost entirely from the composition. The band gap is learned to a moderate level
and stably across seeds (the Materials Project band gap reaches 0.883 under the same recipe); in a MOF about 95 %
of the atoms are C, H, N or O, so the metal node is a small part of the composition input. The band edges are less
stable, being absolute DFT levels without a common reference. An HSE06 band-gap task was also run (R² 0.41 ± 0.06)
before that column was dropped from the dataset on 2026-10-12.
