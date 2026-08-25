# RIKYU hyper-parameter tuning campaign

**Question.** The `HYBRID_RECIPE.md` configuration fixes the *replay* recipe but inherits its
*architecture* and *optimiser* settings unchanged from the original 24-task sweep — they were
never tuned. How much of the remaining gap to the single-task ceilings is a tuning gap, and what
is the best backbone + head configuration to freeze as the base of every future model?

**Method (as directed).** Grid search, staged so each stage fixes what the next one assumes:

| Stage | What is tuned | Probe | Held fixed |
|---|---|---|---|
| **A** | encoder / shared trunk + its optimiser | single-task `formation_energy` | heads at baseline |
| **B** | one grid per head family (regression · kernel-regression · classification) | single-task, per family | encoder = stage-A winner |
| **C** | nothing — final 24-task run with hybrid replay + end-of-run consolidation | full sequence | everything from A+B |

Stage C runs **two arms that differ only in the tuned knobs** (tuned vs. untuned baseline), so
the campaign's headline number is a like-for-like delta measured on the same hardware, the same
container, the same seed and the same replay recipe.

---

## Stage A — encoder / shared trunk

Probe: `fm pretrain` with `pretrain.task_sequence = ["formation_energy"]` (a length-1 sequence,
so no replay is involved). Config: [`configs/single_task.toml`](configs/single_task.toml) — the
same 24-task catalog, split seed and data handling as the final run, so a winner transfers
verbatim.

| Sub-stage | Grid | Runs |
|---|---|---|
| **A1** | `latent_dim` {64, 128, 256} × `encoder_hidden_dims` {[256], [512,256], [1024,512,256]} × `encoder_lr` {1e-3, 5e-3} | 18 |
| **A2** | A1 short list (top 3) × `batch_size` {512, 1024} | 6 |
| **A3** | winner × `descriptor.n_grids` {4, 16} | 2 |
| **A4** | winner and untuned baseline × seeds {2026, 2027} — the noise band | 4 |
| **A5** | short list × {`volume`, `final_energy`} + baseline control — does the choice transfer? | 8 |

### The measurement problem, stated up front

`formation_energy`'s single-task ceiling is **R² = 0.995** (from
`experiments/rikyu_replay_sweep/results/warm_restart.csv`). At that level R² has no resolution
left: the whole plausible spread between a good and a bad encoder is smaller than the ±0.02
single-seed noise band measured in the replay campaign. Ranking 18 configs on it alone would
rank noise.

The instruction to tune the encoder on `formation_energy` is kept. What changes is *which
statistic on that run is read*:

1. **Primary — `mae`** on the fixed test split (recorded alongside `r2` in every
   `<task>_metrics.json`). MAE keeps its dynamic range where R² saturates.
2. **Tie-break — `r2`**, reported but only trusted at the 3rd decimal when A4's seed repeats say
   the gap exceeds noise.
3. **Transfer gate — A5.** The short list is re-run on `volume` (ceiling 0.569) and
   `final_energy` (0.687), two same-dataset big tasks with real headroom. A config that wins on
   `formation_energy` but loses on both is not adopted; the A5 winner is.

A4 is what makes any of this decidable: no configuration is declared better than another unless
its margin exceeds the seed-to-seed spread measured there on the *same* probe.

---

## Stage B — task heads

Encoder frozen at the stage-A winner; one grid per head family, each on tasks with headroom.

| Sub-stage | Grid | Probe tasks | Runs |
|---|---|---|---|
| **B-reg** | `head_hidden_dims` {[64], [128,64], [256,128,64]} × `head_lr` {1e-3, 5e-3, 1e-2} | `formation_energy`, `volume`, `final_energy` | 27 |
| **B-kr** | `n_kernel` {15, 32, 64} × `kr_x_hidden_dims` {[128,64], [256,128,64]} × `kr_lr` {5e-4, 2e-3} | `seebeck`, `dos_density` | 24 |
| **B-kr2** | conditioned on the B-kr winner: `kr_t_hidden_dims` {[16,8], [32,16], [64,32,16]} × `kr_weight_decay` {5e-5, 5e-4} | same | 10 |
| **B-clf** | `head_hidden_dims` {[64], [128,64], [256,128]} × head LR {1e-3, 5e-3, 1e-2} | `material_type` | 9 |

Selection: mean over the probe tasks of (metric − that task's single-task ceiling), so a config
must win across tasks rather than exploit one.

Two properties of the current code shape this stage and are worth stating plainly:

- **`[training].head_lr` is shared by regression and classification heads.** A per-task
  `[[tasks]].lr` override exists and wins over it (`TaskCatalog.build_task_config`), so B-clf
  tunes `material_type`'s LR independently — but the *adopted* value must be written into the
  final config as a per-task override, not as a global `head_lr`.
- **`material_type` accuracy is 0.984 at the ceiling** and the replay campaign found it
  insensitive to replay settings (±0.005). B-clf is therefore ranked on **`macro_f1`**, not
  accuracy — 5 imbalanced classes make macro-F1 the only statistic with resolution. Expect a
  small or null result here and treat it as such.

---

## Stage C — final model, two arms

Recipe frozen from [`HYBRID_RECIPE.md`](../replay_epoch_sweep/HYBRID_RECIPE.md): `resample =
"epoch"`, `amount = max(1500, 0.3·N_t)`, `patience 24`, `max_epochs 150`, followed by one
full-data joint retrain (consolidation).

| Arm | Pretrain | Consolidate |
|---|---|---|
| **C-base** (control) | [`configs/final_hybrid.toml`](configs/final_hybrid.toml) as-is | [`configs/final_consolidate.toml`](configs/final_consolidate.toml) |
| **C-tuned** | same file + stage A/B winners via `--set` | same file + the same `--set` |

One file per role for both arms, so the two arms provably differ only in the tuned knobs.
The published H200 numbers (`mt_hybrid_r03_f1500.csv`, mean R² 0.652 → 0.658 after
consolidation) are a *reference*, not the control: C-base re-establishes the baseline on RIKYU
hardware with the same container, removing the hardware/version confound from the headline delta.

---

## Execution model

Every run is one array task of [`scripts/fm_array.sbatch`](scripts/fm_array.sbatch), driven by a
grid file (`configs/grid_<stage>.txt`, generated by
[`scripts/make_grids.py`](scripts/make_grids.py)):

```
<runid>\t<shell-quoted fm overrides>
```

- 1 GPU / 18 CPUs per array task; independent grid points run concurrently across the partition.
- **Idempotent**: a completed run writes `DONE`; resubmitting the array skips it. A TIMEOUT is
  recovered by resubmitting the identical command (`--resume` is added for stage C).
- Wall-clock and exit code of every unit are appended to `<OUTROOT>/_timing.tsv`.

Paths on RIKYU:

| What | Where |
|---|---|
| repo (bound read-only at `/workspace`) | `$HOME/projects/foundation_model` @ branch `exp/rikyu-hparam-tuning` |
| container | `$HOME/containers/foundation-model_rikyu-0.2.1.sif` (linux/arm64) |
| run outputs | `/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning/<stage>/<runid>` |
| Slurm logs | `$HOME/jobs/hparam` |

**Container/source provenance.** The container's installed `foundation_model` differs from the
branch checkout in exactly one file, `workflows/pretrain.py`: master `352d376` wrapped
`trainer.fit` in `try/finally` so a raised exception cannot leave disabled heads disabled. For
`replay.interval = 1` (this campaign) the disabled-head set is always empty, so the two are
behaviourally identical on every run here. The container is the code that executes.

---

## Execution log

- **2026-08-26** — RIKYU account `rku00225` (project `rkp00067`) prepared from scratch: repo
  cloned, 4 parquet inputs rsynced (271 MB), group workspace `/data1/rkp00067/rku00225/fm`
  created. Pre-existing container smoke (`job 46899`, GB200) confirmed PASS.
- **2026-08-26** — chain smoke submitted at `--sample 400 --max-epochs 2`: single-task probe
  (`47424`) and a 4-task replay sequence covering regression / kernel-regression /
  classification (`47425`).

## Outcome

_pending_
