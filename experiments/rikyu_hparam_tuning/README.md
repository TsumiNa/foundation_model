# RIKYU hyper-parameter tuning campaign

**Question.** The `HYBRID_RECIPE.md` configuration fixes the *replay* recipe but inherits its
*architecture* and *optimiser* settings unchanged from the original 24-task sweep — they were
never tuned. How much of the remaining gap to the single-task ceilings is a tuning gap, and what
is the best backbone + head configuration to freeze as the base of every future model?

**Method (as directed).** Grid search, staged so each stage fixes what the next one assumes:

| Stage | What is tuned | Probe | Held fixed |
|---|---|---|---|
| **A** | encoder / shared trunk + its optimiser | 3-task sequence, one per size group | heads at baseline |
| **B** | one grid per head family (regression · kernel-regression · classification) | one multi-task probe per family | encoder = stage-A winner |
| **C** | nothing — final 24-task run with hybrid replay + end-of-run consolidation | full sequence | everything from A+B |

Stage C runs **two arms that differ only in the tuned knobs** (tuned vs. untuned baseline), so
the campaign's headline number is a like-for-like delta measured on the same hardware, the same
container, the same seed and the same replay recipe.

---

## Stage A — encoder / shared trunk

Probe: [`configs/probe3.toml`](configs/probe3.toml) — a **3-task replay sequence**, one task per
REPORT_20260809 size group (that report reports deficit per group because replay and capacity
requirements differ with task size):

| group | task | N | single-task ceiling R² | dataset |
|---|---|---:|---:|---|
| big ≥20k | `formation_energy` | 23,180 | 0.995 | qc |
| mid 3k–8.1k | `tc` | 7,207 | 0.799 | superconductor |
| small ≤1.2k | `magnetization` | 1,160 | 0.746 | magnetic |

**Why a sequence and not three single-task runs.** The encoder is shared, so the thing being
tuned only exists under multi-task pressure. The probe therefore uses the *final* recipe —
`resample = "epoch"`, `amount = max(1500, 0.3·N_t)`, patience 24, `max_epochs 150`, tasks
introduced in descending size — making it a miniature of stage C. An encoder that wins here wins
in the regime it will actually serve. All three probe tasks are plain regression by design: the
kernel-regression and classification heads are stage B's subject, and including them here would
confound head capacity with encoder capacity.

| Sub-stage | Grid | Runs |
|---|---|---|
| **A1** | `latent_dim` {64,128,256,384} × `encoder_hidden_dims` {[256],[512],[512,256],[1024,512],[1024,512,256]} × `encoder_lr` {1e-3,2e-3,5e-3,1e-2} | 80 |
| **A2** | A1 short list × `batch_size` {512, 1024} | 2·k |
| **A3** | winner × `descriptor.n_grids` {4, 16} | 2 |
| **A6** | winner × `ae_lr` {1e-3, 1e-2} — the AE head's gradients reach the shared trunk | 2 |
| **A4** | short list **and** the untuned baseline × seeds {2026, 2027} — the noise band | 2·(k+1) |

### Why the probe changed, with the measurement that forced it

The campaign originally probed the encoder with single-task `formation_energy`. Two full-data
runs settled it (archived under `probe_singletask/`):

| encoder | R² | MAE | epochs |
|---|---:|---:|---:|
| baseline — L128, [256], lr 5e-3 | 0.99323 | 0.05786 | 150 (cap) |
| large — L256, [1024,512,256], lr 5e-3 | 0.99423 | **0.05018** | 145 |

R² separates the two by 0.001 — inside the ±0.02 single-seed noise band from the replay
campaign, i.e. not a measurement. MAE separates them by 13%. Ranking 80 configs on that R² would
have ranked noise. The 3-task probe fixes this structurally (mid/small tasks have real R²
headroom) and MAE remains the primary statistic on the big task.

### Scoring

Per-task MAE lives on different scales across the three tasks, so absolute deltas cannot be
averaged — the big task would contribute almost nothing to the mean. Grid points are ranked on
the **mean relative MAE improvement over the untuned baseline run of the same probe**, with
per-task R² deltas reported alongside. A config is only preferred when its margin exceeds the
seed-to-seed spread measured in A4.

## Stage B — task heads

Encoder pinned to the stage-A winner. Each head family keeps stage A's shape: its own multi-task
probe config, one run per grid point, ranked on the mean over that probe's tasks.

| Sub-stage | Probe | Grid | Runs |
|---|---|---|---|
| **B-reg** | [`probe3.toml`](configs/probe3.toml) (as stage A) | `head_hidden_dims` {[64],[128,64],[256,128],[256,128,64],[512,256,128]} × `head_lr` {5e-4,1e-3,2e-3,5e-3,1e-2} | 25 |
| **B-kr** | [`probe3_kr.toml`](configs/probe3_kr.toml) | `n_kernel` {15,32,64,128} × `kr_x_hidden_dims` {[128,64],[256,128,64]} × `kr_lr` {2e-4,5e-4,1e-3,2e-3} | 32 |
| **B-kr2** | same | conditioned on the B-kr winner: `kr_t_hidden_dims` {[16,8],[32,16],[64,32,16]} × `kr_weight_decay` {5e-5,5e-4} | 5 |
| **B-clf** | `single_task.toml` + `material_type` | `head_hidden_dims` {[64],[128,64],[256,128],[256,128,64]} × head LR {5e-4,1e-3,2e-3,5e-3,1e-2} | 20 |

The kernel-regression probe spans a different axis than stage A's, and deliberately: **every** KR
task in the catalog sits in the mid band (3.4k–8.1k labels), so there is no big/small axis to
span. What varies instead is the meaning of the `t` coordinate the kernel is fitted over —
`seebeck` (8,072, temperature), `dos_density` (7,009, DOS energy), `zt` (3,445, temperature).

Two code facts shape this stage:

- **`[training].head_lr` is shared by regression and classification heads.** A per-task
  `[[tasks]].lr` override exists and wins over it (`TaskCatalog.build_task_config`), so B-clf can
  be tuned independently — but the adopted value must be written into the final config as a
  per-task override, which is what [`scripts/make_tuned_config.py`](scripts/make_tuned_config.py)
  `--task-lr` is for (`--set` cannot address `[[tasks]]` array entries).
- **`material_type` is ranked on `macro_f1`, not accuracy.** The measured single-task probe hit
  accuracy 0.989 with macro-F1 **0.551** — the head is near-perfect on the dominant class and
  poor on the minority ones, so accuracy has no resolution while macro-F1 has a lot of headroom.

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
- **2026-08-26** — chain smoke at `--sample 400 --max-epochs 2`: single-task pretrain (`47424`),
  4-task replay sequence covering all three head kinds (`47425`), consolidation finetune
  (`47426`) — all PASS.
- **2026-08-26** — full-data timing probes (`47427`, `47428`). Single-task run ≈ 5 min on one
  GB200 (302 s baseline encoder / 322 s largest / 288 s `material_type`), which is what allowed
  A1 to be densified from 18 to 80 grid points. These runs also produced the R²-saturation
  measurement that motivated replacing the stage-A probe. `probe_kr_seebeck` FAILED at 528 s —
  operator error, its output directory was moved while the job was still writing; resubmitted as
  `47462`. Outputs archived under `probe_singletask/`.
- **2026-08-26** — **A1 launched**: job `47461`, 80 grid points on `probe3.toml`, 40 concurrent.

## Outcome

_pending_
