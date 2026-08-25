# RIKYU hyper-parameter tuning campaign

**Question.** The `HYBRID_RECIPE.md` configuration fixes the *replay* recipe but inherits its
*architecture* and *optimiser* settings unchanged from the original 24-task sweep — they were
never tuned. How much of the remaining gap to the single-task ceilings is a tuning gap, and what
is the best backbone + head configuration to freeze as the base of every future model?

**Method (as directed).** Grid search, staged so each stage fixes what the next one assumes:

| Stage | What is tuned | Probe | Held fixed |
|---|---|---|---|
| **A** | encoder / shared trunk + its optimiser | 3-task sequence, one per size group | heads at baseline |
| **B** | every task's own head, independently | single-task probe, one per task | encoder = stage-A winner |
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
| **A1b** | short list × `encoder_lr` {5e-4, 2e-4} — re-opens the LR edge (see below) | 2·k |
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

### Two things the ranking is not

**The optimum must not sit on the grid boundary.** A1's leaders all landed at `encoder_lr` 1e-3,
the smallest value the grid contained — which is the grid running out, not an optimum. A1b extends
that axis downward for the short list, so the adopted LR is either interior or is reported as
still edge-bound.

**Configs are compared at a fixed budget, not at their own convergence.** Every grid point gets
`max_epochs 150` with patience 24, and they use that budget very differently — the A1 leader
early-stops around 50 epochs while some competitive points run to 149. This is the intended
comparison, because stage C spends exactly the same budget, so the deployment budget is the right
one to rank under; but it is *not* a statement about which encoder would win given unlimited
epochs, and the report says so.

### Scoring

Per-task MAE lives on different scales across the three tasks, so absolute deltas cannot be
averaged — the big task would contribute almost nothing to the mean. Grid points are ranked on
the **mean relative MAE improvement over the untuned baseline run of the same probe**, with
per-task R² deltas reported alongside. A config is only preferred when its margin exceeds the
seed-to-seed spread measured in A4.

## Stage B — task heads, one grid per task

Encoder pinned to the stage-A winner; each of the 24 tasks then gets its **own** head grid,
ranked within that task. This is directly expressible in the final config: `TaskSpec` already
carries per-task `hidden_dims` / `x_hidden_dims` / `t_hidden_dims` / `n_kernel` / `lr`, and a
task's own value wins over the `[model]` / `[training]` defaults
(`TaskCatalog.build_task_config`), so a winner transfers verbatim.

| Sub-stage | Tasks | Grid per task | Runs |
|---|---|---|---|
| **B-reg** | 16 regression | `head_hidden_dims` {[64],[128,64],[256,128],[256,128,64]} × head LR {1e-3,2e-3,5e-3,1e-2} | 256 |
| **B-kr** | 7 kernel-regression | `n_kernel` {15,32,64} × `kr_x_hidden_dims` {[128,64],[256,128,64]} × `kr_lr` {5e-4,1e-3,2e-3} | 126 |
| **B-clf** | 1 classification | `head_hidden_dims` {[64],[128,64],[256,128],[256,128,64]} × head LR {1e-3,2e-3,5e-3,1e-2} | 16 |

**Stated limitation, accepted by design.** A head tuned on its task alone is not guaranteed to be
the best head under 24-task continual training. The campaign takes this trade deliberately: it
buys a real, per-task tuning step at a cost that fits the schedule, and stage C is where the
combination is actually measured end-to-end against the untuned control.

### Stage B-mt — the joint-tuning control arm

Per-task tuning is a deliberate trade, and this arm is where it gets priced instead of assumed.
One **shared** head config is tuned jointly on a multi-task probe, and three arms are then read
off that same probe:

| arm | head configuration |
|---|---|
| `mt_base` | untuned shared head (a grid point) |
| `mt_joint` | best shared head from the joint grid (a grid point) |
| `mt_pertask` | each task's own stage-B winner, applied as `[[tasks]]` overrides |

| Probe | Grid | Runs |
|---|---|---|
| **B-mt-reg** — [`probe3.toml`](configs/probe3.toml) | shared `head_hidden_dims` × head LR (4×4) | 16 |
| **B-mt-kr** — [`probe3_kr.toml`](configs/probe3_kr.toml) — `seebeck` / `dos_density` / `zt` | shared `n_kernel` × `kr_x_hidden_dims` × `kr_lr` (3×2×3) | 18 |
| **mt_pertask** | generated configs, one per probe | 2 |

Why joint tuning is a *control* and not the main method: over 24 tasks the task-subset
combinations are neither affordable nor explicable — any particular grouping would need its own
justification, and there is no principled one. Scoping it to two probes avoids that entirely,
because both probes' compositions are inherited rules rather than fresh choices: the regression
triple is stage A's big/mid/small size sampling, and the kernel triple spans `t` semantics
(temperature vs DOS energy) since every kernel task in the catalog is mid-sized.

[`analysis/pertask_vs_joint.py`](analysis/pertask_vs_joint.py) prints the resulting per-task
table and the mean advantage — the campaign's own statement of what tuning heads in isolation
gained or cost.

### Which metric ranks which task

The 24 tasks do not share a regime, so
[`analysis/pick_heads.py`](analysis/pick_heads.py) picks per task and prints the rule it used:

- **classification** → `macro_f1`. The measured `material_type` probe hit accuracy 0.989 with
  macro-F1 **0.551** — near-perfect on the dominant class, poor on the minority ones, so accuracy
  carries no signal while macro-F1 has headroom.
- **regression / kernel-regression** → `r2`, unless that task's R² spread across its whole grid
  is below 0.005, i.e. saturated (`formation_energy` 0.995, `density` 0.988) or degenerate
  (`magnetic_susceptibility`, 58 labels). Those fall back to **`mae`**, which still resolves —
  the two measured encoder probes differ by 0.001 in R² and 13% in MAE.

Winners are emitted as JSON and written into the stage-C config by
[`scripts/make_tuned_config.py --task-overrides`](scripts/make_tuned_config.py); keys that equal
the untuned default are dropped, so the generated config's header diff shows only what tuning
actually changed.

## Stage C — final model, two arms

Recipe frozen from [`HYBRID_RECIPE.md`](../replay_epoch_sweep/HYBRID_RECIPE.md): `resample =
"epoch"`, `amount = max(1500, 0.3·N_t)`, `patience 24`, `max_epochs 150`, followed by one
full-data joint retrain (consolidation).

| Arm | Pretrain | Consolidate |
|---|---|---|
| **C-base** (control) | [`configs/final_hybrid.toml`](configs/final_hybrid.toml) as-is | [`configs/final_consolidate.toml`](configs/final_consolidate.toml) |
| **C-tuned** | `final_hybrid_tuned.toml`, generated from the control's own file | `final_consolidate_tuned.toml`, likewise |

The tuned arm's config is *generated by patching the control's file*, not authored separately, so
every untouched line is byte-identical and the generated header lists exactly which keys differ.
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
