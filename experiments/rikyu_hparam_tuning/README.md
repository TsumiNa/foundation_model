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

### The adoption rule

Ranking produces an order; adopting needs a rule, fixed before A4 is read so the noise band cannot
be reinterpreted after the fact:

> **Adopt the cheapest configuration whose score is within the A4 seed band of the best score.**

Two reasons this is the right rule here rather than "take the argmax". A1 found a *plateau*, not a
peak — at `encoder_lr` 1e-3 every cell is positive and the whole `latent_dim` 384 row sits at
+15…+21%, so the argmax cell is not meaningfully separated from a dozen others. And the encoder
chosen here is paid for 24 times over: it is the backbone of all 398 stage-B runs and of both
stage-C arms, so a deep encoder that ties a shallow one on quality loses on cost.

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

### B4 — confirming each per-task pick against its own seed band

Stage B ranks each task on a single seed, and stage A measured what that is worth: its leader
scored +23.9% at seed 2025 and +15.9%/+15.5% at 2026/2027, against a seed band of 8.5% relative.
Several stage-B per-task gains are 1–2%, which without a band cannot be told apart from seed luck.

**B4** re-runs every task's winner *and* its untuned baseline at seeds 2026 and 2027 (96 runs,
~17 GPU-h). The rule, fixed before B4 was read and identical in principle to stage A's:

> A task keeps its tuned head only if (mean winner − mean baseline) exceeds that task's own seed
> band; otherwise it reverts to the untuned default.

Stage A preferred the simplest configuration among ties for the same reason — do not pay for what
you cannot measure. Stage C deploys the **confirmed** set; the raw point-estimate set is kept and
reported alongside so the deck can state how much of the grid's output survived.

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
- **2026-08-26** — **A1 launched**: job `47461`, 80 grid points on `probe3.toml`, 40 concurrent
  (raised to 80 mid-flight). **80/80 completed**, no failures; 27.0 GPU-h, mean 17.8 min/run.
- **2026-08-26** — kernel-regression cost measured: `seebeck` single-task ran 23.8 min and used the
  full 150-epoch cap (R² 0.663). KR heads are budget-limited, and B-kr is the campaign's most
  expensive stage.
- **2026-08-26** — **A1 result** (see "Stage A outcome" below). Wave 2 submitted: A1b (`47578`,
  8 points, LR edge) and A2 (`47579`, 6 points, batch size).

## Stage A outcome (A1, 80 points, single seed 2025)

**`encoder_lr` dominates, and the untuned value was the problem.** Marginal means over the whole
grid, as relative MAE improvement against the untuned baseline run:

| `encoder_lr` | mean Δ MAE | note |
|---|---:|---|
| 0.001 | **+14.8%** | smallest value in the grid — still improving, hence A1b |
| 0.002 | +11.8% | |
| 0.005 | +6.0% | **the untuned value** |
| 0.010 | −8.6% | diverges on deep encoders (worst cell −45.6%) |

`latent_dim` is a weak effect (384 +9.5% · 256 +6.5% · 64 +5.8% · 128 +2.7%) and
`encoder_hidden_dims` is nearly flat (+7…+9%) except the deepest `[1024,512,256]`, whose mean
falls to +1% because it carries every divergence.

- **Best point: `L256 / [1024,512,256] / lr 1e-3`, +23.9% mean relative MAE** (also #2 on R²,
  +0.025 absolute). The R²-ranked leader is `L256 / [512] / lr 1e-3` (+0.027), #8 on MAE — the two
  metrics agree on the region, not on the single cell.
- **Learning rate alone recovers about two thirds of the gain.** Holding the untuned architecture
  `L128 / [256]` and changing only the LR to 2e-3 gives +15.2%, against +23.9% for the full winner.
- **The optimum is a plateau, not a peak.** At lr 1e-3 every cell is positive (worst +4.6%) and the
  whole `latent_dim` 384 row sits at +15…+21%. The final choice does not need to be a precise one.
- **The LR optimum interacts with width**: `L128` prefers 2e-3, `L256`/`L384` prefer 1e-3 — which is
  why A1b extends the LR axis per architecture rather than globally.

### The gain is metric-dependent (report both)

同一份 A4 数据换成 R² 口径，结论会变：

| 口径 | 采纳配置 | 未调基线 | 净收益 | 噪声带 | 收益/带 |
|---|---|---|---|---|---|
| MAE（相对） | +18.24% | +2.43% | +15.8% | 8.48% | **1.9× 确认** |
| R²（相对） | +2.80% | −0.43% | +3.23% | 3.74% | **0.86× 未确认** |

绝对 R² 上，三任务平均约 **+0.023**。差异来自 `formation_energy`：它的 MAE 可以改善 25%，但 R²
只能从 0.984 挪到 0.991 —— MAE 口径放大了这个饱和任务的贡献。

因此本报告的表述是：**encoder 调参在 MAE 上确认有效，在 R² 上落在噪声带内**。只报 MAE 那一半会
高估结论强度。放进参照系：replay 配方本身值 +0.068 mean R²，末端固化 +0.006 —— 本次调参约 +0.023
介于两者之间。

### Stage A adopted configuration (A4 seed band applied)

The A1 single-seed leader did **not** survive seed repeats: `L256/[1024,512,256]/1e-3` scored
+23.9% at seed 2025 but +15.9% and +15.5% at 2026/2027, mean **+18.45%**.

| configuration | mean (3 seeds) | seed range | wall/run |
|---|---:|---:|---:|
| `L256 / [1024,512,256] / 1e-3` | +18.45% | 8.48% | 18.2 min |
| **`L384 / [256] / 1e-3`  ← adopted** | **+18.24%** | **4.80%** | 16.4 min |
| `L256 / [512] / 1e-3` | +16.70% | 8.25% | 16.3 min |
| untuned baseline | +2.43% | 5.32% | 20.3 min |

- **Seed band = 8.48%** (largest within-arm range). All three candidates are tied; the ordering
  among them is noise and is not reported as a ranking.
- **The tuning gain is real**: adopted − untuned = **+15.8%**, ≈ 1.9× the band.
- `descriptor.n_grids` and `ae_lr` both keep their defaults — A3 found 16 no better (+20.9% vs
  +24.0% single-seed, inside the band) and 4 clearly worse (+3.7%); A6's ae_lr variants (+16.1%,
  +13.5%) are inside the band of the default.

**Adopted: `latent_dim = 384`, `encoder_hidden_dims = [256]`, `encoder_lr = 1e-3`** — two numbers
different from the untuned config, with the hidden layer unchanged, and it trains *faster* than
the config it replaces (16.4 vs 20.3 min/run).

**Rule amendment, disclosed.** The pre-registered rule said "cheapest among the tied". Wall-clock
separated the two leading tied candidates by 0.6% — inside run-to-run variation — so the rule was
deciding on measurement noise, and picked the candidate that was worse on both mean and stability.
The rule now treats costs within 5% as equal and resolves on the smallest seed range. This was
added *after* seeing the A4 output; it is recorded here and in `analysis/adopt.py` rather than
folded in silently.

## Defects found and fixed during the campaign

Recorded because each was silent — none announced itself as an error, and each would have shipped
a wrong number into the report.

| What | How it surfaced | Fix |
|---|---|---|
| Per-task winners could be selected from a *different probe*. `stage_b/` holds the per-task grids, the B-mt control's multi-task probes and the B4 repeats side by side, and their rows carry the same task names; `pick_heads.py` grouped by task alone, so `magnetization`'s winner was taken from a `bmtreg` 3-task run. | a `KeyError: 'bmtreg'` while rendering the stage-B figure — the numbers themselves looked entirely normal | whitelist `breg_`/`bkr_`/`bclf_` prefixes and print how many rows were ignored (48 here) |
| The A4 baseline arm would have been credited with one fewer seed than every other arm: its seed-2025 run is the A1 grid point, while A4 writes the other two as `a4_base_s*`. That narrows the measured band, which biases the rule toward declaring differences significant. | reading `adopt.py`'s arm mapping before trusting its output | pass the baseline runid into `arm_of()` |
| Stage B would have tuned heads on top of an encoder that was not the adopted one, had A3/A6 changed `n_grids`/`ae_lr` — the grid generator had no channel to pass an adopted `ae_lr` through. | auditing what stage B inherits from stage A | `--ae-lr` added to `make_grids.py` |

One operator error, not a code defect: the `seebeck` kernel-regression timing probe was killed
because its output directory was moved while the job was still writing (`probe_kr_seebeck`, rc 1).
Re-run clean.

## Outcome

_pending_
