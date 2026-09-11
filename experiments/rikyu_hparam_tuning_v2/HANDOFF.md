# v2 handoff

## Status

**COMPLETE**

Every stage is finished and verified. All measurements in this round ran on code after PR #45, which
fixed `ReduceLROnPlateau` firing per batch rather than per epoch; hyper-parameters chosen before that
and the ceilings quoted alongside them were void and have been remeasured, so no pre-fix number
appears in the deliverables.

The `v1-complete` tag exists and no v1 artefact was rerun or modified.

## Adopted parameters

```
model.latent_dim            = 384        <- changed (was 128)
model.encoder_hidden_dims   = [256]      = default
training.encoder_lr         = 2e-3       <- changed (was 5e-3)
training.scheduler.min_lr   = 1e-5       <- changed (was 1e-4)
training.scheduler.patience = 5          = default
training.scheduler.factor   = 0.5        = default
model.head_hidden_dims      = [64]       = default
training.head_lr            = 5e-3       = default
model.kr_x_hidden_dims      = [128,64]   = default
training.kr_lr              = 5e-4       = default
training.early_stopping.patience = 24    = default
training.learnable_loss_balancer = false <- explicitly off
```

Everything else stays at its default. **After 1,080 + 120 + 100 + 10 runs, three numbers actually
changed** — and every other "stayed at the default" was measured, not skipped.

## Key numbers

Each carries its seed count and noise convention. The noise band is the single-run range; verdicts
use the standard error of the difference.

| Stage | Config | Metric | Mean | Seeds | Resolvable at | Resolved? |
|---|---|---|---|---|---|---|
| Stage 0 anchor | untuned + container 0.3.2 | probe6 relative score | reference | 9 | sigma = 2.05% | - |
| Stage 0 | previously chosen encoder vs untuned | same | -0.83% | 9 | 2SE 2.73% | **no** |
| A' finals | `L384_E0p002_M1e-05_P5` | same | +1.83% | 25 | - | - |
| A' finals | same vs the finals' untuned control | same | **+1.56%** | 25 vs 25 | 2SE 0.84% | **yes** (1.9x) |
| A' finals | 2nd place `E0p001003...` | same | +1.44% | 25 | 98 seeds to separate from 1st | no |
| A' finals | 3rd place `E0p0042...` | same | +1.35% | 25 | 62 seeds | no |
| A' grid | configs tied with the leader | - | **29 / 296** | 5 | 2*sqrt(sem1^2+sem2^2) per pair | - |
| A4 | scheduled vs fixed LR | same | +4.12% | 60 runs | 2SE 0.78% | **yes** |
| a2b | early stopping 40 vs 24 | same | +0.252% | 5+5 | 2SE 0.403% | **no** (needs 6 seeds) |
| B' grid | leader vs default head | same | +0.608% | 5+5 | 2SE 1.045% | **no** |
| B' finals | leader vs default head | same | **+0.007%** | 25+25 | **172,360** seeds | **no** |
| balancer | best off-arm vs best on-arm | same | +3.61% | 5+5 | 2SE 1.56% | **yes** |

### The six Stage C' arms

| Arm | mean R2 | big | mid | small |
|---|---|---|---|---|
| `c2_top1_cons` | **0.7274** | +0.0206 | -0.0119 | -0.0283 |
| `c2_top3` | 0.7262 | +0.0109 | -0.0071 | -0.0194 |
| `c2_top1` | 0.7229 | +0.0244 | -0.0071 | -0.0243 |
| `c2_top2` | 0.7225 | +0.0197 | -0.0067 | -0.0076 |
| `c2_base_cons` | 0.7189 | +0.0209 | +0.0004 | -0.0218 |
| `c2_base` (untuned) | 0.7155 | +0.0214 | +0.0065 | -0.0286 |

Deficits are against the **same-regime** ceilings; 1 seed per arm, so small differences are not
resolvable.

## Three conclusions that must travel together

**1. Tuning buys +0.0074 / +1.03% across 24 tasks** (untuned 0.7155 -> tuned 0.7229, mean R2), and
consolidation lifts it to 0.7274. Same magnitude and direction as the probe's +1.56%, **but with 1
seed per arm none of it is resolvable**.

The three configs promoted to deployment scale span only 0.0037 (0.51% relative), below the 1e-2
practical threshold — **the probe's ranking carries no actionable information at deployment scale**.
That is exactly the value of promoting three rather than one: promoting one would only have shown
"the tuned arm beats the baseline".

**2. Transfer is negative at deployment scale, and it retracts the probe's conclusion.**

24 tasks, the task under test placed last, **10 shuffled repeats**, matched test rows: **18 of 24
tasks do worse than training alone**, 5 are unresolved, and only `material_type` does better
(+13.41%). On the six-task probe, zt +6.85% and magnetization +4.40% looked like positive transfer;
at 24 tasks both fall back to unresolved, and magnetic_moment turns to -5.75%. **A probe is a cheap
sieve, not a substitute for the real regime.**

Tripling the repeats from 3 to 10 moved tasks in one direction only: the split went from 1 / 15 / 8
to 1 / 18 / 5, with all three newly-resolved tasks landing on the single-task side. Nothing moved
the other way.

**3. The single-task ceilings were remeasured in this regime; the inherited set cannot be reused.**
The old ceilings predate PR #45 and are **too low for 17 of 23 regression/KR tasks, by +0.0275 on
average** — and the offset is **not constant** (seebeck 0.104 low, dielectric_ionic 0.017 high), so
it cannot be corrected with a shift. The new ceilings: 24 tasks x 5 seeds, container 0.3.2, adopted
config, differing from a campaign run only in `pretrain.task_sequence`
(`summary/ceilings_adopted.json`). Every deficit this round is computed against them.

> **The shortlist was a budget truncation, not noise-aware selection.** At 5 seeds, 29 configs were
> tied with the leader and the shortlist took only the top 8, so **22 tied configs never reached the
> finals** — purely because their noisy sample means ranked below 8th. The finals winner is the best
> of those 8, not of 29. This does not overturn the conclusion (the finals show the top four still
> indistinguishable at 25 seeds, so the region is flat), but a stricter next round should either keep
> every tie or state the truncation explicitly.

## Rejected and not adopted

| Item | Verdict | Basis |
|---|---|---|
| learnable loss balancer | **rejected** | mechanically inverted (sigma^2 = L, corr(sigma, loss) = +0.970, AE weight 20,075 vs seebeck 1.5); still inverted 112x with AE excluded |
| PCGrad | **not adopted** | per-task encoder gradients measured directly, no directional conflict found (the method acts only on negative-cosine pairs) |
| head tuning | **not adopted** | 16 of 24 configs tied; "change nothing" ranks 8th with the smallest 2SE; in the finals the leader's edge collapses 99% |
| early stopping 24 -> 40 | **not adopted** | +0.252% unresolvable, and it costs 11% more wall clock |
| extending the LR search downward | **not needed** | the below-floor optimum is unresolvable (-0.11%) |
| turning the LR schedule off | **rejected** | scheduled beats fixed LR by 4.12% |

## What the next round should do, in priority order

**1. Give the descriptor a cell-scale feature.** `formula_to_composition` returns **atomic
fractions** summing to 1, so `Fe2O3` and `Fe4O6` produce an identical descriptor — the model cannot
see how many atoms are in the cell. Meanwhile corr(Volume, atoms per cell) = **+0.868, 75.3% of the
variance**. This is not label noise (only 7 of 33,822 reduced formulas repeat); it is a
generalisation gap. Volume's 0.619 ceiling is the limit of a scale-free input, not under-training.
Add an extensive feature and remeasure the extensive targets. **Confirmed (experiment 8):** with the notebook's XenonPy classic descriptor volume trains to
R² 0.997 alone, and as volume per atom it reaches 0.979 on KMD (2026-09-12 baselines). What is
still open is the policy — per-atom targets or a scale-carrying descriptor — before phase B.

**2. Separate "placed last" from "ordering design".** Attribution is currently impossible: position
correlates with cost at only +0.216, and front-of-sequence tasks lose more (7.48% vs 4.86%), so
position is not the main driver. But the comparison confounds three things — 1 seed vs 3 repeats,
hand-designed vs random ordering, and position. It needs a dedicated experiment: hold one task at a
fixed position and vary only the order of the other 23.

**3. Extend xfer to n=10.** **Done.** 240 runs complete, all carrying 24 steps. It took two
walltime handovers (`74033` -> `76757` -> `78201`, chained with `--dependency=afterany` so no time
was idled) plus one three-hour login-node outage that did not touch the running jobs. Eleven tasks
now show spread wider than their own seed noise, up from nine at n=3 — expected, since an sd from 3
samples underestimates spread.

**4. The pre-trained model library.** **Built**: 240 models, 1.56 GiB, at
`$OUTBASE/model_library/` with `MANIFEST.json` and `INDEX.md`. Each model is an encoder trained
continuously on all 24 tasks and finished on one particular task, and for downstream transfer
learning "which task it finished on" is a usable prior — so the task under test, the executed
ordering, the seed, the hyper-parameters and every model's score on all 24 tasks are in the
manifest. Warm-starting from it has now been measured (experiments 5 and 11 below): 4 better / 4
worse / 14 unresolved against training alone, and an encoder that never saw the task does as well
as one that did — the library's value is the 23-task representation, not the finishing task.

## Methodology worth reusing beyond this project

1. **A probe must span large / medium / small**: sigma fell from 5.01% to 2.05%, dropping the seeds
   needed to resolve 1% from 101 to 17.
2. **Rankings are bought with seeds, not grid points**: this round reproduced **winner's curse twice
   independently** — A''s 5-seed leader finished 10th at 25 seeds, and B''s 5-seed leader lost 99% of
   its edge. More grid points only buy more lottery tickets.
3. **Tie detection must use both arms' standard errors**: `2*sqrt(sem1^2 + sem2^2)`. Using only the
   leader's understates the uncertainty and under-reports ties — correcting it took the count from 11
   to 29.
4. **Put "change nothing" in the grid as a formal config**: it costs 1 of 24 grid points and turns
   "the heads need no tuning" into a ranking result rather than an argument.
5. **An inherited baseline must be remeasured in the current regime**: the old ceilings were too low
   for 17 of 23 tasks, and the offset is not constant.
6. **Any experiment run on the current best has to be redone when the best changes**: this round hit
   it twice (early stopping, and the 24-task ceilings).
7. **A group mean hides cancellation**: one small-group -0.019 was actually +0.060 against -0.022,
   and represented neither.
8. **A component that cannot fail loudly is a systemic risk**: five instances this round — the
   half-wired loss balancer; `parse_point` silently dropping unknown tags (which would have turned
   100 runs into the same baseline); `PACK=1` overriding the environment variable (misfiring 100
   GPUs); an ssh broken pipe returning 0; and an analysis script scoring an unfinished arm at step 23.
9. **A range is not a sigma**: `E[range] = d2(n)*sigma`, and d2 grows with n, so comparing ranges
   across different seed counts compares seed counts.
10. **Do not extrapolate with a median step time when steps grow**: replay makes each step more
    expensive, the median is dominated by the cheap early steps, and this once underestimated Stage
    C's remaining time by an order of magnitude (1.3h vs 11-17h).
11. **`--resume` is decided by a run's length, not its stage name**: xfer inherited "probe stages do
    not resume" while being a Stage-C-length run, so a kill lost a full day — although the
    checkpoints had been written all along.
12. **Two arms must be compared on the same test rows**: the data module splits by the union of the
    tasks it loads, so single-task and multi-task test sets differ (multi-task is 3-7% larger). When
    the containment is a strict subset, restricting to the shared rows corrects it exactly.
13. **Calibrate the packing factor on the real workload**: PACK=8 measured 7.1x on probe6, but on the
    24-task workload **PACK=24 measures 2.98x (75% efficiency) and PACK=32 measures 3.68x (69%)**.
    Neither GPU nor host memory is the constraint (at PACK=32: 32 of 185 GB of GPU memory, 316 of
    1,691 GB of host memory) — the CPU is.

## Where the artefacts are

| Artefact | Path |
|---|---|
| Summary JSON (19 files, **in git**) | `experiments/rikyu_hparam_tuning_v2/summary/*.json` |
| Report | `results/REPORT_v2_20260830.md` (gitignored) |
| Deck (26 slides) | `results/REPORT_v2_20260901.pptx` (gitignored) |
| Figures (10) | `results/*.png` (gitignored; redrawable from the summary JSON via `analysis/plots.py`) |
| Raw run output | RIKYU `/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2/` |
| Methodology notes | `NOTES.md` |
| Transferability summary page (trilingual EN / 中 / 日, **in git**) | `summary_page/gen_page.py` + `page.js` → `summary_page/transferability_summary.html`; published as a claude.ai artifact |
| Per-run scores behind that page's figures | `summary/position_runs.json` (four tasks × 240 runs), `summary/rep_raw.json` |
| Budget / optimisation checks (experiments 6–7) | `summary/long_budget.json` |
| Labels page (trilingual, **in git**) | `summary_page/gen_labels_page.py` + `labels_page.js` → `summary_page/mp_labels_summary.html`; data `summary/mp_labels_page_data.json`, `summary/baselines_mp2026.json`, `summary/space_group_classes.json` |
| Descriptor contrast (experiment 8) | `summary/descriptor.json`; tables `data/desc_xenonpy_{classic,nosum}_trans.parquet` (gitignored, rebuilt by `scripts/make_comp_descriptors.py`) |

**The report and figures are not in git** (`experiments/**/results/` is ignored) and travel by rsync
— but the summary JSON is in git, so every figure can be redrawn.

## Why only material_type gained — and what the warm-start stage found

The transfer stage said 18 of 24 tasks are materially worse when they arrive last, and that the
smaller a task's dataset, the larger its relative loss (corr(log rows, relative change) = +0.636).
**The warm-start stage (`stage_ft`, 480 + 40 runs) resolves most of that.** At step 24 of a
continual sequence the new task shares the encoder with replay from 23 others and owns about 1% of
the gradient if it is small; early stopping watches the sum over all 24 and fires when the replayed
tasks stop improving, so the new task gets ~60 epochs where alone it takes 90-150. Taking the same
final checkpoint and training the same task with replay removed:

| Comparison | Better | Worse | Unresolved |
|---|---|---|---|
| xfer (replay, placed last) vs alone | 1 | 19 | 4 |
| frozen encoder vs alone | 4 | 12 | 7 |
| **warm-start (encoder + head) vs alone** | **4** | **4** | **14** |
| warm-start vs the replay step | **19** | **0** | 2 |

So the loss in the transfer stage was mostly **replay dilution at the task's own step**, not a bad
representation. Warm-starting from the 24-task encoder is roughly break-even against training alone:
material_type +21.9%, zt +6.7%, magnetization +4.9%, dielectric_total +3.4% gain; final_energy -9.1%,
volume -6.9%, dos_density -4.2% lose. zt and magnetization are the probe's original winners, back as
real ones. The frozen encoder is not a drop-in feature extractor (12 worse) — except for
material_type, where frozen (0.7245) beats warm-start (0.6960): it wants the shared "ordinary
material" representation left alone.

Full table: `summary/ft.json`. Baselines and both fine-tune arms were audited for convergence; the
two KR tasks that hit the 150-epoch cap (seebeck, power_factor) were rerun at 400 in every arm and
the ceilings patched (`summary/ceilings_adopted_v2.json`); no verdict changed.

What remains unexplained, and the leads for it:

### Lead one: extensive vs intensive properties (a mechanism, not just a correlation)

| Kind | Tasks | Median delta R2 | Median relative |
|---|---|---|---|
| Extensive (`final_energy` / `volume` / `total_magnetization`) | 3 | **-0.1052** | **-17.0%** |
| Intensive | 20 | -0.0168 | -2.2% |
| Classification (`material_type`) | 1 | **+0.1245** | **+21.8%** |

The mechanism is independently verified: the KMD descriptor goes through `formula_to_composition`,
which returns **atomic fractions**, so `Fe2O3` and `Fe4O6` are identical to it and cell scale is
invisible; `corr(Volume, atom count) = +0.868`, 75.3% of the variance. An extensive target can only
be fitted through a within-dataset composition-to-size correlation, and multi-task training pulls the
encoder toward what the tasks **share** — which is exactly the scale-free part, erasing that
correlation. This explains why single-task training can do it and multi-task cannot.

**The confound must be acknowledged**: within the same 23,678-row dataset the only intensive control
is `density`, and at 0.9898 single-task it is already at the ceiling with nothing to lose. So "9x
worse within the same dataset" **is not usable evidence**; only the 6.3x ratio across all 24 tasks
is, and it rests on 3 extensive tasks.

### Lead two: material_type's gain cannot generalise to regression

Recomputed on matched rows (both arms on the same 7,354 test rows), the decomposition is clear:
**rare-class recall does not improve at all** (IQC even drops 2.14), while **precision** roughly
doubles (IAC 30.2 -> 53.5, IQC 51.1 -> 62.2). The source is `others` being misfiled into the rare
classes falling from 1.16% to 0.49% — 84 rows per run down to 36. So what the shared encoder improves
is its model of the **majority class**, which cuts false alarms.

R2 has no false-alarm axis. There is no mechanism in regression by which "crying wolf less often"
converts into score. So material_type's path to a gain **structurally cannot transfer** to the other
23 regression tasks, which weakens any inference that multi-task training has general value.

### Experiments, in cost order

1. **Add a control where the descriptor can see cell scale** (atom count or volume as an extra input
   feature) and rerun `final_energy` and `volume`, single-task and multi-task. If the extensive-property
   loss closes substantially, lead one holds. **This is the decisive experiment.**
2. **Introduce one or two more classification tasks** (even ones binned from existing continuous
   targets) and see whether they gain too. If they do, the specialness lies in the task type rather
   than in this task; if they do not, a finer explanation is needed.
3. **Recompute material_type's gain under a class-balanced metric** (or a binary collapse). If the
   gain shrinks sharply, much of it is a property of macro-F1 under extreme imbalance rather than a
   representation-learning benefit.
4. **Measure task-specificity of the encoder's representation per task**: compare linear separability
   or CKA similarity between the single-task and multi-task encoders, testing the hypothesis that
   multi-task training pulls the representation into a shared subspace.
5. **Measure warm-starting directly** — **done for the 24 in-distribution tasks** (`stage_ft`,
   table above). Still open: a task *outside* the 24, and a sweep over **low data volumes**, which
   is where a warm start should matter most and where nothing has been measured yet.
6. **Is warm-start's residual loss on the extensive properties just undertraining?** — **done
   (2026-09-07, jobs 85149 / 85150): no.** Both arms were given 500 epochs with early stopping OFF
   (`ftfl_<task>_o<k>` in `stage_ft`, 10 orderings; `stL_<task>_s<seed>` in `stage_single`, 5 seeds;
   configs `ft_full_long.toml` / `probe6_long.toml`, grids from `scripts/make_grid_long.py`, scored
   by `analysis/long_budget.py` → `summary/long_budget.json`). Training alone is insensitive to the
   budget (final_energy −0.1%, volume −0.1%, dos_density +1.0% against the early-stopped baseline).
   Warm-start stays resolvably below the same-budget single-task control: final_energy −10.0%,
   volume −15.9%, dos_density −9.6%. The per-epoch logs say why: warm-start's validation loss bottoms
   out early (median best epoch 143 / 72 / 62 against 130 / 101 / 112 alone) and then drifts up
   (+5% / +22% / +13% by epoch 500), while its training loss ends LOWER than the single-task
   model's (final_energy 0.011 vs 0.015, volume 0.012 vs 0.063, dos_density 0.0098 vs 0.0120). The
   pretrained encoder fits the training set faster and more completely and generalises worse — a
   generalisation gap, not a budget gap. This makes experiment 1 (cell scale in the descriptor)
   the decisive one rather than a longer schedule.
7. **Optimisation or representation?** — **done (2026-09-12, jobs 85797 / 85798): the
   representation.** Two arms on the same three tasks, scored by `analysis/long_budget.py`:
   - `stC_<task>_s<seed>` — alone, 500 epochs, early stopping off, scheduler patience 100000 so the
     learning rate never decays. R² 0.7495 / 0.6121 / 0.6307
     (final_energy / volume / dos_density) against 0.7735 / 0.6187 / 0.6315 with the
     schedule; the final training loss is HIGHER without annealing (0.052 vs 0.015, 0.090 vs 0.063,
     0.0121 vs 0.0120). A fresh model does not reach warm-start's training loss (0.011 / 0.012 /
     0.0098) with or without the schedule, so the low training loss is not an optimisation artefact.
   - `ftflr_<task>_o<k>` — warm-start with the encoder LR 2e-4 instead of 2e-3, early stopping on,
     400-epoch cap. Worse than the default warm-start: 0.6261 vs 0.7033 (-11.0%),
     0.5314 vs 0.5767 (-7.9%), 0.5801 vs 0.5988 (-3.1%). Early stopping fired at epoch
     51–69 with the training loss still at 0.14 / 0.22 / 0.016: underfitting, landing between the
     frozen arm and the default warm-start. Slowing the encoder does not help; the more it is allowed
     to move, the better, and the default still trails training alone.

   Taken with 6: the residual loss is not budget (6), not the LR schedule, and not controllable by
   slowing the encoder (7). The pretrained encoder fits the training set faster and further than a
   fresh one and generalises worse on exactly the labels that depend on cell scale. What remains to
   separate is exposure at step 24 from the pretrained representation itself — the unseen arms
   (stage_xu → ftzu / ftfu) do that — and whether the descriptor is the reason, which is experiment 8.
8. **Descriptor contrast on the extensive properties** — **done (2026-09-12, jobs 86328 / 86329).**
   Single task, the stage_single recipe, 5 seeds, three tasks. The descriptors are the ones
   `data/data/scripts/calculate_compositional_desc.ipynb` produced — XenonPy
   `Compositions(featurizers="classic")` (weighted sum / average / variance / max / min, 290 columns)
   followed by StandardScaler → PowerTransformer(yeo-johnson) — re-keyed by the pipeline's canonical,
   non-reduced composition (`scripts/make_comp_descriptors.py`); `nosum` drops the weighted-sum
   block, the only block whose raw cell amounts carry scale. Output `stage_desc` (`stXc_*`,
   `stXn_*`), scored by `analysis/descriptor.py` → `summary/descriptor.json`. R², last-epoch weights:

   | task | KMD | XenonPy classic (vs KMD) | XenonPy without sum (vs KMD) |
   |---|---|---|---|
   | volume | 0.6191 | 0.9966 (+61.0%*) | 0.5919 (-4.4%*) |
   | final_energy | 0.7739 | 0.7677 (-0.8%) | 0.7895 (+2.0%*) |
   | dos_density | 0.6250 | 0.6241 (-0.1%) | 0.6095 (-2.5%*) |

   - **volume's ceiling was the descriptor.** `Volume (normalized)` is `volume_scaler`
     (StandardScaler + Yeo-Johnson, λ ≈ 0, i.e. ~log) of the CELL volume, and cells run to 10,000
     atoms; KMD sees only atomic fractions. With the scale-bearing sum block the single-task R² goes
     0.62 → 0.997 (five seeds within 0.0002); without it, 0.59. Everything said earlier about volume
     as a "negative-transfer extensive property" is therefore about a label the descriptor could not
     see, not about transfer; its transfer behaviour has to be re-measured with a scale-aware
     descriptor.
   - **final_energy is not a scale problem.** It is per atom, the sum block does not help, and the
     descriptor family lands at 0.77–0.79 either way (gradient boosting on the same features: 0.76
     on the pipeline split and on a random split alike). The label itself is the open question:
     the dataset's `Final energy per atom` does not match MP's PBE `energy_per_atom` (Si −8.77 vs
     −5.42; Pt −51.5 vs −6.1; Au −50.6 vs −3.3; NaCl −6.9 vs −3.5), 18.8% of rows lie below −14
     eV/atom and actinides reach −86, while `Formation energy per atom` looks normal. An element-
     dependent energy reference of that size dominates the variance and is hard to learn from
     composition aggregates. **Where this column came from in the 2026-05-15 reformat, and whether
     it is the same quantity as the earlier final-energy models were trained on, is to be confirmed
     with the data owner before final_energy is interpreted further.**
   - dos_density: the family is level with KMD; the sum block adds +2.4%.
   - First attempt, discarded: a re-implementation of the classic descriptor with plain z-scoring
     (numerically identical to XenonPy's blocks, but the raw sum block's |z| reaches 83 and the
     network memorised training rows; volume R² 0.16) and two invented "bounded" variants. Their
     run directories sit in `stage_desc/_discarded_reimpl/` and are not used anywhere. Lesson kept
     in the working notes: descriptors and preprocessing come from the data notebooks, never
     re-implemented.
9. **The final_energy label itself, traced to its source** — **done (2026-09-12).** With the user's
   temporary MP API key the column was traced to `summary.energy_per_atom`, which in today's
   Materials Project is the GGA / GGA+U / r2SCAN *mixed* thermo scheme: ~20% of entries carry an
   r2SCAN total energy tens of eV below the GGA one (Pt −51.5 vs −6.1, Au −50.6 vs −3.3), so the
   column mixed two energy references. The 2025-04-10 export copied it faithfully — no collection bug.
   The same mixing sits behind `formation_energy_per_atom`, and the summary's structure and magnetism
   come from r2SCAN tasks for ~23k of the 33.8k entries. The dataset was rebuilt on GGA / GGA+U only
   (`data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet`, script
   `data/data/scripts/rebuild_mp_gga_20260912.py`, details in `..._CHANGES.md`), with per-cell values
   rescaled to the dataset's cell, electronic-structure values kept only where MP's own origin task is
   GGA-family, and the extra MP properties added (band gap and its labels, per-atom volume, per-volume
   magnetisation, magnetic ordering, elastic, dielectric + refractive index, piezoelectric maxima).
   **Single-task final_energy on the rebuilt label, same recipe, same descriptor (KMD), 5 seeds:
   R² 0.9986 ± 0.0001, against 0.7739 ± 0.0096 on the old label** (`stM_final_energy_s*`,
   `configs/probe6_mp2026.toml`). The 0.77 ceiling, and everything the transfer stages said about
   final_energy, was a label artefact. Every stage of this campaign used the old dataset; volume,
   final_energy, formation_energy, density, total_magnetization, efermi and band gap have different
   labels in the new file, so their verdicts are not comparable across versions and the affected
   stages have to be re-run on the 2026-09-12 dataset before anything more is concluded about them.
10. **Single-task baselines on the 2026-09-12 dataset** — **done (jobs 87957, 88115; `stage_single_mp2026`,
   `stN_*`, 130 runs)**: the nine relabelled MP tasks and the seventeen added properties (band gap, CBM,
   VBM, is-metal, is-gap-direct, per-atom volume, magnetisation per volume / per f.u., magnetic
   ordering, reaction energy, bulk / shear modulus, Poisson ratio, anisotropy, refractive index,
   piezoelectric maximum, space group), five seeds, the stage_single recipe, scored by
   `analysis/baselines_mp2026.py` → `summary/baselines_mp2026.json`. final_energy 0.9986, formation
   0.9963, density 0.9903, efermi 0.9271, dielectric_electronic 0.9091; volume 0.6058 with KMD while
   its per-atom form reaches 0.9786; magnetisation 0.72–0.74 in every form, magnetic ordering macro-F1
   0.556; band gap 0.883, CBM 0.881, VBM 0.923, is-metal 0.927, bulk modulus 0.928, shear 0.790,
   refractive index 0.893; Poisson 0.313, anisotropy 0.354, reaction energy 0.342, piezoelectric 0.010;
   space group (151 classes, ≥ 10 rows and a row in both splits, rarer groups missing like any other
   task's NaN) macro-F1 0.202 / accuracy 0.241 — only high-symmetry families are recognisable from
   composition. Configs `probe6_mp2026.toml` (41 tasks, dataset 20260912), grids `grid_singlen.txt`,
   `grid_singlesg.txt`. Presented in `summary_page/mp_labels_summary.html` §7–8.

11. **The unseen arms — does the encoder need to have seen X? — done (2026-09-10, stage_xu job
   89380 → ftzu / ftfu jobs 92372 / 92373).** 72 encoders that never saw X (the first 23 steps of
   three transfer orderings per task, X dropped, same seed), each fine-tuned with a fresh head,
   frozen (ftzu) and unfrozen (ftfu); n = 3 orderings per task against n = 10 for the seen arms.
   Scored by `analysis/ft.py` (2×SE with both arms' SE, |Δ| ≥ 0.01), direct seen-vs-unseen counts in
   `summary/seen_vs_unseen_counts.json`.
   - Warm-start, never saw X, vs training alone: 2 better (material_type +21.5%, magnetization
     +4.6%) / 4 worse (final_energy −10.2%, volume −6.5%, seebeck −2.9%, magnetic_susceptibility) /
     15 unresolved — the same shape as the seen warm-start (4 / 4 / 14).
   - Warm-start, seen once vs never: 2 better (seebeck +0.023, zt +0.021) / 2 worse
     (total_magnetization −0.019, dos_density −0.018) / 19 unresolved. material_type 0.6960 seen vs
     0.6939 unseen. With the encoder unfrozen the step-24 exposure — and the head trained there —
     changes nothing measurable.
   - Frozen, seen once vs never: 4 better (final_energy, total_magnetization, magnetization,
     magnetic_susceptibility; 5 by ft.py's one-sided rule, adding electrical_resistivity) / 0 worse /
     19 unresolved. On a fixed representation a head trained at step 24 beats a fresh one; that is the
     head, not the encoder. material_type frozen: 0.7245 seen vs 0.6698 unseen, both far above alone
     (0.5710).
   - Consequence: for a new task the recipe is "pretrain on the existing tasks, then warm-start
     fine-tune"; the continual step with replay is the most expensive part of the pipeline and adds
     nothing the fine-tune does not recover. Warm-start is the fixed transfer method for phase B.
   - Caveats: n = 3, so the unresolved column is wide; final_energy and volume rows are void (mixed
     label / scale-blind descriptor, experiments 9 and 8); the xu encoders are not bit-identical to
     the transfer run's step-23 state (below). Presented in
     `summary_page/transferability_summary.html` §5.

12. **Space group: why 0.24 here and 0.60 in the ShotgunCSP paper — done (2026-09-11).** The paper
   (Liu et al., npj Comput. Mater. 2024, Fig. 3: 33,040 stable MP entries, 213 groups, XenonPy 290,
   FC-NN, plain cross-entropy, top-1 60.22 ± 0.87%) and the pipeline's stN_space_group (151 groups,
   KMD, encoder [256]→384 + head [64], inverse-frequency class weights N/(K·N_c) in the PyTorch cross-entropy — the formula sklearn calls "balanced" — last-epoch weights) were
   compared with a local factorial on the SAME rows, label and split (`analysis/space_group_study.py`,
   16 arms × 3 seeds, `summary/space_group_study{,_table}.json`). The arm built from the project's own
   encoder/head classes reproduces the RIKYU run (0.247 ± 0.006 vs 0.241 ± 0.005, same stopping
   epoch). Findings, as matched-pair effects on top-1 accuracy:
   - **Loss weighting is the first-order cause**: unweighted cross-entropy +21 pt at the pipeline
     shape (+27 mean over four pairs). With 151 classes the inverse-frequency weights range ×0.067 (Fm-3m) to
     ×22 (10-row groups); the three largest groups are 26% of rows but 2% of the loss, and the
     as-run head recalls 29% of Fm-3m, 4% of Pnma, 0% of P2_1/c and C2/c. Macro-F1 does not benefit
     either (0.20 → 0.31 without weights). The weighted validation loss bottoms at epoch 6 and climbs,
     so early stopping fires at 31 epochs with the training loss at 1.6.
   - **The descriptor is the second cause**: XenonPy classic (290) +11 pt over KMD in every setting,
     but classic WITHOUT its weighted-sum block only +1 — the gain is cell size. Atoms per cell alone
     predicts the space group at 0.242 (majority 0.102; 1.9 bits MI). Same root as volume: KMD is
     scale-blind. Third task that needs the cell scale.
   - Model: the paper's 4-layer GELU/dropout net +4 pt (and −9 pt under balanced weights). Optimiser
     recipe −0.3, input scaling −0.7, random split +1.6, best-val-loss vs last epoch −4 (last is right).
   - All three switched: 0.611 on our data = the paper's 0.602.
   - **Pipeline confirmation on RIKYU (5 seeds each, `summary/space_group_confirm.json`)** through the
     new `[[tasks]] class_weights = "balanced" | "none"` knob (default unchanged), run via SRC_OVERRIDE on
     a patched copy of the 0.3.2 container's package: stN as run 0.2412 ±
     0.0053 / F1 0.2022; stW (none, KMD)
     0.4650 ± 0.0064 / F1 0.3127; stX (none, XenonPy classic
     precomputed) 0.5568 ± 0.0087 / F1 0.4002. Configs
     `probe6_mp2026_sgnw.toml` / `_sgxc.toml`, grids `grid_singlesgnw/xc.txt`, stages `singlesgnw` /
     `singlesgxc`. The space-group baseline to quote is 0.465 / 0.313 (KMD) or 0.557 / 0.400
     (scale-aware descriptor), not 0.242 / 0.202; the labels page's "only high-symmetry families are
     recognisable" is withdrawn. Presented in `summary_page/space_group_summary.html`.
   - Open: whether magnetic_ordering / is_metal / is_gap_direct want the weights (two runs each);
     the head shape for many-class tasks (the remaining 4–5 pt to the paper's net).

**stage_xu, cost and caveats (2026-09-09).** One xu run is the matching transfer run minus its last
step: 23 steps, 1,578 epochs, 24k → 78k rows per epoch as replay accumulates, 78.5 M sample-epochs —
about 33 single-task trainings; the stage is 72 of them (≈ 28% of the transfer stage, ~200 GPU-hours
at PACK=6). Submitted at PACK=24 it ran ~70 h per run and TIMED OUT at 48 h with 1/72 done (steps 17–22
reached); resumed at PACK=6 as job 89380. Two consequences: (1) long continual stages go out at
PACK ≤ 6; (2) a resumed run restarts its interrupted step, so its RNG stream differs from an
uninterrupted run — steps 1–21 of xu_curie_o2 reproduce xf_curie_o2 epoch for epoch, steps 22–23 do
not (49 / 69 epochs vs 61 / 63). The xu encoders are therefore "same ordering, same seed, never saw
X" but not bit-identical to the transfer run's step-23 state. Keeping the penultimate step's
checkpoint in future transfer stages makes this whole stage unnecessary.

### What can and cannot be said now

- "Multi-task training hurts data-poor tasks" was a replay artefact: with replay removed the small
  tasks are level or better (magnetization +4.9%, magnetic_moment unresolved).
- "The shared encoder is a general feature extractor" still cannot be said — frozen, it is worse for
  12 of 24 tasks.
- "final_energy is hard" cannot be said: on the GGA-only label it trains to R² 0.9986 alone. Its −9.1%
  warm-start loss, its position curve and its ledger were measured on the mixed-scheme label and are
  void until re-measured on the 2026-09-12 dataset.
- "The extensive-property losses would close with more training" cannot be said: at 500 epochs
  without early stopping warm-start is still 10–16% below the same-budget single-task control, with
  a lower training loss and a higher validation loss than that control.
- "volume is an extensive property the shared encoder cannot serve" cannot be said any more: with
  a descriptor that carries cell scale, training alone reaches R² 0.997. Its transfer verdicts
  (xfer −16.1%, frozen −16.6%, warm-start −6.9%) were measured on a label the descriptor could not
  see and have to be re-measured.
- "The fitting advantage is an optimisation artefact" cannot be said either: a fresh model with the
  learning rate never decayed ends with a higher training loss, not a lower one, and slowing the
  warm-started encoder underfits rather than regularises.
- "The encoder has to have seen the task during pretraining" cannot be said: warm-started from an
  encoder that never saw X, the counts against training alone (2 / 4 / 15) and against the seen
  warm-start (2 / 2 / 19) are the same picture, and material_type's gain is intact (0.694 vs 0.696).
  What the step-24 exposure buys is a trained head, which only matters when the encoder is frozen
  (4 tasks better, 0 worse).
- "Space group is barely learnable from composition" cannot be said: the 0.24 was the inverse-frequency
  class weights (−21 pt on 151 classes) plus KMD's blindness to cell size (−11 pt); the pipeline
  reaches 0.47 with the weights off and 0.56 with a scale-aware descriptor, and the paper's 0.60 is
  reproduced on our rows with its network.
- "Warm-starting from the model library beats training alone" can be said for four tasks and
  denied for three; for the rest it is a wash. The library is a reasonable starting point, not a
  free win, and the extensive properties (final_energy, volume) should not be warm-started from it
  until the descriptor can see cell scale.
