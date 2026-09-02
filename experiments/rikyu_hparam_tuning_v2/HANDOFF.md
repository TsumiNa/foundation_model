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
Add an extensive feature and remeasure the extensive targets.

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
manifest. Note the caveat in the investigation section below: **warm-starting a new task from this
library has never been measured**, and the nearest evidence runs against it.

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

**The report and figures are not in git** (`experiments/**/results/` is ignored) and travel by rsync
— but the summary JSON is in git, so every figure can be redrawn.

## The next round's first investigation: why only material_type gains

The cost-side measurement is unambiguous: of the 24 tasks, **18 are materially worse, 5 are
unresolved, and only material_type gains** — and **the smaller a task's dataset, the larger its
relative loss** (corr(log rows, relative change) = **+0.636**, n=23), the exact opposite of the
premise that a shared encoder helps data-poor tasks. That deserves a real investigation. Below are
the leads that already have evidence and the experiments that would settle them.

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
5. **Measure warm-starting a new task directly** — never done. Take an encoder from the model library,
   fine-tune it on a task outside these 24, compare against training from scratch, and sweep **low
   data volumes**. This is the library's intended use, and the existing evidence (the less data, the
   larger the loss) runs **against** it, so it must be measured rather than assumed.

### What cannot be said right now

- Not "multi-task training helps data-poor tasks" — this round's data says the opposite.
- Not "the shared encoder learned a general materials representation" — not one of the 23 tasks
  improved because of it.
- Not "warm-starting from the model library will do better" — **that case has never been measured**,
  and the nearest evidence points the other way.
