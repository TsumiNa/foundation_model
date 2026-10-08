# Six-run MLP regression learning curves

Use one paired protocol to estimate the training-size dependence of transfer for all
13 scalar regression targets. The user approved six repeats at every eligible point;
classification is unchanged. The historical September curves remain separate controls.

## Registered comparison

- Training labels: 10, 20, 50, 100, 300, 1,000, 3,000 and 10,000, subject to each
  task's historical count plan and an 80% limit on its canonical training-label pool.
- 93 task/count points × six repeats × scratch/full encoder fine-tuning = **1,116 fits**.
  Reuse the 312 completed October 8 paired fits; train **804 additional fits / 402 pairs**.
- Six repeats use subset seeds 0–5 and the first six source models from the frozen
  October 1 uniform draw of 10 out of 240 models. No target score selects these models.
  Every one of the 13 targets must be absent from each checkpoint's task sequence.
- Each seed uses one fixed source checkpoint across counts, nested training-label subsets,
  training seed 2025 + subset seed, and an isolated target-head seed 12025 + subset seed.
- Encoder 464→256→384, head hidden width 64; package 0.5.0, production reconstruction
  workflow, historical learning rates, 150 epochs maximum, early-stopping patience 24,
  final workflow weights. Paired scratch/FT fits share target-head initialization and test data.

The budget counts **training labels**, with historical validation/test labels, archived
transformations and an unlabeled reconstruction pool still available. This is not a test
with only ten measurements in total or composition-disjoint pretraining. Run SD combines
subset/model/training variation and is not predictive uncertainty.

## Explicit inputs and reuse proof

Inputs: `data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet`, its three auxiliary
datasets, `data/agis_pretrained_20261001/{selection,population}_20261001.json` and
the first six drawn checkpoints, this folder's two historical TOML recipes, and
`experiments/rikyu_hparam_tuning_v2/summary/lowdata_n_plan.json`.

Reuse only `experiments/mlp_few_label_regression_20261008/artifacts/formal`, whose
input manifest is `data/mlp_few_label_regression_20261008/manifest.json` (SHA-256
`fda38283df55e0292072cad30876023c9d08551c974b39cb222525db3a9e63b8`).
Preparation verifies identical scientific worker bytes, resolved recipes, source data,
auxiliary inputs, the first three checkpoint hashes, and exact nested subset data at
10/20/50/100 for seeds 0–2. Those parquet bytes are preserved. The old 156 case markers
and their runtime identity are pinned. The collector accepts only these explicitly
registered reusable cases and the new campaign; it never rewrites either identity.
The launcher requires its sparse workspace checkout's HEAD to match the merged runtime
revision, and verifies the actual SIF against both the supplied hash and the registered
reuse-image hash before starting workers. The sparse checkout contains no `src/` tree;
stage this experiment's merged scripts at the workspace root and preserve their hashes.

September results fail this reuse proof (different checkpoint cohort, masks and seed/head
protocol), so no old/new averaging at 100 is performed. The unified curve's 100-label
point instead contains six repeats of the current paired protocol.

## Analysis and slides

Report all six individual runs, mean and sample SD (ddof=1), plus paired effects.
Incomplete coverage must show seed counts. Preserve every result in full learning curves.
For the descriptive task-count summaries corresponding to slides 13 and 56, exclude a
regression task/count comparison when **either method's mean test R² < 0.2**. Compute
each mean from all its repeats; never remove individual poor runs. Report retained and
excluded counts. This test-score filter does not select models, change training, or
establish an overall transfer success rate. Classification summaries remain unchanged.

Slides: one regression task per page, full observed training-size range on the left and
10–100-label detail on the right; historical curves in separate Appendix controls.

## Execution

`scripts/prepare.py` registers new and reusable cases and generates date-suffixed parquet
inputs. `scripts/run.py` is byte-identical to the completed paired worker.
`scripts/array.sbatch` covers only 402 new cases. `analysis/collect.py` combines pinned
reuse and new completions without weakening provenance checks.

RIKYU is under scheduled maintenance; use the verified official x86 CUDA 13 image on
the R-CCS public `ai-h200-brc-pu` GPU partition. No login-node training, source bind or
PYTHONPATH. Maximum four GPUs, 16 CPUs per allocation, four CPU threads per lane.
Use separate `artifacts/smoke`, `artifacts/formal`, and `artifacts/packing` output roots.
A merged-code GPU smoke precedes full-budget unpacked calibration at representative
larger training sizes; matched PACK=3 repeats must agree before sizing the remaining fleet.
Completed fits skip automatically; permit only the worker's single bounded recovery for
incomplete fits after preserving failure evidence. Keep all remote scientific weights.

## Execution log

- 2026-10-08: User approved a unified six-repeat protocol and separate historical controls.
  The previous 312 fits are complete; 804 new fits are registered, pending PR review/merge
  and GPU calibration. User also requested the mean-R² ≥ 0.2 filter on summary slides.
- 2026-10-08: PR #95 passed review after seven findings were addressed and squash-merged
  as `71b22ece3e58e3431e6186969f70442c45f7448b`. The 31 focused tests, Ruff checks,
  package-wide mypy and batch-script syntax checks passed. The training runtime remains
  pinned to this merge; subsequent execution-log commits do not change its identity.
- GPU smoke **396182** completed both paired fits for case 33 (band gap, 10,000 labels,
  subset seed 3) with two actual epochs. The installed package, runtime/input identity,
  paired head initialization, common test hash, finite metrics and nonzero training steps
  were verified. Smoke results are excluded from the scientific collector.
- Unpacked full-budget calibration **396184**, cases 30–32 and 399–401, completed all
  six pairs / twelve fits. These band-gap 10,000-label and piezoelectric 300-label cases
  count toward the formal study. The collector accepted all six; eighteen prediction
  tables were finite, all twelve fits stopped before 150 epochs (maximum 117), and
  eighteen required nonempty `.pt` files were retained remotely.
- Matched PACK=3 jobs **396200** and **396201** completed the same cases in a separate
  calibration output root. All twelve fit records, excluding elapsed seconds, and all
  eighteen prediction tables agreed exactly with the unpacked runs. Worker throughput
  increased **1.9726× / 2.3246×**, with maximum per-case slowdowns **1.2467× / 1.2591×**;
  allocations lasted 390 / 162 seconds. GPU-utilization and CPU accounting were unavailable,
  so no utilization percentage is claimed. Gates and calibrations used **0.49 GPU-hours**.
- Formal batch **396208** started with PACK=3, array 0–59%4, covering cases 0–179 and
  automatically skipping completed calibration cases. Batches 60–119 and 120–133 remain
  pending submission. Each allocation requests one GPU, sixteen CPUs and one hour;
  at most four GPUs run globally. Bounded sequential batches respect the public partition's
  cumulative requested-node-hour limit. No running or completed case is submitted twice.
- The remaining 132 nonempty allocations have a **9.06 GPU-hour** point estimate and
  **13.60 GPU-hours** with 50% margin, using the mean measured pair time, the slower
  measured packing slowdown and twenty seconds of allocation overhead. A scenario in
  which every fit reaches 150 epochs is **23.85 remaining GPU-hours**, using the slower
  measured seconds per epoch. These are extrapolations from two tasks / six cases,
  not guaranteed completion times; the fixed reconstruction pool and variable stopping
  epochs mean training-label count alone does not determine cost. The estimates were
  communicated before the fleet started. Stored evidence is in
  `artifacts/packing_audit_resource_estimate_20261008.json` and the execution state.
- At deployment, the local scientific snapshot contains **162 / 558 pairs**: 156 reused
  pairs and six new calibration pairs. It is partial and is not the live fleet count.
  Final six-repeat curves and the revised presentation require all 558 pairs to pass
  identity, numerical, coverage and retention checks; historical curves remain separate.
