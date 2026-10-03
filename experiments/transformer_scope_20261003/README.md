# When does attention change cross-task transfer?

This expanded study tests whether reproducible Transformer-specific patterns emerge when input
representation, source-task identity, training budget and downstream optimization are varied.
It follows `transformer_transfer_controls_20261003`; its scripts are an experiment-local adaptation
of that reviewed harness, not imports from the older experiment. Old runs are not resumed or mixed
into this matrix. The installed foundation-model 0.5.0 package and official ARM image are unchanged.

## Registered scope

- **Representations:** KMD (464 inputs, 58 groups of eight), and raw atomic fractions in the
  package's **94-element order**, with one element-specific scalar token per position. The latter
  uses existing feature tokenization; zero-fraction positions remain present and there is no new
  masking or tokenizer implementation. Every encoder gets the same input within a representation.
- **Encoders:** MLP, mean-pooling Transformer, and capacity-matched token-wise MLP without
  attention. Transformer: four Pre-LN/GELU layers, width 192, six heads, feed-forward width 768,
  dropout 0.1, latent width 384. The no-attention control has feed-forward width 1,154. MLP uses
  two hidden layers of width 1,040 (KMD) or 1,152 (composition). Parameter counts are respectively
  1,970,912 / 1,954,176 / 1,954,184 for KMD and 1,885,824 / 1,890,048 / 1,890,056 for composition.
  Matching parameters within 1% does not match FLOPs or optimization dynamics.
- **Twelve source properties:** formation energy per atom, band gap, Efermi, volume, density,
  final energy per atom, total magnetization, energy above hull, equilibrium reaction energy per
  atom, atomic density, CBM and VBM. Some are correlated or algebraically related; twelve labels
  are not twelve independent physical families. Target aliases (dielectric components/refractive
  index and elastic-derived properties) remain excluded from sources.
- **Four targets:** dielectric total, bulk modulus, shear modulus and piezoelectric max. Elastic
  properties count as one physical family when aggregating, not two independent confirmations.
- **Source sets:** three single-property sets; three three-property sets; the previous seven
  sources; all twelve sources; independently shuffled twelve-source labels. Their exact indices
  are registered in `configs/study.toml`. The three-task sets deliberately vary identity and are
  not treated as three replicates of the same intervention. Random encoders provide the baseline.
- **Source budgets:** independent 6,000 and 24,000-update runs, batch 128, uniform task sampling,
  5% warmup followed by cosine decay. The schedules span their respective full budgets; the short
  run is not the first quarter of the long schedule. Both use the same selected source LR. Save
  source checkpoints at 6,000/24,000 when reached, plus final weights, loss, gradient norm, tanh
  saturation and latent variance. Joint supervised training uses no reconstruction/replay.
- **Splits and seeds:** three composition-hash 70/15/15 train/validation/test splits, seeds
  20261011/12/13; three optimization seeds 20/21/22 per split. Composition aliases collapse before
  splitting. Source training excludes all validation/test compositions for the active split.
  These are repeated holdouts of an already explored material pool, not new external data or
  nine independent dataset replicates. All targets share the active global composition split.
- **Training sizes:** fixed nested 1%, 10%, 100% target subsets per split, independent of encoder,
  pretraining and optimization seed. Normalization uses the active training subset only. The
  piezoelectric 1% subset has about ten compounds; its instability is part of the low-data test.

## Tuning and endpoint selection

Source pilot: six representation/encoder arms × four LRs × two separate seeds (2/3) = **48 fits**,
6,000 updates, first split only, twelve real sources. Select one source LR per arm using only
full-training target-validation frozen-ridge MSE, weighted equally across dielectric, elastic
and piezoelectric families (weights 1/3, 1/6, 1/6, 1/3 across the four endpoints). No pilot test
prediction or test score is evaluated. All arms receive equal numbers of source LR candidates.

Formal matrix: six arms × three splits × three optimization seeds × [one random baseline plus
nine source conditions × two budgets] = **1,026 lanes**, with **972 source fits**.

- Frozen ridge is evaluated everywhere: four targets × three training sizes, seven validation-
  selected regularization strengths, giving **12,312 selected ridge endpoints**.
- Full fine-tuning is registered only for random, real7, real12 and shuffled12: **4,536 selected
  endpoints** and **27,216 LR candidates**. Encoder LR is independently searched over
  1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 for every encoder, independent of its source LR. The common
  128→64 scalar head uses LR 0.002. Maximum 400 epochs, patience 50; select best epoch and LR by
  validation MSE only. This removes the earlier source-LR-relative fine-tuning restriction.
- Total: **16,848 validation-selected endpoints**. Report all registered outcomes. Do not rank
  encoders by source loss or select a source set/budget using target test error.

## Questions and analysis fixed before the run

The primary attention contrast is the real12-vs-random improvement for Transformer minus that
for the capacity-matched no-attention control, in target-training-SD-normalized RMSE units.
Negative values mean a larger pretraining improvement for Transformer; absolute RMSE/MAE must
also be compared. MLP is a second control. Random/full is scratch; random/ridge is random features.
Real12 versus shuffled12 tests dependence on source-label information.

Report these interactions by representation, source budget and target training size, both per
property and with equal weighting of the three physical families. Check whether the direction
persists across all three splits and changes between 6,000 and 24,000 updates. Ridge source-set
comparisons test task identity at equal count (1 and 3) and the 7-to-12 change. Different source
counts change identity and per-task exposure; these experiments do not by themselves establish
a universal scaling law. A different prediction alone is not evidence of useful transfer.

`analysis/collect.py` audits identity, budget, image and endpoint completeness, then reports partial
coverage, absolute errors by split, paired contrasts and encoder×pretraining interactions. Its
hierarchical bootstrap samples splits and then paired optimization seeds; only complete 3×3
matrices get intervals. Three overlapping holdouts are a small, dependent sample of this material
pool. Intervals are pointwise exploratory summaries, not external generalization guarantees or
multiplicity-adjusted discoveries. Seed SD is never called predictive uncertainty. Same-composition
predictions, best/last predictions and candidate histories are saved for later diagnostic figures.

## Deployment and resource gates

Prepare with `scripts/prepare.py --input data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet
--output data/transformer_scope_20261003`. The dated Parquets, two descriptor arrays and checksummed
manifest live in the shared data pool; each lane records the exact protocol, data, pilot selection,
scripts, merged revision and image identity. The CLI and Slurm env arguments follow `array.sbatch`.
The analysis CLI takes `--root`, `--config`, `--selection`, `--output` within this experiment.

PR review, fixes and squash merge precede RIKYU execution. Only the installed package in the
verified official ARM SIF may run; no src bind/PYTHONPATH or login-node training. The six-arm full-
path GPU smoke precedes pilot execution. Measure unpacked utilization and calibrate packing with
identical completed cases in a separate output root, first for pilots and then full-budget lanes.
Do not extrapolate the previous experiment's packing ratio. Maximum formal concurrency: 16 GPUs.

Initial budget envelope: **100 GPU-hours / JPY 30,000 before tax**, pending measured pilot and
full-budget throughput. Revise the estimate and notify the user before exceeding it. Check project
storage before expansion and retain at least 20 GiB of quota headroom; the complete campaign is
expected to add approximately 60 GiB, subject to the measured checkpoint footprint. Do not delete
old experiments to make room. Source checkpoints and selected **best** downstream weights remain
remote; last predictions/losses are kept, while last/unselected target weights from this new study
are removed only after an atomic completed-endpoint marker. Previous studies' retained weights
remain untouched. Routine rsync excludes weights, logs and temporary files but includes loss CSVs.

Completion is identity-bound and completed lanes/candidates are reused. A failed query is not an
empty queue: retry up to three times. Diagnose any failed or nonfinite fit before a bounded same-
identity recovery; no unbounded requeue loop. No scientific result includes CPU or GPU smoke or
packing repetitions. Update partial analysis while the fleet runs; pause the monitor on completion.

## Execution log

- 2026-10-03: registered this expansion after user authorization. Remote image is present,
  scheduler is available, and shared storage is 910.4 GiB of 1 TiB before this campaign.
  Prepared 48,402 composition-isolated records and three hash splits; all target training-only
  scales (including 1%) are finite and nonzero. Capacity matching was checked by instantiating
  all six installed-package encoder configurations. RIKYU training has not started.
- PR **#82** reviewed and squash-merged as `482babebdb198d3f401e026f16d2766ca734086c`;
  all five findings were addressed. Eighteen focused tests, Ruff, package-wide mypy and shell
  syntax checks passed. The 94-element Transformer CPU functional smoke completed all 24
  endpoints using two source updates and two epochs per downstream candidate; it is not a
  scientific result. The four staged data files were verified against the manifest.
- Submitted six-arm GPU smoke **162517** (pilot cases 0/8/16/24/32/40, PACK=1, six concurrent
  GPUs, 20-minute cap). Submitted initial unpacked pilot **162523**, cases
  0–2/8–10/16–18/24–26/32–34/40–42, at most twelve GPUs and one hour per case, with a strict
  `afterok:162517` dependency. No remaining pilot or formal fleet has been submitted yet.
  All runtime scripts retain the PR82 merged revision; documentation updates do not change it.
- GPU smoke **162517** completed all six arms in 28–34 seconds per allocation. All 144 endpoints
  and best/last exported predictions were finite; CUDA, installed-package image and merged-script
  provenance matched. The dependency released **162523**. Pilot results remain validation-only.
- All eighteen unpacked pilots **162523** completed: 6,000 source updates each, finite diagnostics,
  no low-variance latent dimensions, and declining source validation loss. Slurm average GPU
  utilization was 14–55%. Matched PACK=3 calibration jobs **162544–162549** completed in a separate
  output root; all 72 target-validation metrics matched exactly. Worker throughput improved
  1.51–2.46 times, with arm-wise maximum per-worker duration ratios of 1.22–2.02 and
  92–97% GPU utilization.
- Submitted remaining pilot array **162552** (PACK=3, at most twelve GPUs, 30-minute cap), covering
  all 48 registered indices with the eighteen completed cases skipped. Formal runs await the
  complete validation-only selection. Full-budget calibration will first cover each arm's
  scratch, real7/24k, real12/6k and real12/24k lanes; the three contiguous pretrained cases allow
  matched PACK=3 repetitions before sizing the formal fleet.
- All 48 pilot cases completed and passed the selector's provenance and budget audit.
  Selected source learning rates: KMD MLP/Transformer/no-attention = 0.001/0.0001/0.00001;
  composition MLP/Transformer/no-attention = 0.001/0.0001/0.001. Both selected Transformer
  seeds had zero low-variance dimensions throughout. The KMD no-attention and composition
  no-attention choices lie on opposite grid boundaries; selection uses only two pilot seeds
  and is not proof of a global optimum. Some rejected high-LR Transformer runs had transient
  low-variance dimensions, which remain in the diagnostic record.
- Staged and checksum-verified the selection artifact; submitted full-budget unpacked array
  **162569**, 24 formal cases with at most twelve GPUs and a one-hour cap per allocation.
  It covers scratch and the three pretrained calibration cases above for every arm, first
  split and seed 20. Completed cases count toward the 1,026-lane formal matrix. No expanded
  formal fleet has been submitted; its size and cost remain gated on full-budget calibration.
  Shared storage at submission was 912.1 GiB of 1 TiB.
- Partial full-budget audit: 19 of the 24 calibration lanes completed, yielding 456 finite
  endpoints under the registered common revision, image and selection identity. All eighteen
  pretrained lanes finished their source budget with zero low-variance dimensions in recorded
  probes. Among the completed lanes, all 1,368 neural candidates stopped before the 400-epoch
  cap (maximum 297); exported best/last predictions were finite. This is one split and one seed,
  so the collector reports partial effects without confidence intervals or architecture ranking.
- Submitted matched full-budget PACK=3 jobs **162623–162626** for the already completed KMD
  MLP/Transformer/no-attention and composition MLP triples (first cases 14/185/356/527).
  Each uses one GPU, a one-hour cap and a separate calibration root. Source checkpoints were
  verified before submission. The other two arms await their completed unpacked targets;
  the main fleet remains gated on all packing/cost/storage checks. Shared storage was
  914.5 GiB of 1 TiB at this check; no failed Slurm allocation was observed.
- The composition no-attention triples subsequently completed and passed identity/checkpoint
  checks, bringing the formal collector to 20 complete lanes and 480 endpoints. Submitted its
  matched PACK=3 job **162630** (first case 869). The composition Transformer is the only arm
  still awaiting complete unpacked downstream results before its packing calibration.
