# Does attention improve cross-task transfer?

This investigation tests reproducible effects, not immediate model optimization. It follows
`transformer_encoder_20261002` but uses a separate campaign, new training seeds and four targets.
Only the installed 0.5.0 encoder and LinearBlock components are used; package code is unchanged.

## Registered design

- Encoders: tuned-size MLP, larger MLP, grouped-token mean-pooling Transformer, and the same
  grouped-token architecture without attention. Latent width 384; report actual parameter counts.
- Seven scalar source tasks: formation energy, band gap, Efermi, volume, density, final energy,
  total magnetization. Targets: dielectric total, bulk modulus, shear modulus, piezoelectric max.
  Bulk and shear are related endpoints, not independent task families. Dielectric components and
  refractive index are excluded from sources to avoid near-deterministic target relationships.
- Conditions: random encoder (frozen probes and trainable scratch), real labels for 1/3/7 sources,
  and independently shuffled labels for seven sources. The nested 1/3 task sets use the first
  entries above. Task identity remains part of the contrast; this is not a universal scaling law.
- Each source fit uses 6,000 optimizer updates and batch size 128, sampling tasks uniformly and
  sampling labeled training compositions within a task. Warmup + cosine decay, clipping 1.0,
  weight decay 0.001. Joint supervised training deliberately removes reconstruction/replay as
  confounders; findings must not be described as results of the original continual workflow.
  Equal updates and batch size do not imply equal FLOPs or unique examples; record timing and
  per-task exposure. Retain source weights at 2,000 and 6,000 updates plus loss/representation logs.
- Shuffle labels independently inside training and validation partitions, preserving each task's
  missingness and value distribution. Target labels are never shuffled. No target label enters
  source training. Source training never sees global validation/test compositions.
- Composition aliases share the strictest original split (test > val > train), retaining the first
  raw record without averaging phases. Preserve that record's original composition string.
  KMD uses the installed 94-element vocabulary and 8 grids. Scalar normalization uses training
  labels only; each target's 10% subset is fixed across encoder/seed and nested in 100%.
- Pilot: two distinct seeds × four encoders × three source LRs = 24 fits, 2,000 updates each.
  Choose one source LR per encoder by mean **target-validation standardized MSE** of frozen ridge
  probes across four tasks at full training size. Pilot never evaluates target test labels.
  This is bounded transfer-aware tuning with equal candidate counts, not selecting on source loss.
- Formal: ten fresh seeds × four encoders × five conditions = 200 lanes, 160 source fits.
  At the final budget, each lane evaluates four targets × two training sizes using frozen ridge,
  frozen nonlinear head and full fine-tuning. Three LR candidates per neural mode are selected
  by target validation loss; neural fits retain best and last weights and their losses. Frozen
  BatchNorm buffers must remain unchanged. For random encoders, full fine-tuning is scratch.
  4,800 selected endpoints, 9,600 neural candidate fits, plus 11,200 ridge candidates. Random
  frozen features distinguish useful pretraining from architecture-only priors. The primary
  source-count comparison fixes total updates; checkpoints also permit later compute curves.
- Full fine-tuning uses the same explicit identity-output 128→64 head. No test metric selects
  epochs, LR, architecture or which completed seeds to report. Report all registered endpoints.

## Interpretation before looking at results

A potentially useful Transformer must show seed-repeatable downstream effects, not merely
prediction disagreement. Report paired transfer-vs-scratch changes, real-vs-shuffled source
labels, seven-vs-one sources, and the encoder × pretraining interaction. Always compare absolute
errors against both MLP scratch baselines and the no-attention control. Use paired seed-level
bootstrap intervals (10 training replicates, not thousands of independent test replicates),
RMSE/MAE on original target scales, per-composition predictions and cross-seed variation. Report
all target/fraction strata; related mechanical tasks are not two independent confirmations.

This is exploratory: the parent data's dielectric test set has already been inspected. No
positive result is proof of universal Transformer superiority. A null result only constrains
this descriptor, model size, supervised objective and optimization budget. Diagnose loss,
saturation, latent variance and gradient norms before interpreting a negative result.

## Execution

Preprocess with `scripts/prepare.py --input data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet
--output data/transformer_controls_20261003` (one command). It writes a dated Parquet, descriptor
NPZ and checksummed manifest. Run the installed ARM image only, no bound src or PYTHONPATH.
`array.sbatch` accepts runtime-only workspace/image/data/output paths and a merged revision.
GPU functional smoke precedes the pilot; unpacked cases precede a separate PACK calibration.
Only after those pass may the remaining matrix launch. Initial maximum concurrency: 16 GPUs;
confirm costs from calibration before formal submission. At 300 JPY/GPU-hour, the provisional
100 GPU-hour campaign envelope is 30,000 JPY before tax, not a measured runtime prediction.
Do not exceed that envelope without revising the estimate and informing the user.

Completion markers are identity-bound; never mix smoke/calibration/pilot/formal results. Mirror
scientific outputs every 30 minutes, excluding weights routinely. Retain checkpoints remotely.
Failed lanes require diagnosis and bounded same-identity recovery; completed fits are skipped.

Storage: retain every source budget checkpoint and the validation-selected best/last target
weights; delete only unselected LR candidates' weights after prediction export. Keep every
candidate's loss history and validation score. Shared quota was checked before staging; avoid
copying source datasets or storing all unselected full-encoder weights per trial.

Local acceptance: train-only scaling, alias split isolation, shuffled-label masks, case identity,
frozen-buffer preservation, pilot selection completeness and partial collector validation have
focused tests. A CPU two-update/two-epoch functional run exercised the Transformer source and all
four targets, both training sizes, frozen/full heads and ridge; it is not scientific evidence.
GPU compatibility and performance remain gated on review/merge and real Slurm smoke.
