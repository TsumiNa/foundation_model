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

## Execution log

- 2026-10-03: PR #78 reviewed, findings resolved and squash-merged as
  `f46db926f622be5642c0ccad1c27ef1346f06662`. Twelve focused tests, Ruff, package-wide mypy
  and shell syntax validation passed. No package change; use the registered 0.5.0 ARM image.
- Prepared 48,402 composition-isolated rows with 464 descriptor dimensions. Data-file checksums
  are recorded in the registered manifest. Target train/validation/test counts are dielectric
  3,051/656/697, bulk modulus 4,430/943/997, shear modulus 4,370/925/991 and piezoelectric
  946/202/192. The shared storage quota was checked before staging.
- Submitted GPU functional smoke array **162343**, cases 0/6/12/18, one per encoder arm,
  each using two source updates and two target epochs. Its output root is `artifacts/gpu_smoke`;
  these outputs are excluded from scientific analysis.
- Submitted initial unpacked pilot array **162347**, cases 0–2/6–8/12–14/18–20, with
  `afterok:162343`. It cannot start unless every smoke case succeeds. At most 12 GPUs run
  concurrently, each with a one-hour walltime limit. Pilot outputs live in `artifacts/pilot`.
  Remaining pilot cases and the 200-lane formal campaign are not submitted yet; they remain
  gated on functional validation, measured utilization/packing throughput and a cost estimate.
- Smoke 162343 completed all four cases; all 96 selected smoke endpoints were finite and the
  installed-package/image/revision provenance matched. Initial pilot 162347 completed all 12
  cases. Transformer validation loss decreased without detected latent collapse; this is an
  optimization check, not a transfer result.
- PACK=3 calibration **162359** repeated those same 12 pilot cases in a separate
  `artifacts/pack_calibration` root. All completed with exactly matching validation metrics.
  Measured worker throughput improved 2.19–2.43 times; individual workers took 23–37% longer.
  Slurm averaged GPU utilization was 74–95%, with GPU memory 3.8–5.9 GB. The shorter unpacked
  MLP jobs were below the accounting sampling interval; their zero utilization values are not
  evidence of zero GPU use. Submitted the remaining pilot cases in PACK=3 array **162365**.
- All 24 pilot cases completed, with finite diagnostics and no detected low-variance collapse.
  Validation-only selection chose source LRs 0.0005 (MLP), 0.002 (large MLP), 0.00001
  (Transformer), and 0.0003 (no attention). Pilot scores are tuning evidence, not held-out
  evidence of architecture superiority. The selection record lives in
  `artifacts/pilot_selection.json` and is part of formal run identity.
- Submitted unpacked formal calibration **162372**, cases 1–3/51–53/101–103/151–153:
  seed 10, all four encoder arms, and real 1/3/7-source conditions. These 12 full-budget lanes
  belong to `artifacts/formal` and will count toward the registered 200-lane matrix. Each has
  a one-hour walltime cap. Remaining formal lanes await this complete downstream cost measure;
  any packed repetition for calibration must use a separate root.
- Formal calibration **162372** completed all 12 lanes: 288 finite selected endpoints, common
  registered provenance, and all 36 expected source checkpoint files verified. Slurm elapsed
  times were 136–537 seconds per lane (16–33% average GPU utilization). None of the 576 neural
  LR candidates reached the 250-epoch cap; median stopping epochs were 47–79 by encoder/mode.
  Source diagnostics were finite without detected low-variance collapse. These are seed-10
  results only, with no scratch/random or shuffled-label conditions yet; they cannot establish
  a reproducible transfer advantage.
- Submitted full-budget PACK=3 calibration jobs **162407/162408/162409/162410**, respectively
  repeating cases 1–3/51–53/101–103/151–153 in `artifacts/formal_pack_calibration`. Each reserves
  one GPU for at most 30 minutes. These repetitions are excluded from the formal result root.
  The remaining 188 lanes await measured full-budget packing throughput and its cost estimate.
- Full-budget packing calibration completed: worker throughput improved 2.07/2.10/2.28/2.28
  times for MLP/large MLP/Transformer/no attention. All 288 endpoint metrics matched the
  unpacked runs exactly. Slurm GPU utilization averaged 94–96%, with 3.8–6.0 GB recorded GPU
  memory. The whole campaign is now estimated at 10–15 GPU-hours (JPY 3,000–4,500 before tax,
  including calibration and a contingency), rather than the provisional 100-hour envelope.
  This estimate uses matched measured runtimes; it is not a guarantee for every seed/condition.
- Submitted formal array **162424**, indices 0–66, PACK=3, at most 16 concurrent GPUs, and a
  one-hour limit per array task. It covers the registered 200 cases and skips the 12 completed
  cases under the same identity, leaving 188 new lanes. The last pack is bounded at case 199.
  The shared quota was 877.8 GB of 1 TB before this submission. Scientific results stay under
  `artifacts/formal`; packing repetitions stay separate. No old campaign was restarted.
- The next synchronized snapshot verifies 110/200 lanes and 2,640/4,800 endpoints: both MLP
  arms have all 50 lanes, while Transformer and no-attention coverage is still incomplete.
  No final cross-encoder ranking is inferred from this unequal coverage.

## Reproduce the complete report

After every registered lane is complete, run `analysis/report.py` with `--root artifacts/formal`,
`--config configs/study.toml`, `--selection artifacts/pilot_selection.json`, and `--output results`
using paths relative to this experiment (or equivalent repository-root paths). It first reruns
the provenance-aware collector and refuses an incomplete campaign. It writes paired relative
effects, source-task-count curves, real-versus-shuffled controls, dielectric parity plots and
same-composition examples, plus an English report. Parity plots retain all points on explicitly
labeled symmetric-log axes. Composition examples are selected by ground-truth quantiles, never
by model error; their error bars show seed SD, not predictive uncertainty. Confidence intervals
resample paired seeds and remain pointwise exploratory intervals, without multiplicity correction.
