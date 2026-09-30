# AGIS compound-holdout preprocessing

Prepare the 8 nickelate compounds at nominal 0, 10 and 20 GPa for the agreed transfer
experiment. Each pressure is its own kernel-regression task, with temperature as the single
coordinate. All three pressures of the held-out compound are excluded from scaler fitting.

From the repository root:

```bash
uv run python data/scripts/process_agis_data.py \
  --input-dir data/AGIS --output-dir data --date 20261001
```

The date defaults to the local execution date (`YYYYMMDD`). Existing outputs for that date
are never overwritten. Results and raw data remain ignored by Git.

## Curve preparation

- Read resistivity from the file's `rho` or `rho_xx` column, in **μΩ·cm**. Retain the original
  temperature/resistivity arrays, source hashes, headers and measured magnetic-field range.
- Check the folder composition against the header, allowing reordered element names and
  the `sample name - formula` header variant.
- Select nominal **0, 10, 20 GPa** using the header; the measured pressure drift remains in the
  original file. The previously agreed approximation of the 0.012 T ambient measurement is
  retained, with the measured field recorded.
- Drop and count nonfinite temperature/resistivity pairs. Sort by temperature and average
  readings with exactly equal temperatures. Preserve finite negative readings around zero.
- Linearly interpolate onto **300 equally spaced points from 6 to 290 K**, shared by every
  curve. Require complete coverage and perform no extrapolation. No smoothing is applied:
  the starrydata notebook's 30 K Savitzky–Golay window could broaden a narrow transition.

`data/agis_resistivity_20261001.pd.parquet` holds the 24 raw/resampled curves. Assets are grouped
under `data/agis_preprocessing_20261001/`. Each `fold_01` through `fold_08` holds three
pressure-specific parquet datasets, each with 7 `train` compounds and 1 `test` compound.
The root audit file has multiple pressures per composition; train from the pressure-specific
fold files, which have one row per composition. The manifest maps fold numbers to compounds.

## Standardization and inverse transform

The sequence flattening, fitted-pipeline persistence and inverse-transform interface follow
`data/scripts/process_ac_qe_te_data.ipynb`. For AGIS, fit this pipeline **independently for each
fold and pressure, using only the 7 training compounds**:

```text
StandardScaler(with_mean=False) → asinh → StandardScaler
```

The first scaler divides resistivity by its training standard deviation `s` without subtracting
the mean. The final scaler fits the mean `μ` and standard deviation `σ` of the asinh values:

```text
z = (asinh(ρ / s) - μ) / σ
ρ = s * sinh(μ + σ*z)
```

For constant training values, sklearn's unit-scale fallback keeps the transform valid.
Asinh accepts zero and negative values and has an inverse over the real line. Extremely large
predictions can still overflow floating-point arithmetic. The notebook's Yeo-Johnson pipeline
was checked on all 24 fold/pressure combinations: the fitted lambdas were negative, making
its inverse domain bounded. Unconstrained model outputs beyond that bound yield NaN.

The DOS notebook's per-curve maximum normalization is not used here: restoring the absolute
resistivity of an unseen compound would require its unknown curve maximum.

Each fold contains:

| File | Purpose |
|---|---|
| `agis_resistivity_fold01_0gpa_20261001.pd.parquet` (and 10/20 GPa, each fold) | Composition, explicit split, temperature in K, resistivity in original units and normalized targets. |
| `scalers_20261001.joblib` | Three fitted sklearn pipelines, reusable by every model/arm in this fold. |
| `tasks_20261001.toml` | Dataset/task fragment with `column = "rho_normalized"`, `t_column = "temperature_K"` and the fitted `tasks.scaler` path/key. |
| Root `manifest_20261001.json` | Grid, units, versions, fit compositions/counts, fitted parameters and inverse round-trip errors. |

Merge the datasets/tasks from `tasks_20261001.toml` into the experiment's existing configuration, retaining
the checkpoint's descriptor/model settings and all original tasks needed for warm-start/replay.
Generated relative paths are resolved from the command's working directory, like existing
`fm` configs. Regenerate the fragments at the destination if its dataset paths differ.

The standard `fm predict` and post-training evaluation paths load `tasks.scaler` and restore
both predictions and true targets to **μΩ·cm**. Temperature stays in K throughout. Metrics and
curve plots should use those restored values.

The outer fold has **no validation compound**. Do not use the test curve for early stopping,
hyperparameter selection or refitting preprocessing. Use a fixed epoch budget, or introduce
validation strictly inside the 7 training compounds and refit the scaler on that inner training
subset. For single-pressure fits, set the batch size to at most 7: the existing training loader
drops incomplete batches, so a batch size of 256 would drop every AGIS training sample. Replay
stages can retain their larger batch size because their active dataset includes the old tasks.

For the fixed-budget outer-fold fits, explicitly disable validation-monitored callbacks:

```toml
[data]
batch_size = 7

[training.early_stopping]
enabled = false

[training.checkpoint]
enabled = false
```

The default early-stopping monitor is `val_final_loss`, which does not exist for these
train/test-only folds. Keep the scheduler's training-loss monitor. Warm-start replay may
use validation from the original non-AGIS tasks, while excluding the held-out compound.
The preprocessing script rejects any missing or unexpected compound, including a completely
missing directory, before writing outputs.

## Verification

```bash
uv run pytest data/scripts/process_agis_data_test.py \
  src/foundation_model/workflows/_engine_test.py \
  src/foundation_model/workflows/predict_test.py
```

Tests cover pressure/header/unit validation, duplicate temperatures, missing/nonfinite inputs,
held-out isolation, zero/negative values, inverse extrapolation, scaler serialization and the
existing prediction/evaluation paths reporting both targets and predictions in original units.

## Training campaign

`campaign.py` implements the agreed three routes. The base catalog is the original 24-task
non-AGIS catalog in `base_pretrained.toml`. The dated selection file records a uniform random
sample of ten checkpoints without replacement (sampling seed `20261001`) and their hashes.
The current package loads all ten selected checkpoint states without changing any weights.

```bash
uv run python experiments/agis_transfer/campaign.py plan \
  --base-config experiments/agis_transfer/base_pretrained.toml \
  --output data/agis_campaign_20261001
```

The plan writes an immutable manifest and four grids of unit indices:

The manifest hashes the preprocessing manifest, audit curves, fold task fragments, parquet
files, fitted scalers, checkpoint selection and original replay data. Every unit checks those
hashes before running or reusing outputs. Completion markers are bound to the exact campaign
manifest; a changed manifest or unbound existing output requires a new output root.
The plan also fingerprints the package source, campaign files and dependency specification.
Units and their fit subprocesses reject source changes, and the Slurm worker requires a clean
tracked checkout. The per-job Git commit is recorded alongside this content fingerprint.
The first worker atomically establishes a shared campaign runtime identity containing the
container image hash, Python/architecture/CUDA and resolved package versions. New units,
completion reuse and fit subprocesses must match that identity; changing runtime requires a
new output root. This prevents later submissions from silently mixing container environments.

| Grid | Units | Final models |
|---|---:|---:|
| `direct.txt` | 8 folds × 10 checkpoints × 3 pressures = 240 | 480, frozen/unfrozen |
| `warm_first.txt` | 8 × first 5 checkpoints × 3 targets = 120 | 240, frozen/unfrozen |
| `warm_rest.txt` | 8 × remaining 5 checkpoints × 3 targets = 120 | 240, frozen/unfrozen |
| `scratch.txt` | 8 folds × 3 pressures = 24 | 24, no checkpoint repeats |

The first wave totals **744 final models**. Completing the optional remaining warm-start
checkpoints brings it to **984**. A warm unit continues from its selected checkpoint by adding
the other two pressure tasks in ascending pressure order, then shares that intermediate model
between frozen/unfrozen target fits. Its target pressure head must be absent during continuation.

Final fits use **1000 fixed epochs**, a full batch of **7 training compounds**, encoder LR
`0.002`, KR LR `0.0005`, and the training-loss plateau scheduler (patience 20, minimum LR
`1e-5`). Neither early stopping nor validation checkpoint selection uses an AGIS holdout.
The target head is initialized identically for each fold/pressure across every route and
checkpoint, including scratch; the initial state is saved. Scratch starts with a fresh encoder
and autoencoder, then uses the same final fitting path, with no original task heads or data.
The autoencoder follows the package's existing fine-tuning policy and remains trainable.
Frozen final encoders, including BatchNorm buffers, are checked against their initial states.

Warm stages retain the original replay recipe: maximum 150 epochs per appended task,
epoch-wise replay, 30% of original task labels with the existing per-task 1500-label floors.
The previously appended AGIS pressure replays all seven training curves. Early stopping
monitors only validation from original non-AGIS tasks (patience 24); no held-out AGIS label
enters validation. All AGIS pressure datasets share the same heldout composition split.
La₃Ni₂O₇ has a Tc label in the original pretraining corpus: this experiment holds out AGIS
resistivity labels, and does not claim every test composition was unseen during pretraining.

`array.sbatch` runs independent units on one allocated GPU, each with its own output directory,
logs and completion markers. Set `PROJ`, `IMAGE`, `MANIFEST`, `GRID`, `OUTROOT` and `PACK` when
submitting; supply the project account outside the repository. It checks package version 0.4.1,
the mounted source path and single-GPU visibility before training. Calibrate packing using
identical completed units in separate roots (`CALIBRATE=1`), and read utilization from `sacct`.
The worker requires `sbatch --array`: for `N` grid entries and pack size `P`, submit array indices
`0` through `ceil(N/P)-1`, with a concurrency limit suited to the measured workload.
Submit the remaining five warm-start checkpoints only when measured throughput leaves time
for completion and reporting. No unbounded resubmission is performed.

Each final output must have 300 finite predictions on the shared grid, one heldout composition,
the full epoch budget and a checkpoint before receiving its completion marker. Partial units
are resumable; successful final fits are verified again before a unit is marked complete.

```bash
uv run pytest experiments/agis_transfer/campaign_test.py \
  src/foundation_model/workflows/_engine_test.py
```
