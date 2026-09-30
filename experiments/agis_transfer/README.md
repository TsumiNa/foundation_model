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

## Verification

```bash
uv run pytest data/scripts/process_agis_data_test.py \
  src/foundation_model/workflows/_engine_test.py \
  src/foundation_model/workflows/predict_test.py
```

Tests cover pressure/header/unit validation, duplicate temperatures, missing/nonfinite inputs,
held-out isolation, zero/negative values, inverse extrapolation, scaler serialization and the
existing prediction/evaluation paths reporting both targets and predictions in original units.
