# MLP transfer with 10–100 regression training labels

Extend the regression learning curves in the October 8 MLP status presentation to
10, 20 and 50 training labels. A new 100-label bridge is required because the
original three source checkpoints are inaccessible during RIKYU maintenance.

## Registered comparison

- Thirteen scalar regression targets from the September fixed-count study; no classification.
- Exact, nested 10 / 20 / 50 / 100 training labels per task, with three subset seeds 0 / 1 / 2.
- Scratch versus full encoder fine-tuning, new target head, 150 maximum epochs,
  validation early stopping patience 24, and the historical learning rates.
- The same subset, validation labels and test labels within each pair. Training label
  counts are checked **after** the package's keep-first composition canonicalization.
- Three locally cached source models: the first three models in the pre-existing
  uniform random draw `data/agis_pretrained_20261001/selection_20261001.json`.
  No target scores are used to select these models. Each seed uses one fixed source
  model at all four sizes. All 13 targets must be absent from its task sequence.
- 13 targets × 4 sizes × 3 seeds × 2 methods = **312 fits**.

Explicit inputs: September dataset
`data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet`, the three source checkpoints
and their selection manifest, and the historical catalog/recipes
`experiments/rikyu_hparam_tuning_v2/configs/probe6_mp2026_lowdata.toml` and
`experiments/rikyu_hparam_tuning_v2/configs/ft_lowdata_mp2026.toml`. Resolved recipes are
copied into this experiment's generated input manifest, together with hashes.
Source datasets and old results are never rewritten.

This is an independent replication, not a seamless extension of the old source
checkpoint cohort. Historical points at 100 and above remain identified separately.
The budget counts **training labels**: historical validation labels remain available
for early stopping, the archived target transformations are retained, and unlabeled
training compositions still participate in the production reconstruction objective.
It is therefore not an experiment with only ten labels available in total, nor a
composition-disjoint pretraining test. Report these limits alongside the curves.
Both methods now use the paired training seed 2025 + subset seed; the older fine-tuning
grid used seed 2025 for every source checkpoint. Scores use final workflow weights,
not a retrospectively selected best test checkpoint.

## Files and execution

- `scripts/prepare.py`: exact-count nested subsets, source exclusion/hash checks,
  resolved historical recipes, and deterministic case registry.
- `scripts/run.py`: two production package workflows per case, provenance, finite
  metrics and nonzero-training checks; completed fits are retained and skipped.
- `scripts/array.sbatch`: image-only GPU worker; independent cases may be packed.
- `analysis/collect.py`: complete paired fits only, per-seed errors and R², mean ± SD.

Prepare locally, then stage this folder's scripts and generated data to the chosen
GPU workspace. Use the official installed package 0.5.0 image; never bind `src` or
set `PYTHONPATH`. A GPU smoke and matched full-budget packing calibration precede
the fleet. Keep source/model weights remotely; synchronize scientific outputs.

## Execution log

- 2026-10-08: RIKYU reports scheduled maintenance through October 13. User approved
  locally cached source models and a 100-label bridge, then restricted the extension
  to regression tasks. Dataset/recipe preparation underway; no scientific results yet.
