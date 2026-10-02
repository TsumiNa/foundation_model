# Reproducible transfer benchmark

## Goal

Complete a first test round measuring transfer performance and its dependence on source task
count for the approved encoder variants. Source training accuracy is a diagnostic only.

## Scope

- Preprocessing in `data/scripts` following the user's shared-data convention, with composition-level train/validation/test separation,
  training-only fitted scalers, date-suffixed Parquet data and provenance manifests.
- Eight encoder variants, three paired seeds, source checkpoints at 1/3/7 tasks, two held-out
  target tasks, two target-label budgets, and scratch/frozen/full fine-tuning comparisons.
- Matched latent width, heads and reconstruction decoder; parameter counts and compute recorded.
- Image-only RIKYU smoke, measured GPU packing, launch/collection scripts and result analysis.

## Non-goals

No JEPA objective, new production APIs, universal scaling-law claim, exhaustive hyperparameter
search, or training with a source checkout substituted for the image-installed package.

## Acceptance

- Begin implementation only after PR 01 is remotely confirmed merged and fetch its default branch.
- Preprocessing tests verify split isolation, train-only normalization and nested label budgets.
- Review and merge benchmark code before GPU submission; pin the published ARM image to its SHA.
- Allocated-node smoke passes pretrain, full/frozen fine-tuning, prediction and inverse.
- Measure GPU utilization and packing against unpacked reference runs before sizing the fleet.
- Reconcile all planned fits and locally mirror metrics/predictions. Analyze paired target errors,
  transfer relative to matched scratch baselines and dependence on source task count; identify
  failed or incomplete fits accurately.
