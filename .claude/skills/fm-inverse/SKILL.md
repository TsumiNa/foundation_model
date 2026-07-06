---
name: fm-inverse
description: Configure and run fm inverse — user-specified multi-target inverse design with latent/composition optimisation paths, seed strategies and trajectory recording. Use when asked to design compositions toward property targets or analyse inverse-design results.
---

# Inverse design with `fm inverse`

`fm inverse --config inv.toml --checkpoint <final_model.pt> [--output-dir D] [--steps N]
[--scenario NAME]`. Reference config: `experiments/rikyu_task_scaling/make_configs.py`
(`inv.toml`). The objective is fully user-specified — there is no built-in target.

## Targets (per scenario, any mix)

```toml
[[inverse.scenarios]]
name = "fe_down_total_up"
[[inverse.scenarios.targets]]
task = "formation_energy"
value = -1.0          # regression: value OR direction = "high"/"low" (unbounded — weight is the control)
[[inverse.scenarios.targets]]
task = "dielectric_total"
value = 1.0
weight = 2.0          # >1 prioritises this term; too high starves the others
# kernel_regression: points = [[t, y], ...]   classification: classes = [..] (+ direction)
```

**Units are z-scored**: `y_z = (y − μ_task) / σ_task` with μ, σ from the task's TRAINING split —
so `value = ±1.0` means one standard deviation from the training mean. Say this explicitly in
reports. Separate scenarios per target pair beat one joint many-target scenario when the user
wants per-target difficulty visible.

## Seeds

```toml
[inverse.seeds]
strategy = "weighted_random"   # or top_objective / random / explicit
weight_task = "dielectric_total"
# weight_direction = "high" (default) | "low", or weight_value = <v> (nearest-to-v); mutually exclusive
n = 20
split = "test"
dedup_by_element_system = true
```

- `weighted_random` samples by rank of the TRUE label (p_i = rank(s_i)/Σrank) — chemically
  diverse AND **model-independent**: every run gets the identical seed set, which is what makes
  candidate ensembles comparable across checkpoints/modes/orders. Prefer it for experiments.
- `top_objective` (model-ranked) clusters seeds into one chemical family (e.g. 18/20 oxide
  perovskites) — fine for "best shot" design, wrong for comparisons.

## Paths

- `latent` — latent-space optimisation + AE alignment, KMD-decoded composition. Strong at
  refining near-optimal starts; weak across long distances.
- `composition` — direct simplex optimisation; `max_elements` caps the recipe;
  `diversity_scale`: 1 = no entropy penalty, 0 = strongest. **0.2 collapses candidates to binary
  systems; 0.5 is a good default** for ≤4-element recipes. Better than latent from diverse seeds.

Batch runs: `record_trajectory = true`, `animation_formats = []` (build the interactive viewer
later — see the results-viewer-html skill).

## Outputs & analysis

`seeds.json` (order = candidate index in all arrays), per scenario `summary.json` /
`results.json` / `trajectories/<path>.npz` (`targets (S,B,T)`, `weights (S,B,94)`, `labels`),
`inverse_design.json` marker (use for skip-if-done). Analysis stack under
`experiments/rikyu_task_scaling/analysis/`: `collect.py` (inverse.csv with scenario column),
`plot_inverse.py`, `plot_diversity.py`, `candidate_regularity.py` (per-seed L1 fingerprint
analysis — requires the model-independent seed design).

## Interpretation guide (validated findings)

- The objective score is **model-self-scored** — comparable across runs of the same config, not
  an absolute quality; report achieved per-target channels alongside it.
- Design quality was independent of pretraining task count; path choice and seed strategy
  dominate. Candidate compositions carried no checkpoint/mode/order fingerprint.
- Watch for physically meaningful target conflicts (e.g. FE↓ × dielectric_electronic↑ was
  persistently hard even though electronic had the best prediction R²).
