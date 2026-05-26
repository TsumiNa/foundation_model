# Inverse-design code map + extension notes

Code map for `optimize_composition`'s constraint surface. See
[inverse_design_algorithms.md](inverse_design_algorithms.md) for the *math*; this
doc is the *code map* and is kept in sync with the implementation.

## The two entry points

| Method | Where | Optimisation variable | Method-specific loss term |
|---|---|---|---|
| `optimize_latent` | [flexible_multi_task_model.py:1735](../src/foundation_model/models/flexible_multi_task_model.py#L1735) | latent `h` | `α · ‖h − tanh(E(D(h)))‖²` (AE-alignment) |
| `optimize_composition` | [flexible_multi_task_model.py:2227](../src/foundation_model/models/flexible_multi_task_model.py#L2227) | element-weight logits `θ`, with `w = softmax(θ)` | `(1 − d) · H(w)` (entropy penalty) |

Both share: regression-MSE + classification-cross-entropy backbone, opt-in
`record_*_trajectory` flag for per-step capture (used by
[paper_inverse_trajectory.py](../src/foundation_model/scripts/paper_inverse_trajectory.py)).

## User-facing kwargs (composition path)

Validated in `optimize_composition`'s argument block; the whole surface composes
orthogonally — any subset of A/B/C plus the existing knobs can be used together.

### Existing knobs (PR #18)

| Kwarg | Range | What it does | Implementation |
|---|---|---|---|
| `task_targets` | `{task: value}` | MSE target per regression head | inner loop |
| `class_targets` + `class_target_weight` | `{task: class_idx}`, `> 0` | maximise softmax prob of given class | inner loop |
| `diversity_scale` | `[0, 1]`, default 1.0 | 0 = peaky few-element; 1 = no penalty | `(1 − d) · H(w)` added to loss |
| `seed_blend` | `[0, 1]`, default 0.95 | how much seed kept vs uniform-over-allowed at init | `w₀ ← s·seed + (1−s)·uniform` |
| `allowed_elements` | `"all"` or symbol list | hard whitelist | logit mask to `-inf` |
| `element_step_scale` | `float` or `{symbol: float}` | soft per-element gradient scale; `0` = hard-lock to seed value | grad multiplied per element; lock implemented in `_w_from_logits` — paste seed values back over softmax + renormalise unlocked positions |

### Constraint knobs added in this PR (A / B / C)

| Kwarg | Range | What it does | Implementation |
|---|---|---|---|
| `max_elements` (A) | `int` ∈ `[1, n_components]` or `None` | "at most K non-zero elements" cardinality cap | differentiable Plötz–Roth iterative soft top-K mask multiplies `softmax(lg)` inside `_w_from_logits`; final hard top-K projection at the very end so the returned recipe is exactly K-hot (subject to floor C dropping below-floor positions further down) |
| `annealing_scale` (A) | `[0, 1]`, default 0.5 | single-knob "softness" of the K-hot annealing schedule; maps to `τ_start = 25**scale` (0→1, 0.5→5, 1→25) | drives the τ schedule for the soft top-K mask; default schedule is geometric from `25**scale` down to `τ_end = 0.01` |
| `annealing_schedule` (A) | `dict` or `None` | advanced piecewise override — `{"step": [...], "scale": [...], "annealing_func": [...]}` with per-segment normalised scales and interpolation funcs (`geometric`/`linear`/`cosine`/`constant`) | overrides the front of the simple schedule; if `step[-1] < 1.0`, the tail falls back to a geometric drop to `τ_end = 0.01` |
| `fixed_amounts` (B) | `{symbol: float}` or `None` | pin elements at user-specified absolute amounts (e.g. `{"Au": 0.65, "Ga": 0.20}`); does **not** require `initial_weights` | reuses the existing `locked_mask` / `locked_w0` lock-paste machinery; merged with `element_step_scale=0` locks (validated disjoint) |
| `min_nonzero_weight` (C) | `[0, 1]`, default 0.0 | drop unlocked positions with `0 < w < floor` and re-distribute the freed mass | applied at the very end of `_w_from_logits` (after lock-paste) and again after the final hard projection; locked positions are exempt; per-row fallback when the floor would empty unlocked mass — that row is left unfloored to keep the simplex valid |

## The pipeline inside `_w_from_logits`

```python
# Per-step pipeline (every Adam step) — see flexible_multi_task_model.py near the
# ``_w_from_logits`` definition for the live source.
lg = mask_disallowed(lg, allowed_elements)                # whitelist → -inf
w_soft = softmax(lg)                                       # natural simplex
if max_elements is not None:
    m = soft_topk_mask(lg, K=max_elements, τ=current_τ)    # Plötz-Roth, force-locked positions in
    w = (w_soft * m) / Σ                                    # K-hot weighted by softmax ratios
else:
    w = w_soft
w = apply_lock_paste(w)                                    # paste pinned values (B and step_scale=0)
w = apply_min_floor(w)                                     # zero unlocked below floor, renorm
```

After the optimisation loop, the final state additionally runs a **hard top-K
projection** + a re-paste + re-floor so the returned recipe is clean — at τ_end ≈ 0.01
the soft state is already near K-hot, so this just cleans residual sub-threshold mass.

## How the constraints compose

Designed to be orthogonal — any subset can be used together. The validation enforces
the few impossible combinations up-front so a bad mix raises before model state is
touched:

| Constraint pair | Validation | Behaviour |
|---|---|---|
| A × `allowed_elements` | `max_elements ≤ |allowed|` | only allowed positions can enter the K-hot mask |
| A × `element_step_scale=0` | `max_elements ≥ n_locked` | locked positions counted toward K; force-selected into the mask |
| A × B | `max_elements > n_locked_total` (strict — `fixed_amounts` has `Σ < 1`, leftover mass needs a free slot) | B locks count toward K |
| A × C | `min_nonzero_weight ≤ 1 / max_elements` | floor compatible with K-element simplex |
| B × C | `min(fixed_amounts.values()) ≥ floor` | floor cannot override a user pin |
| B × `element_step_scale=0` | disjoint symbol sets | one lock mechanism per element |
| C × `element_step_scale=0` | per-row locked seed values ≥ floor (runtime) | floor cannot drop a locked seed |

Edge case (C): if dropping every below-floor position would leave a row with zero
unlocked mass, the floor is **skipped for that row only** — preserving the simplex
invariant. The "at most K" promise still holds; some rows can land below K.

## Annealing schedule (A)

`annealing_scale ∈ [0, 1]` is the single-knob shortcut. Internally each scale value
maps to a raw temperature via `τ = 25**scale`:

| scale | τ_start (raw) | Calibration notes |
|---|---|---|
| 0.0 | 1.0 | minimal exploration — constraint nearly hard from step 0 |
| **0.5** | **5.0** | **default**; safe choice — QC stays within ±0.02 of unconstrained baseline across all three paper scenarios |
| 1.0 | 25.0 | max exploration — best for escaping local optima at the cost of slower QC refinement |

The full default schedule is **geometric** from `τ_start(scale)` to `τ_end = 0.01`. For
finer control, supply `annealing_schedule = {"step": [...], "scale": [...], "annealing_func": [...]}`
— see the kwarg docstring.

**Calibration source**: reproducible via [`logs/sweep_tau_schedule.py`](../logs/sweep_tau_schedule.py)
+ [`logs/plot_sweep.py`](../logs/plot_sweep.py) — the (scale × schedule × K) sweep on the
inverse-design fine-tuned model that placed the 0.5 default in the safe region. JSON / PNG
outputs are git-ignored; rerun the scripts to regenerate.

## Tests

All behaviour is contract-tested in
[`flexible_multi_task_model_test.py`](../src/foundation_model/models/flexible_multi_task_model_test.py).
Search patterns:

- `test_optimize_composition_max_elements_*` — A's contract (≤ K, annealing, K=n no-op,
  locked interaction, validation).
- `test_optimize_composition_fixed_amounts_*` — B's contract (exact pin, no-init mode,
  combined with A/C, validation).
- `test_optimize_composition_min_nonzero_weight_*` — C's contract (≥ floor, no-op at 0,
  fallback, validation against fixed_amounts / step_scale=0 locks).
- `test_optimize_composition_annealing_*` — annealing knob endpoints + dict override.

Reference contract: `test_optimize_composition_runs_and_returns_simplex_weights` (rows
sum to 1, non-negative — invariant across every combination of the constraints).

## End-to-end behavioural evidence (reproducible scripts)

All evaluation outputs are git-ignored; rerun the scripts below to regenerate them.

- [`logs/eval_abc_intuition.py`](../logs/eval_abc_intuition.py) +
  [`logs/plot_abc_intuition.py`](../logs/plot_abc_intuition.py) — 80+ runs on the
  inverse-design fine-tuned model across A/B/C and their combinations × two paper
  scenarios. Verifies every contract (≤K, exact pins, ≥ floor) and prints PASS/FAIL
  per intuition check. Original run (this PR): all 13 contract checks pass; 11/12
  monotone-intuition checks pass (the one "failure" is a legitimate multi-objective
  trade-off, not a bug — FE flat while klat improves with K under fixed-Au+Ga).
- [`logs/sweep_tau_schedule.py`](../logs/sweep_tau_schedule.py) — the calibration grid
  (τ_start × schedule × K × target-set) used to pick `annealing_scale = 0.5` as the
  safe default.
- [`logs/test_max_elements_smoke.py`](../logs/test_max_elements_smoke.py) — minimal
  smoke test confirming the byte-identical reproducibility of K=3/K=5 default
  (`annealing_scale=0.5` ≡ the previous `τ_start=5.0` calibration).
- [`logs/eval_combined_abc.py`](../logs/eval_combined_abc.py) +
  [`logs/plot_combined_abc.py`](../logs/plot_combined_abc.py) — the 9-config combined
  evaluation chart (baseline + each of A/B/C alone + every pair + full stack at two
  annealing settings).
- `paper_inverse_3scenarios` with `--output-dir
  artifacts/inverse_design_run/inverse_design_max_elements/` — the three paper
  scenarios rerun with the new A bars added (`paper_inverse_comparison.py` now threads
  `max_elements` / `annealing_scale` / `annealing_schedule` from each comp-config
  row); existing 5 comp + 3 latent bars are byte-identical to before.
