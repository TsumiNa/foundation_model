# Inverse-design code map + extension notes

Snapshot of the inverse-design surface as of PR #18, written so a future session
can extend it (e.g. "exactly K elements", "fix Au at 65 %", "min-weight floor")
without having to reverse-engineer the design. See
[inverse_design_algorithms.md](inverse_design_algorithms.md) for the *math*; this
doc is the *code map*.

## The two entry points

| Method | Where | Optimisation variable | Method-specific loss term |
|---|---|---|---|
| `optimize_latent` | [flexible_multi_task_model.py:1735](../src/foundation_model/models/flexible_multi_task_model.py#L1735) | latent `h` | `α · ‖h − tanh(E(D(h)))‖²` (AE-alignment) |
| `optimize_composition` | [flexible_multi_task_model.py:2227](../src/foundation_model/models/flexible_multi_task_model.py#L2227) | element-weight logits `θ`, with `w = softmax(θ)` | `(1 − d) · H(w)` (entropy penalty) |

Both share: regression-MSE + classification-cross-entropy backbone, opt-in
`record_*_trajectory` flag for per-step capture (used by
[paper_inverse_trajectory.py](../src/foundation_model/scripts/paper_inverse_trajectory.py)).

## What's already there on the composition path

User-facing kwargs (validated in `optimize_composition`'s argument block at lines
~2393–2465):

| Kwarg | Range | What it does | Implementation |
|---|---|---|---|
| `task_targets` | `{task: value}` | MSE target per regression head | inner loop |
| `class_targets` + `class_target_weight` | `{task: class_idx}`, `> 0` | maximise softmax prob of given class | inner loop |
| `diversity_scale` | `[0, 1]`, default 1.0 | 0 = peaky few-element; 1 = no penalty | `(1 − d) · H(w)` added to loss |
| `seed_blend` | `[0, 1]`, default 0.95 | how much seed kept vs uniform-over-allowed at init | `w₀ ← s·seed + (1−s)·uniform` |
| `allowed_elements` | `"all"` or symbol list | hard whitelist | logit mask to `-inf` |
| `element_step_scale` | `float` or `{symbol: float}` | soft per-element gradient scale; `0` = hard-lock to seed value | grad multiplied per element; **lock implemented in `_w_from_logits`** (line 2576) — paste seed values back over softmax + renormalise unlocked positions |

## The single point of leverage: `_w_from_logits` inside `optimize_composition`

[flexible_multi_task_model.py:2576-2591](../src/foundation_model/models/flexible_multi_task_model.py#L2576-L2591)

```python
def _w_from_logits(lg: torch.Tensor) -> torch.Tensor:
    """Softmax over logits; mask disallowed elements; hard-lock the chosen ones at seed."""
    w = softmax_with_mask(lg, elem_mask)              # whitelist
    if locked_mask is None:
        return w
    # rewrite locked positions to seed values + renormalise unlocked positions to fill 1 − Σ_locked
    ...
```

**Every simplex-projection / hardening rule belongs here.** It runs once per step,
on every (B × n_components) row, and the gradient flows correctly through any
differentiable rewriting (the existing lock branch is `.detach()`-constant, so
its gradient is 0; new differentiable steps would let gradient flow naturally).
Adding a new constraint = (a) accept a new kwarg in the signature, (b) validate
it in the arg-block, (c) compute any per-step state once before the loop, (d)
apply it inside `_w_from_logits`.

## Three extensions the user has flagged

### A. "Specify number of elements" — top-K mass constraint

**Use case**: "give me exactly 3-element recipes" / "at most K elements".

**Suggested API**:
```python
optimize_composition(..., max_elements: int | None = None)
```

**Implementation sketch** (inside `_w_from_logits`):
```python
if max_elements is not None and max_elements < n_components:
    # Top-K hardening: keep the K largest weights per row, zero the rest, renormalise.
    topk_vals, topk_idx = w.topk(max_elements, dim=-1)
    mask = torch.zeros_like(w).scatter_(-1, topk_idx, 1.0)
    w = w * mask
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
```

Notes:
- `topk` returns the K largest indices — this is non-differentiable at the "K-th
  vs (K+1)-th" boundary, but the gradient through the K kept values is correct.
  In practice, with `diversity_scale < 1` to drive peakiness *before* the hard
  cutoff, the boundary doesn't oscillate.
- Validate `1 ≤ max_elements ≤ n_components` in the arg-block.
- Tests to add: pattern after
  [test_optimize_composition_element_step_scale_locks_symbols](../src/foundation_model/models/flexible_multi_task_model_test.py)
  — assert that `(w > 1e-6).sum(dim=-1) <= max_elements` for every row of the
  output.

### B. "Fix Au at exactly 65 %" — explicit fixed-amount API

**Use case**: chemistry-driven prior says "I want exactly 65 % Au, 20 % Ga,
optimiser picks the remaining 15 % freely".

**Already half-possible** via `element_step_scale = {"Au": 0.0, "Ga": 0.0}` +
seed has those amounts. But it requires constructing a seed; cleaner standalone
API:

```python
optimize_composition(..., fixed_amounts: Mapping[str, float] | None = None)
```

**Implementation sketch**:
- Validate that `sum(fixed_amounts.values()) < 1.0` (need free mass) and each
  symbol resolves in `DEFAULT_ELEMENTS`.
- Compute `fixed_w0: (n_components,)` with those positions set, zeros elsewhere.
- Reuse the existing `locked_mask` / `locked_w0` infrastructure
  ([line 2557-2574](../src/foundation_model/models/flexible_multi_task_model.py#L2557-L2574))
  — basically: set `locked_mask = (fixed_w0 > 0)` and `locked_w0 = fixed_w0` for
  every row in the batch, skip the "needs `initial_weights`" requirement that
  the `element_step_scale=0` branch has.
- The existing `_w_from_logits` already does the right paste + renormalise; no
  change needed there.

Tests: assert `w[:, fixed_idx] ≈ fixed_amount` exactly after every step.

### C. Min-weight floor / "if you use Au, use ≥ 10 %"

**Use case**: avoid trace-amount appearances (`Pt = 0.5 %`) that are not
synthesisable.

**Suggested API**: `min_nonzero_weight: float = 0.0`. After top-K (B) or simplex
projection, zero out any weight below the floor, renormalise.

Implementation goes in the same `_w_from_logits` block, after any top-K /
locking. Same test pattern.

## What lives where (for the future agent)

| Concern | Location |
|---|---|
| Method docstring (the user-facing contract) | `optimize_composition` docstring, lines 2243–2370 |
| Kwarg validation | arg-block lines 2393–2465 (mirror the pattern: per-kwarg validation block + a `*_arg` local prepared for the inner loop) |
| One-time setup (locked indices, scaled steps, …) | lines 2536–2574 (before the `for _ in range(steps)` loop) |
| Per-step constraint application | `_w_from_logits` (line 2576) — single point |
| Loss term additions (entropy etc.) | inner loop, line ~2614 (`if diversity_scale < 1.0:`) |
| Per-step trajectory recording | already wired via `record_weights_trajectory` (line 2603 + 2622) — new constraints automatically reflected in the trajectory because we record post-`_w_from_logits` weights |
| Tests | [flexible_multi_task_model_test.py](../src/foundation_model/models/flexible_multi_task_model_test.py) — search `test_optimize_composition_*` (38 existing tests cover the current surface) |

The latent path (`optimize_latent`, line 1735) is more rigid: the optimisation
variable is `h`, not a simplex, so the same constraints don't translate
naturally. Most new constraint features will only make sense for
`optimize_composition` — call this out in any new kwarg's docstring.

## Pre-merge checklist

When extending: keep the surgical-edits pattern. The existing PR already added a
lot; future extensions should be one-kwarg-per-PR with the validation +
`_w_from_logits` change + at least one test that pins the contract end-to-end
(input kwarg → output `w` rows satisfy the constraint).

Reference tests to mimic:
- `test_optimize_composition_element_step_scale_locks_symbols` —
  contract test for an existing constraint kwarg.
- `test_optimize_composition_runs_and_returns_simplex_weights` — smoke test
  that the simplex is preserved (rows sum to 1, all ≥ 0).
