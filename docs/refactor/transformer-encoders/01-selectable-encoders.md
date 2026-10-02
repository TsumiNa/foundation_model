# Selectable feature-token encoders

## Goal

Enable controlled comparisons of MLP, legacy Transformer, feature-token Transformer and grouped
feature encoders, including CLS, mean and concat pooling, through the normal model workflows.

## Scope

- Validated encoder settings in `models/model_config.py`, exposed through `[model]` and
  `[model.transformer]` for every workflow.
- Legacy shared scalar tokenizer, independently learned scalar/group tokenizers, modern
  attention blocks and the no-attention group control, with an optional output projection.
- Explicit reconstruction hidden dimensions for matched encoder comparisons.
- Focused tests for configuration failures, gradients, pooling, initialization, state restoration,
  masks, training and inference/inverse compatibility; synchronized user/architecture docs.
- Bump the package version to 0.5.0 and publish fresh versioned images through the existing workflow.

## Non-goals

No JEPA objective, variable-support element tokenizer, new task heads, global data rewrite,
unrelated production cleanup, default architecture change, or benchmark runner implementation.

## Acceptance

- Targeted colocated tests pass, followed by the full source suite where practical.
- Touched Python files pass `uv run ruff format` / `uv run ruff check`.
- `uv run mypy src/` and `uv run python scripts/check_config_docs.py` pass, or independent
  pre-existing failures are identified with evidence.
- Old MLP configuration reconstructs the same architecture; modern encoders round-trip
  through checkpoints and support finite input/composition gradients.
- PR checks and all review surfaces are audited, actionable findings addressed, and the remote
  squash merge confirmed before beginning PR 02.
- The version and lockfile agree; the RIKYU image is published from the merge commit.
