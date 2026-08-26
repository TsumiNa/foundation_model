# PR3 — test fixtures under `check_untyped_defs`

## Goal

Make the five remaining `*_test` modules type-check and remove them from the override list.

## Scope

26 errors, all fixture/mock looseness rather than defects:

| File | n | Shape of the problem |
|---|---:|---|
| `models/flexible_multi_task_model_test.py` | 13 | `MagicMock` passed where a `Tensor` is declared; assignment to bound methods (`model.training_step = ...`) |
| `data/composition_sources_test.py` | 9 | pandas `.loc` indexed with `str \| None`; an unannotated `calls` list |
| `workflows/task_catalog_test.py` | 2 | fixture return types |
| `data/datamodule_test.py`, `data/dataset_test.py` | 2 | fixture return types |

Approach: annotate fixtures and local collections, and narrow the `str | None` values that reach
pandas indexers with an assertion that documents why they cannot be `None` at that point. Where a
test deliberately monkey-patches a method, use `mocker.patch.object` (already used elsewhere in
these files) instead of direct assignment.

Split further if the diff grows past comfortable review — one PR per file is acceptable, and the
override list makes partial progress visible.

## Non-goals

- Changing what any test asserts. A test whose assertions change is not a typing fix.

## Acceptance

```bash
uv run mypy src/                # clean with only the override block left
uv run pytest src/ -q           # same count, same passes
```
