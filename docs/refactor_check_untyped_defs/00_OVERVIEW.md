# Enabling `check_untyped_defs` — overview

**Status:** ratchet in place; backlog open.
**Owner PR:** the one adding this document.

## Why

`mypy` does not look inside a function that has no annotations. Every `def f(self, batch, batch_idx):`
in this repository is therefore **completely unchecked**, and those are the functions that carry
the training loop.

This is not hypothetical. In PR #45, `training_step` had:

```python
train_logs: dict[str, torch.Tensor] = {}
...
train_logs[f"train_{name}_all_missing"] = 1.0   # a float, not a Tensor
```

`uv run mypy` reported the file clean. Adding `--check-untyped-defs` reported it immediately. The
same audit found an `opt.step()` the branch's own log message said it would not take, in the same
unchecked function. Annotating those two step methods is what let the type checker see them at all.

The goal is that no function in `src/` is invisible to the type checker.

## What is actually in the way

Measured on `5455da0` with `uv run mypy --check-untyped-defs src/foundation_model/`:

**32 errors in 7 files.** Only **6 are in production code**, and none of the six is a runtime bug —
they are places where a declared type does not describe the real data:

| File | n | What |
|---|---:|---|
| `data/composition_sources_test.py` | 9 | pandas `.loc` indexing with `str \| None` keys; an unannotated `calls` list |
| `models/flexible_multi_task_model_test.py` | 13 | mock/fixture looseness — assigning to methods, `MagicMock` passed where tensors are declared |
| `data/datamodule.py` | 4 | `batched_y_dict` is annotated `list[Any]` but genuinely holds `Tensor \| list[Tensor]`; `on_train_epoch_start` guards with `getattr(..., None) is not None`, which mypy cannot narrow through |
| `models/flexible_multi_task_model.py` | 2 | `encoder_config` declared `BaseEncoderConfig` where only the two concrete subclasses are valid; `head.compute_loss` unresolvable because `ModuleDict.__getitem__` returns `Module` |
| `workflows/task_catalog_test.py` | 2 | fixture typing |
| `data/datamodule_test.py`, `data/dataset_test.py` | 2 | fixture typing |

The production errors are worth fixing on their own merits: `batched_y_dict: list[Any]` actively
tells a reader that every value is a list when half of them are stacked tensors, and that is the
dictionary the collate function hands to every training step.

## Decision

**Turn the flag on globally now and quarantine the seven known-failing modules**, rather than fix
32 errors in one unreviewable PR or leave the flag off until someone finds time.

```toml
[tool.mypy]
check_untyped_defs = true

[[tool.mypy.overrides]]
module = [ ... the seven ... ]
check_untyped_defs = false
```

The ratchet is the point: from the moment this lands, **a new untyped-body error cannot enter any
module that is already clean** — which is 48 of the 55 files. The backlog is the override list, it
only shrinks, and `pyproject.toml` shows exactly how much is left.

### Alternatives rejected

- **One big PR fixing all 32.** Mixes production type corrections with test-fixture churn across
  seven files; unreviewable, and the two categories deserve different scrutiny.
- **Leave it off, fix opportunistically.** This is the status quo that let PR #45's defects sit in
  an unchecked function. Without the flag on, nothing stops the next one.
- **`--check-untyped-defs` only in CI, not in `pyproject.toml`.** Then `uv run mypy` locally and CI
  disagree, and contributors get failures they cannot reproduce with the documented command.

## Non-goals

- Annotating every function signature. The flag checks *bodies*; full signature coverage
  (`disallow_untyped_defs`) is a separate, much larger decision.
- `strict = true`, `disallow_any_*`, or any other tightening.
- Changing runtime behaviour. Every PR in this sequence must be behaviour-preserving; where a type
  fix would change behaviour, that is a bug fix and belongs in its own PR with its own tests.

## Ordered PRs

Production first: those errors describe real data, and the modules are the ones the training loop
runs through. Test files after, one per PR, so fixture churn never lands in the same diff as a
production type correction.

| # | Plan | Removes from the override list |
|---|---|---|
| 1 | [`01_pr1_datamodule.md`](01_pr1_datamodule.md) | `data.datamodule` |
| 2 | [`02_pr2_flexible_multi_task_model.md`](02_pr2_flexible_multi_task_model.md) | `models.flexible_multi_task_model` |
| 3 | [`03_pr3_test_fixtures.md`](03_pr3_test_fixtures.md) | the five `*_test` modules |
| 4 | [`04_pr4_remove_the_override.md`](04_pr4_remove_the_override.md) | the override block itself |

Each PR's acceptance is the same shape: remove its entries from the override list, and
`uv run mypy src/` passes with no new `# type: ignore`.
