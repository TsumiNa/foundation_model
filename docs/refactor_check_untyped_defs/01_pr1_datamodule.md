# PR1 — `data/datamodule.py` under `check_untyped_defs`

## Goal

Make `foundation_model.data.datamodule` type-check with untyped bodies enabled, and remove it from
the override list in `pyproject.toml`.

## Scope

Four errors, none of them a runtime bug:

| Line | Error | Reality |
|---|---|---|
| 87, 88 | `Incompatible types in assignment (expression has type "Tensor", target has type "list[Any]")` | `batched_y_dict` / `batched_mask_dict` hold **`Tensor` for stacked tasks and `list[Tensor]` for kernel-regression tasks**. The annotation says every value is a list. |
| 538 | `Item "None" of "DistributedSampler \| None" has no attribute "set_epoch"` | Guarded by `getattr(self, "_train_sampler", None) is not None`; mypy cannot narrow through `getattr`. |
| 538 | `Item "None" of "Trainer \| None" has no attribute "current_epoch"` | Same, for `getattr(self, "trainer", None)`. |

Fix by describing the data, not by silencing:

- Widen the batched-dict annotations to `dict[str, torch.Tensor | list[torch.Tensor]]`. This is the
  dictionary the collate function hands to every `*_step`; the current annotation misleads anyone
  reading how kernel-regression batches differ from the rest.
- Replace the `getattr(...) is not None` guards with direct attribute tests so narrowing works.

## Non-goals

- Changing what the collate function produces.
- Touching `datamodule_test.py` (PR3).

## Acceptance

```bash
uv run mypy src/foundation_model/data/datamodule.py   # clean, with the override removed
uv run pytest src/foundation_model/data/ -q           # unchanged
```

No new `# type: ignore` in the diff.
