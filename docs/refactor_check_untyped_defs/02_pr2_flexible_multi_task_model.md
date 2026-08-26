# PR2 — `models/flexible_multi_task_model.py` under `check_untyped_defs`

## Goal

Make the model module type-check with untyped bodies enabled and remove it from the override list.

This is the module the flag exists for: PR #45's defects — a float assigned into a
`dict[str, Tensor]`, and an `opt.step()` the branch's own log said it would skip — both lived in
functions mypy was not reading.

## Scope

Two errors:

| Line | Error | Cause |
|---|---|---|
| 301 | `Argument "encoder_config" to "FoundationEncoder" has incompatible type "BaseEncoderConfig"` | `self.encoder_config` is declared as the abstract base, but only `MLPEncoderConfig` and `TransformerEncoderConfig` are ever valid — `EncoderConfig` is already that union. |
| 962 | `"Tensor" not callable` on `head.compute_loss(...)` | `ModuleDict.__getitem__` returns `Module`, and `Module.__getattr__` is typed `Tensor \| Module`, so any method reached through `self.task_heads[name]` is unresolvable. |

Approach:

- Narrow the `encoder_config` attribute to `EncoderConfig` (the existing union). `build_encoder_config`
  already returns exactly that.
- For the head lookup, introduce one typed accessor — e.g. `_head(name) -> BaseTaskHead` — used by
  the shared pipeline instead of indexing `self.task_heads` directly. One cast in one place, rather
  than a `# type: ignore` at every call site, and it gives the heads a named protocol boundary.

## Non-goals

- Annotating every signature in the file (that is `disallow_untyped_defs`, out of scope).
- Any change to loss composition, masking or optimizer behaviour — this module was just rewritten
  in #45 and its three characterization tests must keep passing **unmodified**.

## Acceptance

```bash
uv run mypy src/foundation_model/models/flexible_multi_task_model.py   # clean, override removed
uv run pytest src/foundation_model/models/ -q                          # unchanged
```

The characterization tests (`test_step_output_is_stable`) must pass without edits — if they need
editing, the PR has changed behaviour and is out of scope.
