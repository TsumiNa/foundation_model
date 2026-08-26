# PR4 — remove the override block

## Goal

Delete the `check_untyped_defs = false` override from `pyproject.toml` and this plan directory.

## Scope

By this point the override list is empty. The PR removes:

- the `[[tool.mypy.overrides]]` block that carried the backlog,
- `docs/refactor_check_untyped_defs/`, following the precedent of `docs/refactor_fm_cli/`, which was
  deleted once its sequence merged.

## Acceptance

```bash
uv run mypy src/    # clean, with no check_untyped_defs override anywhere
```

`grep -r "check_untyped_defs" pyproject.toml` returns only the single global `true`.
