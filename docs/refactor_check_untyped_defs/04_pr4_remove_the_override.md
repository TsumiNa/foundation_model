# PR4 — remove the override block, and enforce mypy per commit

## Goal

Delete the `check_untyped_defs = false` override from `pyproject.toml`, and make the type check
run as a pre-commit hook so the ratchet is enforced before code reaches CI rather than after.

## Scope

By this point the override list is empty. This PR:

- removes the `[[tool.mypy.overrides]]` block that carried the backlog;
- adds a `mypy` hook to `.pre-commit-config.yaml`, scoped to `src/` and using the project
  environment so it sees the same settings as `uv run mypy`;
- updates `AGENTS.md`, which currently states that the commit hook *"intentionally does not run
  Mypy"* — that sentence is the thing being changed, and leaving it would make the repository
  contradict itself;
- deletes `docs/refactor_check_untyped_defs/`, following the precedent of `docs/refactor_fm_cli/`,
  which was removed once its sequence merged.

Hook design matters here. `mypy` is not a per-file linter: checking only the staged files gives
different answers than checking the package, because inference crosses module boundaries. The hook
therefore runs once over `src/` with `pass_filenames: false`, triggered by any staged `.py` change.
It is slower than `ruff format`, which is the reason the original decision went the other way — the
ratchet is what changes the trade, since a violation now blocks the whole repository rather than
one file.

## Non-goals

- **Tightening beyond `check_untyped_defs`.** No `strict`, no `disallow_untyped_defs`, no
  `disallow_any_*`. Signature coverage is a separate decision with a much larger diff.
- **Changing runtime behaviour.** This PR touches configuration and documentation only.
- **Adding hooks for other tools** (pytest, ruff check). One hook, one purpose; a slow test hook is
  a different trade-off and deserves its own discussion.
- **Removing the `joblib` `ignore_missing_imports` override**, which is unrelated to this sequence.

## Acceptance

```bash
uv run mypy src/                       # clean, with no check_untyped_defs override anywhere
uv run pre-commit run mypy --all-files # passes
uv run pytest src/ -q                  # unchanged
```

`grep -rn "check_untyped_defs" pyproject.toml` returns only the single global `true`, and
`grep -n "Mypy" AGENTS.md` no longer claims the hook skips it.
