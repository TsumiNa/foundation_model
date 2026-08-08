# Agent Instructions

## Instruction Loading

- `AGENTS.md` is the canonical agent entry point.
- Before acting, inspect `.github/instructions/*.instructions.md` and use the table below to select files
  whose stated purpose matches the task.
- For selected files, use frontmatter `applyTo` only to narrow their file scope. An `applyTo: "**"` value
  does not make a purpose-specific instruction relevant to every task.
- A selected instruction without `applyTo` applies repository-wide.
- Follow all applicable supplemental instructions alongside this file.

| Instruction | Use for |
|---|---|
| `branch-and-pr-workflow.instructions.md` | Any repository modification |
| `implementation-and-tests.instructions.md` | Features, fixes, refactors, and generated code |
| `repository-doc-boundaries.instructions.md` | `README.md`, `ARCHITECTURE.md`, `AGENTS.md`, contributor docs |
| `shell-environment.instructions.md` | Terminal use; read before the first command |
| `rikyu-supercomputer.instructions.md` | RIKYU jobs; preferred compute platform |
| `riken-rccs-supercomputer.instructions.md` | Experimental R-CCS Cloud jobs |
| `ism-gpu-a100-training.instructions.md` | Training on the remote A100 host |

## Project

- Multi-task PyTorch Lightning model for material/polymer property prediction.
- Source: `src/foundation_model/`; tests are colocated as `<module>_test.py`.
- CLI: `fm pretrain`, `fm finetune`, `fm inverse`, `fm predict`.
- Architecture: `ARCHITECTURE.md`; usage: `README.md`; config schema: `docs/configuration.md`.
- Sample TOML configs: `samples/*.toml`; experiment notebooks: `notebooks/`.

## Package Manager

- Use **uv** with Python 3.11–3.13.
- Install: `uv sync --frozen --all-groups`
- Add dependency: `uv add <pkg>` or `uv add --dev <pkg>`

## File-Scoped Commands

| Task | Command |
|---|---|
| Format | `uv run ruff format path/to/file.py` |
| Lint | `uv run ruff check path/to/file.py` |
| Typecheck | `uv run mypy path/to/file.py` |
| Test file | `uv run pytest path/to/module_test.py` |
| Test case | `uv run pytest path/to/module_test.py::test_name` |

## Key Conventions

- Use TOML configs normalized into `@dataclass` objects; validate in `__post_init__`.
- Use `str`-based enums for closed choices and explicit type hints on public APIs.
- Keep task/encoder config dataclasses and shared enums in `models/model_config.py`.
- Use `loguru`; do not add another logging framework.
- LoRA support is removed; legacy `lora_*` keys are invalid.
- Preserve the composition-keyed data model: each task owns its data files and missing targets are masked.
- Keep finalized logic in `src/`; notebooks must remain non-critical.
- Keep documentation synchronized when entry points, config fields, or data conventions change.

## Testing

- Add or update the colocated `<source>_test.py` for implementation changes.
- Cover the primary path and likely failures: missing/malformed input, boundaries, NaNs/masks, shape errors,
  and enforced preconditions.
- Run targeted tests first; run the full suite when the change warrants it.

## Commits and Pull Requests

- Commit messages: `<type>: <imperative summary>`, preferably under 72 characters.
- PRs state scope, motivation, validation, and backward-incompatible behavior.
- Follow `.github/instructions/branch-and-pr-workflow.instructions.md` before editing.

## Commit Attribution

- AI commits MUST include its own attribution:

```text
Co-Authored-By: <agent model name> <agent attribution email>
```
