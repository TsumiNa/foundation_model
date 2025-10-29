# Repository Guidelines

## Project Structure & Module Organization
- Core source lives under `src/foundation_model`, with task heads in `src/foundation_model/models/task_head` and shared layers in `src/foundation_model/models/components`.
- Integration and regression tests sit at the repo root as `test_*.py`; keep new suite files alongside existing ones for quick discovery (e.g., `test_prediction_writer_kernel_regression.py`).
- Reference notebooks reside in `notebooks/` for experimentation; mirror finalized logic into `src/` modules and keep notebooks informative but non-critical.
- Sample Hydra/Lightning configs and data fixtures are in `samples/` and `data/`; avoid committing large datasets beyond lightweight fixtures.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create a local virtual environment (recommended for contributors).
- `pip install -e ".[dev]"`: install the package and dev extras defined in `pyproject.toml`.
- `pytest`: run the full test suite; respects `pyproject` options (`--maxfail=2`, benchmark disabled).
- `pytest test_integration_kernel_regression.py`: quick smoke check covering multi-task model wiring.
- `fm-trainer --config <config.yaml>`: launch the primary training CLI (entry point declared under `[project.scripts]`).

## Coding Style & Naming Conventions
- Follow Ruff formatting (`ruff format`) and lint checks (`ruff check`); both enforce 4-space indentation, 120-character lines, and double quotes.
- Prefer explicit type hints on public APIs; mirror patterns used in `src/foundation_model/models/task_head`.
- Module and file names stay snake_case; classes use CapWords (`KernelRegressionHead`), functions snake_case, constants UPPER_SNAKE.

## Testing Guidelines
- Tests rely on `pytest` with benchmark plugins; name files `test_*.py` and functions `test_*`.
- Include targeted unit tests near the feature (`test_kernel_regression_refactor.py`) and integration flows when touching the trainer or data pathways.
- Run `pytest` before submitting; for long-running suites, at least execute the affected module tests and document any skips.

## Commit & Pull Request Guidelines
- Commit messages generally follow `<type>: <summary>` (see `git log`, e.g., `refactor(task-head): use raw t input in kernel regression`); keep summaries imperative and under 72 characters when possible.
- Squash fixups locally; each PR should describe scope, motivation, and validation (tests run, configs modified). Link issues or tasks in the description.
- Provide screenshots or logs when changes impact CLI output or training metrics, and flag any backward-incompatible behavior for reviewer attention.
