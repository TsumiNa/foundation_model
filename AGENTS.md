# Repository Guidelines

This is the canonical contributor/agent guide for this repository. Other agent files (e.g. `CLAUDE.md`) point here — keep this file authoritative and up to date.

## What this is

A multi-task PyTorch Lightning model (`FlexibleMultiTaskModel`) for predicting material/polymer properties from formula descriptors (and optional structure descriptors). One architecture serves both pre-training (with self-supervised auxiliary losses) and downstream fine-tuning.

## Project Structure & Module Organization
- Core source lives under `src/foundation_model`: task heads in `models/task_head/`, shared layers in `models/components/`, config dataclasses/enums in `models/model_config.py`, data loading in `data/`, the thin `fm` CLI in `cli/`, and the workflow engines (task catalog, run recorder, pretrain/finetune/inverse/predict) in `workflows/`.
- Tests are **co-located** with the code they cover as `<module>_test.py` (e.g. `datamodule.py` → `datamodule_test.py`, `workflows/pretrain.py` → `workflows/pretrain_test.py`). There is no separate `tests/` tree.
- Reference notebooks reside in `notebooks/` for experimentation; mirror finalized logic into `src/` modules and keep notebooks informative but non-critical.
- Sample Lightning/suite configs and data fixtures are in `samples/` and `data/`; avoid committing large datasets beyond lightweight fixtures.
- Detailed model/component design lives in `ARCHITECTURE.md`; user-facing usage lives in `README.md`.

## Build, Test, and Development Commands

Environment is managed with **uv** (Python 3.11–3.13).

```bash
uv sync --frozen --all-groups   # install runtime + dev deps from uv.lock
uv add <pkg>                    # add a runtime dep; uv add --dev <pkg> for dev
```

Lint, format, and type-check:

```bash
ruff format        # 4-space indent, 120-col, double quotes
ruff check         # Pyflakes + pycodestyle (see pyproject.toml for ignored rules)
mypy src           # type checking
```

Tests (pytest, with benchmark plugin):

```bash
pytest                                              # full suite (--maxfail=2, benchmarks skipped)
pytest src/foundation_model/data/datamodule_test.py # single test file
pytest path/to/file_test.py::test_name              # single test
```

Run `pytest` (or `uv run pytest`) before submitting; for long-running suites, at least execute the affected module tests and document any skips.

## Entry Points

A single console script `fm` (declared in `pyproject.toml [project.scripts]` →
`cli/main.py:main`), a [click](https://click.palletsprojects.com/) group with four subcommands:

- `fm pretrain --config <toml>` — continual-rehearsal pre-training (rehearsal-interval replay +
  optional `n_runs` sweep). Engine: `workflows/pretrain.py`.
- `fm finetune --config <toml> --checkpoint <ckpt>` — frozen-encoder fine-tuning of selected task
  heads (AE head stays trainable). Engine: `workflows/finetune.py`.
- `fm inverse  --config <toml> --checkpoint <ckpt>` — inverse design (scenarios × latent/composition
  algorithm paths). Engine: `workflows/inverse.py` (+ `workflows/inverse_trajectory.py`).
- `fm predict  --config <toml> --checkpoint <ckpt>` — evaluate / predict with an arbitrary
  checkpoint (`strict=False` load). Engine: `workflows/predict.py`.

Config conventions (no YAML / LightningCLI): every config is **TOML**, normalized into `@dataclass`
config objects by per-subcommand `build_*_config` builders (shared `[data]`/`[descriptor]`/
`[datasets.*]`/`[[tasks]]`/`[model]`/`[training]` sections + a per-subcommand section). Common CLI
flags on every subcommand: `--config` (required), `--output-dir`, `--set section.key=value`
(repeatable, value parsed with TOML semantics), `--seed`, `--accelerator`, `--sample`. Each run
writes `run_provenance.json` (resolved config + package versions + git + argv + seeds) and
`run.log` into its output dir via `workflows/recording.py::RunRecorder`. Sample configs live in
`samples/*.toml` (formal + `*_smoke.toml` variants); the full per-key schema (every section, type,
default, constraint) is [`docs/configuration.md`](docs/configuration.md) — keep it in sync when you
add or change a config field.

## Architecture

Full details in `ARCHITECTURE.md`. Key flow:

```
x_formula ──▶ FoundationEncoder ──▶ latent ──tanh──▶ h_task ──▶ task heads ──▶ {task_name: pred}
              (shared backbone)       (applied in model.forward)  (dict of heads)
```

- **FoundationEncoder** (`models/components/foundation_encoder.py`): a configurable shared backbone that produces a `latent` representation. `FlexibleMultiTaskModel.forward()` then applies `torch.tanh(latent)` to produce `h_task`, the single representation fed to **all** task heads. (The Tanh is applied at the model level, not inside the encoder; `ARCHITECTURE.md` still describes it as a separate `deposit` layer.) Backbone mode is chosen by `EncoderType` in the encoder config:
  - `MLPEncoderConfig` — feed-forward stack; `hidden_dims[0]` is the input dim, `hidden_dims[-1]` is `latent_dim`.
  - `TransformerEncoderConfig` — treats each scalar feature as a token; aggregates via learnable `[CLS]` or mean pooling (`use_cls_token`); `latent_dim == d_model`.
- **Task heads** (`models/task_head/`): each is an `nn.Module` registered in an `nn.ModuleDict`. Types are enumerated by `TaskType` and configured by matching dataclasses in `models/model_config.py`:
  - `REGRESSION` → `RegressionTaskConfig` / `regression.py`
  - `CLASSIFICATION` → `ClassificationTaskConfig` / `classification.py`
  - `KernelRegression` → `KernelRegressionTaskConfig` / `kernel_regression.py` — handles variable-length `(t, target)` sequences (e.g. DOS, temperature-dependent properties) via Gaussian-kernel mixtures and a Fourier/FC `t` encoder.
  - `AUTOENCODER` → `AutoEncoderTaskConfig` / `autoencoder.py`
- **Config is dataclass-based**, not pydantic. `model_config.py` defines all encoder/task/optimizer dataclasses plus `build_encoder_config()`, which normalizes dict/mapping input into the right dataclass. Per-task `OptimizerConfig` is supported.
- **Loss weighting**: supervised tasks use learnable homoscedastic-uncertainty weighting (Kendall et al. 2018). The model learns `log σ_t` per task (`model.task_log_sigmas`); the final per-task loss is `0.5·w_t·exp(−2logσ_t)·L_t + logσ_t`, where `w_t` is the static `loss_weight`. Monitored metrics: `train_final_loss` / `val_final_loss`. Disable to get plain `w_t·L_t`.
- **Dynamic task surgery**: `model.add_task(*cfgs)` and `model.remove_tasks(name)` let you swap heads after loading a checkpoint (used for fine-tuning).

## Data

- `CompoundDataModule` / `CompoundDataset` (`data/`) are **composition-keyed**: each task owns its own data file(s), joined to the others by a **composition** column. There is no monolithic `attributes_source` — adding a property task means adding one file + one task config.
- `CompoundDataModule(task_configs, descriptor_fn, *, task_frames=None, default_data_files=None, composition_column="composition", val_split, test_split, test_all, swap_train_val_split, random_seed, batch_size, num_workers)`:
  - `descriptor_fn: Callable[[list[str]], pd.DataFrame]` computes descriptors from the union of compositions (results cached per unique composition). `PrecomputedDescriptorSource(path)` in `data/composition_sources.py` is a YAML/CLI-friendly callable for already-computed, composition-indexed descriptor files.
  - Per-task data comes from `cfg.data_files` (paths; helpers in `data/composition_sources.py`), or an in-memory `task_frames` mapping, or a shared `default_data_files` fallback. `composition_column` may be a column name or the file's index name (per-task override via `cfg.composition_column`).
- Per-task config fields (`model_config.py` `BaseTaskConfig`): `data_files`, `data_column` (+ `t_column` for kernel regression), `split_column`, `task_masking_ratio`, `predict_idx`. **Column names must match the source exactly.**
- Splits: a single composition-level train/val/test split overlays each task file's `split` column (precedence `test > val > train`) with a random fallback (`val_split`/`test_split`/`random_seed`).
- Prediction: each task's `predict_idx` (literal `train`/`val`/`test`/`all` or an explicit composition sequence) selects a subset; the predict set is their union, exposed as `datamodule.predict_compositions` (the prediction writer attaches these as the output index).
- List-valued cells (sequences, multi-dim targets) must be strings parseable by `ast.literal_eval`, e.g. `"[1.0, 2.5, 3.0]"`.
- Missing targets / NaNs (incl. compositions absent from a task's frame) are **masked out** per task rather than dropped; placeholders fill `y_dict`.

## Coding Style & Naming Conventions
- Follow Ruff formatting (`ruff format`) and lint checks (`ruff check`): 4-space indentation, 120-character lines, double quotes.
- Module and file names stay snake_case; classes use CapWords (`KernelRegressionHead`), functions snake_case, constants UPPER_SNAKE.
- Prefer explicit type hints on public APIs; mirror patterns used in `models/task_head/`.
- Configure behavior through `@dataclass` objects with `__post_init__` validation (mirror `model_config.py`); accept dicts via a normalizer like `build_encoder_config`. Use `str`-based enums for closed choice sets.
- Use the existing `loguru` logger rather than introducing a new logging mechanism.
- LoRA support has been removed; drop any legacy config keys like `lora_rank`/`lora_enabled` (they are no longer recognized and unknown keys will error during config parsing).

## Testing Guidelines
- Tests use `pytest`; name test functions `test_*` and prefer `pytest.mark.parametrize` for input variations.
- Every implementation change ships with a co-located `<module>_test.py` covering the primary path plus the most likely failure patterns (empty/missing/malformed input, NaN/masked targets, tensor-shape or dimension mismatches, enforced preconditions).
- Include targeted tests when adding or modifying features; exercise the trainer and data pathways when touching them.

## Commit & Pull Request Guidelines
- Commit messages follow `<type>: <summary>` (see `git log`, e.g. `refactor(task-head): use raw t input in kernel regression`); keep summaries imperative and under 72 characters when possible.
- Squash fixups locally; each PR should describe scope, motivation, and validation (tests run, configs modified). Link issues or tasks in the description.
- Provide screenshots or logs when changes impact CLI output or training metrics, and flag any backward-incompatible behavior for reviewer attention.
