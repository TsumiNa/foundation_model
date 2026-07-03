# PR2 — `fm pretrain`: rehearsal engine + plots + CLI skeleton

> Branch: `feat/fm-cli-pretrain`. Depends on: PR1. Read `00_OVERVIEW.md` first.
>
> **Source references verified against `master` @ 532a4aa.** Corrected line numbers and
> migration-source divergences are folded in below.

## A. Config schema (`[pretrain]` + shared sections)

```toml
[pretrain]
task_sequence = ["density", "dos_density", "..."]   # ordered; empty/omitted → [[tasks]] order
n_runs = 1
task_order = "fixed"          # "fixed" | "random"  (random: reshuffle per run, seeded)

[pretrain.rehearsal]
interval = 1                  # every Nth step includes old tasks; other steps: new task + AE only
default_replay = 0.05         # float < 1 → fraction of valid labels; int >= 1 → label count

[pretrain.rehearsal.per_task]  # optional overrides (replaces fixed_tail/replay_ratio_high)
formation_energy = 0.10
tc = 500                      # count form
```

Dataclasses in `workflows/pretrain.py`:

```python
@dataclass(kw_only=True)
class RehearsalConfig:
    interval: int = 1                      # __post_init__: >= 1
    default_replay: float | int = 0.05     # same validation as TaskSpec.replay
    per_task: dict[str, float | int] = field(default_factory=dict)

@dataclass(kw_only=True)
class PretrainConfig:
    catalog: TaskCatalogConfig
    model: ModelSectionConfig              # latent_dim / encoder_hidden / head_hidden_dim / n_kernel
    training: TrainingSectionConfig        # epochs, early-stop, lrs (incl. ae_lr), accelerator, devices, seed
    rehearsal: RehearsalConfig
    task_sequence: list[str] = field(default_factory=list)   # __post_init__: subset of catalog tasks
    n_runs: int = 1
    task_order: str = "fixed"              # enum TaskOrder(str, Enum): FIXED | RANDOM
    output_dir: Path
```

`ModelSectionConfig` / `TrainingSectionConfig` are shared with finetune (PR3) — put them in
`workflows/pretrain.py` for now if only used there, or a small `workflows/_sections.py` once
PR3 needs them (do NOT create it preemptively in this PR unless finetune fields are certain).

## B. Engine (`workflows/pretrain.py :: run(cfg: PretrainConfig)`)

Migrate from `scripts/continual_rehearsal_full.py` `ContinualRehearsalFullRunner`
(training `run()` **L773–889** — NOT L2065–2217, which is `_write_slide_prep_md`;
`_build_empty_model` **L754**; `_evaluate_task` **L1001–1096**; per-step loop L789–867) with
these changes. Note the legacy tiered replay this PR replaces: `fixed_tail` (default
`DEFAULT_FIXED_TAIL` L202) + `replay_ratio` (0.05) + `replay_ratio_high` (0.10), selected at
L794–803; today **every** already-learned task participates every step (`active =
task_sequence[:step+1]`) — there is no interval gating, so the interval schedule is entirely new.

1. **Multi-run sweep** (new, covers `dynamic_task_suite` / `progressive_clf` N-run design):
   - `n_runs == 1`: outputs directly under `<output_dir>/` (`training/...`).
   - `n_runs > 1`: `<output_dir>/runs/runNN/training/...`; run seed = `training.seed + NN`;
     `task_order == "random"` reshuffles `task_sequence` per run with that seed
     (log the realized order into the run's provenance + records).
   - Cross-run aggregate: `<output_dir>/experiment_records.json` merging per-run records
     (fields: run, step, new_task, epochs_run, per-task metrics) + a top-level provenance.
2. **Per-step loop** (per run):
   - `model.add_task(task_config)`; new task `masking_ratio = 1.0`.
   - **Rehearsal interval**: on step `s` (1-based), old supervised tasks participate iff
     `s % rehearsal.interval == 0`. Participating old tasks get their replay amount
     (per-task override else default); **count form is converted to a ratio here** using the
     task's number of valid (non-NaN) training labels: `ratio = min(1.0, count / n_valid)`.
     Non-participating old tasks are **excluded from the datamodule entirely** (cheaper than
     ratio=0). The AE head is always on.
   - Train with `Trainer` + `EarlyStopping(monitor="val_final_loss", patience, min_delta)`
     (verified: this is the exact legacy monitor string, `continual_rehearsal_full.py:814–820`);
     `max_epochs` is the ceiling. **LR mechanism:** the model has no native
     `encoder_lr/head_lr/kr_lr/ae_lr` split — `encoder_lr` maps to the model's
     `shared_block_optimizer.lr` (which also carries `task_log_sigmas`), and every per-head LR
     (head_lr / kr_lr / ae_lr, and any per-task `TaskSpec.lr` override) is realized by setting
     that head config's `.optimizer = OptimizerConfig(lr=...)`. Assign these by head type when
     building the configs. Reuse `_DropLastTrainCompoundDataModule` (`full.py:540`) if BatchNorm
     size-1 tail batches still crash training.
   - **Evaluate ALL learned heads every step** (regardless of interval) on the fixed test
     split. The legacy `_evaluate_task` (L1001–1096) does **not** call `model.forward()`; it runs
     a manual per-head path — `h = torch.tanh(model.encoder(x))`, then `model.task_heads[name](h)`
     (reg/clf) or the KR path via the private `model._expand_for_kernel_regression`. Preserve
     this (or switch to `model.forward()` and select the task from the returned dict) — but do
     not silently change the numerics. Via `RunRecorder`: `<task>_pred.parquet` (scaler
     inverse-transformed through `TaskCatalog.inverse_transform`), `<task>_metrics.json`, the
     newest-head plot (parity / confusion / kr-sequences via `plots.py`), `checkpoint.pt`.
     Metric definitions (keep legacy exactly): reg → `{r2, mae, samples, primary=r2}`; clf →
     `{accuracy, macro_f1, samples, primary=accuracy}`; kr → `r2`+`mae` over flattened
     (concatenated) sequences. **The legacy clf metrics dict does NOT include confusion counts**
     (confusion is plot-only). If you want confusion counts in the JSON, add them as a *new*
     field and say so in the PR — do not describe it as "legacy".
3. **Post-run artifacts**: `forgetting_trajectory.png` (per-step × per-task primary metric),
   `experiment_records.json`, `metrics_table.csv`, `final_model.pt` +
   `final_model_taskconfigs.json` (schemas from PR1 recorder). The forgetting plot is currently
   `ContinualRehearsalFullRunner._plot_forgetting` (`continual_rehearsal_full.py:1686`), a
   **bound method that depends on `self._task_colors` and `self.training_dir`** — migrating it
   into `plots.py` means refactoring those `self` dependencies into plain parameters (task→color
   map + data in, figure out).
4. Keep `_DropLastTrainCompoundDataModule` behaviour if migrating it (check whether
   drop_last is still needed; if yes, move it into `workflows/pretrain.py` as a private class).

## C. `workflows/plots.py`

Migrate the pure plotters from `scripts/continual_rehearsal_common.py`: `plot_parity` (L140),
`plot_confusion` (L173, matplotlib, row-normalized, has a `special_material_type` flag),
`plot_kr_sequences` (L240), `plot_element_frequency_heatmap` (L305); plus constants
`SCATTER_COLOR` (L50), `DISCOVERED_ELEMENT_COLOR`, `MATERIAL_TYPE_CLASSES`,
`MATERIAL_TYPE_DISPLAY_ORDER`. Two corrections vs. the original draft:

- **The forgetting plot is NOT in `common.py`** — it lives in `continual_rehearsal_full.py:1686`
  as a `self`-dependent method (see §B.3); migrate it from there, not from `common.py`.
- **`common.py` does not currently call `matplotlib.use("Agg")`** (it's set in
  `continual_rehearsal_full.py:54`). Adding `matplotlib.use("Agg")` at the top of `plots.py` is a
  new line, not a migration — do it anyway (headless-safe).

`multi_task_progressive_clf.py` has **no reusable** confusion/report function — its rendering is
**inline inside `_run_finetune_eval`** and uses `seaborn.heatmap`. There are therefore two
different confusion renderers (matplotlib `common.plot_confusion` vs. seaborn inline). **Pick
the matplotlib `plot_confusion` as canonical** (avoids adding a seaborn dependency to the
workflows layer); fold the classification-report JSON writing (`sklearn.classification_report`,
`output_dict=True`) into the recorder/metrics path rather than into `plots.py`.

Pure functions: take data + return/write a matplotlib figure; no file-layout knowledge (paths
come from the recorder).

## D. `cli/main.py` + packaging

Use [`click`](https://click.palletsprojects.com/) (add `click>=8.1` to `pyproject.toml`
dependencies in this PR). One `fm` group + one subcommand per workflow:

```python
# src/foundation_model/cli/main.py
import click

@click.group()
def main() -> None:
    """Unified foundation-model CLI."""

def common_options(f):
    """Shared decorator: --config (required), --output-dir, --set (multiple), --seed,
    --accelerator, --sample. Applied to every subcommand."""
    for opt in reversed([
        click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False)),
        click.option("--output-dir", default=None),
        click.option("--set", "overrides", multiple=True, metavar="SECTION.KEY=VALUE"),
        click.option("--seed", type=int, default=None),
        click.option("--accelerator", default=None),
        click.option("--sample", type=int, default=None),
    ]):
        f = opt(f)
    return f

@main.command("pretrain")
@common_options
def pretrain_cmd(config_path, output_dir, overrides, seed, accelerator, sample):
    ...   # PR3-5 add finetune/inverse/predict commands the same way
```

- Common options on every subcommand: `--config <path>` (required), `--output-dir`,
  `--set section.key=value` (repeatable via `multiple=True`; value parsed with `tomllib`
  semantics — `tomllib.loads(f"_={value}")["_"]`), `--seed`, `--accelerator`, `--sample`
  (maps to all datasets' `sample`).
- Flow per subcommand: `tomllib.load` → apply `--set`/flag overrides onto the raw tree →
  `build_*_config` → `RunRecorder(root)` → `write_provenance(...)` → `workflows.<mod>.run(cfg)`.
  Keep the command body THIN — a helper `_load_and_override(config_path, overrides, **flags)`
  shared across subcommands is fine; no business logic in `cli/`.
- `pyproject.toml`: add `fm = "foundation_model.cli.main:main"` (click groups are directly
  usable as console-script entry points). **Do not remove** the legacy entries yet (PR6).

## E. Tests

`workflows/pretrain_test.py`:
- Config: interval < 1 → ValueError; `task_sequence` containing unknown task → ValueError;
  `task_order` invalid → ValueError; per_task replay referencing unknown task → ValueError.
- Replay math: count→ratio conversion (count > n_valid clamps to 1.0; fraction passes through).
- Interval schedule: with interval=2 and 4 tasks, the participating-task sets per step are
  exactly {t1}, {t2,+old}, {t3}, {t4,+old} (pure function — factor the schedule computation
  into a testable helper, e.g. `active_old_tasks(step, learned, interval)`).
- Random order: same seed → same permutation; different runs → logged orders differ.
- Smoke: 2 tiny synthetic tasks (20 rows, kmd n_grids=4, latent_dim=8, max_epochs=1, cpu)
  end-to-end; assert output layout (step dirs, parquet, metrics json, checkpoints,
  final_model.pt, experiment_records.json, forgetting png, run_provenance.json).

`workflows/plots_test.py`: each plot function renders on synthetic input without error and
writes a file (smoke-level; migrate relevant cases from `continual_rehearsal_common_test.py`).

`cli/main_test.py`: `--set training.max_epochs=3` overrides the TOML value in the built config;
unknown subcommand exits non-zero; `--config` missing file → clear error.

## F. Acceptance

- `fm pretrain --config samples/pretrain_smoke.toml` (add this sample: 2–3 tasks, kmd,
  `sample=200`, `interval=2`, `n_runs=2`, `max_epochs=2`) completes on CPU.
- Full `pytest` / `ruff` / `mypy src` green. Legacy entry points untouched and still working.
