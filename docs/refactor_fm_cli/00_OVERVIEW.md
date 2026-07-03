# Refactor: Unified `fm` CLI — Master Plan

> **Status**: approved plan, not yet implemented.
> **Goal owner**: TsumiNa. Executed in stages by coding agents — each numbered doc in this
> directory is one PR. Read this overview first, then the PR doc you are executing.

## 1. Goal

Consolidate all console entry points into a single `fm` command:

```
fm pretrain  --config <toml> [overrides]   # continual rehearsal pre-training (multi-run sweep)
fm finetune  --config <toml> [overrides]   # frozen-encoder fine-tuning of selected task heads
fm inverse   --config <toml> [overrides]   # inverse design (scenarios × algorithm paths)
fm predict   --config <toml> [overrides]   # evaluate / predict with an arbitrary checkpoint
```

This **fully replaces and deletes**:

| Legacy entry | Script | Replaced by |
|---|---|---|
| `fm-trainer` | `scripts/train.py` (LightningCLI) | `fm pretrain` / `fm finetune` (fit), `fm predict` (validate/test/predict) |
| `fm-pretrain-suite` | `scripts/dynamic_task_suite.py` | `fm pretrain` (n_runs sweep) + `fm finetune` |
| `fm-progressive-clf` | `scripts/multi_task_progressive_clf.py` | `fm pretrain` (reg+kr sequence) + `fm finetune` (clf head) |
| — | `scripts/continual_rehearsal_demo.py` | `fm pretrain` (smoke config) |
| — | `scripts/continual_rehearsal_full.py` | `fm pretrain` + `fm inverse` |
| — | `scripts/finetune_inverse_heads.py` | `fm finetune` |
| — | `scripts/paper_inverse_comparison.py` / `paper_inverse_3scenarios.py` | `fm inverse` |
| — | `scripts/eval_inverse_methods.py` | helpers folded into `workflows/inverse.py` |

## 2. Non-negotiable decisions (already confirmed with the owner)

1. **Single `fm` command** built with [`click`](https://click.palletsprojects.com/) (one group +
   four subcommands). No `fm-*` intermediates. (`click` chosen over stdlib argparse — popular,
   ergonomic, extensible for multi-subcommand CLIs; added as a direct dep in PR2.)
2. **`LightningCLI` is dropped entirely.** No jsonargparse / YAML configs anywhere. All configs
   are TOML + argparse CLI overrides. Lightning is kept only as the internal training engine
   (`Trainer`, `EarlyStopping`, etc.). `StrictFalseLightningCLI` / `LoggerSaveConfigCallback`
   are deleted, not migrated; checkpoint loading uses `torch.load` +
   `load_state_dict(strict=False)` directly.
3. **Provenance log on every execution** (all four subcommands): `run_provenance.json` written
   into the output dir at startup, containing the fully-resolved config, core package versions
   (python / torch / lightning / numpy / pandas / scikit-learn / foundation-model), execution
   datetime (ISO), git commit hash + dirty flag, full `sys.argv`, and random seeds. Plus a
   human-readable `run.log` (loguru sink) in the same dir.
4. **Task catalog is fully TOML-driven** (`[datasets]` + `[[tasks]]`). No hardcoded
   `TASK_SPECS` / task registries in code.
5. **Rehearsal interval semantics**: `interval = N` → old tasks join training only on every Nth
   step (with replay subsampling); other steps train the new task + AE only. `N = 1` reproduces
   current behaviour. Evaluation always covers **all** learned heads at every step (forgetting
   trajectory must not be affected by the interval).
6. **Replay amount**: float `< 1` = fraction of labels, int `>= 1` = absolute label count.
   Global default + per-task override. This replaces the old `fixed_tail` / `replay_ratio_high`
   two-tier design.
7. **AE head always trains** in both pretrain and finetune; only its learning rate (`ae_lr`) is
   configurable. (Behaviour change vs `finetune_inverse_heads`, which froze AE — intentional.)
   `task_log_sigmas` are frozen during finetune.
8. **`n_runs` sweep built into `fm pretrain`**: `task_order = "fixed" | "random"`, per-run
   subdirectory + derived seed, aggregated `experiment_records.json`.
9. **Research provenance is first-class**: per-step checkpoints, metrics json + csv, prediction
   parquets, forgetting trajectory, inverse seeds/results/trajectory arrays + plots/animations
   are all preserved and centralised in one recording module.
10. **All legacy scripts, sample configs, shell wrappers and stale docs are deleted** at the end
    (PR6). Historical `artifacts/*/README.md` files are left untouched.

## 3. Target architecture

```
src/foundation_model/
  cli/
    __init__.py
    main.py              # `fm` entry: argparse subparsers → dispatch. THIN — no business logic.
  workflows/
    __init__.py
    task_catalog.py      # TOML → DatasetSpec/TaskSpec; data loading; descriptor sources; scalers
    recording.py         # RunRecorder: output layout, provenance, checkpoints, metrics, parquet
    pretrain.py          # rehearsal pretrain engine (interval replay, n_runs sweep)
    finetune.py          # freeze policy + finetune engine
    inverse.py           # scenario × path inverse-design engine, seed selection
    inverse_trajectory.py# trajectory analytics/plots/animations (migrated)
    plots.py             # parity / confusion / kr-sequences / forgetting / comparison / heatmap
    predict.py           # arbitrary-checkpoint evaluation & prediction
```

Module boundaries:

- `cli/` parses argv, loads TOML, applies overrides, builds the workflow config dataclass, calls
  exactly one `workflows.<mod>.run(cfg)` function. Nothing else.
- `workflows/*` never parse argv. Each `run()` takes a validated config dataclass.
- Only `recording.py` writes artifacts (files) for training/predict flows; other modules hand it
  dataframes/dicts/figures. `inverse.py` may delegate figure rendering to `plots.py` /
  `inverse_trajectory.py` but file paths still come from the recorder.
- Config dataclasses follow repo conventions: `@dataclass(kw_only=True)`, defaults via `field`,
  validation in `__post_init__`, dict→dataclass via builder functions (mirror
  `build_encoder_config`), `str`-based enums for closed choices.
- Imports: single-dot relative inside `workflows/` (`from .recording import RunRecorder`),
  absolute across packages (`from foundation_model.models.model_config import ...`).
- Every module ships a colocated `<module>_test.py` (pytest, parametrized).

## 4. Config schema (shared sections)

One TOML file can hold everything; each subcommand reads the shared sections plus its own.
See per-PR docs for the full schemas. Shared:

```toml
# ---- data & tasks (consumed by all subcommands) ----
[data]
composition_column = "composition"
val_split = 0.1
test_split = 0.1
split_random_seed = 42
batch_size = 256
num_workers = 0

[descriptor]
kind = "kmd"                    # "kmd" (on-the-fly, invertible) | "precomputed"
n_grids = 8                     # kmd only
# path = "data/descriptors.pd.parquet"   # precomputed only (composition-indexed)

[datasets.qc]
path = "data/qc_ac_te_mp_dos_reformat_20260515.pd.parquet"
# preprocessing_path = "..."    # optional dropped-row index cache
# min_elements = 4              # optional composition filter
# sample = 2000                 # optional row cap (smoke runs)

[datasets.supercon]
path = "data/NEMAD_superconductor_20260425.parquet"

[[tasks]]
name = "density"
kind = "regression"             # regression | kernel_regression | classification
dataset = "qc"
column = "Density (normalized)"
# t_column = "DOS energy"       # kernel_regression only
# num_classes = 3               # classification only
# lr = 5e-3                     # per-task LR override
# replay = 0.10                 # per-task rehearsal override (fraction or count)
# scaler = { path = "data/scalers.pkl.z", key = "density" }  # optional inverse-transform

[model]
latent_dim = 128
encoder_hidden = 256
head_hidden_dim = 64
n_kernel = 15

[training]
max_epochs = 100
early_stop_patience = 8
early_stop_min_delta = 1e-4
encoder_lr = 5e-3
head_lr = 5e-3
kr_lr = 5e-4
kr_weight_decay = 5e-5
ae_lr = 5e-3
accelerator = "auto"
devices = 1
seed = 2025

[output]
dir = "artifacts/my_run"        # subcommand appends its own structure under this
```

CLI override convention: `--set section.key=value` (repeatable, dotted path into the TOML tree,
values parsed as TOML), plus first-class flags for the most common knobs (`--output-dir`,
`--checkpoint`, `--max-epochs`, `--sample`, `--accelerator`, `--seed`).

## 5. PR breakdown

| PR | Doc | Branch | Contents | Depends on |
|---|---|---|---|---|
| 1 | [01_pr1_task_catalog_and_recording.md](01_pr1_task_catalog_and_recording.md) | `feat/fm-cli-foundation` | `workflows/task_catalog.py`, `workflows/recording.py` + tests | — |
| 2 | [02_pr2_pretrain.md](02_pr2_pretrain.md) | `feat/fm-cli-pretrain` | `workflows/pretrain.py`, `workflows/plots.py`, `cli/main.py` skeleton, register `fm` | PR1 |
| 3 | [03_pr3_finetune.md](03_pr3_finetune.md) | `feat/fm-cli-finetune` | `workflows/finetune.py`, `fm finetune` subcommand | PR2 |
| 4 | [04_pr4_inverse.md](04_pr4_inverse.md) | `feat/fm-cli-inverse` | `workflows/inverse.py`, `workflows/inverse_trajectory.py`, `fm inverse` | PR2 (PR3 for e2e chain) |
| 5 | [05_pr5_predict.md](05_pr5_predict.md) | `feat/fm-cli-predict` | `workflows/predict.py`, `fm predict` | PR2 |
| 6 | [06_pr6_cleanup.md](06_pr6_cleanup.md) | `chore/fm-cli-cleanup` | delete legacy scripts/configs/docs/wrappers, update AGENTS/README/ARCHITECTURE, remove old console scripts | PR2–5 |

Rules for every PR:

- Branch off `master` (rebase on the previous PR's merge). One PR per doc; do not combine.
- Old entry points keep working until PR6 — do not break `fm-trainer` / `fm-pretrain-suite` /
  `fm-progressive-clf` before then. It is fine for new `workflows/` code to import from legacy
  `scripts/` modules *temporarily only if unavoidable*; prefer copying logic into the new module
  so PR6 can delete `scripts/` files without surgery.
- Per repo conventions: `ruff format`, `ruff check`, `mypy src`, `pytest` must pass; colocated
  `<module>_test.py` with primary-path + failure-pattern coverage; commit style
  `<type>: <summary>`.
- No new markdown docs beyond what each PR doc specifies.

## 6. Functional coverage checklist (must hold at PR6)

- [ ] `fm-trainer` fit → `fm pretrain` / `fm finetune`; validate/test/predict → `fm predict`;
      `strict=False` checkpoint loading preserved.
- [ ] `fm-pretrain-suite`: n_runs sweep, random task order, frozen-encoder finetune stages,
      precomputed descriptors, per-task scaler inverse-transform on predictions, per-task
      masking ratios, per-stage parquet + metrics + parity plots, `experiment_records.json`.
- [ ] `fm-progressive-clf`: reg+kr pretrain sequence + clf finetune; kernel center/sigma
      init-from-data; confusion matrices + classification reports; `min_elements` filter.
- [ ] `continual_rehearsal_full`: tiered replay (per-task overrides), per-step raw artifacts,
      forgetting trajectory, final model + taskconfigs json, inverse-only mode
      (= `fm inverse` on an existing checkpoint).
- [ ] `finetune_inverse_heads`: `fm finetune --set finetune.tasks=[...]`; AE now trainable.
- [ ] `paper_inverse_comparison` + `3scenarios`: scenarios × paths, seeds.json, results.json,
      comparison.png, element-frequency heatmap, qc-vs-secondary scatter, seed→optimized maps,
      trajectory npz + static/animated plots (incl. per-seed variants). SLIDE_PREP/ANALYSIS
      auto-writers are dropped intentionally.

## 7. End-to-end acceptance (run at PR6)

```sh
# CPU smoke chain, ~minutes
fm pretrain --config samples/pretrain_smoke.toml            # 2-3 tasks, sample cap, interval=2, n_runs=2
fm finetune --config samples/finetune_smoke.toml \
    --checkpoint <run>/runs/run00/training/final_model.pt
fm inverse  --config samples/inverse_smoke.toml \
    --checkpoint <finetune>/final_model.pt                  # 1 scenario, latent+comp, steps=5
fm predict  --config samples/predict_smoke.toml \
    --checkpoint <finetune>/final_model.pt
```

Each output dir must contain `run_provenance.json` + `run.log`. Then `pytest`, `ruff check`,
`mypy src` green.
