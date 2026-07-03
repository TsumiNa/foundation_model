# PR3 — `fm finetune`: freeze policy + finetune engine

> Branch: `feat/fm-cli-finetune`. Depends on: PR2. Read `00_OVERVIEW.md` first.
>
> **Source references verified against `master` @ 532a4aa.** Corrected line numbers and
> source-attribution fixes are folded in below.

## A. Config schema

```toml
[finetune]
checkpoint = "artifacts/run/training/final_model.pt"   # or --checkpoint flag
tasks = ["formation_energy", "klat", "material_type"]  # heads to train (must be in [[tasks]])
epochs = 20
# freeze_encoder = true      # default true; expose for completeness (suite always froze it)
# add_new_tasks = true       # default true: tasks not present in the checkpoint are added via
#                            # model.add_task() before training (downstream-finetune use case)
```

```python
@dataclass(kw_only=True)
class FinetuneConfig:
    catalog: TaskCatalogConfig
    model: ModelSectionConfig
    training: TrainingSectionConfig
    checkpoint: Path
    tasks: list[str]                     # __post_init__: non-empty, all in catalog
    epochs: int = 20
    freeze_encoder: bool = True
    add_new_tasks: bool = True
    output_dir: Path
```

If PR2 kept `ModelSectionConfig`/`TrainingSectionConfig` local to `pretrain.py`, move them now
to `workflows/_sections.py` (shared vocabulary module) and update imports.

## B. Engine (`workflows/finetune.py :: run(cfg)`)

Sources to migrate/replace:

| Legacy | What to take |
|---|---|
| `scripts/finetune_inverse_heads.py` `freeze_except` (**L44–67**) + `finetune()` (**L76–164**) | freeze bookkeeping, disable/re-enable of non-target heads (`disable_task`/`enable_task`), save flow. **Caveat:** this file loads with `load_state_dict` at **strict (default `True`)** and **errors on missing heads — it does NOT `add_task`**. |
| `scripts/dynamic_task_suite.py` `_run_finetune_stages` (**L353–427**) | frozen-encoder finetune pattern **and** the `load_from_checkpoint(strict=False, freeze_shared_encoder=...)` + `add_task` behaviour this PR wants. NOTE it `remove_tasks(*all)` then adds a *single* head (so its checkpoint holds only that head) and writes **no** metrics.json (only a `records` list + prediction plots). |
| `scripts/multi_task_progressive_clf.py` `_run_finetune_eval` (**L576–742**) | clf finetune (`strict=False` + `add_task(deepcopy(clf_cfg))`) + confusion matrix (seaborn) / classification report (`clf_report_all.json`, `clf_report_ge4.json`, `metrics.json`). |

> Source-attribution fix: the `strict=False` + `add_task` design in the flow below comes from
> `_run_finetune_stages` / `_run_finetune_eval`, **not** from `finetune_inverse_heads` (which is
> strict + no add_task). Take the *disable/re-enable-then-save* pattern from
> `finetune_inverse_heads` and the *strict=False + add_task* pattern from the suite/clf scripts.

Flow:

1. Rebuild the full model from `TaskCatalog` (all checkpoint tasks + AE), load
   `load_checkpoint_state(...)["model"]` with `load_state_dict(strict=False)` (log
   missing/unexpected keys at INFO).
2. If `add_new_tasks` and some `finetune.tasks` are absent from the checkpoint's task set:
   `model.add_task(...)` for them (this is how progressive-clf attaches the clf head). If
   `add_new_tasks=False` and a requested head is missing → raise, listing the missing heads.
3. **Freeze policy** (rewrite of `freeze_except`, adapted — note the AE change). Attribute names
   (verified): encoder is `model.encoder` (backbone `model.encoder.shared`); heads live in the
   `nn.ModuleDict` **`model.task_heads`** (there is no `model.heads`); disabled heads move to
   `model.disabled_task_heads`; loss-balancer weights are `model.task_log_sigmas`
   (`nn.ParameterDict`). The AE head is keyed by the reserved literal **`"__reconstruction__"`**
   (no `ae_head` attribute, no shared constant today — consider extracting one).
   - `freeze_encoder=True` → `for p in model.encoder.parameters(): p.requires_grad_(False)`.
   - Every task head in `model.task_heads` NOT in `finetune.tasks` → `requires_grad_(False)`,
     **except `"__reconstruction__"` which always stays trainable** (LR = `training.ae_lr`).
     ← intentional behaviour change vs `finetune_inverse_heads` (which froze AE); owner-confirmed.
   - `model.task_log_sigmas` → freeze every scalar `requires_grad_(False)` (weights must not
     drift). **Note:** `task_log_sigmas` are optimized by the encoder/"main" optimizer group
     (`shared_block_optimizer`), not a per-head one — freezing `requires_grad` is sufficient (no
     grad flows), but if `freeze_encoder=True` also skips building the main optimizer, ensure the
     AE head still gets an optimizer (see §B.5 LR note).
   - Return + log a `{param_name: requires_grad}` snapshot into the records for provenance.
4. Datamodule: only `finetune.tasks` (+ AE input) active, `masking_ratio=1.0` for all; other
   heads disabled for the fit via `model.disable_task(*others)` then restored via
   `model.enable_task(*others)` before saving (mirror `finetune_inverse_heads` L102–105 /
   L135–137 so the saved state_dict keeps every head). Do **not** use the suite/clf
   `remove_tasks(*all)+add_task(one)` pattern here — that would drop heads from the checkpoint.
5. Train `epochs` with `Trainer` (EarlyStopping optional — reuse `training.early_stop_*`;
   `epochs` acts as the ceiling). **LR-split mechanism (verified):** the model has **no** native
   `encoder_lr`/`head_lr`/`kr_lr`/`ae_lr` concept. `configure_optimizers` (L1318–1399) builds one
   "main" optimizer for the encoder + learnable `task_log_sigmas` using
   `model.shared_block_optimizer` (an `OptimizerConfig`), and one optimizer per head from that
   head's `config.optimizer or OptimizerConfig()`. So the 4-way LR split must be realized by:
   encoder_lr → `shared_block_optimizer.lr`; and per-head LR (head_lr / kr_lr / ae_lr, plus any
   `TaskSpec.lr` override) → set each head config's `.optimizer = OptimizerConfig(lr=...)` by head
   type when `TaskCatalog.build_task_config` builds it / when the AE head is created. The AE head
   (`"__reconstruction__"`) must receive `OptimizerConfig(lr=training.ae_lr)`.
6. Evaluate the finetuned heads (plus optionally all heads) on the test split; dump
   parquet + metrics + parity/confusion plots via `RunRecorder`; write
   `final_model.pt` + `finetune_summary.json` (heads, epochs run, frozen-param counts,
   before/after metrics).

## C. CLI

Add an `fm finetune` click command to `cli/main.py` (reuse `common_options`): + `--checkpoint`,
`--tasks a,b,c` (comma list overriding `finetune.tasks`), `--epochs`.

## D. Tests (`workflows/finetune_test.py`)

- Freeze policy on a tiny 3-task model: target head trainable; non-target heads frozen;
  **AE head `requires_grad=True`**; `task_log_sigmas` frozen; encoder frozen;
  `freeze_encoder=False` leaves encoder trainable.
- `add_new_tasks`: checkpoint without the clf head + `tasks=["clf"]` → head added and trained;
  `add_new_tasks=False` + missing head → ValueError.
- Config validation: empty `tasks` → ValueError; task not in catalog → ValueError.
- Smoke: pretrain-smoke checkpoint (reuse PR2 fixture helper) → finetune 1 head 1 epoch on CPU
  → `final_model.pt` loadable, `finetune_summary.json` schema, provenance file present.
- Regression guard: saved state_dict contains ALL heads (disable/re-enable round-trip).

## E. Acceptance

- `fm finetune --config samples/finetune_smoke.toml --checkpoint <pretrain-smoke ckpt>`
  completes on CPU (add the sample config).
- `pytest` / `ruff` / `mypy src` green. Legacy entries untouched.
