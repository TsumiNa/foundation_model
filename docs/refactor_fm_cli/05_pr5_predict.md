# PR5 — `fm predict`: arbitrary-checkpoint evaluation & prediction

> Branch: `feat/fm-cli-predict`. Depends on: PR2. Read `00_OVERVIEW.md` first.
> Replaces `fm-trainer`'s `validate` / `test` / `predict` subcommands.
>
> **Source references verified against `master` @ 532a4aa.** `predict_idx` /
> `predict_compositions` mechanics confirmed; metric-source and strict=False caveats folded in.

## A. Config schema

```toml
[predict]
checkpoint = "artifacts/run/training/final_model.pt"
tasks = []                    # empty → every head present in the checkpoint
split = "test"                # "train" | "val" | "test" | "all"
# compositions = ["Fe2O3", "..."]   # explicit list; overrides split when given
with_metrics = true           # compute metrics when true targets exist (validate/test use case)
```

```python
@dataclass(kw_only=True)
class PredictConfig:
    catalog: TaskCatalogConfig
    model: ModelSectionConfig
    checkpoint: Path
    tasks: list[str] = field(default_factory=list)
    split: str = "test"                       # enum; ignored when compositions given
    compositions: list[str] = field(default_factory=list)
    with_metrics: bool = True
    output_dir: Path
```

## B. Engine (`workflows/predict.py :: run(cfg)`)

1. Rebuild model from `TaskCatalog`, `load_checkpoint_state` + `load_state_dict(strict=False)`
   (log missing/unexpected keys at INFO). NOTE: the legacy `StrictFalseLightningCLI`
   (`scripts/train.py:18–22`) is **dead code** — `cli_main()` wires the base `LightningCLI`
   (L34), so the strict=False behaviour is intent, not an active legacy path. Implement it fresh
   here rather than trying to "preserve" a code path that isn't used.
2. Resolve the prediction set: `split` via the datamodule's split column / random-split overlay
   (reuse `CompoundDataModule` `predict_idx` mechanics — literal split name or explicit
   composition sequence, exposed as `datamodule.predict_compositions`).
3. Batch-predict all requested heads; per task write `<task>_pred.parquet`
   (composition, pred[, true][, t]) with scaler inverse-transform applied via
   `TaskCatalog.inverse_transform`.
4. `with_metrics` and true targets available → `<task>_metrics.json` + a flat `metrics_table.csv`.
   **There is no single canonical "pretrain eval" metrics module to import.** Use the same metric
   *definitions* as the PR2 `RunRecorder`/`_evaluate_task` path (reg `{r2, mae}`; clf
   `{accuracy, macro_f1}`; kr `r2`+`mae` over flattened sequences). The closest legacy reference
   is the inline block in `dynamic_task_suite.py:721–742` (`samples/mae/mse/rmse/r2`), but it
   **drops None rows** — this PR instead requires NaN/missing targets **masked out, not dropped**
   (a composition with a NaN target still appears in the parquet with `true=NaN`, just excluded
   from the metric). Re-implement; do not import the legacy block.
5. Provenance as everywhere: `run_provenance.json` + `run.log`.

Note: single-device only for this refactor. The legacy distributed gather in
`dynamic_task_suite` / `callbacks/prediction_writer.py` is NOT migrated (out of scope; the
unused callback is deleted in PR6). If multi-GPU predict is needed later it is a separate task.

## C. CLI

Add an `fm predict` click command (reuse `common_options`): + `--checkpoint`, `--tasks a,b,c`,
`--split`, `--compositions "Fe2O3,BaTiO3"` (comma list), `--no-metrics`.

## D. Tests (`workflows/predict_test.py`)

- Config: invalid split → ValueError; unknown task name → ValueError; compositions + split
  both given → compositions wins (documented behaviour).
- Smoke: pretrain-smoke checkpoint → predict on `split="test"`; parquet exists per head,
  metrics json has finite values; explicit `compositions=[...]` returns exactly those rows in
  order; head absent from checkpoint but requested → clear error listing available heads.
- Scaler round-trip: task with a configured scaler produces inverse-transformed predictions
  (compare against manual `scaler.inverse_transform`).
- Masked targets: composition with NaN target appears in parquet with `true=NaN` and is
  excluded from metrics.

## E. Acceptance

- `fm predict --config samples/predict_smoke.toml --checkpoint <smoke ckpt>` completes on CPU
  (add the sample).
- `pytest` / `ruff` / `mypy src` green. Legacy entries untouched.
