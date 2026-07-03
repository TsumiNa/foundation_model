# PR1 — Foundation: `workflows/task_catalog.py` + `workflows/recording.py`

> Branch: `feat/fm-cli-foundation`. Depends on: nothing. Read `00_OVERVIEW.md` first.
> This PR adds **library modules + tests only** — no CLI, no deletion of legacy code.
>
> **Source references verified against `master` @ 532a4aa.** Line numbers below are corrected
> from the original draft; where the legacy behaviour differs from what this PR proposes, the
> divergence is called out explicitly (kernel init, `min_elements`, unknown-key rejection).

## A. `src/foundation_model/workflows/task_catalog.py`

Replaces every hardcoded task registry in the repo:

| Legacy source | What to migrate |
|---|---|
| `scripts/continual_rehearsal_demo.py` `TASK_SPECS` (**L93–111**) | task → (source, kind, column, t_column, num_classes) mapping becomes TOML-driven. **`t_column` is present only for `kind=="kr"`, `num_classes` only for `kind=="clf"`** (kind-conditional, matching the new `TaskSpec.__post_init__`). Legacy `kind` values are the short strings `"reg"/"kr"/"clf"` — `build_task_catalog_config` must normalize them to the `TaskKind` enum (`"regression"/"kernel_regression"/"classification"`). |
| `scripts/continual_rehearsal_demo.py` `ContinualRehearsalRunner._load_data` (**L350**) / `descriptor_fn` (**L452**) | composition-keyed frame loading + on-the-fly KMD-1d descriptor (`utils/kmd_plus.py`: `KMD`, `element_features` — a module-level `DataFrame`, not a fn —, `formula_to_composition`, `DEFAULT_ELEMENTS`) with per-composition caching (KMD built once as `KMD(element_features.values, method="1d", n_grids=..., sigma="auto", scale=True)`; cache is a `dict[str, np.ndarray]`). |
| `scripts/continual_rehearsal_demo.py` `_build_task_config` (**L478**) | TaskSpec → `RegressionTaskConfig` / `ClassificationTaskConfig` / `KernelRegressionTaskConfig` builder |
| `scripts/dynamic_task_suite.py` `SuiteConfig` descriptor/scaler fields (**L91–153**) + `_load_datasets` (**L818**) | precomputed-descriptor path (`PrecomputedDescriptorSource` in `data/composition_sources.py`, L306), per-task scaler loading — `joblib.load` returns a **dict-of-scalers keyed by task name** (`_load_scalers` L914–957) — for inverse-transform (`_maybe_inverse_transform` L959–966, applied to targets and preds). |
| `scripts/multi_task_progressive_clf.py` `_init_centers_and_sigmas` (**L79–118**) **or** `scripts/continual_rehearsal_demo.py` `_init_kernels` (**L324–331**); `n_elements` filter | kernel center/sigma init-from-data helper (**two legacy implementations exist — pick one, see §A note below**); `min_elements` dataset filter (**behaviour-change — see note**). |

### Dataclasses (all `@dataclass(kw_only=True)`, `__post_init__` validation)

```python
class TaskKind(str, Enum):
    REGRESSION = "regression"
    KERNEL_REGRESSION = "kernel_regression"
    CLASSIFICATION = "classification"

@dataclass(kw_only=True)
class ScalerSpec:
    path: Path
    key: str | None = None          # key inside a dict-of-scalers pickle; None = whole object

@dataclass(kw_only=True)
class DatasetSpec:
    name: str
    path: Path
    preprocessing_path: Path | None = None   # dropped-row index cache (qc dataset)
    min_elements: int | None = None          # composition filter (progressive-clf `>= 4`)
    sample: int | None = None                # row cap for smoke runs
    # __post_init__: path exists (defer to load time is OK but validate extension),
    #                min_elements >= 1, sample >= 1

@dataclass(kw_only=True)
class TaskSpec:
    name: str
    kind: TaskKind                            # accepts str, normalized in __post_init__
    dataset: str                              # key into datasets
    column: str
    t_column: str | None = None               # required iff kind == KERNEL_REGRESSION
    num_classes: int | None = None            # required iff kind == CLASSIFICATION
    lr: float | None = None                   # per-task LR override
    replay: float | int | None = None         # per-task rehearsal override
    scaler: ScalerSpec | None = None
    # __post_init__: kind-conditional requirements above; replay: float in (0,1) or int >= 1

@dataclass(kw_only=True)
class DescriptorConfig:
    kind: str = "kmd"                          # "kmd" | "precomputed"
    n_grids: int = 8                           # kmd only
    path: Path | None = None                   # precomputed only; required then

@dataclass(kw_only=True)
class DataConfig:
    composition_column: str = "composition"
    val_split: float = 0.1
    test_split: float = 0.1
    split_random_seed: int = 42
    batch_size: int = 256
    num_workers: int = 0

@dataclass(kw_only=True)
class TaskCatalogConfig:
    data: DataConfig
    descriptor: DescriptorConfig
    datasets: dict[str, DatasetSpec]
    tasks: list[TaskSpec]
    # __post_init__: task names unique; every task.dataset in datasets; >= 1 task
```

### Builder + catalog API

```python
def build_task_catalog_config(raw: Mapping[str, Any]) -> TaskCatalogConfig:
    """Normalize the parsed-TOML tree ([data]/[descriptor]/[datasets.*]/[[tasks]]) into the
    dataclass. Reject unknown keys with a ValueError naming the offending key."""
    # NOTE: build_encoder_config() does NOT actually validate unknown keys — it forwards
    # leftover keys via **config_dict, so an unknown key surfaces as a dataclass
    # TypeError("unexpected keyword argument ..."), not a ValueError. Do NOT rely on that
    # behaviour: add an explicit `set(raw) - KNOWN_KEYS` check per section here so the
    # error message names the key, per the OVERVIEW's config-hygiene requirement.

class TaskCatalog:
    """Loads frames + descriptors once; hands out task configs and datamodules."""
    def __init__(self, config: TaskCatalogConfig) -> None: ...
    @property
    def task_names(self) -> list[str]: ...
    def task_spec(self, name: str) -> TaskSpec: ...
    def task_frames(self, names: Sequence[str]) -> dict[str, pd.DataFrame]:
        """Composition-keyed per-task frames (lazy per-dataset load, min_elements/sample applied)."""
    def descriptor_fn(self) -> Callable[[list[str]], pd.DataFrame]:
        """KMD on-the-fly (cached per composition) or precomputed-file lookup."""
    def kmd(self) -> KMD | None:
        """The invertible KMD object (None when descriptor.kind == 'precomputed').
        Needed by fm inverse (optimize_composition kernel + KMD.inverse decode)."""
    def build_task_config(
        self, name: str, *, latent_dim: int, head_hidden_dim: int, n_kernel: int, lr: float,
        weight_decay: float = 0.0, masking_ratio: float = 1.0, init_from_data: bool = True,
    ) -> RegressionTaskConfig | ClassificationTaskConfig | KernelRegressionTaskConfig:
        """Kernel-regression heads get center/sigma initialized from the task's t/y distribution.
        Field-name mapping into the real dataclasses (model_config.py) — these params are an
        abstraction, they do NOT match the dataclass field names 1:1. NOTE the head input dim must
        equal the model `latent_dim` (heads consume `tanh(latent)`), so:
          - reg/clf `dims=[latent_dim, head_hidden_dim, out]` (out=1 for reg, num_classes-ish for clf)
          - kr `x_dim=[latent_dim, ...]`, `t_dim=[...]`
          - n_kernel     -> kr `kernel_num_centers` (default 32 in the dataclass)
          - lr/weight_decay -> attach `optimizer=OptimizerConfig(lr=lr, weight_decay=weight_decay)`
            (there is no bare `lr` field on task configs; per-head LR lives on `.optimizer`)
          - masking_ratio -> `task_masking_ratio`
          - init_from_data -> kr `kernel_centers_init`/`kernel_sigmas_init` (None => equal-spaced)
        clf heads also carry `num_classes` (from the TaskSpec) and optional `class_weights`."""
    def scaler(self, name: str):  # -> fitted scaler or None
    def inverse_transform(self, name: str, values: np.ndarray) -> np.ndarray:
        """Apply the task's scaler inverse-transform if configured, else identity."""
    def build_datamodule(
        self, active_tasks: Sequence[str], *, masking_ratios: Mapping[str, float] | None = None,
        predict_idx: str | Sequence[str] | None = None,
    ) -> CompoundDataModule: ...
```

Notes:
- Reuse `CompoundDataModule` / `data/composition_sources.py` as-is; this module only assembles
  their inputs.
- **`predict_idx` is a per-task field on `BaseTaskConfig`, not a `CompoundDataModule.__init__`
  argument.** `build_datamodule(..., predict_idx=...)` must set it onto the built task configs
  (literal `train/val/test/all` or an explicit composition sequence), then hand the configs to
  `CompoundDataModule`. The DataModule exposes the resolved set as `datamodule.predict_compositions`.
- Replay **count → ratio** conversion happens where the number of valid labels is known
  (pretrain engine, PR2) — `TaskSpec.replay` is stored raw here.
- Keep `normalize_composition` usage identical to the rehearsal scripts so composition joins
  behave the same.
- **Kernel init — choose ONE canonical helper (owner decision, then delete the other's copy in
  PR6).** Two legacy implementations exist and produce *different* centers/sigmas:
  `multi_task_progressive_clf._init_centers_and_sigmas` (L79–118: histogram density → CDF →
  quantile-interp centers, neighbour-gap sigmas, called with `inverse_density=True`) vs.
  `continual_rehearsal_demo._init_kernels` (L324–331: plain quantile centers, constant span).
  **Decision (owner-delegated):** adopt the demo's `_init_kernels` (quantile centers + constant
  span) as canonical, because it is what the pretrain path actually exercises today
  (`continual_rehearsal_full` builds heads via the demo runner), so the migration reproduces
  current behaviour. The histogram-density `_init_centers_and_sigmas` variant stays available as
  an alternative if a future finetune path needs it, but is not the default. Document the choice
  in the module docstring.
- **`min_elements` is a behaviour change, not a lift-and-shift.** In the legacy code the
  `n_elements >= 4` filter is an *evaluation-time* test-subset report
  (`multi_task_progressive_clf._run_finetune_eval` L651–654 → `clf_report_ge4.json`), NOT a
  row-dropping load filter. This PR intentionally reinterprets it as a `DatasetSpec.min_elements`
  load filter that drops rows before splitting. Call this out in the PR description as a
  deliberate semantics change so reviewers don't expect identical eval numbers.

## B. `src/foundation_model/workflows/recording.py`

The **only** artifact writer for training/predict flows. Centralises what today is scattered
across `continual_rehearsal_common.py` (dump helpers), `continual_rehearsal_full.py`
(per-step checkpoint/metrics/records), `dynamic_task_suite.py` (prediction parquet + records).

```python
@dataclass(kw_only=True)
class RunPaths:
    root: Path                      # <output.dir>[/runs/runNN]
    training: Path                  # root/training
    # helpers: step_dir(step:int, task:str) -> training/stepNN_<task>

class RunRecorder:
    def __init__(self, root: Path) -> None: ...   # mkdir -p, attach loguru file sink root/run.log

    def write_provenance(self, *, config: Any, argv: list[str], seeds: Mapping[str, int]) -> Path:
        """Write root/run_provenance.json:
        - resolved_config: dataclasses.asdict(config), JSON-sanitized. NOTE asdict recurses into
          nested dataclasses but leaves BOTH Path AND Enum (TaskKind/TaskOrder/InverseMethod)
          values non-serializable — dump with a coercing default (e.g.
          `json.dumps(..., default=lambda o: o.value if isinstance(o, Enum) else str(o))`) so
          Path→str and Enum→its `.value`.
        - packages: {python, torch, lightning, numpy, pandas, scikit-learn, foundation-model}
          via importlib.metadata.version (missing → "unknown")
        - datetime_utc / datetime_local (ISO 8601)
        - git: {commit, dirty} via subprocess `git rev-parse HEAD` / `git status --porcelain`
          (repo not found → nulls, never raise)
        - argv, seeds
        Called at startup by EVERY fm subcommand."""

    def save_step_checkpoint(self, step: int, task: str, model, active_tasks: list[str]) -> Path:
        """training/stepNN_<task>/checkpoint.pt — {"model": state_dict, "task_sequence": [...],
        "step": N, "new_task": task, "active_tasks": [...]} (exact schema from
        continual_rehearsal_full.py L851–861 — note `step` is stored 1-based there — so old
        checkpoints stay loadable)."""

    def save_final_model(self, model, task_sequence: list[str], task_spec_dump: dict) -> Path:
        """training/final_model.pt + training/final_model_taskconfigs.json. Legacy schema
        (continual_rehearsal_full.py `_save_final_model` L891–908): final_model.pt =
        {"model": state_dict, "task_sequence": [...]}; final_model_taskconfigs.json =
        {task_name: {"kind": ..., "column": ..., "source": ...}}. Keep both so fm finetune /
        fm inverse can consume either era of checkpoint."""

    def dump_predictions(self, step_dir: Path, task: str, frame: pd.DataFrame) -> Path:
        """<task>_pred.parquet — columns (composition, true, pred[, t]); scaler inverse already
        applied by caller via TaskCatalog.inverse_transform."""

    def dump_metrics(self, step_dir: Path, task: str, metrics: dict) -> Path:   # <task>_metrics.json
    def save_figure(self, path_rel: str, fig) -> Path:                          # any matplotlib fig
    def append_record(self, record: dict) -> None:                              # in-memory
    def write_records(self) -> Path:        # training/experiment_records.json
    def write_metrics_table(self) -> Path:  # training/metrics_table.csv (flat per-task table)
```

Checkpoint-loading helper (used by finetune/inverse/predict later, but lives here):

```python
def load_checkpoint_state(path: Path) -> dict:
    """torch.load(map_location='cpu'); returns the raw dict. Accepts both the rehearsal schema
    ({"model": ..., "task_sequence": ...}) and a bare state_dict (fm-trainer era) — normalize to
    {"model": state_dict, "task_sequence": list|None, ...}."""
```

## C. Tests (colocated)

`workflows/task_catalog_test.py`:
- `build_task_catalog_config`: happy path from a TOML string (use `tomllib.loads`);
  unknown key → ValueError; missing kind-conditional field (`t_column` for kr, `num_classes`
  for clf) → ValueError; duplicate task names → ValueError; `task.dataset` not in `[datasets]`
  → ValueError; replay validation (0 < float < 1 ok, int >= 1 ok, 0/negative → ValueError).
- `TaskCatalog` with tiny synthetic parquet fixtures (tmp_path): frames are composition-keyed;
  `min_elements` filter drops rows; `sample` caps rows; NaN targets preserved (masking is
  downstream); `build_task_config` returns the right dataclass per kind; kernel init-from-data
  produces finite centers/sigmas; `inverse_transform` round-trips with a fitted StandardScaler
  and is identity when no scaler configured; precomputed descriptor path returns the stored
  rows and `kmd()` is None; kmd descriptor is deterministic and cached (second call, same id
  or equal frame).

`workflows/recording_test.py`:
- `write_provenance`: file exists; contains resolved config values, all package-version keys,
  ISO datetime, argv, seeds; git fields present (null tolerated).
- `save_step_checkpoint` / `save_final_model`: schema keys exact; reload via
  `load_checkpoint_state` normalizes both new-schema and bare-state_dict inputs.
- `dump_predictions` / `dump_metrics` / `write_records` / `write_metrics_table`: files appear
  under the documented layout; parquet round-trips.

## D. Acceptance

- `pytest src/foundation_model/workflows/ -q` green; `ruff format` / `ruff check` / `mypy src` clean.
- No changes to `pyproject.toml`, no legacy files touched.
