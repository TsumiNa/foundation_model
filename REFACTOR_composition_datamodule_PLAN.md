# Refactor Plan — Composition-keyed, per-task data sources

Status: **PROPOSED** · Owner: TBD · Created: 2026-05-21

## 1. Motivation

Today `CompoundDataModule` takes a single `attributes_source` file that must contain
**every** task's target column (plus an optional `split` column), aligned positionally
to a single precomputed `formula_desc_source`. Adding one new property task means
rebuilding the whole monolithic attributes file.

Goal: let each property task own its **own data file(s)** (`csv` / `pd.xz` / `parquet`),
joined to the others by a **composition** column. Descriptors are no longer pre-baked
into a file — the DataModule computes them on demand from the union of compositions via
a user-supplied function. Adding a task becomes "drop in one more file + one more task
config", with no global rebuild.

## 2. Confirmed design decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Old `formula_desc_source` / `attributes_source` API | **Hard remove**, migrate all callers in sync (no deprecated coexistence in the final state) |
| 2 | train/val/test split source (after `attributes_source` removal) | Each task file **may carry a `split` column**; compositions with no label fall back to **random split** |
| 3 | `predict_idx` (moved onto `BaseTaskConfig`) | Each task names **its own composition subset** to predict; predict set = union of per-task subsets, each task aligned/masked to its own subset |
| 4 | descriptor function contract | `Callable[[list[str]], pd.DataFrame]` indexed by composition; DataModule **caches** per unique composition |

## 3. Target architecture

### 3.1 Config surface (`models/model_config.py`)

`BaseTaskConfig` gains (all keyword-only, validated in `__post_init__`):

```python
data_files: str | Sequence[str] = ()      # per-task source file(s); concatenated by rows
composition_column: str | None = None      # override DataModule's global default
split_column: str = "split"                 # optional split column name inside the file(s)
task_masking_ratio: float | None = None      # replaces DataModule.task_masking_ratios
predict_idx: Sequence[str] | str | None = None  # composition subset / "train|val|test|all" / file path
```

- `data_column` / `t_column` keep their meaning, but now reference columns **inside the
  task's own file(s)**.
- `AutoEncoderTaskConfig` needs no `data_files` (its target is the input `x`); it "rides
  along" whatever compositions are present in the batch.
- Validation: a non-AE supervised task with empty `data_files` is an error; ratios must be
  in `[0, 1]`; `predict_idx` string literals restricted to `{train,val,test,all}`.

### 3.2 DataModule signature (`data/datamodule.py`)

```python
CompoundDataModule(
    task_configs,
    descriptor_fn: Callable[[list[str]], pd.DataFrame],   # NEW, required
    *,
    composition_column: str = "composition",               # global default, per-task overridable
    val_split: float = 0.1,
    test_split: float = 0.1,
    test_all: bool = False,
    swap_train_val_split: float = 0.0,
    random_seed: int | None = 42,
    batch_size: int = 32,
    num_workers: int = 0,
)
```

**Removed**: `formula_desc_source`, `attributes_source`, `task_masking_ratios`, `predict_idx`.

### 3.3 Data assembly pipeline (inside DataModule `setup`)

1. **Load** each enabled non-AE task's `data_files` (csv/pd.xz/parquet), set the composition
   column as index. Multiple files for one task → concatenated by rows; duplicate
   compositions deduped (keep-first + warn).
2. **Composition universe** = union of all task-file compositions ∪ all task `predict_idx`
   compositions (so predict-only compositions still get descriptors).
3. **Descriptors**: call `descriptor_fn(unique_uncached_compositions)` → DataFrame indexed by
   composition; merge into an instance cache. Compositions with no valid descriptor are
   dropped (+warn); the surviving set is the **master composition index**.
4. **Per-task alignment**: reindex each task's target / `t` / `split` to the master index;
   absent compositions → NaN (→ masked out). This reproduces the old "attributes_df" role
   but assembled from many files, keyed by composition.
5. **Split resolution** (composition-level, single global split shared by all tasks):
   overlay every task file's `split` column with precedence `test > val > train` (warn on
   conflict); unlabeled compositions get the random fallback (`val_split`/`test_split`/seed).
   `test_all=True` short-circuits to all-test. `swap_train_val_split` preserved.
6. **Predict** (`setup("predict")`): predict composition universe = union of per-task
   `predict_idx`; build a predict dataset where each task is masked to its own subset.

Reproducibility: keep the existing deterministic `_seed_for(purpose)` offset scheme.

### 3.4 CompoundDataset rewrite (`data/dataset.py`)

New constructor consumes already-loaded, composition-keyed inputs (produced by §3.3):

```python
CompoundDataset(
    compositions: list[str],                 # sample order + dataset length for this split
    descriptors: pd.DataFrame,                # indexed by composition
    task_frames: dict[str, pd.DataFrame],     # per-task data, indexed by composition
    task_configs,
    task_masking_ratios: dict[str, float] | None = None,
    task_masking_seed: int | None = None,
    is_predict_set: bool = False,
    dataset_name: str = "dataset",
)
```

- **Alignment by composition happens at construction**: each task frame is reindexed to
  `compositions`; `x_formula` is `descriptors.loc[compositions]`. `__getitem__(idx)` then
  fetches positionally from these aligned structures.
- **Output tuple is unchanged**: `(x, y_dict, task_masks_dict, t_sequences_dict)` — so
  `CollateFnWithTaskInfo`, `forward`, `training/validation/test/predict_step` need **no
  change**. This deliberately contains the blast radius.
- Reuse `_parse_structured_element` for list-valued / kernel-regression cells.
- Per-task masking (`task_masking_ratio`) applied here exactly as today, but sourced from
  the task config instead of a DataModule dict.

### 3.5 Splitter (`data/splitter.py`)

`MultiTaskSplitter` is currently positional/DataFrame-based. Adapt it to operate on a
composition-indexed task-availability matrix and serve as the **random fallback** in §3.3
step 5 (so each task keeps representation across train/val/test). Optional polish — a plain
random split is acceptable for the first cut.

### 3.6 Prediction writer (`scripts/callbacks/prediction_writer.py`)

`predict_step` emits every head for every sample in the batch. With per-task `predict_idx`,
the writer must filter each task's rows by that task's mask so only requested compositions
are written. Add composition keys to the written output for downstream joins.

## 4. PR breakdown (each independently testable & green)

> The hard-remove + sync-migrate decision makes the Dataset+DataModule switch naturally
> atomic (the old DataModule calls the old Dataset constructor). PR3 is therefore the large
> "core switch". PR1/PR2 de-risk it by landing and fully testing the new pieces first.

### PR1 — Config surface (additive, green)
- Add the new `BaseTaskConfig` fields + `__post_init__` validation.
- **Tests**: `model_config_test.py` (new/extended) — field defaults, ratio bounds,
  predict_idx literal validation, AE-without-files allowed, supervised-without-files errors.
- No behavior change anywhere else; tree stays green.

### PR2 — Composition data layer (new module, green, pure unit tests)
- New `data/composition_sources.py` (name TBD) with pure helpers:
  multi-file loading, composition-universe build, **cached** `descriptor_fn` application,
  per-task reindex/alignment, composition-level split resolution (overlay + random fallback).
- **Tests**: `composition_sources_test.py` with a fake `descriptor_fn`, tiny in-memory
  frames, and temp files for each format. Cover dedupe, missing descriptors, split conflict
  precedence, predict-only compositions, NaN→mask.
- Not yet wired into DataModule → tree green.

### PR3 — Core switch (atomic, green)
- Rewrite `CompoundDataset` (§3.4) and `CompoundDataModule` (§3.2/§3.3) on top of PR2.
- Migrate **both** consumers: `scripts/dynamic_task_suite.py`,
  `scripts/multi_task_progressive_clf.py` (provide a `descriptor_fn`; move file paths /
  masking ratio / predict subset into task configs; suite wraps its precomputed descriptor
  table as a lookup `descriptor_fn`).
- Update `prediction_writer.py` (§3.6) and sample TOMLs under `samples/`.
- Rewrite `dataset_test.py` + `datamodule_test.py` for the new API; update
  `dynamic_task_suite_test.py` / prediction-writer tests.
- **Optional sub-split** if the diff is too large: (3a) Dataset with a temporary dual-mode
  constructor → (3b) DataModule + scripts switch and remove the temporary mode. Final state
  is still hard-removed.

### PR4 — Splitter + docs (green)
- Adapt `MultiTaskSplitter` (§3.5) and wire as the random fallback.
- Refresh `AGENTS.md` "Data" section, `ARCHITECTURE.md`, `README.md`; add a short
  end-to-end example (descriptor_fn + 2 task files joined by composition).

## 5. Open edge cases to handle (tracked in PR2/PR3)

- Composition in multiple task files with conflicting `split` labels → precedence + warn.
- Composition with no valid descriptor → drop + warn.
- `predict_idx` referencing compositions absent from all task files → still added to the
  descriptor universe.
- Pure-AE / unsupervised config (no supervised task files) → needs an explicit composition
  source; document as unsupported-for-now or add a minimal composition-only channel.
- In-memory task data for tests/suite (avoid forcing temp files): allow the
  Dataset/DataModule layer to accept in-memory frames in addition to `data_files` paths.
- DDP: descriptor computation/cache placement (`setup` runs per-rank) — compute-once vs
  per-rank; keep simple (per-rank cache) and note for later.
- KernelRegression `t_column` now lives in the same per-task file as its `data_column`.

## 6. Validation per PR

Run `ruff format`, `ruff check`, `mypy src`, and the affected `*_test.py` for every PR;
run full `pytest` before PR3 and PR4 merge. Each PR ships its co-located tests per
`AGENTS.md` testing policy.
