# Composition input and split integrity repair (2026-10-16)

## Deliverables

Generate, without training:

```bash
uv run python data/scripts/repair_task_datasets_20261016.py
```

Defaults: source `data/fm_task_datasets_20261013/`, new output
`data/fm_task_datasets_20261016/`. Existing output directories are rejected rather
than overwritten. Data remain local/git-ignored; the generator and tests are tracked.

The initial complete build contains 64 tasks, ten repaired training tables,
238,545 retained table records and 168,339 composition groups. It rejects 84
**table-record incidences**: 36 each in the material-type and transport views,
10 in NEMAD magnetic, two in NEMAD superconducting. The two qc-derived views
repeat affected source rows; these correspond to the 48 source rows in the
seven-source audit, not 84 distinct source materials. No supported record is
averaged or removed by the generator. Raw provenance sidecars are copied byte
for byte and are explicitly not repaired training inputs.

The build changes 8,269 table-record split assignments. This includes replicated
views, exact-key conflicts and proportional aliases; it is not the audit's count
of unique training formula keys moving to holdout. Original formulas, target
values, curves and archived normalization values are unchanged.

Output evidence:

- `split_manifest_20261016.csv`: one unique `composition_group` / `split` per group.
- `row_manifest_20261016.parquet`: original file, row position/ID, formula key,
  group, original split and final split for every retained record.
- `split_changes_20261016.csv`, `rejected_rows_20261016.csv`: movements and exclusions.
- `task_index.csv`: updated observed-label counts for all 64 tasks.
- `provenance_20261016.json`: input/output hashes, script hash, package version,
  policy and validation counts.

The generator checks target/curve equality against supported source records and
asserts zero cross-split composition groups. Independent validation also checks
source/output hashes, every output record's group assignment and all task counts.

## Runtime behavior in 0.6.0

Record keys retain absolute stoichiometry. Split identity uses full atomic
fractions at 12 decimal places, including fractional stoichiometries that
`reduced_formula` alone does not collapse. Non-formula lookup IDs remain separate.
Explicit labels use `test > val > train`; duplicate held-out labels are resolved
before keep-first record handling. Random fallback and train/validation swaps
operate on groups, with a final isolation assertion. Unequal group sizes may
change row counts during swaps.

TaskCatalog resolves the whole campaign scope before dataset filtering or
sampling. Explicit `pretrain.active_tasks` defines the fixed scope when old
checkpoint heads are retained without their original data. Effective splits also
drive kernel initialization and training-label statistics. For separate catalogs,
use the same frozen manifest:

```toml
[data]
split_manifest = "data/fm_task_datasets_20261016/split_manifest_20261016.csv"
```

Missing groups, duplicate/null manifest entries and invalid labels fail. KMD
source formulas are validated before canonical formatting can round away an
unsupported trace element. Default element conversion rejects unsupported or
invalid compositions; explicit custom projection layouts and generic mixture KMD
remain supported (0.5.1, PR #98).

## Limits

No research training or cluster deployment was run. Existing checkpoints can
retain prior held-out composition exposure; new files cannot undo that exposure.
Train-only target normalization is a separate unresolved change: this repair
preserves archived transformations. Do not pool evaluations under the revised
split protocol with historical results as if their conditions were identical.
