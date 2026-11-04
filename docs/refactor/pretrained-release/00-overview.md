# Pretrained-model release: store, publish, reuse — the 1.0 interface

The user approved this plan on 2026-11-04. When it is complete, `foundation-model` becomes 1.0. From then
on, the release bundle, the `foundation_model.hub` Python API and the `--pretrained` CLI flags below are the
stable public interface.

Licensing is decided in `docs/licensing/`:

- weights are released under CC BY-NC-SA 4.0 in a gated Hugging Face repository;
- the pretraining data is not released, so post-training that needs it (rehearsal / replay) is out of scope;
- models trained from scratch with the code are not covered by the weights licence.

Execution status (2026-11-04):

- Merged: PR 0 (#102), the run-seed option (#103), PR 1 (#104), PR 2 (#105), PR 3 (#106), PR 4 (#107),
  PR 5 (#108) and PR 6 (#109). PR 7, the 1.0.0 release, closes the sequence.
- The NEMAD / phonix-db transforms are recovered and recorded (#104).
- Still open, outside the code:
  - train the release suite: fixed run seed, the post-#99 splits, and the 31 unresolved Materials Project
    rows excluded;
  - the owner's Hugging Face account, token and gating text, before the first `fm publish`.

## Problem

A run writes `training/final_model.pt`, which holds a `state_dict` plus `task_sequence`, and
`final_model_taskconfigs.json`, which holds only `kind`, `column` and `source` per task. Every consumer
rebuilds the model from the TOML task catalog, so it needs the original datasets:

- classification heads read `num_classes` and class weights from the data;
- kernel-regression heads initialise kernel centres from the training `t` values;
- predictions are mapped back to physical units by joblib-pickled sklearn scalers;
- `fm inverse` takes targets in normalised units and draws seed compositions from the task data.

Weights are then loaded with `strict=False`, and missing keys are only logged at info level.

So nobody without our datasets can predict, fine-tune or run inverse design from a checkpoint, and a wrong
architecture can load silently.

## Decisions

| Topic | Decision |
|---|---|
| Code location | new subpackage `foundation_model.hub`; model code stays in this package and Hugging Face holds data only |
| Weights format | `safetensors` (new required dependency); no pickle anywhere on the load path |
| Hub access | `huggingface_hub`, optional extra `foundation-model[hub]`; the token is read from `HF_TOKEN` / `hf auth login` |
| Units at the interface | users give and receive **physical units**. The bundle's target transforms convert to and from the training scale. This applies to prediction outputs, fine-tuning data for existing heads, and inverse-design targets. The exception is shape-only heads (below) |
| Shape-only heads | a head whose training target was normalised per row has no shared transform back to physical units. It is marked `output_scale = "shape"`, and its outputs, fine-tuning data and inverse targets are all in that per-row normalised form. A helper applies the same normalisation to user data. `dos_density` is one: `DOS_smoothing.ipynb` resamples E − E_F onto `linspace(-6, 10, 300)` eV and divides each curve by its own max abs value (`InterpSmooth(normalize=True)`) |
| Kernel-regression `t` | physical units, never transformed; only `y` has a target transform. The model accepts any `t` values and lengths per row; the constraint is that a row's `t` and `y` have equal length. Per task the bundle records two things. The **observed range** is the union of `t` over training rows. The **default grid** is used by `predict_curve(t=None)`: it is the shared grid when every training row uses the same one (DOS), else `n` points evenly spaced over the observed range, where `n` is the most common row length. Transport curves are resampled to 300 points between each curve's own rounded temperature bounds (`process_ac_qe_te_data.ipynb`), so their grids differ per row. A `t` outside the observed range warns (extrapolation) and does not fail |
| Licence of a bundle | a per-bundle `license` field, set at export (`fm export --license`); there is no default. A bundle derived from another inherits the source licence, and a ShareAlike source cannot be relicensed. The model card shows the bundle's own licence. CC BY-NC-SA 4.0 therefore applies to the released weights and their derivatives only, never to a model a user trains from scratch and exports |
| Suite | one release = one repository holding N members (up to 100) that differ only in pretraining task order. Shared `config.json`; per-member weights. Members are exchangeable; the default member is `m000` |
| Composition hashes | not published |
| Legacy | `.pt` checkpoints and the catalog-based CLI paths keep working unchanged; the bundle is an additional path |

## Bundle format, schema v1

```
<bundle>/
  README.md                         model card (YAML: license cc-by-nc-sa-4.0, library_name foundation-model, gating fields)
  config.json                       shared by all members
  manifest.json                     members, default member, per-member task order / seed / metrics
  members/m000/model.safetensors
  members/m000/member.json
  ...
```

A single exported model is a bundle with one member, so there is only one format.

`config.json`:

| Key | Contents |
|---|---|
| `schema_version`, `fm_version` | the loader refuses a newer schema and names the `fm` version that wrote it |
| `descriptor` | `kind = "kmd"`, `n_grids`, the element list, and the sha256 of `element_features.csv`. The loader checks the hash so the input is identical to training. Precomputed descriptors cannot be exported |
| `model` | the full `[model]` section: encoder type and widths, Transformer settings, autoencoder, head and kernel-regression widths, `n_kernel` |
| `heads` | per task: name, kind, display name, unit, per-task width overrides. Classification adds `num_classes` and class names. Kernel regression adds `t` name and unit, `t_min` / `t_max`, the training mesh and `n_kernel`. Kernel centres and widths are parameters inside the weights |
| `targets` | per regression / kernel-regression task, the target transform as an ordered list of plain-number steps. Supported steps:
  - `standard_scaler` (`with_mean`, `with_std`, `mean`, `scale`);
  - `power_transformer` (`yeo-johnson` / `box-cox`, `lambdas`, `standardize` and its inner mean / scale);
  - `log1p` (with an optional lower clip of the input);
  - `arcsinh`;
  - `clip` (bounds on the transformed value; its inverse saturates at the bounds, which the head's metadata reports);
  - `identity`. Classification heads carry `LabelEncoder` classes. Export rejects any other step by name rather than approximating it. Shape-only heads have no entry |
| `license`, `provenance` | the bundle licence (see Decisions). Provenance holds the pretraining and consolidation configs, git commit, data snapshot names, split seed, and `derived_from` (repository, revision, member, licence) for derived bundles |

## Public interface (1.0)

Python, `foundation_model.hub`:

```python
model = load_pretrained("TsumiNa/fm-inorganic", member="m003", revision="v1.0.0")   # or a local path
model.predict(["Fe2O3", "NaCl"], tasks=["band_gap"])        # DataFrame in physical units
model.predict_curve(["Bi2Te3"], task="seebeck", t=None)     # None → training mesh; out-of-range t warns
suite = load_suite("TsumiNa/fm-inorganic", revision="v1.0.0")   # all members, lazily downloaded
model.save("out/my-bundle")                                 # bundle, provenance and licence carried over
export_run(run_dir, catalog_config, "out/bundle")           # pretrain / finetune run → bundle
```

CLI, where `SOURCE` is a local bundle path or `repo[@revision][#member]`:

| Command | Change |
|---|---|
| `fm export --config <toml> --run <dir> [--run <dir> ...] --out <bundle>` | new. One or more runs become one bundle; with several runs, each run is one member |
| `fm predict --pretrained SOURCE` | the model and units come from the bundle; the TOML lists only compositions and tasks |
| `fm finetune --pretrained SOURCE` | the TOML catalog lists only the user's tasks. Data for an existing head is given in physical units and transformed with the bundle's transform. A new head fits its own transform on the training split (default recipe `standard_scaler → power_transformer(yeo-johnson)`), and the transform is stored in the output bundle. The run writes a bundle with `derived_from` and the **source bundle's** licence |
| `fm inverse --pretrained SOURCE` | targets in physical units; classes by name; seeds from a user composition list or an element-system generator, since the bundle carries no data |
| `fm publish <bundle> --repo <id> --tag <v>` | new. Writes the model card and pushes to a private or gated repository |

Weights are loaded with `strict=True` on every bundle path. A missing or unexpected key is an error.

## PR sequence

Each PR passes review and is squash-merged before the next one starts.

| PR | Scope | Acceptance |
|---|---|---|
| 0 | this design | matches the plan the user approved on 2026-11-04; review findings addressed |
| 1 | `foundation_model/data/targets.py`: target transforms as data. Covers the supported steps; conversion from the existing sklearn objects (`Pipeline[StandardScaler, PowerTransformer]`, the AGIS `StandardScaler(with_mean=False) → arcsinh → StandardScaler`, `LabelEncoder`); `fit` with the default recipe; the per-row max-abs shape normalisation | forward and inverse match sklearn to within 1e-12 on every scaler in `data/*preprocessing*.pkl.z` and on the AGIS pipeline. The recorded NEMAD / phonix-db transforms reproduce their `_norm` tables to within 1e-12. JSON round-trip; an unsupported step is rejected by name |
| 2 | `hub/bundle.py` schema + `export_run` + `fm export --license`; records observed range and default grid per kernel-regression task, including heterogeneous per-row grids; refuses members whose architecture or heads differ | a bundle loaded back predicts exactly what the source checkpoint predicts on a fixed composition set (torch `equal` on CPU); the default grid is checked for a shared-grid task and a per-row-grid task |
| 3 | `load_pretrained` / `load_suite` / `predict` / `predict_curve`; `fm predict --pretrained`; strict loading; hub download behind the optional extra | runs in a clean environment with no dataset files and no network (hub calls mocked); out-of-range `t` warns |
| 4 | `fm finetune --pretrained` and `model.save`: user-only catalogs, unit conversion for existing heads, new heads, `derived_from` / licence propagation | end to end: tiny model → export → fine-tune an existing head and a new head on new data only → save → load → predict |
| 5 | `fm inverse --pretrained`: physical-unit targets, class names, seed lists / element-system seeds | inverse from a bundle with no data reproduces the catalog path's result on the same seeds and targets after unit conversion |
| 6 | `fm publish` + model card generator (licence, gating form, citations from `docs/licensing/`) | push to a private test repository, then `load_pretrained` from it |
| 7 | 1.0.0: version bump, README / ARCHITECTURE / `docs/configuration.md` / CHANGES, a "Using a pretrained model" guide | documentation matches the CLI; full test suite |

Independent of PRs 1–7, and needed before the release suite is trained:

- **NEMAD and phonix-db transforms, recorded.** The tables `NEMAD_magnetic_20260419_norm`,
  `NEMAD_superconductor_20260425_norm` and `phonix-db-filtered_20260425_norm` cover 7 tasks. Their code and
  transforms are not in the repository, but the transforms were recovered exactly, to within 1.3e-15 on
  every row:
  1. keep the first row per composition;
  2. `log1p` of the value clipped below at 0;
  3. z-score with the training split's mean and population std;
  4. for phonix `kp` / `klat` only, clip at +5.

  The tables therefore stay as they are. A script in `data/scripts/` records these transforms with the
  PR 1 steps, checks them against the tables, and saves them beside the tables. A full rebuild, which the
  user approved on 2026-11-04 as the fallback, is not needed;
- a pretraining option that keeps the run seed fixed across members while the task order changes;
- exclusion of the 31 unresolved Materials Project rows (`docs/licensing/data_sources.md`, open item 1);
- the post-#99 composition-grouped splits.

## Hugging Face steps that need the owner

These happen at PR 6. Nothing is pushed before then.

1. Choose the account or organisation that owns the repositories.
2. Create a write token and store it outside the repository, for example in the macOS keychain, read
   through `HF_TOKEN` at call time.
3. Approve the gating form text and switch on manual approval in the repository settings.

## Data preprocessing in the CLI: what 1.0 includes

The user asked whether data preprocessing should join the `fm` CLI, with a plugin path for new data
sources. The decision is to split it:

- **In 1.0, the generic part lives in `src/`:**
  - target transforms (fit, apply, invert, serialise), PR 1;
  - curve resampling (`utils/interpolate.InterpSmooth`, already there);
  - the shape normalisation;
  - composition normalisation (`data/composition_sources`, already there).

  `fm finetune --pretrained` applies these to a user's raw table. The user-facing preprocessing that
  reuse needs is therefore inside `fm` from 1.0.
- **After 1.0, source adapters as plugins.** Parsing a specific database is per-source code: NEMAD's
  text-mined units, Starrydata's curve bundles, the Catalysis-Hub API. A later `fm data build` would
  discover adapters through a `foundation_model.datasets` entry-point group. Each adapter turns one raw
  source into a composition-keyed raw table plus a recipe, and the shared code above does the rest. Until
  then the adapters stay in `data/scripts/` and call the `src/` API, starting with the NEMAD / phonix-db
  rebuild.

Why not in 1.0: the plugin contract is only worth fixing once there are two or three adapters written
against the shared API. Freezing it now would make 1.0 promise an interface that has not been tried.

## Non-goals

- Releasing training data, and rehearsal / replay on top of released weights.
- Precomputed-descriptor bundles.
- Hosting model code on the Hub (`trust_remote_code`).
- Changing the training recipe.
