# Pretrained-model release: store, publish, reuse — the 1.0 interface

The user approved this plan on 2026-11-04. When it is complete, `foundation-model` becomes 1.0. From then
on, the release bundle, the `foundation_model.hub` Python API and the `--pretrained` CLI flags below are the
stable public interface.

Licensing is decided in `docs/licensing/`:

- weights are released under CC BY-NC-SA 4.0 in a gated Hugging Face repository;
- the pretraining data is not released, so post-training that needs it (rehearsal / replay) is out of scope;
- models trained from scratch with the code are not covered by the weights licence.

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
| Units at the interface | users give and receive **physical units**. The bundle's target transforms convert to and from the training scale. This applies to prediction outputs, fine-tuning data for existing heads, and inverse-design targets |
| Kernel-regression `t` | physical units, never transformed; only `y` has a target transform. The model accepts any `t` values and lengths per row; the constraint is that a row's `t` and `y` have equal length. The bundle records the training `t` range and mesh, which are the default prediction grid. A `t` outside the training range warns (extrapolation) and does not fail |
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
| `targets` | per regression / kernel-regression task, the target transform as plain numbers: StandardScaler `mean`, `scale`; PowerTransformer `method`, `lambdas`, `standardize` and its inner scaler |
| `provenance` | pretraining and consolidation configs, git commit, data snapshot names, split seed, licence, and `derived_from` (repository, revision, member) for fine-tuned bundles |

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
| `fm finetune --pretrained SOURCE` | the TOML catalog lists only the user's tasks. Data for an existing head is given in physical units and transformed with the bundle's transform; a new head fits its own. The run writes a bundle with `derived_from` and the CC BY-NC-SA 4.0 licence |
| `fm inverse --pretrained SOURCE` | targets in physical units; classes by name; seeds from a user composition list or an element-system generator, since the bundle carries no data |
| `fm publish <bundle> --repo <id> --tag <v>` | new. Writes the model card and pushes to a private or gated repository |

Weights are loaded with `strict=True` on every bundle path. A missing or unexpected key is an error.

## PR sequence

Each PR passes review and is squash-merged before the next one starts.

| PR | Scope | Acceptance |
|---|---|---|
| 0 | this design | matches the plan the user approved on 2026-11-04; review findings addressed |
| 1 | `hub/transforms.py`: target transforms as data, converted from the existing sklearn `Pipeline[StandardScaler, PowerTransformer]` and `LabelEncoder` objects | forward and inverse transforms match sklearn to within 1e-12 on every scaler in the current preprocessing files; round-trip through JSON |
| 2 | `hub/bundle.py` schema + `export_run` + `fm export`; records the `t` range and mesh per kernel-regression task; refuses members whose architecture or heads differ | a bundle loaded back predicts exactly what the source checkpoint predicts on a fixed composition set (torch `equal` on CPU) |
| 3 | `load_pretrained` / `load_suite` / `predict` / `predict_curve`; `fm predict --pretrained`; strict loading; hub download behind the optional extra | runs in a clean environment with no dataset files and no network (hub calls mocked); out-of-range `t` warns |
| 4 | `fm finetune --pretrained` and `model.save`: user-only catalogs, unit conversion for existing heads, new heads, `derived_from` / licence propagation | end to end: tiny model → export → fine-tune an existing head and a new head on new data only → save → load → predict |
| 5 | `fm inverse --pretrained`: physical-unit targets, class names, seed lists / element-system seeds | inverse from a bundle with no data reproduces the catalog path's result on the same seeds and targets after unit conversion |
| 6 | `fm publish` + model card generator (licence, gating form, citations from `docs/licensing/`) | push to a private test repository, then `load_pretrained` from it |
| 7 | 1.0.0: version bump, README / ARCHITECTURE / `docs/configuration.md` / CHANGES, a "Using a pretrained model" guide | documentation matches the CLI; full test suite |

Independent of PRs 1–7, and needed before the release suite is trained:

- a pretraining option that keeps the run seed fixed across members while the task order changes;
- exclusion of the 31 unresolved Materials Project rows (`docs/licensing/data_sources.md`, open item 1);
- the post-#99 composition-grouped splits.

## Hugging Face steps that need the owner

These happen at PR 6. Nothing is pushed before then.

1. Choose the account or organisation that owns the repositories.
2. Create a write token and store it outside the repository, for example in the macOS keychain, read
   through `HF_TOKEN` at call time.
3. Approve the gating form text and switch on manual approval in the repository settings.

## Non-goals

- Releasing training data, and rehearsal / replay on top of released weights.
- Precomputed-descriptor bundles.
- Hosting model code on the Hub (`trust_remote_code`).
- Changing the training recipe.
