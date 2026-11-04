# Using a pretrained model

A released model is a **bundle**. It holds the weights, a self-describing `config.json` and a model
card, and needs no training data and no TOML catalog. This guide covers loading a bundle, predicting
in physical units, fine-tuning it on your own data and running inverse design. The last two sections are
for maintainers who export and publish bundles. Format and design:
[`docs/refactor/pretrained-release/00-overview.md`](refactor/pretrained-release/00-overview.md).

## Install and access

```bash
pip install 'foundation-model[hub]'   # the extra pulls huggingface_hub for downloads
hf auth login                          # gated repositories: request access on the model page first
```

A *source* is either a local bundle directory or a Hub repository written
`owner/name[@revision][#member]`, for example `<owner>/<repo>@v1.0.0#m003`. `<owner>/<repo>` stands for the
release repository named on each model card.

## Predict

```python
from foundation_model.hub import load_pretrained

model = load_pretrained("<owner>/<repo>@v1.0.0")   # the default member, downloaded on its own
model.tasks                                              # head names
model.units()                                            # unit of each head ("shape" for shape-only heads)

model.predict(["Fe2O3", "NaCl"])                         # every scalar head, physical units
model.predict(["Fe2O3"], ["material_type"], probabilities=True)   # class name + one column per class

model.predict_curve(["Bi2Te3"], "seebeck")                # on the head's default grid
model.predict_curve(["Bi2Te3"], "seebeck", t=[300, 400, 500])
model.predict_curve(["Bi2Te3", "PbTe"], "seebeck", t=[[300, 350], [400, 450, 500]])  # one grid per row
```

- **Units.** Regression and curve outputs are in physical units. Classification gives class names.
  Shape-only heads (`units()[name] == "shape"`, e.g. the DOS) were trained on curves each divided by its
  own maximum absolute value. They predict on that dimensionless scale: a shape, not an absolute value.
  The prediction itself is not renormalised, so its peak is near but not exactly 1.
- **Extrapolation.** A curve head knows the `t` range it was trained on (`model.config.head(name).t_range`).
  Points outside that range raise `ExtrapolationWarning` but are still predicted.
- **Inputs.** Compositions are formula strings (`"Fe2O3"`, `"Fe0.5 Ni0.5"`). Anything the KMD descriptor
  cannot parse raises an error that names the composition. `model.latent(compositions)` gives the shared
  latent vectors.

Command line, with no config needed:

```bash
fm predict --pretrained <owner>/<repo>@v1.0.0 \
    --compositions "Fe2O3,NaCl" --compositions-file more.txt --output-dir pred/
# → predictions.csv, <task>_curve.csv per curve head, prediction_info.json (member, licence, units)
```

### Suites

A release can hold many members (up to ~100) that differ only in the order in which pretraining met the
tasks. They are interchangeable, so the first *k* are a random sample:

```python
from foundation_model.hub import load_suite

suite = load_suite("<owner>/<repo>@v1.0.0")   # metadata only; each member downloads + loads on first access
for member_id in suite.member_ids[:5]:          # only these five are loaded
    print(member_id, suite[member_id].predict(["Fe2O3"], ["band_gap"])["band_gap"].item())
```

## Fine-tune on your own data

The bundle supplies the architecture and its heads. Your TOML lists **only your tasks**. Give the data
in **physical units** and leave out `[tasks.scaler]` and `[model]`.

```toml
[descriptor]            # must match the bundle (KMD); n_grids is checked against it
kind = "kmd"
n_grids = 8

[datasets.mine]
path = "my_measurements.parquet"   # composition, columns in physical units, optional split

[[tasks]]                          # an existing head: same name and kind as in the bundle
name = "band_gap"
kind = "regression"
dataset = "mine"
column = "band_gap_eV"

[[tasks]]                          # a new head
name = "hardness"
kind = "regression"
dataset = "mine"
column = "hardness_GPa"
unit = "GPa"

[training]
accelerator = "auto"

[finetune]
pretrained = "<owner>/<repo>@v1.0.0"
tasks = ["band_gap", "hardness"]
epochs = 50
freeze_encoder = true

[output]
dir = "runs/my_finetune"
```

```bash
fm finetune --config my_finetune.toml
```

- **Existing heads** use the bundle's own transform, so their outputs stay in the same units. A
  classification head accepts labels as the bundle's class names. A shape-only head normalises each of
  your curves the same way it was trained.
- **New heads** fit `StandardScaler → PowerTransformer(yeo-johnson)` on your training split and store it.
- **Output.** `runs/my_finetune/bundle` is a new bundle. Load it with `load_pretrained` like any other.
  It keeps the source licence and records where it came from (`provenance.derived_from`).
- **Not available.** Replaying the pretraining data (rehearsal) is impossible: the data are not
  released. With `freeze_encoder = true` the encoder and every head you do **not** list in `tasks` keep their
  predictions; the heads you list are trained (that is the point). With `false` the unlisted heads may
  drift too, since they share the encoder.

From Python, after loading: `model.save("out/copy")` writes a one-member derived bundle.

## Inverse design

```toml
[inverse]
pretrained = "<owner>/<repo>@v1.0.0"
steps = 300

[inverse.seeds]
strategy = "elements"          # or "explicit" with explicit = ["Fe2O3", ...]; a bundle has no data to rank
elements = ["Fe", "Co", "Ni", "Al"]
n = 20

[[inverse.scenarios]]
name = "wide_gap_quasicrystal"

[[inverse.scenarios.targets]]
task = "band_gap"
value = 2.5                    # eV, physical units

[[inverse.scenarios.targets]]
task = "material_type"
classes = ["IQC", "DQC"]       # class names

[output]
dir = "runs/my_design"
```

```bash
fm inverse --config my_design.toml
```

Targets are converted with the bundle's transforms. Curve targets `points = [[t, y], ...]` take `t` and
`y` in physical units, or in shape units for shape-only heads. Reports add `<task>_after_mean_physical`,
and `scenario.json` keeps the targets as you wrote them.

## Licence

Weights published by the authors are licensed under **CC BY-NC-SA 4.0** (check the model card of each
release):

- They, and **every model derived from them**, including fine-tuned ones, are for non-commercial use,
  with attribution, under the same licence. A derived bundle carries the licence automatically.
- Models you train **from scratch** with this code on your own data are not covered. The code itself is
  Apache-2.0.

Data sources and their licences: [`docs/licensing/`](licensing/).

## For maintainers: export and publish

```bash
# One bundle from a suite of runs: one member per run, each verified bit-identical to its checkpoint.
fm export --config release.toml --run runs/order_00 --run runs/order_01 ... \
    --out bundles/fm-inorganic-v1 --license cc-by-nc-sa-4.0

# Review the card, then push: gated = public page, downloads after manual approval.
fm publish bundles/fm-inorganic-v1 --repo <owner>/<repo> --tag v1.0.0 \
    --visibility gated --card-extra docs/licensing/citations.md --dry-run
fm publish bundles/fm-inorganic-v1 --repo <owner>/<repo> --tag v1.0.0 --visibility gated
```

`release.toml` is the training catalog with `[model]`. Every head needs either a `[tasks.scaler]` (a
fitted scaler pickle or a `*_transforms.json` record) or `output_scale = "shape"`. Classification heads
need their `LabelEncoder`. Set `unit` / `t_unit` so the card and the outputs say what the numbers are.

## Stability (1.0)

From 1.0, these follow semantic versioning:

- bundle schema v1;
- `foundation_model.hub` (`load_pretrained`, `load_suite`, `PretrainedModel`, `export_runs`, `publish`);
- the `--pretrained` CLI paths.

A newer schema is refused by older readers with a message that names the required version; readers
accept every older schema.
