# Using a pretrained model

A released model is a **bundle**. It holds the weights, a self-describing `config.json` and a model
card, and needs no training data and no TOML catalog. This guide covers loading a bundle, predicting
in physical units, fine-tuning it on your own data and running inverse design. The last two sections are
for maintainers who export and publish bundles. Format and design:
[`docs/refactor/pretrained-release/00-overview.md`](refactor/pretrained-release/00-overview.md).

## Install and access

Most uses need **no installation**: the container `ghcr.io/tsumina/yammi:<version>` (or `:latest`)
holds yammi, CUDA 13 and the Hub client, for x86-64 and Grace / AArch64 alike. Every `yammi` command
below runs inside it unchanged; see [container.md](container.md) for Docker / Apptainer commands, the
token, the download cache and offline compute nodes.

```bash
hf auth login     # gated repositories: request access on the model page first
apptainer exec --nv docker://ghcr.io/tsumina/yammi:latest yammi predict --pretrained TsumiNa/yammi@v1.0.0 \
  --compositions "Fe2O3,NaCl" --output-dir pred/
```

Install into your own environment only to call yammi from Python or to develop on it (not on PyPI yet;
the `[hub]` extra pulls `huggingface_hub`):

```bash
pip install 'yammi[hub] @ git+https://github.com/TsumiNa/yammi'
```

yammi logs at INFO; set `YAMMI_LOG_LEVEL=WARNING` for less, `DEBUG` to see the model structure as it is built.

A *source* is a local bundle directory, optionally `<dir>#member`, or a Hub repository written
`owner/name[@revision][#member]`, for example `TsumiNa/yammi@v1.0.0#m003`. The released suites live in
`TsumiNa/yammi`; each tag is one release.

## Predict

**Recommended: all members combined.** A release holds several members that differ only in the order in
which pretraining met the tasks. Combined, they are more accurate than one member (higher held-out R² on
27 of 29 regression heads of v1), and their spread is an uncertainty:

```python
from yammi.hub import load_suite

suite = load_suite("TsumiNa/yammi@v1.0.0")
suite.predict(["Fe2O3", "NaCl"], ["band_gap", "is_metal"])   # band_gap, band_gap:std, is_metal
suite.predict_curve(["Bi2Te3"], "zt", t=[300, 400, 500])     # composition, T, zt, zt:std
```

Regression: the mean of the members whose output stays in range, and `<task>:std` across them.
Classification: the class with the highest mean probability (`probabilities=True` adds the mean
`<task>:<class>` columns). `yammi predict --pretrained owner/name` does this by default; add `#m003` for
one member. The model card's accuracy table lists the held-out R² / accuracy of every head.

One member:

```python
from yammi.hub import load_pretrained

model = load_pretrained("TsumiNa/yammi@v1.0.0")   # the default member, downloaded on its own
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
- **Out of range → NaN.** Some heads' transforms cannot be inverted past the training range (a
  Yeo-Johnson λ < 0 maps every physical value below a bound); the bundle marks them with a final
  `clip` step with `guard: true`. A single member returns a prediction past that range as **NaN** and
  raises `SaturationWarning`. `saturated="clip"` returns the most extreme training value instead, but
  on v1's test split those values were almost always far off (one head's R² fell from 0.25 to −24
  when they were kept). The combined prediction leaves out only the members that are out of range.
- **Inputs.** Compositions are formula strings (`"Fe2O3"`, `"Fe0.5 Ni0.5"`). Anything the KMD descriptor
  cannot parse raises an error that names the composition. `model.latent(compositions)` gives the shared
  latent vectors.

Command line, with no config needed:

```bash
yammi predict --pretrained TsumiNa/yammi@v1.0.0 \
    --compositions "Fe2O3,NaCl" --compositions-file more.txt --output-dir pred/
# → predictions.csv (with <task>:std), <task>_curve.csv per curve head,
#   prediction_info.json (members, licence, units). Add "#m003" to the source for one member.
```

### Members one by one

A release can hold many members (up to ~100). They are interchangeable, so the first *k* are a random
sample:

```python
from yammi.hub import load_suite

suite = load_suite("TsumiNa/yammi@v1.0.0")   # metadata only; each member downloads + loads on first access
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
pretrained = "TsumiNa/yammi@v1.0.0"
tasks = ["band_gap", "hardness"]
epochs = 50
freeze_encoder = true

[output]
dir = "runs/my_finetune"
```

```bash
yammi finetune --config my_finetune.toml
```

- **Existing heads** use the bundle's own transform, so their outputs stay in the same units. A
  classification head accepts labels as the bundle's class names. A shape-only head normalises each of
  your curves the same way it was trained.
- **New heads** fit `StandardScaler → PowerTransformer(yeo-johnson)` on your training split and store it,
  with λ kept in [0, 2] so that every prediction can be inverted (outside it, a zero-inflated target
  saturated 7 % of its test predictions). A new classification head takes its class names from the
  labels; `num_classes` may be left out (an existing head takes the bundle's count).
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
pretrained = "TsumiNa/yammi@v1.0.0"
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
yammi inverse --config my_design.toml
```

Targets are converted with the bundle's transforms. Curve targets `points = [[t, y], ...]` take `t` and
`y` in physical units, or in shape units for shape-only heads. Reports add `<task>_after_mean_physical`;
each scenario's `SUMMARY.md` lists the targets as you wrote them with the achieved values in physical
units, and `scenario.json` keeps the targets as written.

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
yammi export --config release.toml --run runs/order_00 --run runs/order_01 ... \
    --out bundles/fm-v1 --license cc-by-nc-sa-4.0

# Review the card, then push: gated = public page, downloads after manual approval.
yammi publish bundles/fm-v1 --repo TsumiNa/yammi --tag v1.0.0 \
    --visibility gated --card-extra docs/licensing/citations.md --dry-run
yammi publish bundles/fm-v1 --repo TsumiNa/yammi --tag v1.0.0 --visibility gated
```

`release.toml` is the training catalog with `[model]`. Every head needs either a `[tasks.scaler]` (a
fitted scaler pickle or a `*_transforms.json` record) or `output_scale = "shape"`. Classification heads
need their `LabelEncoder`. Set `unit` / `t_unit` so the card and the outputs say what the numbers are.

## Stability (1.0)

From 1.0, these follow semantic versioning:

- the bundle schema: v2 since 1.0.1, which adds the `guard` marker on saturation clips; every reader accepts v1;
- `yammi.hub` (`load_pretrained`, `load_suite`, `PretrainedModel`, `export_runs`, `publish`);
- the `--pretrained` CLI paths.

A newer schema is refused by older readers with a message that names the required version; readers
accept every older schema.
