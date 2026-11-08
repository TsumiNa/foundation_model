---
name: yammi-pretrained
description: Use the released YAMMI models (Hugging Face TsumiNa/yammi) — predict material properties from chemical composition, fine-tune on your own data, run inverse design — via the container or Python. Use when asked to predict properties of compositions, adapt the released models to new targets, or design compositions with them.
---

# Using the released YAMMI models

YAMMI (Yet Another Materials Model, Inverse-ready) predicts material properties from **chemical
composition alone**. The released suite is `TsumiNa/yammi` on the Hugging Face Hub: 10 members that
differ only in the order pretraining met the tasks; 41 heads; licence **CC BY-NC-SA 4.0** — results
and fine-tuned models are non-commercial and ShareAlike. Pin releases: `TsumiNa/yammi@v1.0.0`.

## 1. Access and environment (check before anything else)

- **Gated repository.** The user must request access at https://huggingface.co/TsumiNa/yammi and log in
  once (`hf auth login`, or `HF_TOKEN`). An agent cannot do this for them: ask. Error
  `... is gated: request access ...` means exactly this.
- **Prefer the container**; nothing to install. `ghcr.io/tsumina/yammi:<version>` / `:latest` is
  multi-arch (x86-64 and Grace / AArch64, CUDA 13) and includes the Hub client:

  ```bash
  docker run --rm --gpus all -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$PWD:/workspace" \
    ghcr.io/tsumina/yammi:latest predict --pretrained TsumiNa/yammi@v1.0.0 --compositions "Fe2O3,NaCl" --output-dir pred/
  apptainer exec --nv docker://ghcr.io/tsumina/yammi:latest yammi predict ...   # HPC; $HOME is bound
  ```

  The image's entrypoint is `yammi` (Docker: `... <image> predict ...`; Apptainer: `exec ... yammi predict`).
  Without a GPU drop `--gpus all` / `--nv`; prediction is fast on CPU too.
- **Python** only to script it or develop on yammi: `pip install 'yammi[hub] @ git+https://github.com/TsumiNa/yammi'`,
  or for the locked dependency versions `git clone https://github.com/TsumiNa/yammi.git` and
  `uv sync --frozen --no-dev --extra hub` in the clone (its `.venv`). Not on PyPI yet: the PyPI name `yammi` is
  an empty placeholder, so always give the GitHub URL. Python 3.12–3.14.
  `YAMMI_LOG_LEVEL=WARNING` quiets the INFO log.
- **Offline compute nodes.** Run once where the Hub is reachable (fills `~/.cache/huggingface/hub`, or
  `$HF_HOME/hub`), then `HF_HUB_OFFLINE=1` works from the cache. On clusters point `HF_HOME` at group
  storage.

## 2. Predict

```bash
yammi predict --pretrained TsumiNa/yammi@v1.0.0 --compositions "Fe2O3,NaCl" --output-dir pred/
yammi predict --pretrained TsumiNa/yammi@v1.0.0 --compositions-file comps.txt --tasks band_gap,is_metal,zt --output-dir pred/
yammi predict --pretrained "TsumiNa/yammi@v1.0.0#m003" ...    # one member instead of the combination
```

```python
from yammi.hub import load_suite, load_pretrained
suite = load_suite("TsumiNa/yammi@v1.0.0")
suite.predict(["Fe2O3", "NaCl"], tasks=["band_gap", "is_metal"], probabilities=True)
suite.predict_curve(["Bi2Te3"], "zt", t=[300, 400, 500])       # long table: composition, T, zt, zt:std
model = load_pretrained("TsumiNa/yammi@v1.0.0#m003"); model.tasks; model.units()
```

- **Default = all members combined** (CLI without `#member`, and `load_suite(...).predict`): mean
  of the members whose output stays in range, plus `<task>:std` (member spread, a usable uncertainty;
  it correlates with the error). On the held-out test split it had a higher R² than the default member
  on 27 of 29 regression heads (MAE 13–41 % lower on band gap, formation energy, density, bulk
  modulus). Prefer it; report the `:std`.
- **Choose heads with `tasks=` / `--tasks`.** Scalar heads via `predict`, curve heads
  (`kernel_regression`: `zt`, `seebeck`, `power_factor`, `thermal_conductivity`,
  `electrical_resistivity`, `magnetic_susceptibility`, `dos_density`) via `predict_curve`; the CLI
  writes `predictions.csv` plus `<task>_curve.csv`, and `prediction_info.json` (units, members).
- **Outputs are in physical units** (see `model.units()`). `dos_density` is a **shape** (each curve
  divided by its max |value|), not an absolute DOS. Classification gives class names
  (`probabilities=True` adds `<task>:<class>` columns).
- **Inputs**: formulas like `Fe2O3`, `Fe2 O3`, `Fe0.4O0.6` give identical results (atomic fractions:
  `Fe2O3` ≡ `Fe4O6`). Case matters (`fe2o3` is rejected). Unparseable or unsupported-element inputs
  raise `ValueError` naming every bad entry — clean the list, don't silently drop rows.

## 3. Read results critically (what goes wrong)

- **NaN with `SaturationWarning`**: the output is past the range the head's transform can express
  (heavy-tailed heads: dielectric_*, magnetisations, universal_anisotropy, reaction_energy). A single
  member returns NaN by default; the combination drops only the out-of-range members. Do not switch to
  `saturated="clip"` to "fix" NaNs: clipped values (the training maximum) were almost always far off.
- **Head quality differs a lot.** Held-out R² (combined, physical units) is ≥ 0.9 for formation/final
  energy, density, bulk modulus, VBM/CBM/E_F, band gap; 0.6–0.8 for shear modulus, Curie/Néel
  temperatures, the transport curves; low (< 0.3 in physical units) for dielectric constants,
  magnetisations, piezoelectric_max, universal_anisotropy, reaction_energy, poisson_ratio;
  `electrical_resistivity` and `magnetic_susceptibility` curves are not usable. The model card's
  "Accuracy on held-out compositions" table has every head — quote it when reporting.
- **Composition only**: polymorphs get one prediction; extensive quantities (cell volume,
  `total_magnetization` per cell, `magnetization` per mol, moments per f.u.) cannot see formula size
  (`Fe5` ≡ `Fe10`).
- **Out of domain**: elements rare in training (noble gases, actinides) get confident-looking numbers
  with no warning; curve `t` outside the trained range raises `ExtrapolationWarning`; regression heads
  are unbounded (small negative band gaps occur).

## 4. Fine-tune on your own data

The bundle supplies the architecture. The TOML lists **only the user's tasks**, data in **physical
units**, no `[model]`, no `[tasks.scaler]`:

```toml
[descriptor]
kind = "kmd"
n_grids = 8                       # must match the bundle

[datasets.mine]
path = "mine.parquet"             # columns: composition, your targets, optional split (train/val/test)

[[tasks]]                         # existing head: same name + kind; data in the head's unit
name = "band_gap"
kind = "regression"
dataset = "mine"
column = "gap_eV"

[[tasks]]                         # new regression head
name = "hardness"
kind = "regression"
dataset = "mine"
column = "hardness_GPa"
unit = "GPa"

[[tasks]]                         # new classification head: string labels; num_classes is inferred
name = "phase"
kind = "classification"
dataset = "mine"
column = "phase"

[finetune]
pretrained = "TsumiNa/yammi@v1.0.0#m004"   # one member
tasks = ["band_gap", "hardness", "phase"]
epochs = 40
freeze_encoder = true             # false = also adapt the encoder (often a little better, slower)

[output]
dir = "runs/my_finetune"
```

`yammi finetune --config ft.toml` → `runs/my_finetune/bundle` (a derived bundle; load it with
`load_pretrained`; it keeps CC BY-NC-SA and records `derived_from`). Notes:

- New heads fit `StandardScaler → Yeo-Johnson` on the training split with lambda kept in [0, 2], so
  every prediction inverts (yammi ≥ 2.1.0; earlier versions failed on zero-inflated targets).
- Give a `split` column, or the engine makes one. Score the derived bundle on the user's **test**
  split with `predict` and compare with a train-mean baseline; the fine-tune log's metrics leave out
  non-finite predictions (it warns when it does).
- Measured on three unseen tasks (40 epochs): QMOF band gap R² 0.57–0.64, Catalysis-Hub H* adsorption
  0.59–0.63, HEA Curie temperature 0.82 (with yammi ≥ 2.1.0).
- Rehearsal of the pretraining data is impossible (the data are not released).

## 5. Inverse design

```toml
[inverse]
pretrained = "TsumiNa/yammi@v1.0.0"
steps = 300

[inverse.seeds]
strategy = "elements"             # a bundle has no data to rank seeds; or "explicit" with explicit = [...]
elements = ["Fe", "Co", "Ni", "Al"]
n = 20

[[inverse.scenarios]]
name = "wide_gap_stable"
[[inverse.scenarios.targets]]
task = "band_gap"
value = 2.5                       # physical units with --pretrained (eV)
[[inverse.scenarios.targets]]
task = "formation_energy"
direction = "low"
```

`yammi inverse --config design.toml`. Targets are physical units here (with a training checkpoint
instead of a bundle they are z-scores — see the yammi-inverse skill). Results are model-self-scored:
check candidates with `suite.predict` (combined, with `:std`) and say they are proposals, not
validated materials.
