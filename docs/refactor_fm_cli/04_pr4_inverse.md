# PR4 — `fm inverse`: scenario × path inverse-design engine

> Branch: `feat/fm-cli-inverse`. Depends on: PR2 (PR3 only needed for the e2e chain).
> Read `00_OVERVIEW.md` first.
>
> **Source references verified against `master` @ 532a4aa.** The original draft predates commit
> `e5cca4a` (#20, cardinality cap): `DEFAULT_PATHS` is now **11**, not 8, and `PathConfig` needs
> two extra fields to represent the newest path. Corrections folded in below.

## A. Config schema

```toml
[inverse]
checkpoint = "artifacts/run/finetune/final_model.pt"
steps = 300
lr = 0.05
class_weight = 5.0            # QC/classification primary-objective weight
record_trajectory = true
per_seed_trajectories = false
animation_formats = ["gif"]   # subset of {gif, html, svg}; [] = none

[inverse.seeds]
strategy = "top_qc"           # "top_qc" | "random" | "explicit"
n = 20
split = "test"                # which datamodule split to draw candidates from
explicit = []                 # used when strategy == "explicit"
explicit_append = ["Au65 Ga20 Gd15", "Au65 Ga20 Tb15", "Au65 Ga20 Ho15"]
dedup_by_element_system = true

[[inverse.scenarios]]
name = "scenario1_fe_down_moment_up"
reg_tasks = ["formation_energy", "magnetic_moment"]
reg_targets = [-2.0, 2.0]
# primary classification objective is implicit: material_type → QC class (keep legacy default,
# but expose:)
# class_task = "material_type"
# class_target = 2

[[inverse.paths]]             # omit the whole array → DEFAULT_PATHS (the 11 legacy paths)
name = "latent_align1"
method = "latent"             # "latent" | "composition"
ae_align_scale = 1.0

[[inverse.paths]]
name = "comp_seed_5all_elemlist_k5"
method = "composition"
init = "seed"                 # "seed" | "random"
seed_blend = 0.95
allowed_elements = ["Al", "Ti", "..."]   # or "all"; default palette constant exported
diversity_scale = 1.0
max_elements = 5              # optional cardinality cap
# element_step_scale = 1.0    # float or {element: float}
# fixed_amounts = {}          # {element: fraction} pinned during optimization
```

```python
@dataclass(kw_only=True)
class SeedConfig: ...          # fields above; __post_init__: strategy enum, n >= 1,
                               # strategy=="explicit" requires non-empty explicit

@dataclass(kw_only=True)
class ScenarioConfig:
    name: str
    reg_tasks: list[str]
    reg_targets: list[float]   # __post_init__: same length as reg_tasks, non-empty
    class_task: str = "material_type"
    class_target: int | None = None   # None → legacy QC default

@dataclass(kw_only=True)
class PathConfig:
    name: str
    method: str                       # enum InverseMethod: LATENT | COMPOSITION
    # latent:
    ae_align_scale: float = 0.5
    # composition:
    init: str = "seed"                # "seed" | "random"
    n_starts: int | None = None       # random init uses n_starts=len(seeds) in the legacy path;
                                      # None → default to len(seeds) at call time
    seed_blend: float = 0.95
    allowed_elements: list[str] | str = "all"
    diversity_scale: float = 1.0
    max_elements: int | None = None
    element_step_scale: float | dict[str, float] = 1.0
    fixed_amounts: dict[str, float] = field(default_factory=dict)
    # annealing (needed to represent DEFAULT_PATHS path #8 "K=5, linear"):
    annealing_scale: float = 0.5      # optimize_composition default is 0.5
    annealing_schedule: dict[str, Any] | None = None
    # __post_init__: reject latent-only keys on composition paths and vice versa (explicitly
    # set non-default values on the wrong method → ValueError; keeps configs honest)

@dataclass(kw_only=True)
class InverseConfig:
    catalog: TaskCatalogConfig
    model: ModelSectionConfig
    checkpoint: Path
    seeds: SeedConfig
    scenarios: list[ScenarioConfig]   # __post_init__: >= 1, unique names
    paths: list[PathConfig]           # empty → DEFAULT_PATHS
    steps: int = 300
    lr: float = 0.05
    class_weight: float = 5.0
    record_trajectory: bool = True
    per_seed_trajectories: bool = False
    animation_formats: list[str] = field(default_factory=lambda: ["gif"])
    output_dir: Path
```

`DEFAULT_PATHS`: **11** legacy paths = 3 latent + 8 composition. Sources in
`scripts/paper_inverse_comparison.py`: `LATENT_ALIGN_SCALES` (**L203** = `[0.0, 0.25, 1.0]`) and
`COMPOSITION_CONFIGS` (**L137–202**). Export the alloy palette constant (`DEFAULT_ALLOY_PALETTE`,
`paper_inverse_comparison.py:80–132`, asserted length **48**) from `workflows/inverse.py`.

Exact kwargs (regression-test these — hardcode expected values):

Latent (`ae_align_scale` = each of): `0.0`, `0.25`, `1.0`.

Composition (`P` = `DEFAULT_ALLOY_PALETTE`; unlisted fields = the `optimize_composition` default):

| # | label | init | seed_blend | allowed_elements | element_step_scale | diversity_scale | extra |
|---|---|---|---|---|---|---|---|
| 1 | seed | seed | 1.0 | all | 1.0 | 1.0 | — |
| 2 | seed, 5% all | seed | 0.95 | all | 1.0 | 1.0 | — |
| 3 | seed, 5% all, element list | seed | 0.95 | P | 1.0 | 1.0 | — |
| 4 | seed, list, low diversity | seed | 0.95 | P | 1.0 | 0.0 | — |
| 5 | random | random | 0.95 | all | 1.0 | 1.0 | `n_starts=len(seeds)` |
| 6 | seed, list, K=3 | seed | 0.95 | P | 1.0 | 1.0 | `max_elements=3` |
| 7 | seed, list, K=5 | seed | 0.95 | P | 1.0 | 1.0 | `max_elements=5` |
| 8 | seed, list, K=5, linear | seed | 0.95 | P | 1.0 | 1.0 | `max_elements=5`, `annealing_scale=0.715`, `annealing_schedule={"step":[1.0],"scale":[0.0],"annealing_func":["linear"]}` |

`fixed_amounts` is exposed by `optimize_composition` but **no legacy DEFAULT_PATH sets it** — it
is new capability, keep the default empty.

## B. Engine (`workflows/inverse.py :: run(cfg)`)

Sources:

| Legacy | What to take |
|---|---|
| `scripts/paper_inverse_comparison.py` `run()` (**L759–948**), `_run_composition_config` (**L1095**), figure builders | the per-path engine + comparison.png / element_frequency_heatmap.png / qc_vs_secondary_scatter.png / seed_to_optimized__*.png |
| `scripts/eval_inverse_methods.py` `_run_latent_method` (**L105**), `_qc_prob` (**L58**), `_reg_preds` (**L65**), `_seed_weights_from_compositions`, `_format_weights`, `_decode_latent_path` | the latent path + QC-prob / regression-prediction helpers (these live **here**, not in `paper_inverse_comparison.py` — the comparison script imports them). Its own CLI (`evaluate`/`main`) is unused elsewhere → absorb the helpers, drop the CLI. |
| `scripts/paper_inverse_3scenarios.py` (`main` **L110**, scenario.json **L149–161**) | per-scenario loop + `<output>/<scenario>/` isolation (`dataclasses.replace` per scenario, calls the comparison `run`) |
| seed selection: `scripts/continual_rehearsal_demo.py` `_select_seeds` (**L857**, flat `list[str]` — the one the paper engine actually invokes via `ContinualRehearsalRunner`) or `scripts/continual_rehearsal_full.py` `_select_seeds` (**L1133**, dict-returning) | top-QC seed selection with element-system dedup + explicit-append budget. Pick the demo variant to match current paper behaviour; note it is NOT at ~L2375. |
| `scripts/continual_rehearsal_full.py` `run_inverse_only` (**L910**, exact) | replaced by `fm inverse` itself (checkpoint in, results out). The demo runner also has one (`continual_rehearsal_demo.py:627`). |

Flow per scenario:

1. Rebuild model from `TaskCatalog` + `load_checkpoint_state` (`strict=False`); validate that
   every `reg_task` and the `class_task` have heads in the checkpoint — fail loudly listing
   missing heads.
2. Select seeds once per run (not per scenario) per `SeedConfig`; write `seeds.json` at root.
3. For each path: dispatch on method (both signatures verified —
   `optimize_latent` @ `flexible_multi_task_model.py:1742`, `optimize_composition` @ `:2234`; all
   plan kwarg names are correct) —
   `model.optimize_latent(..., optimize_space="latent", ae_align_scale=..., task_targets=...,
   class_targets=..., steps, lr, record_input_trajectory=record_trajectory)` or
   `model.optimize_composition(kmd_kernel, initial_weights=..., n_starts=..., seed_blend=...,
   allowed_elements=..., diversity_scale=..., max_elements=..., element_step_scale=...,
   fixed_amounts=..., annealing_scale=..., annealing_schedule=..., steps, lr,
   record_weights_trajectory=...)`. **`kmd_kernel` is the positional first arg** — legacy passes
   `runner._kmd.kernel_torch(...)` built from `catalog.kmd()`. Pass `n_starts=len(seeds)` for the
   random-init path and `annealing_scale`/`annealing_schedule` for path #8.
   Note the model's own defaults differ between the two (`optimize_latent`: `steps=200, lr=0.1`;
   `optimize_composition`: `steps=300, lr=0.05`) — the `InverseConfig` defaults (300 / 0.05) are
   passed explicitly to both, so pass them, don't rely on the method defaults.
   Composition method requires `catalog.kmd()` — `descriptor.kind == "precomputed"` + a
   composition path → ValueError at config-build time.
4. Outputs per scenario dir: `scenario.json`, `results.json` (raw per-seed arrays per path),
   `summary.json` + `SUMMARY.md` (per-path mean/std table), `comparison.png`,
   `element_frequency_heatmap.png` (discovered elements highlighted),
   `qc_vs_secondary_scatter.png`, `seed_to_optimized__<path>.png`, `targets.json`.
   Legacy figure builders to migrate: `_plot_comparison` (**L218** → comparison.png),
   `plot_element_frequency_heatmap` (imported from `continual_rehearsal_common` →
   element_frequency_heatmap.png), `_plot_qc_vs_reg_scatter` (**L573** →
   qc_vs_secondary_scatter.png), `_plot_seed_to_optimized_mapping` (**L426** →
   `seed_to_optimized__<slug>.png`, note the double underscore). **This is a proposed layout, not
   a straight copy:** the legacy `run()` does NOT write `scenario.json`, `summary.json`, or
   `targets.json` — it embeds the summary under `results.json["summary"]` and writes
   `final_model.pt` + `SUMMARY.md`. `scenario.json` comes from `paper_inverse_3scenarios.py`
   (L149–161). Adding the split-out JSONs is fine; flag it as new in the PR.
5. Trajectories: keep the `.npz` raw dumps + static/animated outputs; per-seed layout behind
   `per_seed_trajectories`.
6. Root outputs: `seeds.json`, `inverse_design.json` (nested all-scenario dump),
   cross-scenario `SUMMARY.md`, `run_provenance.json`, `run.log`.
   **Do NOT migrate** SLIDE_PREP.md / ANALYSIS.md / README.md auto-writers — dropped by design.

## C. `workflows/inverse_trajectory.py`

Rename-move of `scripts/paper_inverse_trajectory.py` (pure helpers:
`best_seed_by_target_distance`, `normalize_target_trajectories`, `plot_trajectory_static`,
`plot_trajectory_animation`). Migrate its colocated test as
`workflows/inverse_trajectory_test.py`.

## D. CLI

Add an `fm inverse` click command (reuse `common_options`): + `--checkpoint`, `--scenario NAME`
(`multiple=True` filter: run only the named scenarios), `--steps`, `--no-trajectory`,
`--animation-formats`.

## E. Tests (`workflows/inverse_test.py`)

- Config validation: reg_tasks/reg_targets length mismatch; empty scenarios; duplicate
  scenario names; latent kwargs on composition path (and vice versa) → ValueError;
  precomputed descriptor + composition path → ValueError; `animation_formats` outside
  {gif,html,svg} → ValueError.
- `DEFAULT_PATHS`: exactly **11** (3 latent + 8 composition), kwargs equal to the §A table
  (regression test — hardcode expected values, including path #8's `annealing_scale=0.715` and
  `annealing_schedule`).
- Path→model-kwargs mapping: a `PathConfig` produces the exact `optimize_latent` /
  `optimize_composition` call kwargs (use a stub model recording calls).
- Seed selection: top-QC dedup keeps one composition per element system; explicit_append
  reduces the strategy budget; strategy="explicit" uses the given list verbatim.
- Smoke: stub/tiny model + 2 seeds + steps=2, one latent + one composition path, one scenario
  → full per-scenario file set exists, results.json schema sane.

## F. Acceptance

- `fm inverse --config samples/inverse_smoke.toml --checkpoint <smoke ckpt>` completes on CPU
  (add the sample: 1 scenario, 2 paths, steps=5, n seeds=4, animations=[]).
- `pytest` / `ruff` / `mypy src` green. Legacy entries untouched.
