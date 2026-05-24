# Wiring the trajectory plotting module into another runner

Short integration note for any runner that calls `model.optimize_latent` /
`model.optimize_composition` and wants the per-step trajectory artefacts
([`paper_inverse_trajectory`](../src/foundation_model/scripts/paper_inverse_trajectory.py)).
The reference wiring lives in
[`paper_inverse_comparison.run()`](../src/foundation_model/scripts/paper_inverse_comparison.py)
— copy that pattern.

## Where this module lives (and why)

| File | Role |
|---|---|
| [`paper_inverse_trajectory.py`](../src/foundation_model/scripts/paper_inverse_trajectory.py) | **NEW standalone module** — all trajectory helpers live here (`best_seed_by_target_distance`, `normalize_target_trajectories`, `plot_trajectory_static`, `plot_trajectory_animation`). |
| [`paper_inverse_comparison.py`](../src/foundation_model/scripts/paper_inverse_comparison.py) | Calls the helpers from a private orchestrator `_emit_trajectory_outputs()` (search for it). **This is the reference wiring.** |
| [`continual_rehearsal_common.py`](../src/foundation_model/scripts/continual_rehearsal_common.py) | **Untouched.** It hosts training-loop helpers shared between the two training runners. Trajectory plotting is an analysis-time concern, not a training concern. |
| [`continual_rehearsal_demo.py`](../src/foundation_model/scripts/continual_rehearsal_demo.py) / [`continual_rehearsal_full.py`](../src/foundation_model/scripts/continual_rehearsal_full.py) | **Untouched** by the trajectory feature. To opt in, `import` from `paper_inverse_trajectory` directly — same pattern these runners already use to import `_plot_qc_vs_reg_scatter` / `_plot_seed_to_optimized_mapping` from `paper_inverse_comparison` (see e.g. [`continual_rehearsal_full.py:100-101`](../src/foundation_model/scripts/continual_rehearsal_full.py#L100-L101)). |

**Rationale**: the `paper_inverse_*` files form the post-training analysis
layer; `continual_rehearsal_common.py` holds the training-time shared helpers.
A single consumer doesn't justify promoting to `common`. If a second consumer
materialises (and the wiring is genuinely shared, not just the plot helpers),
the wiring itself — not the plotters — can graduate to `common` later.

## What this module produces, per path

| File | Content |
|---|---|
| `trajectories/<slug>.npz` | `targets`: `(steps, B, T_reg)` per-step regression predictions. `weights`: `(steps, B, n_components)` per-step element weights. |
| `trajectories/trajectory__<slug>.png` | Static line plot, x = step, y = normalised progress (0 = seed, 1 = target), all reg targets on one axis. |
| `trajectories/trajectory__<slug>.{gif,html,svg}` | Same line + per-step top-K composition bar chart of the best representative seed. Format-controlled by `animation_formats`. |

The npz file is the **single source of truth** — both plots and any later
replot read from it; no need to rerun the optimisation.

## 3 hook-up steps

### Step 1 — turn recording on at the model call

`optimize_composition` and `optimize_latent` each take an opt-in flag (default
`False`, zero cost when off):

```python
res = model.optimize_composition(
    kmd_kernel, task_targets=reg_targets,
    # … existing args …
    record_weights_trajectory=True,    # ← was the only new line
)
# res.weights_trajectory: (steps, B, n_components)  — None if flag was False

res = model.optimize_latent(
    initial_input=x_seed, task_targets=reg_targets,
    # … existing args …
    record_input_trajectory=True,      # ← was the only new line
)
# res.input_trajectory: (B, R, steps, input_dim)  — None if flag was False
# For latent, decode to weights via runner._kmd.inverse(per_step_inputs[s]) per step.
```

The latent flag stores the AE-decoded per-step input; `KMD.inverse` then gives
the per-step element weights (one extra QP solve per step × seed, ~10 % overhead).

### Step 2 — persist as compressed npz

Inlining `(steps=300, B=20, n_components=94)` into `results.json` balloons it
to ~36 MB / scenario. Persist alongside instead:

```python
import numpy as np
traj_dir = out_dir / "trajectories"
traj_dir.mkdir(exist_ok=True)
np.savez_compressed(
    traj_dir / f"{slug}.npz",
    targets=res.trajectory.cpu().numpy(),                 # composition: (steps, B, T)
    weights=res.weights_trajectory.cpu().numpy(),         # composition: (steps, B, n_components)
)
# For latent, ``targets`` is res.trajectory[:, 0, :, :].permute(1, 0, 2) → (steps, B, T)
# and ``weights`` is the per-step KMD.inverse stack → (steps, B, n_components).
```

The composition path's slug helper is
[`paper_inverse_comparison._path_slug(r)`](../src/foundation_model/scripts/paper_inverse_comparison.py)
— reuse it so filenames match the existing convention (`latent_align0p25`,
`comp_seed_5_all_element_list`, …).

### Step 3 — render the figures

One helper call per path; the module handles both axes (the line plot on
mean-across-seeds, the comp panel on the best representative seed):

```python
from foundation_model.scripts.paper_inverse_trajectory import (
    best_seed_by_target_distance, normalize_target_trajectories,
    plot_trajectory_static, plot_trajectory_animation,
)
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

# 1. Normalise per-step targets to progress fractions (0 = seed, 1 = target):
progress = normalize_target_trajectories(
    qc_trajectory=np.tile(qc_after_decode[None, :], (steps, 1)),  # see Note A below
    reg_trajectory={t: traj_targets[:, :, j] for j, t in enumerate(reg_names)},
    reg_targets=reg_targets,
    seed_qc=before_qc, seed_reg=before_reg,
)
progress.pop("QC", None)   # we don't have per-step QC; drop the flat synthesised line

# 2. Pick the representative seed for the animation's comp panel:
best_idx = best_seed_by_target_distance(qc_after_decode, reg_after_decode, reg_targets)

# 3. Static + animated:
plot_trajectory_static(progress, out_dir / "trajectory.png", title="…")
plot_trajectory_animation(
    progress,
    per_step_weights=traj_weights[:, best_idx, :],      # (steps, n_components)
    element_symbols=list(DEFAULT_ELEMENTS),
    out_paths_by_format={"gif": out_dir / "trajectory.gif",
                         "html": out_dir / "trajectory.html"},   # any of gif/html/svg
    title="…",
)
```

**Note A** — per-step QC: the model's `optimize_*.trajectory` only records the
reg-target predictions, not the QC head's per-step probability. We synthesise a
flat QC line from the end-state `qc_after_decode` so `normalize_target_trajectories`
has something to return, then drop `progress["QC"]` from the plot. If you need the
real per-step QC curve, post-process the per-step weights yourself:

```python
qc_traj = np.stack(
    [_qc_prob(model, torch.tensor(traj_weights[s] @ kmd_kernel_np, dtype=...))
     for s in range(traj_weights.shape[0])]
)   # (steps, B)
```

That's an extra `B × steps` forward pass — cheap for composition path; for the
latent path it's redundant because the predicts are already on the decoded x.

## Worked example — `continual_rehearsal_full.py`

The runner's existing inverse-design layout (a `paths: dict[str, dict[str,
Any]]` per scenario, populated by `_run_latent_path` / `_run_composition_path`,
then plotted in one shot via the existing
`_plot_inverse_scenario` + `_element_frequency_heatmap` +
`_plot_qc_vs_reg_scatter` block) is exactly the right shape — just three
edits:

### Edit A — `_run_latent_path` (around [continual_rehearsal_full.py:1405](../src/foundation_model/scripts/continual_rehearsal_full.py#L1405))

```python
def _run_latent_path(self, model, x_seed, seeds, reg_targets, path_dir, *,
                    ae_align_scale, label, _qc_prob_fn, _reg_preds_fn,
                    record_trajectory: bool = False):     # ← new arg
    # … existing setup …
    res = model.optimize_latent(
        # … existing args …
        record_input_trajectory=record_trajectory,        # ← new line
    )
    # … existing post-processing populates ``result`` dict …

    if record_trajectory and res.input_trajectory is not None:
        # (B, R=1, steps, input_dim) → (steps, B, input_dim) via permute+squeeze
        per_step_inputs = res.input_trajectory[:, 0, :, :].cpu().numpy().transpose(1, 0, 2)
        per_step_weights = np.stack(
            [self._kmd.inverse(per_step_inputs[s]) for s in range(per_step_inputs.shape[0])]
        )  # (steps, B, n_components) — one QP per step × seed (~10% overhead)
        # ``res.trajectory`` is (B, R=1, steps, T) — squeeze restart → (steps, B, T)
        result["trajectory_targets"] = res.trajectory[:, 0, :, :].cpu().numpy().transpose(1, 0, 2)
        result["trajectory_weights"] = per_step_weights
    return result
```

### Edit B — `_run_composition_path` (around [continual_rehearsal_full.py:1465](../src/foundation_model/scripts/continual_rehearsal_full.py#L1465))

```python
def _run_composition_path(self, model, kmd_kernel, w_seed, seeds, reg_targets,
                          path_dir, *, init, blend, allowed, diversity, label,
                          _qc_prob_fn, _reg_preds_fn,
                          record_trajectory: bool = False):  # ← new arg
    # … existing setup …
    res = model.optimize_composition(
        kmd_kernel, task_targets=reg_targets,
        # … existing args …
        record_weights_trajectory=record_trajectory,           # ← new line
    )
    # … existing post-processing populates ``result`` dict …

    if record_trajectory and res.weights_trajectory is not None:
        # Composition path: trajectories are already on the right surface, no decoding needed.
        result["trajectory_targets"] = res.trajectory.cpu().numpy()             # (steps, B, T)
        result["trajectory_weights"] = res.weights_trajectory.cpu().numpy()     # (steps, B, n_components)
    return result
```

### Edit C — scenario loop (the `paths` dict block around [continual_rehearsal_full.py:1230](../src/foundation_model/scripts/continual_rehearsal_full.py#L1230))

After the existing `_plot_qc_vs_reg_scatter` / `_plot_seed_to_optimized_mapping`
calls, persist the trajectory arrays and emit the new figures:

```python
from foundation_model.scripts.paper_inverse_comparison import _path_slug
from foundation_model.scripts.paper_inverse_trajectory import (
    best_seed_by_target_distance, normalize_target_trajectories,
    plot_trajectory_static, plot_trajectory_animation,
)
from foundation_model.utils.kmd_plus import DEFAULT_ELEMENTS

if record_trajectory:
    traj_dir = sc_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    for path_key, p in paths.items():
        if "trajectory_targets" not in p:
            continue
        slug = _path_slug({"method": p["method"], "label": p["label"],
                           "align_scale": p.get("ae_align_scale")})
        np.savez_compressed(
            traj_dir / f"{slug}.npz",
            targets=p["trajectory_targets"].astype(np.float32),
            weights=p["trajectory_weights"].astype(np.float32),
        )

        # --- plots ---
        reg_names = list(reg_targets)
        traj_targets = p["trajectory_targets"]  # (steps, B, T)
        traj_weights = p["trajectory_weights"]  # (steps, B, n_components)
        qc_after = np.asarray(p["qc_after_decode"], dtype=float)
        reg_traj = {t: traj_targets[:, :, j] for j, t in enumerate(reg_names)}
        progress = normalize_target_trajectories(
            qc_trajectory=np.tile(qc_after[None, :], (traj_targets.shape[0], 1)),
            reg_trajectory=reg_traj, reg_targets=reg_targets,
            seed_qc=before_qc, seed_reg=before_reg,
        )
        progress.pop("QC", None)
        best_idx = best_seed_by_target_distance(
            qc_after, {t: np.asarray(p["reg_after_decode"][t]) for t in reg_names},
            reg_targets,
        )
        plot_trajectory_static(progress, traj_dir / f"trajectory__{slug}.png",
                               title=f"Trajectory · {p['label']}")
        out_paths = {fmt: traj_dir / f"trajectory__{slug}.{fmt}" for fmt in animation_formats
                     if fmt != "none"}
        if out_paths:
            plot_trajectory_animation(
                progress, traj_weights[:, best_idx, :], list(DEFAULT_ELEMENTS),
                out_paths_by_format=out_paths,
                title=f"Trajectory · {p['label']} (best seed: {best_idx})",
            )

        # Free memory before the next path — the trajectories are now on disk.
        del p["trajectory_targets"], p["trajectory_weights"]
        p["trajectory_file"] = str((traj_dir / f"{slug}.npz").relative_to(sc_dir))
```

`record_trajectory`, `per_seed_trajectories`, and `animation_formats` come from
the CLI flags below; thread them down from `_parse_args` → the inverse-design
entry method that owns the scenario loop. The `before_qc` / `before_reg`
arrays are already computed in that same loop for the existing scatter plot,
so no extra forward passes.

## Reference wiring

The full pattern is in
[`paper_inverse_comparison.run()`](../src/foundation_model/scripts/paper_inverse_comparison.py)
(search `_emit_trajectory_outputs`). It also handles the
`--per-seed-trajectories` flag (one plot + animation per `(path × seed)` instead
of the across-seed mean) — same helpers, looped per seed.

## CLI flags to forward

If your runner has its own CLI, mirror these three on it (or read them from the
existing config):

```python
parser.add_argument("--record-trajectory", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--per-seed-trajectories", action="store_true")
parser.add_argument(
    "--animation-formats", nargs="+",
    choices=["gif", "html", "svg", "none"], default=["gif"],
)
```

Pass them through to the runner's inverse-design loop so users can switch
formats without code changes.
