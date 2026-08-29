# rikyu_hparam_tuning_v2

The v2 hyper-parameter campaign on RIKYU. Execution plan: [PLAN.md](PLAN.md). What the v1 session
established, and the mistakes it corrected: [SESSION_HANDOVER.md](SESSION_HANDOVER.md).

## Why there is a v2

Every v1 tuning conclusion was measured under a broken LR-scheduler cadence. Before PR #45,
`ReduceLROnPlateau` stepped once per BATCH, so at ~90 batches/epoch a `patience = 5` schedule drove
the LR from 5e-3 to the `min_lr = 1e-4` floor inside the first epoch and the whole run trained on
the floor. `min_lr` was the de-facto training LR; `patience` and `factor` were inert.

Fixing it moved the baseline by +0.0263 mean R² (1.99x the seed band, three seeds non-overlapping).
So v1's numbers are not wrong, they are answers about a different training regime — and its tuned
settings were chosen inside that regime.

## Layout

| path | what |
|---|---|
| `configs/probe6.toml` | the 6-task probe every A'/B' run uses |
| `configs/final_hybrid_v2.toml`, `final_consolidate_v2.toml` | stage C', 24 tasks |
| `configs/grid_*.txt` | generated grids — the registry of what was actually tried |
| `scripts/make_grids.py` | grid generation, including the validity check below |
| `scripts/submit.sh` | stage -> (config, grid, walltime, image); the only submit path |
| `scripts/fm_array.sbatch` | the array worker, with the image gate |
| `scripts/verify_runs.py` | did these runs really TRAIN, or just exit 0 |
| `scripts/status.sh` | queue, completion, timing, and which image each stage used |
| `analysis/common.py` | scoring conventions, inherited from v1 unchanged |
| `analysis/stage0.py`, `stage_a.py`, `finals.py` | per-stage scoring -> `summary/*.json` |
| `summary/*.json` | **the only data that crosses sessions reliably** — see below |

`results/`, `*.png` and `*.csv` are all gitignored. Raw run output lives on RIKYU under
`/data1/rkp00067/rku00225/fm/rikyu_hparam_tuning_v2/` and moves by rsync. Figures are regenerated
from the summary JSON, so the analysis scripts must never need the raw runs to draw.

## Three things this setup refuses to do silently

**Run on the wrong image.** v1's array worker defaulted `IMAGE` to the 0.2.1 container and v1's
`submit.sh` never exported `IMAGE`. Inheriting that would have run the entire v2 campaign under the
per-batch cadence v2 exists to escape — exiting 0, dropping DONE markers, and reporting plausible
numbers from the wrong world. `fm_array.sbatch` now requires `IMAGE`, asks the container its own
version, and refuses to train on a mismatch. Verified by running it against 0.2.1 while expecting
0.3.2: exit 3, no DONE marker, so the point re-runs rather than being skipped.

**Generate a grid point that cannot be constructed.** `OptimizerConfig` rejects `min_lr >= lr`,
and one `[training.scheduler]` block serves all four optimizer groups — so the binding constraint
is the *smallest* group LR, `kr_lr = 5e-4`, not the encoder LR the grid is nominally about.
`make_grids.py` checks every point against all four groups and prints what it dropped instead of
emitting runs that would die on a ValueError.

**Call a green run a trained run.** A run that exits early still exits 0 and still looks green in
`sacct`. `verify_runs.py` requires a DONE marker, the expected fm version, one step directory per
task in the sequence, and a finite metric for every task in the last step.

## Rules carried over from v1, not to be relaxed

1. Every ordering is quoted against the seed band. Differences inside the band are reported as
   inside it, and are not conclusions.
2. Metric dependence is disclosed. v1's stage-A gain was 1.9x the band on MAE and 0.86x on R² —
   real on one, absent on the other. Both are always reported.
3. Saturated tasks do not carry a ranking. `formation_energy`'s ceiling is R² 0.995, so R² there
   measures rounding; per-task metric selection falls back to MAE when R² has no spread left.
4. Read per-step JSON, never `metrics_table.csv` — a `--resume`'d run writes a partial table.
5. An optimum on a grid edge is the grid running out, not an optimum. `stage_a.py` exits non-zero
   when the short list is edge-bound.
6. All remote Slurm commands go through a login shell: `ssh rikyu-login 'bash -lc "..."'`.
   Without it Slurm cannot find its controller and fails in a way that reads exactly like a site
   outage.
