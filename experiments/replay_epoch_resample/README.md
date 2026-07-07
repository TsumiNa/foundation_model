# replay_epoch_resample — per-epoch replay resampling validation (local)

**Question**: with `pretrain.replay.resample = "epoch"` (redraw the n-label replay subset every
epoch instead of freezing one subset per step), does small-n replay resist forgetting like large-n
frozen replay does? Coverage over E epochs ≈ `N·(1−(1−n/N)^E)` labels vs a frozen `n`, at the same
per-epoch cost — so n200/n500-epoch *should* behave closer to n1000-step.

**Runs** (Mac Studio, MPS, `run.sh`): the unchanged rikyu sweep configs (`sweep_n200.toml`,
`sweep_n500.toml`: 24 tasks, fixed order, seed 2025, max 100 epochs + early stop) with only
`--set 'pretrain.replay.resample="epoch"'`, output under `artifacts/replay_sweep_epoch/`.

**Baselines**: the rikyu A100 frozen-subset sweep (`artifacts/replay_sweep/replay_{n100..n2500}_rikyu`),
mean final R² 0.371@n100 → ~0.60@n1000+. Caveats when comparing: different hardware (A100 vs MPS)
and the mask-RNG protocol changed in commit `cd2d0ea` (statistically equivalent subsets, not
bit-identical). The expected effect (epoch-n200 escaping the small-n collapse band) is much larger
than either confound; if results land in-between, add local `resample="step"` controls at the same
n before concluding.

**Analysis**: compare per-task forgetting trajectories and mean final R² from each run's
`training/metrics_table.csv` against the rikyu sweep table (`experiments/rikyu_replay_sweep/analysis/`).

Results live under `artifacts/` and are shared by rsync, not git (repo policy).
