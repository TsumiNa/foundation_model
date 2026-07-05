# rikyu task-scaling experiment — does more pretraining tasks help downstream targets?

**Question:** as the number of continual-pretraining tasks *k* grows, does warm-start / finetune
performance on held-out targets improve (a task-increment scaling law)? And does inverse-design
quality improve with it?

- **Targets** (3, evaluated jointly): `dielectric_total`, `dielectric_ionic`,
  `dielectric_electronic` (qc regression, n_train ≈ 3124 each).
- **Pretrain pool** (21 = the 24 sweep tasks minus the 3 targets): 14 non-KR (13 regression +
  `material_type`) + 7 kernel-regression tasks.
- **Replay**: two branches, fixed count **n=1000** and **n=1500**, `interval=1`. Both branches
  use the SAME three task orders (same `task_order_seed`), so the k-curves are directly
  comparable across replay amounts. The n=1000 branch writes to the untagged dirs
  (`pre_o*`, `ws_o*`, …); n=1500 to `pre_n1500_o*` etc. Submit a branch with
  `bash jobs/submit_all.sh 1000|1500`.

## Design

### Pretraining (3 runs)

`configs/pre_n1000.toml` — `task_order = "random"` with `task_order_groups` pinning:

1. `formation_energy` at step 1 — it is part of the inverse objective, so every step checkpoint
   k ≥ 1 supports the inverse stage;
2. the other 13 non-KR tasks shuffled over steps 2–14;
3. the 7 KR tasks shuffled over steps 15–21 (KR replay dominates wall time — keeping KR late is
   a deliberate compute/randomness trade-off; the k axis therefore partially confounds task
   *count* with task *kind*).

Run `ORD ∈ {0,1,2}` uses `--seed 2025+ORD --set pretrain.task_order_seed=7501+ORD`
(bit-identical to what `n_runs=3` would produce, but as parallel jobs). Realized orders are in
`orders.txt`; they are also recoverable from each run's `metrics_table.csv` (`new_task` column).

### Per-checkpoint evaluation (k = 1..21, both modes)

- **warm-start** (`ws_o{ORD}/k{KK}`): `fm pretrain --checkpoint stepKK` continues training with
  `task_sequence = [dielectric_total, dielectric_ionic, dielectric_electronic]` (fixed order,
  encoder unfrozen, all k old tasks replayed at n=1000). Metrics: last-step rows of the run's
  `metrics_table.csv` (test split).
- **finetune** (`ft_o{ORD}/k{KK}`): `fm finetune --checkpoint stepKK` — frozen encoder, 3 heads
  jointly (they are independent given the frozen encoder), epochs ≤ 100 with early stopping.
  Metrics: `finetune_summary.json` → `metrics_after`.
- **inverse** (`{ws,ft}_o{ORD}/k{KK}/inverse/`): `fm inverse` on the eval output model.
  Objective (z-scored units): `formation_energy → −1.0`, three dielectrics `→ +1.0`; paths =
  `latent_default` (alignment 0.5) + `comp_k4_lowdiv` (`max_elements=4`, `diversity_scale=0.2`);
  seeds `top_objective` n=20 (test split); no animations (trajectories land in
  `trajectories/*.npz` and feed the aggregated HTML viewer at analysis time). Caveat: objective
  achievement is scored by the same model — comparable across k, not absolute.
- **k=0 baseline** (`scratch_s{2025,2026,2027}`): the 3 targets trained from scratch (no
  checkpoint, replay only among themselves). No inverse (no `formation_energy` head).

### Jobs (all `1n1gpu`, GPU-only; logs in `/home/ea0094/jobs/task_scaling/`)

| Job | Count | Depends on | Est. | Wall |
|---|---|---|---|---|
| `ts_smoke` (`jobs/smoke.sbatch`) | 1 | — (manual gate: wait for `SMOKE_PASS`) | ~20 min | 1 h |
| `ts_pre_o{0,1,2}` | 3 | — | 8–11 h | 24 h |
| `ts_ws_o{0,1,2}` | 3 | `afterok` its own pre | 8–14 h | 24 h |
| `ts_ft_o{0,1,2}` | 3 | `afterok` its own pre | 3–6 h | 12 h |
| `ts_scratch` | 1 | — | ~1.5 h | 4 h |

Submit: `sbatch jobs/smoke.sbatch` → check `SMOKE_PASS` in the log → `bash jobs/submit_all.sh`.
Every worker is idempotent (skip-if-done per k + `--resume`), so any failed/timed-out job is
fixed by resubmitting the same command. If a pretrain job fails, its two dependents stay
**PENDING with reason `DependencyNeverSatisfied` forever** (verified live on rikyu 2026-07-05:
`afterok` releases correctly on success, but the scheduler does *not* auto-cancel on failure —
no `kill_invalid_depend`). Recovery: `scancel` the two stuck eval jobs, resubmit the pretrain,
then resubmit the eval jobs with a fresh `--dependency=afterok:<new pre id>`.

## Analysis (after the runs; scripts land in `analysis/`)

1. Collector → `results/scaling.csv`: `(mode, ord, k, task_added_at_k, target, r2, mae)` from
   the 126 ws `metrics_table.csv` + 63 ft `finetune_summary.json`; inverse collector →
   `results/inverse.csv`: `(mode, ord, k, path, objective_mean, per-target channel means,
   convergence steps)` from `summary.json`/`results.json`/npz.
2. Training curves: per-target test R² vs k (3 orders thin + mean bold), ws/ft panels, scratch
   baseline as a horizontal band. Secondary x-axis variant: cumulative pretraining samples seen
   (orders differ at fixed k — separates "task diversity" from "data volume").
3. Inverse curves: achieved objective / per-target channels vs k, latent vs composition;
   trajectory stats (steps to 90 % of final objective improvement; per-seed final-objective
   spread) vs k.
4. Interactive HTML viewer (`ws_trajectories.html` / `ft_trajectories.html`): dropdowns for
   order / k / path + step slider over the (subsampled) 20-candidate cloud in objective space,
   hover = composition. Built from the npz trajectories; raw data stays on rikyu.

## Provenance

- Configs generated by `make_configs.py` from `../rikyu_replay_sweep/configs/sweep_n1000.toml`
  (same data files, model dims, and training hyper-parameters as the replay sweep).
- Needs `task_order_seed`/`task_order_groups` (PR #31 range, commit 8e15be6) and the
  user-specified inverse targets (PR #32).
- Data: synced to the local `data/` (originally on rikyu Phase 1; conventions in
  `../rikyu_replay_sweep/README.md` "Data conventions").
