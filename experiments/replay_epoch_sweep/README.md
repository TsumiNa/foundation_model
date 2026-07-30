# replay_epoch_sweep — full fixed-count sweep with per-epoch replay resampling

**Question**: rerun the L1 replay-amount sweep (fixed-count family only) with
`pretrain.replay.resample = "epoch"` and compare against the frozen-subset (`"step"`) results —
does per-epoch redrawing of the replay subset shift the per-task saturation curves and the
event-driven forgetting trajectories?

Follow-up to the local 10-step validation (`../replay_epoch_resample/`, n200/n500 on MPS:
seen-task mean R² +0.06 at matched n; n500-epoch beat frozen n1000). This is the full-scale run.

## Design

- **Runs (7)**: fixed-count family n = 100 / 200 / 500 / 1000 / 1500 / 2000 / 2500.
  The ratio family (0.05/0.10/0.15/0.20) is deliberately skipped to save compute.
- **Configs unchanged**: the canonical `../rikyu_replay_sweep/configs/sweep_n*.toml`
  (24 tasks, fixed order, seed 2025, 100 epochs/step + early stop, batch 256, 1 GPU).
  The ONLY deltas per run: `--set 'pretrain.replay.resample="epoch"'` and a fresh
  `--output-dir artifacts/replay_sweep_epoch/replay_n*_epoch`.
- **Baselines reused, not rerun**: the single-task baseline (`../rikyu_replay_sweep/results/warm_restart.csv`)
  involves no replay; at-intro levels come from each run's own `metrics_table.csv`.
- **Comparison target**: the existing step-mode results `../rikyu_replay_sweep/results/mt_n*.csv`
  (rikyu GB200, pre-`cd2d0ea` mask RNG). Known confounds vs this rerun: different hardware
  (A100 vs GB200) and the mask-RNG protocol change (statistically equivalent subsets, not
  bit-identical). Per user decision 2026-07-29: treated as statistically negligible; no extra
  step-mode controls unless the comparison turns ambiguous.

## Execution (ism-gpu-a100, 2026-07-29)

- Machine: ism-gpu-a100 (megalith3), 4× A100-SXM4-40GB, 128 cores; repo clone at
  `/data/claude/foundation_model`, commit `de711ed` (recorded in
  `artifacts/replay_sweep_epoch/COMMIT.txt` on the machine).
  R-CCS Cloud was staged as fallback (`~/projects/foundation_model-gh200` + data) but unused:
  qc-gh200 fully allocated with queue, ai-h200-brc drained ("Kill task failed" 2026-07-29),
  ai-h100l-pu limited to 30-min MIG slices.
- Scheduling: `launch_ism.sh` starts 4 tmux workers (one per GPU) sharing a heavy-first FIFO
  queue (`worker.sh`): n2500 → n2000 → n1500 → n1000 → n500 → n200 → n100. Wall-clock ≈ the
  n2500 run; light runs backfill freed GPUs.
- Every run is idempotent (`--resume`); a failed tag is rerun manually with the same command.
- Remote logs: `logs/replay_n*_epoch.log` + `logs/worker_gpu*.log` under the repo clone.

## Variant: 3x early-stopping patience (p24, RIKEN R-CCS)

Same 7 fixed-count runs and the same two-way setup, plus ONE extra delta:
`--set 'training.early_stopping.patience=24'` (8 → 24). Question: does patience=8 cut the
resampling benefit short (each extra epoch adds fresh replay coverage), or does longer
training just overfit? Three-way comparison: step-p8 (rikyu) vs epoch-p8 (ism) vs epoch-p24.

- Machine: R-CCS Cloud `ai-h200-brc` (H200, 1 GPU + 28 cores per job), clone
  `~/projects/foundation_model-x86-cuda` at commit `25a58b1`, data hardlinked from the
  gh200 clone. Job script: `p24_h200.sbatch` (TAG env selects the config; idempotent
  `--resume`, resubmit on walltime kill).
- Submission staggered under the per-user nodes×72h cap: 4 heavy jobs at 18 h first
  (2026-07-30 ≈11:20 JST: n2500 #263688, n2000 #263689, n1500 #263690, n1000 #263691),
  lights follow as slots free.
- Outputs: `artifacts/replay_sweep_epoch_p24/replay_n*_epoch_p24/` on R-CCS, collected as
  `results/mt_n*_epoch_p24.csv`.

## Results & provenance (not in git, rsync policy)

- Raw run outputs: `artifacts/replay_sweep_epoch/replay_n*_epoch/` on ism-gpu-a100, rsync'd
  back to the same path locally (metrics at `<dir>/training/metrics_table.csv`).
- Collected per-run metric tables: `results/mt_n*_epoch.csv` (same format as the step sweep's
  `mt_n*.csv`, one copy of each run's `metrics_table.csv`).
- Analysis outputs land in `analysis/` (step-vs-epoch versions of `per_task_saturation` and
  `replay_trajectories`, plus the comparison figures for the report/PPT).
