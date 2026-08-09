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

## Variant: raised epoch budget (m150, RIKEN R-CCS)

Motivated by the p24 finding that 100% of completed steps hit the 100-epoch cap (patience is
no longer the binding constraint): `m150_h200.sbatch` = epoch resampling + patience 24 +
`--set 'training.max_epochs=150'`. Probes the marginal return of a 1.5x epoch budget.
Rollout order: n2500/n1000 first (the informative extremes), n1500/n2000 later, all under the
partition's hard per-user quota of 72 node-hours of *submitted* work (sbatch REJECTS above it —
observed live), so submissions trickle in as running jobs complete. Outputs:
`artifacts/replay_sweep_epoch_m150/replay_n*_epoch_m150/` → `results/mt_n*_epoch_m150.csv`.

## Extension: ratio-replay family + no-replay/joint-retrain baseline (2026-08-08, R-CCS)

Follow-up to the report's ratio-parameterization evidence. Two additions, both on ai-h200-brc:

1. **Ratio family under the m150 recipe** — replay `amount` ∈ {0.1, 0.2, 0.3, 0.5} (fraction of
   each old task's labels, redrawn every epoch) with the same overrides as the m150 arm
   (`resample="epoch"`, patience 24, max_epochs 150). Reuses `m150_h200.sbatch` verbatim with
   `TAG=0p10/0p20/0p30/0p50` — configs `sweep_0p10/0p20.toml` predate this experiment
   (their frozen-subset step-p8 runs exist: `../rikyu_replay_sweep/results/mt_0p10/0p15/0p20.csv`);
   `sweep_0p30/0p50.toml` are new copies differing only in `amount`. Outputs
   `artifacts/replay_sweep_epoch_m150/replay_0p*_epoch_m150/` → `results/mt_0p*_epoch_m150.csv`.
2. **Baseline: no replay + one full joint retrain** — arm A `configs/noreplay_full24.toml`
   (`noreplay_h200.sbatch`): sequential 24 tasks with replay disabled via `interval = 999`,
   patience 24 / max 150 (the pure-forgetting trajectory). Arm B
   `configs/joint_retrain_full24.toml` (`joint_retrain_h200.sbatch`): `fm finetune` with
   `freeze_encoder = false` on all 24 heads at full data from arm A's `final_model.pt` — i.e.
   plain multi-task training, ≤150 epochs, patience 24. `finetune_summary.json` carries
   before (= arm A final) and after metrics. No `--resume` in finetune: must fit in 24 h.

Smoke first (`smoke_ratio_baseline.sbatch`: all three paths at `--sample 400`, 1 epoch), then
relay-submitted under the 72 node-hour quota, heaviest first (0p50 ≈ 2× the n2500 replay volume).

Arm B convergence probe (2026-08-08 follow-up): the 150-epoch joint retrain ran to its cap
without early-stopping (epochs_run = 150), so caps 200/250/300 are added — same checkpoint,
same config, `--epochs` override via `EP` in `joint_retrain_h200.sbatch` (outputs
`joint_retrain_m{200,250,300}/`; the original 150-epoch run stays in `joint_retrain/`).
Patience 24 stays on: an early stop below the cap, or flat final metrics across caps, is the
convergence evidence. The per-epoch val-loss curves (`logs/finetune/*/metrics.csv`) back this
graphically. All baseline cases (arm A, arm B ×4 caps) are part of the final report/PPT
comparison alongside the ratio and fixed-n arms.

## Outcome — extension (2026-08-09)

All 9 extension runs complete (ratio 0p10/0p20/0p30/0p50 + no-replay + joint retrain ×4 caps).
On the mean, ratio joins the fixed-count plateau (0.639–0.653 — no free lunch at matched cost),
but allocation is mirror-imaged: ratio 0.3/0.5 cuts the big-task deficit to ~0.025 (half the old
"multi-task cost" was recoverable forgetting) while starving small tasks (r0.5: 0.085 vs 0.002 at
n2500) ⇒ hybrid `amount = max(floor, r·N)` via `replay.per_task`. Baselines: no replay collapses
to 4% task retention (mean R² −33/−88 by protocol); one full joint retrain at the end CONVERGES
(early stop @214; caps 250≡300) at 0.584 — below every continual-replay arm. Upstream fix
required and merged: PR #36 (interval>1 crash on learned kernel-regression heads). Full report:
`results/REPORT_20260809.md` + `.pptx` (13 slides, builder shared with the 08-02 deck).

## Outcome (2026-08-02)

Extension 2026-08-02: the m150 arm is being completed to all 7 n (n100/n200/n500 submitted on
the same H200 partition as the p24 lights, so the m150-vs-p24 comparison at matched n stays
same-hardware). Report to be refreshed when they land.

All 25 runs complete. Headline (23-task mean final R², vs step baseline): epoch resampling
+0.022…+0.126 at every n (peak @n200); with patience 24 (⇒ full 100 epochs) the n-dependence
nearly flattens (n100-p24 0.592 ≈ n2500-step 0.600); max_epochs 150 adds only +0.009 mean —
the epoch budget saturates near 100. Full report: `results/REPORT_20260802.md` + `.pptx`
(build with `build_report_pptx.py`). Open control: step-p24 (not run).

## Results & provenance (not in git, rsync policy)

- Raw run outputs: `artifacts/replay_sweep_epoch/replay_n*_epoch/` on ism-gpu-a100, rsync'd
  back to the same path locally (metrics at `<dir>/training/metrics_table.csv`).
- Collected per-run metric tables: `results/mt_n*_epoch.csv` (same format as the step sweep's
  `mt_n*.csv`, one copy of each run's `metrics_table.csv`).
- Analysis outputs land in `analysis/` (step-vs-epoch versions of `per_task_saturation` and
  `replay_trajectories`, plus the comparison figures for the report/PPT).
