# Release suite v1 (2026-11-08)

The suite of pretrained models published as the first `foundation-model` release
(`docs/refactor/pretrained-release/00-overview.md`). Ten members that differ **only in pretraining task
order**: same data, same split manifest, same training seed, same hyper-parameters.

## Decisions (owner, 2026-11-08)

| Item | Choice |
|---|---|
| Tasks | the 41 tasks used in earlier training (`trained_in_campaign_v2` in the task index): Materials Project 27 (incl. DOS), HYPOD-X `material_type`, Starrydata 6 curves, NEMAD 5, phonix-db 2 |
| Members | 10 (`m000`–`m009`) |
| Order | scalar + classification heads shuffled first; the 7 kernel-regression curves shuffled in a final block |
| Later | the remaining 23 tasks (KKR-CPA HEA 7, QMOF 4, Catalysis-Hub 12): plan below, not started |

## Inputs

- Data: `data/fm_task_datasets_20261108_release/`, built by `data/scripts/build_release_datasets_20261108.py`
  from the post-#99 folder `data/fm_task_datasets_20261016/` (composition-group splits). It drops the 31
  Materials Project ids whose licence tag could not be read (65 table rows: 31 properties, 3 DOS curves, 31
  `material_type` rows), and adds class names for `is_metal` / `is_gap_direct`.
  Its `provenance_20261108.json` records script sha256 `381bb7c8…`, the file before `ruff format`
  rewrapped one line. Review of #113 later added one branch (an empty exclusion table when no id is
  dropped) that this build never reached, so the committed file hashes differently; the output is the same.
- Split manifest: `split_manifest_20261016.csv` (same file for every stage).
- Recipe: the default hybrid replay (`resample = "epoch"`, `amount = max(1500, 0.3 N)`), then one
  full-model consolidation of all heads. Model / optimiser values are the ones adopted by
  `rikyu_hparam_tuning_v2` (latent 384, encoder LR 2e-3, scheduler min LR 1e-5), early-stopping patience 24,
  150 epochs per step; `class_weights = "none"` on every classification head.
- Container `rikyu-1.0.0` (master 19a5c04).

## Files

| Path | What |
|---|---|
| `scripts/make_configs.py` | writes every config below from the task index; `--set v1` (41 tasks); `--set full` (64) stops until its scaler mappings exist |
| `configs/pretrain_v1.toml`, `consolidate_v1.toml` | training; **no scalers** (see below) |
| `configs/export_v1.toml` | the same catalog with each task's scaler / label encoder, for `fm export` |
| `configs/members_v1.txt` | member id → `task_order_seed` (20261108 + i) |
| `scripts/check_transforms.py` | gate: each configured transform reproduces its training column, and the export head spec builds |
| `scripts/suite.sbatch` | PACK members per GPU; pretrain (`--resume`) → consolidate, DONE markers |
| `scripts/export.sbatch` | `fm export` of all members into one bundle, then `load_suite` / `predict` / `fm predict --pretrained` on it |
| `scripts/submit.sh` | `smoke`, `suite`, `export` |

Outputs (untracked) on RIKYU under `$OUTBASE/{smoke,members}/` (default `$RIKYU_GROUP_DIR/fm/release_suite_v1`); the
published bundle is `members/bundle_v101/` (exported with 1.0.1; the 1.0.0 export was cancelled).

### Why training runs without scalers

With a scaler configured, the per-step evaluation is computed in physical units. For 15 of the 41 tasks the
Yeo-Johnson λ is negative (or above 2), so the inverse is only defined below (above) a bound, and for 14 the
training maximum lies on that bound (e.g. `total_magnetization`: bound z = 2.26, data max z = 2.26). A head
that predicts past it returns NaN, and the evaluation stops the run. Found in the local smoke on
2026-11-08. Scalers never touch the weights, so training uses the normalised space every earlier campaign
reported, and only the export config names them.

The same NaN reached `PretrainedModel.predict` for out-of-range predictions. Fixed in 1.0.1 (PR #111):
export adds a `guard` clip at the training range on each bounded side (bundle schema v2), so those predictions
saturate with a `SaturationWarning`. The published bundle is exported with the 1.0.1 container.

## Gate results

- `check_transforms.py configs/export_v1.toml`: 41/41 pass (max error 4.6e-13; classification labels exact).
- Local CPU smoke (2 members, `--sample 400`, 1 epoch): pretrain → consolidate → export (outputs identical
  on reload) → `load_suite` / `predict` / `predict_curve`: pass; member task orders differ; 8 MB per member.

## Execution log

- 2026-11-08: data folders built; configs generated; local smoke passed.
- 2026-11-08: RIKYU smoke (job 297096, export check 297119) passed on container 1.0.0.
- 2026-11-08: suite submitted: training 297120 (2 GPUs × 5 members), git dbcfc61, container 1.0.0.
- 2026-11-08: 1.0.1 merged (#111); the 1.0.0 export job 297121 cancelled. Smoke export check on container
  1.0.1 (297514) passed: schema 2, 15 guarded heads, no NaN. Final export 297515 (container 1.0.1,
  `members/bundle_v101`) waits on 297120.
- 2026-11-08: training 297120 completed, 00:58 → 20:06 JST (array tasks 18 h 39 min and 19 h 08 min).
  Export 297515 passed in 1 min (10 members, no NaN, 15 guarded heads). Compute: 37.8 GPU-h on GB200,
  about 11,400 JPY before tax at 300 JPY/GPU-h. The 4–5 day estimate was far too high: the seven curve
  steps took about 1–2 h each, not the much longer times assumed.
- 2026-11-08: consolidation (fm finetune, full model) early-stopped at 26–46 epochs and did not help.
  Mean test primary metric over 10 members, before → after: regression R² 0.666 → 0.664 (per member
  −0.009 … +0.004; 6 lower, 2 higher, 2 equal), classification 0.802 → 0.800. The released bundle holds
  the consolidated models, as planned.
- 2026-11-08: published to `TsumiNa/fm` (gated, tag v1.0.0). The first upload was refused: the Hub
  rejects metadata keys containing ".", and the licence checkbox key held `cc-by-nc-sa-4.0`. The card's
  `pip install 'foundation-model[hub]'` would also install an unrelated PyPI project. Both fixed in
  1.0.2 (PR #112); card regenerated from that branch, v1.0.0 re-tagged on the corrected card. Verified:
  a fresh-cache download matches the local bundle bit for bit (default member and `#m004`), and a clean
  venv installed from GitHub predicts scalars and curves from `TsumiNa/fm@v1.0.0`.

## Plan: the remaining 23 tasks (not started)

The 64-task set adds KKR-CPA HEA 7, QMOF 4 and Catalysis-Hub 12 to the 41. Before launching:

1. Add the 23 tasks' scalers / label encoders to `SCALERS` in `make_configs.py`: until then
   `--set full` stops without writing configs. Then run `check_transforms.py configs/export_full.toml`.
   The HEA / QMOF / Catalysis-Hub raw values sit in separate `_raw` tables, so their data check needs a
   row alignment that is not written yet.
2. Decide where the new tasks sit in the order: the HEA tables have 51k training rows per task (7 tasks on
   one composition set of equiatomic quaternaries), Catalysis-Hub 12 tasks have 135–1,063.
3. Smoke, then measure one member's wall clock before choosing PACK: 64 steps with replay growing every step
   is roughly 2.5× the 41-step cost.
4. Publish as a new release (new tag); the v1 bundle stays as it is.
