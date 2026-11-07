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
  rewrapped one line (no code change; the committed file hashes `c69623bb…`).
- Split manifest: `split_manifest_20261016.csv` (same file for every stage).
- Recipe: the default hybrid replay (`resample = "epoch"`, `amount = max(1500, 0.3 N)`), then one
  full-model consolidation of all heads. Model / optimiser values are the ones adopted by
  `rikyu_hparam_tuning_v2` (latent 384, encoder LR 2e-3, scheduler min LR 1e-5), early-stopping patience 24,
  150 epochs per step; `class_weights = "none"` on every classification head.
- Container `rikyu-1.0.0` (master 19a5c04).

## Files

| Path | What |
|---|---|
| `scripts/make_configs.py` | writes every config below from the task index; `--set v1` (41 tasks) or `--set full` (64) |
| `configs/pretrain_v1.toml`, `consolidate_v1.toml` | training; **no scalers** (see below) |
| `configs/export_v1.toml` | the same catalog with each task's scaler / label encoder, for `fm export` |
| `configs/members_v1.txt` | member id → `task_order_seed` (20261108 + i) |
| `scripts/check_transforms.py` | gate: each configured transform reproduces its training column, and the export head spec builds |
| `scripts/suite.sbatch` | PACK members per GPU; pretrain (`--resume`) → consolidate, DONE markers |
| `scripts/export.sbatch` | `fm export` of all members into one bundle, then `load_suite` / `predict` / `fm predict --pretrained` on it |
| `scripts/submit.sh` | `smoke`, `suite`, `export` |

Outputs (untracked) on RIKYU under `/data1/rkp00067/rku00225/fm/release_suite_v1/{smoke,members}/`; the
bundle is `members/bundle/`.

### Why training runs without scalers

With a scaler configured, the per-step evaluation is computed in physical units. For 14 of the 41 tasks the
Yeo-Johnson λ is negative (or above 2), so the inverse is only defined below (above) a bound, and the
training maximum lies on that bound (e.g. `total_magnetization`: bound z = 2.26, data max z = 2.26). A head
that predicts past it returns NaN, and the evaluation stops the run. Found in the local smoke on
2026-11-08. Scalers never touch the weights, so training uses the normalised space every earlier campaign
reported, and only the export config names them.

The same NaN reaches `PretrainedModel.predict` for out-of-range predictions. That is a library fix
(1.0.1) needed before the first `fm publish`; it does not affect training.

## Gate results

- `check_transforms.py configs/export_v1.toml`: 41/41 pass (max error 4.6e-13; classification labels exact).
- Local CPU smoke (2 members, `--sample 400`, 1 epoch): pretrain → consolidate → export (outputs identical
  on reload) → `load_suite` / `predict` / `predict_curve`: pass; member task orders differ; 8 MB per member.

## Execution log

- 2026-11-08: data folders built; configs generated; local smoke passed.

## Plan: the remaining 23 tasks (not started)

`make_configs.py --set full` writes the 64-task configs (KKR-CPA HEA 7, QMOF 4, Catalysis-Hub 12 added to
the 41). Before launching:

1. Run `check_transforms.py configs/export_full.toml`. The HEA / QMOF / Catalysis-Hub raw values sit in
   separate `_raw` tables, so their data check needs a row alignment that is not written yet.
2. Decide where the new tasks sit in the order: the HEA tables have 51k training rows per task (7 tasks on
   one composition set of equiatomic quaternaries), Catalysis-Hub 12 tasks have 135–1,063.
3. Smoke, then measure one member's wall clock before choosing PACK: 64 steps with replay growing every step
   is roughly 2.5× the 41-step cost.
4. Publish as a new release (new tag); the v1 bundle stays as it is.
