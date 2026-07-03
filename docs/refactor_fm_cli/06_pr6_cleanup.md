# PR6 — Cleanup: delete legacy code, configs, docs; update project docs

> Branch: `chore/fm-cli-cleanup`. Depends on: PR2–PR5 merged.
>
> **Delete list verified against `master` @ 532a4aa.** All named files exist. Two orphans the
> original draft missed (`samples/cli_examples/`, `samples/configs/`) and the omegaconf/jsonargparse
> dep reality are folded in below.
> Before starting, run the full functional-coverage checklist in `00_OVERVIEW.md` §6 and the
> end-to-end acceptance chain in §7 — this PR must not land if any item fails.

## A. Delete — `src/foundation_model/scripts/`

| File | Also delete |
|---|---|
| `train.py` | — |
| `dynamic_task_suite.py` | `dynamic_task_suite_test.py` |
| `multi_task_progressive_clf.py` | — |
| `continual_rehearsal_demo.py` | `continual_rehearsal_demo_test.py` |
| `continual_rehearsal_full.py` | `continual_rehearsal_full_test.py` |
| `continual_rehearsal_common.py` | `continual_rehearsal_common_test.py` (cases already migrated to `workflows/plots_test.py` in PR2 — verify before deleting) |
| `finetune_inverse_heads.py` | `finetune_inverse_heads_test.py` |
| `paper_inverse_comparison.py` | `paper_inverse_comparison_test.py` |
| `paper_inverse_3scenarios.py` | — |
| `paper_inverse_trajectory.py` | `paper_inverse_trajectory_test.py` (migrated in PR4 — verify) |
| `eval_inverse_methods.py` | — |
| `callbacks/prediction_writer.py` | delete `callbacks/` entirely if nothing else remains |

If `scripts/` ends up empty (check `__init__.py`), remove the package and any references.
Before each deletion, grep the repo for imports of the module (`grep -r "scripts.<name>"`) —
notebooks may import them; notebook references get a one-line note in the PR description, not
a code fix (notebooks are non-critical per AGENTS.md).

## B. Delete — repo root & samples & docs

Root shell wrappers (all 6 exist):
- `run_dynamic_task_suite.sh`, `run_dynamic_task_suite_radonpy.sh`,
  `run_dynamic_task_suite_retrain_source.sh`, `run_progressive_clf.sh`,
  `run_continual_rehearsal_demo.sh`, `run_continual_rehearsal_full.sh`
- `REFACTOR_composition_datamodule_PLAN.md` (stale plan). **`compare_input_vs_latent.py` — KEEP**
  (verified: it imports only `foundation_model.models.flexible_multi_task_model` +
  `foundation_model.models.model_config`, none of the deleted script modules, so it is standalone).

`samples/` — delete all legacy configs:
- `dynamic_task_suite_config*.toml` (4: base, `_free`, `_radonpy`, `_smoke`),
  `progressive_clf_config*.toml` (2), `continual_rehearsal_demo_config*.toml` (3: base, `_smoke`,
  `_inverse_baseline`), `continual_rehearsal_full_config.toml`, `example_config.yaml`

`samples/` — **two orphaned subdirectories the original draft missed** (both reference the deleted
`fm-trainer` / LightningCLI YAML and must be deleted or rewritten):
- `samples/cli_examples/` — `01_basic_run.sh` (calls `fm-trainer fit/test/predict` 5×),
  `02_config_override.sh`, `03_scaling_law_xargs.sh`. Delete, or rewrite against `fm` if the
  worked examples are still wanted (owner's call — default: delete, README covers usage).
- `samples/configs/test_t_depends_on_mac_pro/` — `fit_config.yaml`, `predict_config.yaml`,
  `test_config.yaml` (LightningCLI YAML configs). Delete the whole directory.

`samples/` — final set (created across PR2–5; verify all exist and run):
- `pretrain.toml` (full 24-task formal run — port values from `continual_rehearsal_full_config.toml`)
- `pretrain_smoke.toml`, `finetune.toml`, `finetune_smoke.toml`,
  `inverse.toml` (3 scenarios, default paths), `inverse_smoke.toml`,
  `predict.toml`, `predict_smoke.toml`

`docs/` — delete stale (content superseded by the new workflows or PR history):
- `continual_rehearsal_full_PLAN.md`, `trajectory_integration.md`,
  `inverse_design_extension_notes.md`
- Review individually and keep if still accurate as *results/method* docs (not script docs):
  `inverse_design_algorithms.md`, `qc_inverse_design_summary.md` — update command references
  to `fm inverse` if kept.
- `docs/refactor_fm_cli/` (this plan) — keep until the refactor is fully merged, then the
  owner decides; do not delete in this PR.

## C. `pyproject.toml`

```toml
[project.scripts]
fm = "foundation_model.cli.main:main"      # added in PR2 — now the ONLY entry
# DELETE: fm-trainer, fm-pretrain-suite, fm-progressive-clf
```

Dependency cleanup — the original draft's "remove omegaconf / jsonargparse" is only half right
(verified):
- **`omegaconf`** is a *direct* dep (`pyproject.toml:14`, `omegaconf>=2.3.0`). Its only real code
  use after this refactor is `scripts/train.py:39` (`parser_mode="omegaconf"`, deleted here) —
  BUT `notebooks/interactive_model_test.ipynb` still does a real `from omegaconf import OmegaConf`.
  Notebooks are non-critical per `AGENTS.md`, so either (a) drop the dep and note the notebook
  breakage in the PR, or (b) keep the dep. Recommend (a) + a one-line PR note. Do **not** claim a
  clean removal.
- **`jsonargparse`** is **not** a declared dependency — it is transitive via
  `lightning[pytorch-extra]` (`pyproject.toml:10`) and imported nowhere in `src/`. There is
  nothing to remove; "remove jsonargparse from deps" is a no-op. If you want it gone from the
  lockfile, drop the `[pytorch-extra]` extra (`lightning[pytorch-extra]` → `lightning`) and
  re-run `uv sync` — but verify nothing else in the extra is needed first.

## D. Update project docs

- `AGENTS.md`: rewrite the **Entry Points** section (three scripts → one `fm` command with
  four subcommands + config conventions); update Project Structure (add `cli/`, `workflows/`);
  update any test-layout references.
- `README.md`: user-facing usage — replace `fm-trainer`/suite examples with `fm pretrain` /
  `fm finetune` / `fm inverse` / `fm predict` quick-start snippets pointing at `samples/`.
  Keep README user-focused per repository-doc-boundaries rules.
- `ARCHITECTURE.md`: add a short "Workflows & CLI" section describing the module boundaries
  (`cli` thin dispatch / `task_catalog` / `recording` provenance / engines).
- `.github/instructions/`: grep for references to deleted commands (e.g. rikyu doc mentions
  `fm-trainer`, `fm-pretrain-suite`, `fm-progressive-clf` symlinks) and update the symlink
  recipe to `fm`.

Do NOT touch `artifacts/*/README.md` — they are historical run records; their stale commands
stay as-is.

## E. Acceptance

1. `grep -rn "fm-trainer\|fm-pretrain-suite\|fm-progressive-clf\|continual_rehearsal\|paper_inverse\|finetune_inverse_heads\|dynamic_task_suite\|eval_inverse_methods"`
   over `src/ samples/ docs/ README.md AGENTS.md ARCHITECTURE.md pyproject.toml *.sh` returns
   only: historical `artifacts/` files, `docs/refactor_fm_cli/`, and git history.
2. `uv sync --frozen --all-groups` then full `pytest`, `ruff format --check`, `ruff check`,
   `mypy src` green.
3. The §7 end-to-end smoke chain from `00_OVERVIEW.md` passes with the final sample configs.
4. `fm --help` lists exactly four subcommands; the three legacy commands are gone from a fresh
   `uv sync` environment.
