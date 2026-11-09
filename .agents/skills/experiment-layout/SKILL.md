---
name: experiment-layout
description: The recursive self-contained folder convention for experiments and experiment groups (configs, scripts, results, analysis, artifacts per experiment; shared pieces hoisted to the group level). Use when creating a new experiment, organizing runs, or deciding where files go.
---

# Experiment folder layout — recursive & self-contained

One experiment = one folder that contains **everything the experiment owns**. Nothing it needs
(beyond `src/` and the shared `data/` pool) lives outside it; nothing inside it is shared with
siblings. Applies to every NEW experiment (existing pre-convention folders are left as-is —
do not retrofit).

There is **no repository-level `artifacts/` or `results/` folder**: every output lands inside the experiment
that produced it (the old root `artifacts/` was dissolved on 2026-11-09). Folder names carry the start date,
`<topic>_<YYYYMMDD>`.

## The index: `experiments/README.md`

One table row per experiment: folder, date, question, outcome, where the data lives (local subfolders and any
remote path), status. It is the entry point for finding data again and for cross-experiment analysis, and the
only file under `experiments/` that is tracked in git.

- **Add the row when the experiment is created** (question, folder, status `planned` / `running`); update it
  when the outcome is known, when raw data moves (mirror, remote only, deleted), or when it is superseded.
- Outcome: one sentence with the key numbers, the same as the experiment README's outcome; never a new claim.
- Deleting or moving an experiment's data is recorded in its row (what is gone, what remains, the date).

## Single experiment

```
experiments/<experiment>/   # nothing here is in git (see "Tracking & provenance")
  README.md          # question, design, execution log, outcome — updated as the experiment runs
  configs/           # this experiment's TOML configs
  scripts/           # launchers, job scripts, workers, collectors
  analysis/          # analysis code; generated figures
  results/           # collected metrics tables, summaries, reports, decks
  artifacts/         # raw run outputs & intermediate results, one subdir per run
    <run>/           #   the workflow's own layout (training/stepNN_*/, checkpoints, logs,
                     #   run_provenance.json) lands here unmodified
```

- A category that grows large subdivides INSIDE itself: `configs/sweep/`, `scripts/rccs/`,
  `analysis/figures/<topic>/`, `artifacts/<arm>/<run>/` — never by spilling into a sibling.
- Configs are referenced by paths inside the experiment (`--config experiments/<exp>/configs/x.toml`);
  job scripts parameterize the run, they don't hardcode another experiment's files.
- Reusing an artifact from ANOTHER experiment (a checkpoint, a baseline CSV) is allowed but must
  be an explicit input recorded in README + the consuming config/script — never a hidden path.

## Experiment group (and groups of groups — recurse)

When several experiments form one investigation, wrap them:

```
experiments/<group>/
  README.md          # group question + index of members + combined outcome
  configs/ scripts/  # ONLY pieces shared by ≥2 members (base configs, common workers)
  analysis/ results/ # ONLY cross-experiment aggregation (comparison figures, joint reports)
  <experiment-a>/    # each member keeps the FULL single-experiment layout above
  <experiment-b>/
```

- Hoist shared files to the NEAREST common ancestor, never higher; a member must stay runnable
  standalone (its configs may extend a group base config by relative path — that dependency is
  part of the group contract, documented in the member README).
- Member-level results stay in the member; the group level holds only the synthesis.

## Tracking & provenance (repo policy)

- **Nothing under `experiments/` is in git except the index `experiments/README.md`** (the folder
  is git-ignored since 2026-11-08): README, configs, scripts, analysis code and results all travel
  by rsync between machines — mirror to the SAME relative path, exclude checkpoints on routine mirrors. A cluster run therefore
  needs the experiment folder rsynced there, not just a `git pull`.
- What an experiment needs from the code is in `src/` (tracked). If a script turns out to be
  reusable, move it into `src/` or `scripts/` (tracked) instead of pointing other code at
  `experiments/`.
- Every run writes `run_provenance.json` (resolved config + git commit); launchers record the
  commit at launch. Execution events (submissions, TIMEOUTs, resumes) go into the README
  execution log as they happen — the README is the experiment's lab notebook.
