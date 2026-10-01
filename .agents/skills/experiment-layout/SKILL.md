---
name: experiment-layout
description: The recursive self-contained folder convention for experiments and experiment groups (configs, scripts, results, analysis, artifacts per experiment; shared pieces hoisted to the group level). Use when creating a new experiment, organizing runs, or deciding where files go.
---

# Experiment folder layout — recursive & self-contained

One experiment = one folder that contains **everything the experiment owns**. Nothing it needs
(beyond `src/` and the shared `data/` pool) lives outside it; nothing inside it is shared with
siblings. Applies to every NEW experiment (existing pre-convention folders are left as-is —
do not retrofit).

## Single experiment

```
experiments/<experiment>/
  README.md          # question, design, execution log, outcome — updated as the experiment runs
  configs/           # this experiment's TOML configs                          [tracked]
  scripts/           # launchers, job scripts, workers, collectors             [tracked]
  analysis/          # analysis code [tracked]; generated figures [untracked]
  results/           # collected metrics tables, summaries, reports, decks     [untracked]
  artifacts/         # raw run outputs & intermediate results, one subdir per run  [untracked]
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

- Tracked: README, configs, scripts, analysis CODE. Untracked (gitignore): results/, artifacts/,
  generated csv/png/pptx — they travel by rsync between machines (mirror to the SAME relative
  path locally; exclude checkpoints on routine mirrors).
- Every run writes `run_provenance.json` (resolved config + git commit); launchers record the
  commit at launch. Execution events (submissions, TIMEOUTs, resumes) go into the README
  execution log as they happen — the README is the experiment's lab notebook.
