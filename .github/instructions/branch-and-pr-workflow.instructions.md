---
description: 'Use when starting work on a change, bug fix, refactor, upgrade, or any modification request. Covers when to create a new branch, when to stay on the current branch, when to open a pull request before continuing, and how to split a complex refactor into a sequence of independently verifiable PRs.'
name: 'Branch and Pull Request Workflow'
applyTo: '**'
---

# Branch and Pull Request Workflow

When the user asks for a code change, feature, fix, refactor, upgrade, or any modification, decide where to do the work by checking the current branch state **before** editing files.

## Decision Order

Check these conditions in order and stop at the first match:

1. **User explicitly opts out.** If the user says they do not want a new branch or PR (for example "just edit on main", "no PR needed", "直接改", "不用开 PR"), honor that and work wherever they indicate.

2. **Current branch already has an open PR.** Continue working on the current branch. Do not create a new branch. Push follow-up commits to the same branch so they land on the existing PR.

3. **Current branch is ahead of the default branch but has no PR yet.** Open a PR for the current branch against the default branch first, then continue working on the same branch. New commits will land on the newly opened PR.

4. **Default case (current branch is the default branch, or is in sync with it, or none of the above apply).** Create a new branch off the default branch, make the change there, and open a PR when the change is ready to share.

## Practical Rules

- Determine the current branch and its PR status before making code changes, not after. Use the repository's PR metadata (for example `currentActivePullRequest`) or `git` commands to check.
- Name new branches descriptively for the change (for example `feat/...`, `fix/...`, `docs/...`, `refactor/...`).
- Do not commit directly to the default branch unless case 1 applies.
- When case 3 applies, do not silently keep committing without a PR; open the PR first so the work is reviewable.
- When case 2 applies, do not open a second PR for the same branch.
- If it is unclear whether an existing branch is "ahead but unpushed" versus "already has a PR", prefer checking remote state before deciding.

## Splitting a Complex Refactor

Treat a change as a **complex refactor** when any of these hold:

- it touches more than one layer (data pipeline, API, frontend, documentation);
- it moves or renames files that other files reference by literal path or literal string;
- its diff would mix mechanical moves with behavior changes;
- it cannot be reviewed in one sitting;
- it would leave the repository failing its own checks partway through.

For these, decide the PR sequence **before editing any file**. Do not open one branch, start changing things, and look for the seams afterward — by then the diff is already entangled.

### Plan first, in the repository

Record the plan under `docs/refactor/<refactor-slug>/` before implementation:

- `00-overview.md` — why the refactor exists, the decision with alternatives and consequences, explicit non-goals, and the ordered list of planned PRs.
- one plan file per planned PR, each with **Goal**, **Scope**, **Non-goals**, and **Acceptance** sections. Follow the plan-filename convention in `repository-doc-boundaries.instructions.md`; documentation validation enforces it.

If the user requests a complex refactor without a plan, propose the split and get agreement before writing code.

### Requirements for every PR in the sequence

1. **One stated purpose.** The title names a single outcome. If stating the goal requires "and", it is probably two PRs.
2. **Independently verifiable.** Each PR lists its own acceptance commands in its plan file and leaves the repository's test suite and validation checks passing when it merges. A PR that only turns green after a _later_ PR is not independently verifiable — merge it with that PR, or reorder the sequence.
3. **Independently reviewable.** Someone reading only this PR and its plan file should understand what changed and why, without reading the rest of the sequence. "Part 2 of the refactor" is not a description.
4. **Independently revertible.** Reverting one PR must not break the PRs that landed before it.
5. **No mixing of mechanical and semantic change.** A pure move/rename is one PR; a behavior change is another. But when a move breaks literal-path or literal-string references, update those references **in the same PR that moves the files** — never split a move from the reference updates it invalidates.

### Ordering

Order the sequence so the repository stays green at every step. Prefer **introduce the new form → migrate the call sites → remove the old form**; never remove first.

Per `in-branch-api-compat.instructions.md`, do not add compatibility shims, aliases, or adapter layers merely to keep an intermediate PR green. If a step cannot be made green without a shim, the sequence is ordered wrong — reorder it.

### Sequential review and squash-merge gate

For every ordered multi-PR plan, process one PR at a time using this fixed sequence:

**review → address feedback → squash merge → begin the next PR**

The current PR is a hard gate for every later PR in the plan. Do not create the next implementation branch, start its changes in another worktree or session, or otherwise develop later stages in parallel while the current PR is open.

Before the gate may advance:

1. Finish the current PR's stated scope and run its acceptance checks.
2. Push the complete change and wait for required CI and configured human or automated review. Passing CI alone does not complete the review gate.
3. Inspect every review surface: submitted reviews, inline review threads, and general PR comments.
4. Address every actionable comment with a code or documentation change and regression coverage where appropriate. If a suggestion should not be implemented, reply with a concrete technical reason instead of silently ignoring it.
5. Push the follow-up commits, wait for the checks on the latest head commit, reply to each handled thread, and resolve it. Recheck that no new or unresolved review thread remains.
6. Squash-merge the PR. Confirm the remote PR state is `MERGED` and record the resulting merge commit; a local worktree warning is not evidence that the remote merge failed.
7. Fetch the merged default branch, then create the next PR's branch or worktree from that updated default branch. Never base the next stage on the unmerged predecessor branch.

Keep every later plan item pending until the preceding PR has passed this complete gate. If review requests changes or the latest checks fail, remain on the current PR and fix it; do not advance the sequence. A separately submitted refactor-plan PR is subject to the same gate before PR1 starts.

## Anti-patterns

- Creating a new branch when the user is already on a branch with an open PR for the same piece of work.
- Committing to the default branch for a non-trivial change without asking.
- Pushing follow-up commits to a feature branch that is ahead of main without ever opening a PR for it.
- Opening a new PR for in-progress follow-up work that belongs on an already-open PR.
- Starting a multi-layer refactor and only discovering the PR boundaries after the diff is already entangled.
- Landing a rename in one PR and its reference updates in another, leaving the default branch red in between.
- Bundling a mechanical move with a behavior change so a reviewer cannot tell which lines actually changed meaning.
- Splitting by file or by commit count rather than by verifiable outcome, producing PRs that individually mean nothing.
- Starting, branching or implementing a later planned PR before its predecessor is remotely confirmed as merged.
- Treating green CI as a substitute for waiting for and auditing review feedback.
- Merging while actionable comments or unresolved review threads remain.
- Advancing from a local branch state without confirming the remote squash merge and updating from the default branch.
