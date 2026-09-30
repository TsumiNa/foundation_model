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

3. **Current branch is ahead of the default branch but has no PR yet.** Continue working on the same branch. When remote PR work is authorized, open a PR for the branch against the default branch before adding further related commits, so the work is reviewable; new commits will land on that PR.

4. **Default case (current branch is the default branch, or is in sync with it, or none of the above apply).** Create a new branch off the default branch, make the change there, and open a PR when the change is ready to share.

Decide the local work location separately from remote actions. A request such as "no PR needed" disables remote PR work but does not by itself select a local branch or authorize commits to the default branch.

## Practical Rules

- Determine the current branch and its PR status before making code changes, not after. Use the repository's PR metadata (for example `currentActivePullRequest`) or `git` commands to check.
- Name new branches descriptively for the change (for example `feat/...`, `fix/...`, `docs/...`, `refactor/...`).
- Do not commit directly to the default branch unless case 1 applies.
- When case 3 applies, do not silently keep committing without a PR; open the PR first so the work is reviewable.
- When case 2 applies, do not open a second PR for the same branch.
- If it is unclear whether an existing branch is "ahead but unpushed" versus "already has a PR", prefer checking remote state before deciding.
- Remote actions are opt-in: push commits, create or update a PR, request review, or merge only when the user requested or explicitly agreed to that remote work. When authorized, update the existing PR for the branch rather than opening a duplicate.
- Do not overwrite, discard, commit, or reformat unrelated user changes. If existing changes overlap the requested files and cannot be preserved safely, stop and ask the user how to proceed.

## Review, CI Attribution, and Merge Eligibility

Judge a PR by the behavior and failures attributable to its own diff, not by whether every Action in the repository's history is green.

### No PR checks

- If no PR checks are triggered, treat the PR as having no CI blocker.
- Do not add, modify, or repair CI merely to manufacture a passing check for the current PR.
- "No checks" is not "CI passed". In the PR description or handoff, state it as "no applicable or triggered PR checks".

### Pre-existing failures

- A failing check is **non-blocking** when there is concrete evidence that the same failure existed on the PR's base commit or before the PR was opened, and the current diff does not touch the related code, content, tests, workflow, configuration, or failure path.
- Record that evidence (run link, commit, or log excerpt) in the PR or handoff. Never describe a non-blocking failure as a passing check.
- Do not expand the current PR's scope because of an unrelated pre-existing failure.

### Blocking failures

A failure blocks merging when any of these hold:

- the PR introduced the failure;
- the PR worsened an existing failure;
- the current diff changed code, content, tests, workflow, configuration, or an execution path that could reasonably affect the failure;
- the failure's independence from the PR cannot be established.

When attribution is unclear, investigate before deciding; do not treat an unexplained failure as passing.

### No cross-PR fixes

- Do not fold fixes for unrelated problems left by other commits or PRs into the current PR to make its checks pass.
- Report unrelated pre-existing problems separately. Fix them only in a separate task or PR, and only when the user authorizes it.
- When fixing a pre-existing failure is the stated purpose of the current PR, that failure is in scope and the PR must include proportionate evidence that the fix works.

### Review

- Review against the latest head commit and the final diff.
- Address every actionable finding that is correct and in scope.
- For findings that are incorrect or belong to another scope, reply with a concise rationale; do not expand the PR to silence a comment.
- Resolve a review thread only after its disposition is clear: the change was made, or the reason for not making it was stated.
- If a requested automated reviewer is unavailable or has exhausted its quota, perform an independent review of the final diff and record the result. Reviewer unavailability alone is not a merge blocker.

### Before an authorized merge

Re-confirm, immediately before merging:

1. The PR head SHA has not changed since the final checks and review.
2. The final diff still has one coherent scope.
3. The local verification that `implementation-and-tests.instructions.md` requires for this change type passed, or anything that could not be run is stated accurately.
4. Every reported PR check is classified as passing, a documented non-blocking pre-existing failure, or blocking.
5. All actionable review findings are addressed.
6. No unresolved review thread still requires a change.
7. The merge method is the one the user authorized.
8. Branch protection and repository permissions are not bypassed to force a merge; report a configuration-level block instead.

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
2. Push the complete change and wait for required CI and configured human or automated review. Passing CI alone does not complete the review gate. Classify every reported check under "Review, CI Attribution, and Merge Eligibility" above; a documented non-blocking pre-existing failure does not hold the gate, an unattributed failure does.
3. Inspect every review surface: submitted reviews, inline review threads, and general PR comments.
4. Address every actionable comment with a code or documentation change and regression coverage where appropriate. If a suggestion should not be implemented, reply with a concrete technical reason instead of silently ignoring it.
5. Push the follow-up commits, wait for the checks on the latest head commit, reply to each handled thread, and resolve it. Recheck that no new or unresolved review thread remains.
6. Squash-merge the PR (or use the merge method the user authorized), after the pre-merge re-confirmation above. Confirm the remote PR state is `MERGED` and record the resulting merge commit; a local worktree warning is not evidence that the remote merge failed.
7. Fetch the merged default branch, then create the next PR's branch or worktree from that updated default branch. Never base the next stage on the unmerged predecessor branch.

Keep every later plan item pending until the preceding PR has passed this complete gate. If review requests changes or the latest checks fail, remain on the current PR and fix it; do not advance the sequence. A separately submitted refactor-plan PR is subject to the same gate before PR1 starts.

## Finishing a Change

Before requesting review or reporting a change as done:

1. Inspect the final diff for unrelated or generated changes.
2. Run the checks that `implementation-and-tests.instructions.md` requires for the change type.
3. Summarize the change and the verification results accurately, including any check that could not be run and any pre-existing failure with its evidence.
4. If remote PR work was requested, check for an existing PR before creating one, and never create a second PR for the same branch.

## Anti-patterns

- Creating a new branch when the user is already on a branch with an open PR for the same piece of work.
- Pushing, opening, or merging a PR when the user did not request that remote action.
- Describing "no checks were triggered" or a pre-existing failure as a passing check.
- Adding or repairing CI only to give the current PR a green check.
- Folding fixes for unrelated pre-existing failures into the current PR so its checks pass.
- Merging on an unattributed failure instead of investigating whether the PR's diff could have caused it.
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
