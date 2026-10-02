# Shared installation

The canonical, version-controlled skill is `.agents/skills/research-slides/`.
The repository's `.claude/skills/research-slides` is a relative directory symlink to it,
so a clone contains one set of instructions for both agents. Commit both the canonical
folder and that symlink when publishing this change.

The personal Codex installation belongs in
`${CODEX_HOME:-$HOME/.codex}/skills/research-slides` (normally
`~/.codex/skills/research-slides`); Claude Code uses `~/.claude/skills/research-slides`.
On the machine where this skill was created, both link to the canonical repository
directory. Editing it therefore updates both installations without a copy step.
The directory and supporting references must remain together.

## Install on another machine

From the repository root, when no personal skill of this name is already installed:

```sh
skill_dir="$(pwd)/.agents/skills/research-slides"
codex_skills_dir="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$codex_skills_dir" "$HOME/.claude/skills"
ln -s "$skill_dir" "$codex_skills_dir/"
ln -s "$skill_dir" "$HOME/.claude/skills/"
```

The commands do not replace existing installations. Inspect any name collision before
changing it. Linking requires this checkout to remain at its location and on a branch
that contains the skill. For an installation independent of a checkout, copy the entire
skill folder into each personal directory and synchronize later revisions deliberately.

If a platform does not preserve Git symlinks, install a copy of the canonical folder
in `.claude/skills/research-slides` after cloning. Keep the canonical folder as the
maintained source and verify the copy when updating it.

## Use

- Codex: `Use $research-slides to prepare a presentation for the intended audience.`
- Claude Code: `/research-slides Prepare a presentation for the intended audience.`

Automatic selection remains enabled through the skill description. If the current
session does not list a newly installed skill, start a fresh session or explicitly ask
the agent to read its `SKILL.md`; do not change unrelated agent settings.

The skill discovers and builds on the presentation skill available to that agent.
It does not bundle a rendering library or a third-party PPTX skill. On the machine
where it was created, the available foundations included Codex's `Presentations`
skill and the installed `pptx` / `pptx-official` skills. Their implementation details
remain owned by those skills and are not copied into this package.

The entry point uses the common Agent Skills `name` and `description` fields.
`agents/openai.yaml` supplies optional Codex interface metadata and contains no
instructions required for Claude Code. Supporting files use relative links and contain
no machine-specific absolute paths.

Claude Code's documented personal/project paths and directory-symlink support are in
[its skills documentation](https://code.claude.com/docs/en/skills#choose-where-skills-load).
