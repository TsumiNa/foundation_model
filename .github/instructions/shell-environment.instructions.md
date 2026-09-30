---
description: "Use at the start of any new session before running terminal commands. Detect the active shell to avoid heredoc and syntax errors caused by shell incompatibility (fish, bash, zsh, sh), and follow the repository's uv-based command and filesystem safety rules."
name: "Shell Environment and Repository Commands"
applyTo: "**"
---
# Shell Environment and Repository Commands

At the start of every new agent session, before executing any terminal command, identify the interpreter that will run your commands.

## How to Confirm

- Read the interpreter from the execution environment or the command runner's metadata (for example the shell the tool declares it uses) before relying on shell-specific syntax.
- Do not make the first probe depend on shell expansions that must be parsed before the interpreter is known: `$$`, `$0`, `$fish_pid`, command substitution, or conditional syntax are not portable across the supported shells.
- If the environment does not expose the interpreter, use an explicitly selected shell through the command runner when available; otherwise keep commands to literal, shell-neutral program invocations until the interpreter is known.
- Treat `$SHELL` only as information about the user's configured login shell; it may not be the interpreter executing the current command.
- Record the running interpreter for the session and do not assume Bash.

## Shell-Specific Rules

**fish shell** — does NOT support POSIX heredocs or all POSIX variable/loop syntax. Avoid all of these patterns:

```bash
# WRONG in fish
cat << 'EOF' >> file.txt
...content...
EOF
```

Use `printf` or `echo` with explicit newlines instead, or write the file directly with an editor tool:

```fish
printf '%s\n' 'line1' 'line2' >> file.txt
```

Or prefer the `create_file` / `replace_string_in_file` agent tools whenever available — they bypass shell syntax entirely.

**bash / zsh / sh** — POSIX heredocs are safe, but prefer the editor or patch tool for file creation and precise edits.

## General Rules

- Never assume `bash` without confirming — the user's default interactive shell may differ.
- Do not launch a different shell (for example `bash -c "..."`) merely to bypass an avoidable syntax issue unless explicitly asked. If a command fails with a shell syntax error, check the active shell before retrying.
- When in doubt, prefer agent file-editing tools over shell redirection for writing file content.
- Quote paths and arguments that may contain spaces, brackets, or glob characters.
- Prefer `rg` and `rg --files` for searching.
- Avoid commands that rewrite broad file sets unless the task requires it; inspect the diff after formatters or generators run.
- Do not delete or overwrite files, caches, build artifacts, branches, or user modifications unless the task requires it and the exact target has been verified.
- Never print secrets, tokens, environment files, or credential-bearing command output, and never write credentials into tracked files.

## Repository Command Rules

- Use `uv`, matching `pyproject.toml` and `uv.lock`. Do not use `pip`, `conda`, or global installs to complete repository work.
- Run tools through `uv run <tool>` (`ruff`, `mypy`, `pytest`, `pre-commit`); the file-scoped commands are listed in `AGENTS.md`.
- Add or update dependencies with `uv add` / `uv add --dev` so `pyproject.toml` and `uv.lock` change together.
- Remote clusters have their own instruction files (`rikyu-supercomputer`, `riken-rccs-supercomputer`, `ism-gpu-a100-training`); follow their login-shell and container rules there.
