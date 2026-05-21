---
description: "Use when creating, editing, or reorganizing repository documentation, especially README.md, ARCHITECTURE.md, or AGENTS.md. Keep README focused on user-facing purpose, usage, and examples, and move developer-facing structure, architecture, and workflow guidance into the dedicated developer docs."
applyTo: "**/{README,ARCHITECTURE,AGENTS,CONTRIBUTING}.md"
---

# Repository Documentation Boundaries

Keep `README.md` user-facing and keep developer guidance in the dedicated docs this repo already uses.

- Step 1: Put user-first content in `README.md` (purpose, audience, install, quick start, CLI/API usage, concise examples).
- Step 2: Put contributor workflow content in `AGENTS.md` (project structure, development/testing workflow, coding and documentation conventions, contribution expectations) and detailed model/component design in `ARCHITECTURE.md`.
- Step 3: If `README.md` starts accumulating deep architecture detail or contributor-only guidance, move that material into `ARCHITECTURE.md` / `AGENTS.md` and leave a short link in `README.md`.
- Keep these docs in sync with the code they describe: when an entry point, config field, or data convention changes, update the affected doc rather than letting it drift.