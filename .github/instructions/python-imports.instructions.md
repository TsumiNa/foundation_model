---
description: 'Use when creating, editing, or refactoring Python imports. Prefer single-dot relative imports only for modules inside the current package, and absolute imports for parent packages, sibling packages, and external packages.'
name: 'Python Import Rules'
applyTo: '**/*.py'
---

# Python Import Rules

- Use relative imports only for modules inside the current package via a single leading dot, such as `from .shared import helper`.
- Do not use parent-relative imports such as `..foo` or `...foo`.
- When importing from a parent package or a sibling package, use an absolute import, such as `from dyneq_py.input_processor.sections.base import _format_int`.
- When importing standard-library or third-party modules, use absolute imports.
- Group imports in the following order, with a blank line between each group:
    1. Standard library imports.
    2. Third-party imports.
    3. Local application imports.

- Always use explicit imports rather than wildcard imports (e.g., `from module import *`).
- Avoid circular imports by restructuring code or using local imports within functions when necessary.
- Keep import statements at the top of the file, except for local imports within functions to avoid circular dependencies.
