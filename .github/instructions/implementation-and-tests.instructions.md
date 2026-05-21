---
description: "Use when implementing user-described features, fixing bugs, refactoring, or generating new code. Enforces minimal abstraction, minimal scope expansion, dataclass-based config conventions, type hints, and core-logic test coverage with colocated `<source>_test.py` files."
applyTo: "**"
---

# Implementation and Core-Logic Tests

When implementing what the user described, follow these principles together. They constrain both the production code and the accompanying tests.

**Key rules at a glance** (use the numbered sections below for the full rules):
- Use the most direct structure; avoid extra abstraction layers.
- Implement only what the description asks for; mention out-of-scope improvements instead of adding them.
- Configure behavior through `@dataclass` config objects (mirror `models/model_config.py`); validate in `__post_init__` and accept dicts via a normalizer like `build_encoder_config`.
- Type-hint public APIs; use enums for closed choice sets.
- Prefer TDD when feasible: write test logic first, then implement. Colocate tests in `<source>_test.py`; cover the primary path and the most likely failure patterns. Run them with `pytest` (or `uv run pytest`).
- If the description is incomplete or ambiguous, clarify with the user before proceeding. If it explicitly conflicts with the abstraction or config-shape guidance, follow the description and note the deviation; scope restrictions still apply unless the user explicitly requests otherwise.

## 1. Minimal Abstraction

Use the most direct code structure that satisfies the described requirement. If the user's description conflicts with these principles, prioritize the user's description and document the deviation.

- Do not introduce abstract base classes, factories, registries, generic protocols, plugin layers, or extra indirection unless the described requirement actually needs them.
- Do not split a single concrete implementation into multiple layers "for future flexibility".
- Inline a value or short helper when extracting it would only be used once and would not improve readability.
- Prefer concrete types and direct function/method calls over abstract types and dispatch when there is one real implementation today.
- Do not split a single-use operation into a public/private wrapper pair or a chain of one-off helper functions just to make individual functions shorter or easier to unit test. If the helper has no independent semantic role, no second caller, and no meaningful name beyond restating the caller, keep the logic in the caller.
- Prefer writing one cohesive function for one cohesive behavior. Extract a helper only when it names a distinct concept, removes meaningful duplication, isolates a genuinely complex sub-step, or is reused by multiple call sites.

If you find yourself adding a layer "in case we need to swap it later", stop and use the concrete form instead.
If you find yourself creating a function that only forwards to another function with nearly the same name, inline it unless there is a concrete lifecycle, validation, or API-boundary reason for the wrapper.

### Configuration Objects

This codebase configures behavior through `@dataclass` objects (see `models/model_config.py`), not ad-hoc dicts or long positional argument lists. When a type needs behavioral tuning, follow the existing conventions.

- Define a `@dataclass` for the config. Use `field(default=...)` / `field(default_factory=...)` for sensible defaults so callers can build it from a literal or adjust a default instance. Prefer `@dataclass(kw_only=True)` or `kw_only=True` fields for optional knobs that should be passed by name.
- Validate and normalize in `__post_init__` (coerce types, raise `ValueError` for invalid combinations) rather than scattering validation across call sites. Mirror the encoder/task config classes.
- When a constructor or factory must accept either a dataclass instance or a plain mapping (e.g. from a YAML/TOML config), provide a normalizer function that returns the dataclass — follow the `build_encoder_config()` pattern instead of branching on `dict` everywhere.
- For closed sets of choices, define a `str`-based `Enum` (like `TaskType`, `EncoderType`) and accept the enum or its string value, normalizing in `__post_init__`.
- Keep required problem inputs (the concrete things a class needs to do its job — e.g. a dataframe, descriptor source, task configs) as explicit leading `__init__` parameters, distinct from the optional config knobs. Do not funnel everything through `**kwargs`.

### Module Organization

Place code near the behavior that gives it meaning.

- Keep a primary class with its methods and tightly coupled helpers in one cohesive module (e.g. a task head and its config-consuming logic).
- Config dataclasses and shared enums for the model live together in `models/model_config.py` because they form a small, stable, cross-module vocabulary. Follow that established split — do not scatter new task/encoder config dataclasses into unrelated modules, and do not create a competing catch-all module for unrelated types.
- Keep small private helper functions next to their single caller unless they are reused across modules.
- Tests live beside their source as `<source>_test.py` (see Section 3), not in a separate `tests/` tree.

## 2. Minimal Scope Expansion

Implement only what the user's description asks for.

- Do not add configuration options, flags, fields, methods, or task types that the description does not mention.
- Do not refactor unrelated code, rename unrelated symbols, or "clean up" nearby files while implementing the requested change.
- Do not add logging, metrics, retries, caching, or validation unless explicitly mentioned in the description or clearly required by the surrounding code. (When you do log, use the existing `loguru` logger rather than introducing a new logging mechanism.)
- If a related improvement seems valuable but is out of scope, mention it briefly to the user instead of silently adding it.

When in doubt, prefer the smaller change. The user can always ask for more.

## 3. Tests Cover Core Logic and Failure Patterns

Every implementation change must ship with at least one test file that exercises the core logic, with priority on the patterns most likely to fail.

When feasible, prefer a test-driven approach: write the test logic first to define the expected behavior, then write the implementation code to make those tests pass. This should be the default for new functions, config classes, and data/model flows; it is optional when changing existing code where the behavior or test boundary is not yet clear.

- Always generate or update a test file for the code you wrote or modified.
- Cover the primary success path of the described behavior.
- Cover the inputs and conditions most likely to break it: empty / missing / malformed input, boundary values, NaN / masked targets, mismatched tensor shapes or dimensions, errors raised by dependencies, and any explicit precondition the code enforces (for example "hidden_dims must include input_dim", "must be positive", "column must exist in attributes_source").
- Do not aim for exhaustive coverage of trivial properties, generated code, or thin pass-through wrappers. Aim for the logic that, if broken, would silently produce wrong behavior.
- Run the new tests with `pytest <path>` (or `uv run pytest <path>`) and confirm they pass before reporting the task done.

### Test File Naming and Location

Colocate tests next to the source file and name them after that source file.

- For a source file `bar.py`, create or update `bar_test.py` in the same directory (e.g. `datamodule.py` → `datamodule_test.py`, `dynamic_task_suite.py` → `dynamic_task_suite_test.py`).
- Do not create a separate `tests/` tree, a generic `helpers_test.py`, or a catch-all file when a focused `<source>_test.py` companion already fits.
- If a single source file's tests grow large enough to warrant splitting, split by feature into additional files that still start with the source file's basename (for example `model_parse_test.py`, `model_loss_test.py`).
- Use pytest conventions: test functions named `test_*`, fixtures over ad-hoc setup, and `pytest.mark.parametrize` for input variations.

## Quick Self-Check Before Finishing

Before reporting an implementation as complete, confirm:

1. No abstraction was added that is not justified by the described requirement.
2. No feature, option, or refactor was added beyond the description.
3. Behavioral configuration goes through a `@dataclass` with defaults and `__post_init__` validation, and required problem inputs stay as explicit constructor parameters separate from optional knobs.
4. Mappings/dicts are normalized into the config dataclass via a builder (like `build_encoder_config`) rather than handled inline, and closed choice sets use enums.
5. Public APIs are type-hinted and follow the existing module organization (model config dataclasses/enums in `models/model_config.py`).
6. A `<source>_test.py` file exists next to the changed source and exercises the core path plus the most likely failure patterns.
7. The new and existing tests for the touched modules pass (`pytest`).
