#!/usr/bin/env python3
"""Fail if docs/configuration.md has drifted from the config dataclasses.

Checks both directions per section: every dataclass field must have a table row, and every table
row must still correspond to a field. Section-scoped on purpose — a plain full-text search reports
a field as documented when its name merely appears in prose somewhere else, which is how
``[[tasks]].weight_decay`` was nearly shipped undocumented.

    uv run python scripts/check_config_docs.py
"""

import dataclasses, re, sys
from pathlib import Path
from foundation_model.workflows import _sections as S
from foundation_model.workflows import task_catalog as TC
from foundation_model.workflows.pretrain import ReplayConfig

doc = Path("docs/configuration.md").read_text()


def section_body(heading_regex: str) -> str:
    m = re.search(heading_regex, doc, re.M)
    if not m:
        return ""
    rest = doc[m.end() :]
    nxt = re.search(r"^#{2,3} ", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


CHECKS = [
    (r"^## `\[data\]`", TC.DataConfig, set()),
    (r"^## `\[descriptor\]`", TC.DescriptorConfig, set()),
    (r"^## `\[model\]`", S.ModelSectionConfig, set()),
    (
        r"^## `\[training\]`",
        S.TrainingSectionConfig,
        {"early_stopping", "checkpoint", "logging", "optimizer", "scheduler"},
    ),
    (r"^### `\[training\.optimizer\]`", S.OptimizerSectionConfig, set()),
    (r"^### `\[training\.scheduler\]`", S.SchedulerSectionConfig, set()),
    (r"^### `\[training\.early_stopping\]`", S.EarlyStoppingConfig, set()),
    (r"^### `\[training\.checkpoint\]`", S.CheckpointConfig, set()),
    (r"^### `\[training\.logging\]`", S.LoggingConfig, set()),
    (r"^## `\[\[tasks\]\]`", TC.TaskSpec, {"scaler", "type"}),
    (r"^### `\[pretrain\.replay\]`", ReplayConfig, set()),
]
bad = 0
for heading, cls, skip in CHECKS:
    body = section_body(heading)
    label = heading.strip("^$").replace("\\", "")
    if not body:
        print(f"MISSING SECTION  {label}")
        bad += 1
        continue
    rows = set(re.findall(r"^\| `([a-z_0-9]+)`", body, re.M))
    fields = {f.name for f in dataclasses.fields(cls)} - skip
    absent = sorted(fields - rows)
    stale = sorted(rows - fields)
    if absent or stale:
        bad += 1
        print(f"GAP  {label}")
        if absent:
            print(f"       undocumented fields : {absent}")
        if stale:
            print(f"       documented but gone : {stale}")
    else:
        print(f"OK   {label:34s} {len(fields)} fields")
sys.exit(1 if bad else 0)
