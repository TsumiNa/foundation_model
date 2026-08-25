#!/usr/bin/env python3
"""Write the stage-C *tuned* config by patching scalar keys of the untuned one.

Stage C must be a like-for-like comparison, so the tuned arm is produced from the control's own
file rather than authored separately: every key not named on the command line is byte-identical,
and the emitted header records exactly which lines changed. That diff is the campaign's claim
that the two arms differ only in the tuned knobs.

`--set` cannot address `[[tasks]]` array entries, which is why a generated file is needed at all:
stage B tunes each task's head separately, and those winners live inside the task's own table.

    python .../make_tuned_config.py configs/final_hybrid.toml configs/final_hybrid_tuned.toml \\
        --set model.latent_dim=256 --set 'model.encoder_hidden_dims=[512,256]' \\
        --task-overrides results/head_winners.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Per-task keys TaskSpec accepts; anything else in the winners JSON is metadata, not config.
_TASK_KEYS = {"lr", "hidden_dims", "x_hidden_dims", "t_hidden_dims", "n_kernel"}


def patch_scalar(text: str, section: str, key: str, value: str) -> tuple[str, str]:
    """Replace ``key = ...`` inside ``[section]``. Returns (new_text, description)."""
    sec = re.search(rf"^\[{re.escape(section)}\]\s*$", text, re.M)
    if not sec:
        raise SystemExit(f"section [{section}] not found")
    end = re.search(r"^\[", text[sec.end():], re.M)
    stop = sec.end() + (end.start() if end else len(text) - sec.end())
    block = text[sec.end():stop]

    pattern = re.compile(rf"^({re.escape(key)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
    match = pattern.search(block)
    if not match:
        raise SystemExit(f"key {key!r} not found in [{section}]")
    old = match.group(2)
    new_block = block[: match.start()] + f"{match.group(1)}{value}{match.group(3)}" + block[match.end():]
    return text[: sec.end()] + new_block + text[stop:], f"[{section}] {key}: {old} -> {value}"


def patch_task_keys(text: str, task: str, values: dict) -> tuple[str, list[str]]:
    """Add or replace keys inside the ``[[tasks]]`` table whose ``name`` is ``task``."""
    for match in re.finditer(r"^\[\[tasks\]\]\s*$", text, re.M):
        end = re.search(r"^\[", text[match.end():], re.M)
        stop = match.end() + (end.start() if end else len(text) - match.end())
        block = text[match.end():stop]
        if not re.search(rf'^name\s*=\s*"{re.escape(task)}"\s*$', block, re.M):
            continue
        notes = []
        for key, value in values.items():
            literal = toml_literal(value)
            existing = re.search(rf"^({re.escape(key)}\s*=\s*)(.*)$", block, re.M)
            if existing:
                block = block[: existing.start()] + f"{existing.group(1)}{literal}" + block[existing.end():]
                notes.append(f"[[tasks]] {task}.{key}: {existing.group(2)} -> {literal}")
            else:
                block = block.rstrip("\n") + f"\n{key} = {literal}\n\n"
                notes.append(f"[[tasks]] {task}.{key}: (unset) -> {literal}")
        return text[: match.end()] + block + text[stop:], notes
    raise SystemExit(f"no [[tasks]] entry named {task!r}")


def toml_literal(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(toml_literal(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return repr(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument("--task-lr", action="append", default=[], metavar="TASK=LR")
    ap.add_argument(
        "--task-overrides",
        type=Path,
        help="JSON from analysis/pick_heads.py: {task: {override: {key: value}}} or {task: {key: value}}",
    )
    ap.add_argument("--note", default="", help="one line describing where the winners came from")
    args = ap.parse_args()

    text = args.source.read_text()
    changes: list[str] = []

    for item in args.sets:
        dotted, _, value = item.partition("=")
        section, _, key = dotted.rpartition(".")
        text, note = patch_scalar(text, section, key, value)
        changes.append(note)

    if args.task_overrides:
        winners = json.loads(args.task_overrides.read_text())
        for task in sorted(winners):
            entry = winners[task]
            values = entry.get("override", entry) if isinstance(entry, dict) else entry
            values = {k: v for k, v in values.items() if k in _TASK_KEYS}
            if not values:
                continue
            text, notes = patch_task_keys(text, task, values)
            changes += notes

    for item in args.task_lr:
        task, _, lr = item.partition("=")
        text, notes = patch_task_keys(text, task, {"lr": lr})
        changes += notes

    if not changes:
        raise SystemExit("nothing to patch")

    header = [
        f"# GENERATED from {args.source.name} by scripts/make_tuned_config.py — do not hand-edit.",
        "# Every other line is byte-identical to the control arm's config. Changed keys:",
        *(f"#   {c}" for c in changes),
    ]
    if args.note:
        header.append(f"# {args.note}")
    args.dest.write_text("\n".join(header) + "\n#\n" + text)
    print(f"{args.dest}", file=sys.stderr)
    for c in changes:
        print(f"  {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
