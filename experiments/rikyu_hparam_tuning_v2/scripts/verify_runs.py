#!/usr/bin/env python3
"""Verify that a stage's runs actually TRAINED — not merely that they exited 0.

A run that exits early still exits 0, still drops a DONE marker, and still looks green in
`sacct`. v1 was bitten by exactly that shape once, which is why PLAN §5 makes "confirm the smoke
really trained" a standing rule rather than a one-off check. Green here means all of:

  * a DONE marker exists;
  * ENV.json reports the EXPECTED fm version — the guard against the whole stage having silently
    run under the 0.2.1 per-batch scheduler cadence;
  * one step directory per task in the sequence, so the continual sequence ran to the end;
  * every expected task has a *_metrics.json in the LAST step directory;
  * those metrics carry real finite numbers, not nulls.

    python scripts/verify_runs.py <outroot> --tasks volume formation_energy ... --version 0.3.2
    python scripts/verify_runs.py <outroot> --expect-runs 18        # also check none are missing

Exit status is 0 only when every run passes, so this is usable as a gate in a pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

PROBE6 = ["volume", "formation_energy", "seebeck", "zt", "magnetization", "magnetic_moment"]


def step_dirs(run: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for d in (run / "training").glob("step*_*"):
        m = re.match(r"step(\d+)_", d.name)
        if m:
            out[int(m.group(1))] = d
    return out


def check(run: Path, tasks: list[str], version: str | None) -> list[str]:
    """Every reason this run is not trustworthy. Empty list = trustworthy."""
    problems: list[str] = []

    if not (run / "DONE").exists():
        problems.append("no DONE marker")

    env_path = run / "ENV.json"
    if not env_path.exists():
        # v1-era runs predate ENV.json; for v2 its absence means the worker never wrote it.
        problems.append("no ENV.json (worker did not reach the training call?)")
    elif version is not None:
        try:
            got = json.loads(env_path.read_text()).get("fm_version")
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"ENV.json unreadable: {exc}")
        else:
            if got != version:
                problems.append(f"WRONG IMAGE: fm_version={got!r}, expected {version!r}")

    steps = step_dirs(run)
    if not steps:
        problems.append("no step directories — it never trained")
        return problems
    if len(steps) < len(tasks):
        problems.append(f"only {len(steps)}/{len(tasks)} step directories — sequence did not finish")

    last = steps[max(steps)]
    for task in tasks:
        jf = last / f"{task}_metrics.json"
        if not jf.exists():
            problems.append(f"{task}: no metrics JSON in {last.name}")
            continue
        try:
            m = json.load(open(jf))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{task}: metrics JSON unreadable: {exc}")
            continue
        # A task is measured on r2 unless it is a classifier; either way SOMETHING finite must be
        # there. All-null metrics is the signature of a run that set up and then bailed.
        values = [m.get(k) for k in ("r2", "mae", "accuracy", "macro_f1")]
        finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        if not finite:
            problems.append(f"{task}: no finite metric in {jf.name} (got {values!r})")

    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outroot", type=Path, help="stage output root (one subdirectory per runid)")
    ap.add_argument("--tasks", nargs="+", default=PROBE6, help="tasks the sequence should have trained")
    ap.add_argument("--version", default="0.3.2", help="fm version ENV.json must report ('' to skip)")
    ap.add_argument("--expect-runs", type=int, default=None, help="fail if fewer run dirs than this")
    ap.add_argument("--glob", default="*", help="restrict to runids matching this glob")
    args = ap.parse_args()

    version = args.version or None
    runs = sorted(d for d in args.outroot.glob(args.glob) if d.is_dir() and not d.name.startswith("_"))
    if not runs:
        print(f"FAIL no run directories under {args.outroot}", file=sys.stderr)
        raise SystemExit(1)

    bad = 0
    for run in runs:
        problems = check(run, args.tasks, version)
        if problems:
            bad += 1
            print(f"FAIL {run.name}")
            for p in problems:
                print(f"       {p}")
        else:
            print(f"ok   {run.name}")

    print(f"\n{len(runs) - bad}/{len(runs)} runs verified as really trained")
    if args.expect_runs is not None and len(runs) < args.expect_runs:
        print(f"FAIL only {len(runs)} run dirs, expected {args.expect_runs}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
