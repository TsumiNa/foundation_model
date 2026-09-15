#!/usr/bin/env python3
"""Turn the transfer stage's finished runs into a reusable pre-trained model library.

The transfer stage trained 24 tasks x 10 orderings. Each run ends with an encoder that has seen all
24 tasks, differing only in the order it saw them, so the 240 final checkpoints are 240 samples of
the same distribution -- exactly what someone starting a NEW task wants to warm-start from.

What makes them usable is not the weights, it is knowing what each one is. A checkpoint whose task
order, seed, hyper-parameters and per-task scores are unrecorded cannot be chosen between, so this
writes a MANIFEST.json carrying all of it plus an INDEX.md a person can read.

WHY THE PER-TASK SCORES TRAVEL WITH THE WEIGHTS
-----------------------------------------------
Picking a warm start is a per-task question: "which of these encoders was good at something like my
task?" The manifest therefore records every run's final score on all 24 tasks, not just on the task
it was built to test, and the index ranks the library by each task so that question is answerable
by looking rather than by re-running anything.

    python scripts/build_model_library.py --runs <outroot>/stage_xfer -o <outroot>/model_library
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

STEP = re.compile(r"step(\d+)_(.+)$")
RUN = re.compile(r"^xf_(.+)_o(\d+)$")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def dig(data, *keys, default=None):
    """Walk a nested dict, returning default the moment a level is missing."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def read_run(run: Path) -> dict | None:
    m = RUN.match(run.name)
    if not m or not (run / "DONE").exists():
        return None
    task_under_test, ordering = m.group(1), int(m.group(2))

    steps: dict[int, str] = {}
    last_dir, last_n = None, -1
    for d in (run / "training").glob("step*_*"):
        sm = STEP.match(d.name)
        if sm:
            n = int(sm.group(1))
            steps[n] = sm.group(2)
            if n > last_n:
                last_n, last_dir = n, d
    if last_dir is None:
        return None

    # Every task's score in the FINAL step directory: what this encoder is worth on each task after
    # the whole sequence, which is the number a future user of the checkpoint actually inherits.
    final_metrics = {}
    for mf in last_dir.glob("*_metrics.json"):
        task = mf.name[: -len("_metrics.json")]
        try:
            data = json.loads(mf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        final_metrics[task] = {
            "primary": data.get("primary"),
            "r2": data.get("r2"),
            "mae": data.get("mae"),
            "samples": data.get("samples"),
        }

    prov_path = run / "run_provenance.json"
    prov = {}
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text())
        except (OSError, json.JSONDecodeError):
            prov = {}
    cfg = prov.get("resolved_config", {})

    model = run / "training" / "final_model.pt"
    # The task order is reconstructed from the step directories, NOT from resolved_config: that
    # field records the string "fixed" (the mode), not the sequence. The step names are the sequence
    # as it was actually executed, which is the thing a user of the checkpoint needs.
    replay = dig(cfg, "replay", default={})
    per_task = replay.get("per_task") or {}
    return {
        "run": run.name,
        "task_under_test": task_under_test,
        "ordering_index": ordering,
        "task_order": [steps[k] for k in sorted(steps)],
        "n_tasks": len(steps),
        "seed": dig(cfg, "training", "seed", default=dig(prov, "seeds", "seed")),
        "hyperparameters": {
            "latent_dim": dig(cfg, "model", "latent_dim"),
            "encoder_hidden_dims": dig(cfg, "model", "encoder_hidden_dims"),
            "head_hidden_dims": dig(cfg, "model", "head_hidden_dims"),
            "encoder_lr": dig(cfg, "training", "encoder_lr"),
            "head_lr": dig(cfg, "training", "head_lr"),
            "encoder_weight_decay": dig(cfg, "training", "encoder_weight_decay"),
            "head_weight_decay": dig(cfg, "training", "head_weight_decay"),
            "max_epochs": dig(cfg, "training", "max_epochs"),
            "early_stopping_patience": dig(cfg, "training", "early_stopping", "patience"),
            "early_stopping_min_delta": dig(cfg, "training", "early_stopping", "min_delta"),
            "replay_interval": replay.get("interval"),
            "replay_amount_fraction": replay.get("amount"),
            "replay_floor_per_task": min(per_task.values()) if per_task else None,
        },
        "descriptor": dig(cfg, "catalog", "descriptor", default={}),
        # run_provenance records git.commit as null, so the checkpoint's identity rests on the
        # package set — foundation-model's version is the container that produced it.
        "packages": prov.get("packages", {}),
        "container_version": dig(prov, "packages", "foundation-model"),
        "git_commit": dig(prov, "git", "commit"),
        "trained_utc": prov.get("datetime_utc"),
        "model_file": str(model) if model.exists() else None,
        "model_bytes": model.stat().st_size if model.exists() else None,
        "final_metrics": final_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--copy", action="store_true",
                    help="copy the checkpoints into the library (default: reference them in place)")
    args = ap.parse_args()

    entries = [e for d in sorted(args.runs.iterdir()) if d.is_dir()
               if (e := read_run(d)) is not None]
    if not entries:
        raise SystemExit(f"no finished runs under {args.runs}")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.copy:
        for e in entries:
            if not e["model_file"]:
                continue
            dest = args.out / "models" / e["task_under_test"] / f"o{e['ordering_index']}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(e["model_file"], dest)
            e["library_path"] = str(dest.relative_to(args.out))
            e["sha256"] = sha256(dest)

    # Per-task view: for each task, how every checkpoint in the library scores on it.
    by_task: dict[str, list] = defaultdict(list)
    for e in entries:
        for task, m in e["final_metrics"].items():
            if m["primary"] is not None:
                by_task[task].append((m["primary"], e["run"]))

    task_summary = {}
    for task, vals in by_task.items():
        scores = [v for v, _ in vals]
        best = max(vals)
        task_summary[task] = {
            "n_models": len(scores),
            "mean": statistics.fmean(scores),
            "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": best[0],
            "best_model": best[1],
        }

    manifest = {
        "library": "rikyu_hparam_tuning_v2 / transfer stage final encoders",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_models": len(entries),
        "what_each_model_is":
            "an encoder trained continuously on all 24 tasks with hybrid replay; models differ only "
            "in the order the tasks were presented and in seed",
        "how_to_use":
            "warm-start `fm finetune` from the .pt; pick by task_summary if your target resembles "
            "one of the 24, otherwise any model is an equally valid draw from the same distribution",
        "task_summary": task_summary,
        "models": entries,
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Pre-trained model library — transfer stage (v2)",
        "",
        f"{len(entries)} encoders, each trained on all 24 tasks with hybrid replay, differing only in "
        "task order and seed. Built "
        f"{manifest['built_utc']}.",
        "",
        "Warm-start `fm finetune` from any `.pt`. If your target task resembles one of the 24, the "
        "table below says which checkpoint was best at it; otherwise the models are interchangeable "
        "draws from the same distribution.",
        "",
        "## What each task looks like across the library",
        "",
        "| task | models | mean | sd | best | best model |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for task in sorted(task_summary, key=lambda t: -task_summary[t]["mean"]):
        s = task_summary[task]
        lines.append(f"| {task} | {s['n_models']} | {s['mean']:.4f} | {s['sd']:.4f} | "
                     f"{s['max']:.4f} | `{s['best_model']}` |")
    lines += ["", "## Models", "",
              "| model | task under test | ordering | seed | size |", "|---|---|---:|---:|---:|"]
    for e in entries:
        size = f"{e['model_bytes'] / 2**20:.1f} MiB" if e["model_bytes"] else "-"
        lines.append(f"| `{e['run']}` | {e['task_under_test']} | {e['ordering_index']} | "
                     f"{e['seed']} | {size} |")
    lines += ["", "Full task orders, hyper-parameters and every model's score on all 24 tasks are in "
              "`MANIFEST.json`.", ""]
    (args.out / "INDEX.md").write_text("\n".join(lines))

    total = sum(e["model_bytes"] or 0 for e in entries)
    print(f"  {len(entries)} models, {total / 2**30:.2f} GiB, {len(task_summary)} tasks scored")
    print(f"  wrote {args.out / 'MANIFEST.json'} and {args.out / 'INDEX.md'}")


if __name__ == "__main__":
    main()
