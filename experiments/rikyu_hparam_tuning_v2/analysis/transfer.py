#!/usr/bin/env python3
"""Does multi-task training beat training each task alone? — summary/transfer.json

The campaign quotes single-task "ceilings" from the replay campaign's warm-restart control, and
five of the six probe tasks sit ABOVE them under multi-task training. That looks like positive
transfer, but those ceilings were measured on different hardware (H200), in a different container,
against different code. A consistent offset is exactly what a change of measurement frame produces
too, and the comparison cannot separate the two.

This compares against single-task runs made in THIS regime: same image, same probe config, same
adopted hyper-parameters, differing only in ``pretrain.task_sequence``. The difference is transfer;
the difference against the old ceilings is transfer plus an unknown offset, and both are reported
so the size of that offset is visible rather than assumed away.

WHY THIS GATES SEVERAL OTHER QUESTIONS
--------------------------------------
Loss balancing and gradient surgery both exist to make multi-task training treat its tasks better.
If multi-task training does not beat training a small task by itself, neither has anything to
repair — the answer for that task is to train it separately. So this runs first, and what it says
decides whether the others are worth measuring at all.

    python analysis/transfer.py --single <outroot>/stage_single --multi <outroot>/stage_bal \\
        --multi-glob 'bal*off_s*' -o summary/transfer.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from common import CEILING, N_TRAIN, PROBE6, final_metrics, fnum, pct_views, size_group

# The prefix is configurable because two single-task sets exist: one on the config that led
# the 5-seed grid, and one on the config the 25-seed finals actually adopted. Matching "st_"
# literally would silently read the wrong set now that "stA_" exists.
def single_re(prefix: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(prefix)}_(?P<task>.+)_s(?P<seed>\d+)$")


def collect_single(root: Path, tasks: list[str], prefix: str = "st") -> dict[str, list[float]]:
    """{task: [r2 per seed]} from runs that trained that task and nothing else."""
    out: dict[str, list[float]] = {t: [] for t in tasks}
    pat = single_re(prefix)
    for run in sorted(root.glob(f"{prefix}_*")):
        if not (run / "DONE").exists():
            continue
        m = pat.match(run.name)
        if not m or m.group("task") not in out:
            continue
        try:
            metrics, _ = final_metrics(run)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        v = fnum(metrics.get(m.group("task"), {}).get("r2"))
        if v is not None:
            out[m.group("task")].append(v)
    return out


def collect_multi(root: Path, glob: str, tasks: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {t: [] for t in tasks}
    for run in sorted(root.glob(glob)):
        if not (run / "DONE").exists():
            continue
        try:
            metrics, _ = final_metrics(run)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        for t in tasks:
            v = fnum(metrics.get(t, {}).get("r2"))
            if v is not None:
                out[t].append(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", type=Path, required=True)
    ap.add_argument("--multi", type=Path, required=True)
    ap.add_argument("--multi-glob", default="bal*off_s*")
    ap.add_argument("--single-prefix", default="st", help="runid prefix of the single-task set")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=PROBE6)
    args = ap.parse_args()

    single = collect_single(args.single, args.tasks, args.single_prefix)
    multi = collect_multi(args.multi, args.multi_glob, args.tasks)

    rows = []
    for t in args.tasks:
        s, m = single.get(t, []), multi.get(t, [])
        if not s or not m:
            rows.append({"task": t, "n_single": len(s), "n_multi": len(m), "skipped": "missing runs"})
            continue
        ms, mm = statistics.fmean(s), statistics.fmean(m)
        ss = statistics.stdev(s) if len(s) > 1 else 0.0
        sm = statistics.stdev(m) if len(m) > 1 else 0.0
        se = math.sqrt(ss**2 / len(s) + sm**2 / len(m)) if len(s) > 1 and len(m) > 1 else None
        separated = bool(se) and abs(mm - ms) > 2 * se
        views = pct_views(mm - ms, ms)
        rows.append({
            "task": t,
            "group": size_group(t),
            "n_train": N_TRAIN.get(t),
            "single_task_r2": ms,
            "multi_task_r2": mm,
            "transfer": mm - ms,
            **views,
            # Statistical resolution says the effect is REAL; the practical gate says it is worth
            # acting on. A task needs both to count as transfer that matters.
            "matters": separated and views["practically_significant"],
            "se_of_difference": se,
            # A difference smaller than its own resolution is not transfer in either direction.
            "separated": separated,
            "n_single": len(s),
            "n_multi": len(m),
            "recorded_ceiling_h200": CEILING.get(t),
            # How far the inherited ceiling sits from a same-regime measurement of the same thing.
            "frame_offset_vs_recorded": ms - CEILING[t] if t in CEILING else None,
        })

    scored = [r for r in rows if "transfer" in r]
    helped = [r for r in scored if r["separated"] and r["transfer"] > 0]
    hurt = [r for r in scored if r["separated"] and r["transfer"] < 0]
    matters = [r for r in scored if r["matters"]]
    offsets = [r["frame_offset_vs_recorded"] for r in scored if r["frame_offset_vs_recorded"] is not None]

    out = {
        "question": "does multi-task training beat training each task alone, in this regime?",
        "per_task": rows,
        "summary": {
            "tasks_helped": [r["task"] for r in helped],
            "tasks_hurt": [r["task"] for r in hurt],
            "tasks_unresolved": [r["task"] for r in scored if not r["separated"]],
            "tasks_that_matter": [r["task"] for r in matters],
            "resolved_but_negligible": [r["task"] for r in scored
                                        if r["separated"] and not r["practically_significant"]],
            "mean_transfer": statistics.fmean([r["transfer"] for r in scored]) if scored else None,
            "mean_relative_pct": statistics.fmean(
                [r["relative_pct"] for r in scored if r["relative_pct"] is not None]) if scored else None,
            "mean_frame_offset_vs_recorded_ceilings": statistics.fmean(offsets) if offsets else None,
        },
        "notes": [
            "single-task runs differ from the multi-task arm ONLY in pretrain.task_sequence — same "
            "image, same config, same adopted hyper-parameters, same seeds.",
            "frame_offset_vs_recorded is single-task-here minus the recorded H200 ceiling. A large "
            "consistent offset means the inherited ceilings cannot be used to judge transfer, "
            "which is why these runs exist.",
            "A task whose difference is not separated is reported as unresolved, not as zero.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'task':20s} {'grp':6s} {'N':>7s} {'single':>8s} {'multi':>8s} {'transfer':>9s} "
          f"{'rel%':>8s} {'err.red%':>9s} {'2SE':>7s}  verdict")
    for r in rows:
        if "transfer" not in r:
            print(f"{r['task']:20s} {'':6s} {'':>7s} {'':>8s} {'':>8s} {'':>9s} {'':>8s} {'':>9s} "
                  f"{'':>7s}  {r.get('skipped')} (single={r['n_single']}, multi={r['n_multi']})")
            continue
        se = r["se_of_difference"] or 0.0
        verdict = ("multi-task better" if r["separated"] and r["transfer"] > 0
                   else "single-task better" if r["separated"] else "unresolved")
        if r["separated"] and not r["practically_significant"]:
            verdict += " (negligible)"
        # An error-reduction figure computed against a residual under 0.05 is not quotable.
        err = ("n/a*" if r["near_ceiling"] or r["error_reduction_pct"] is None
               else f"{r['error_reduction_pct']:+.1f}%")
        print(f"{r['task']:20s} {r['group']:6s} {r['n_train']:7d} {r['single_task_r2']:8.4f} "
              f"{r['multi_task_r2']:8.4f} {r['transfer']:+9.4f} {r['relative_pct']:+7.2f}% "
              f"{err:>9s} {2 * se:7.4f}  {verdict}")
    print("  * error reduction suppressed where the single-task residual is under 0.05 — "
          "a 1e-3 change there moves it by whole percentage points.")

    s = out["summary"]
    print(f"\n  helped: {s['tasks_helped'] or 'none'}")
    print(f"  hurt:   {s['tasks_hurt'] or 'none'}")
    print(f"  resolved AND practically significant (|delta| >= 0.01): "
          f"{s['tasks_that_matter'] or 'none'}")
    if s["resolved_but_negligible"]:
        print(f"  resolved but negligible: {s['resolved_but_negligible']}")
    if s["mean_frame_offset_vs_recorded_ceilings"] is not None:
        print(f"\n  same-regime single-task minus recorded H200 ceiling: "
              f"{s['mean_frame_offset_vs_recorded_ceilings']:+.4f} mean")
        print("  (that offset is what made the inherited ceilings unusable for this question)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
