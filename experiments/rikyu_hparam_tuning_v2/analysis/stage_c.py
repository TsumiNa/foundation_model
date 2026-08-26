#!/usr/bin/env python3
"""Score the stage-C' arms, join them to v1's, and emit summary/stage_c.json.

Reads each arm from its per-step ``training/stepNN_<task>/<task>_metrics.json`` files rather than
from ``metrics_table.csv``: a ``--resume``'d pretrain writes the table from the resuming process's
in-memory records only, so a run recovered from a walltime kill has a PARTIAL table while the step
JSONs are always complete. Stage C' runs are long enough that a resume is likely, so this is the
default path, not a fallback.

WHAT THIS STAGE IS FOR
----------------------
Two questions, and the second is the one v2 was designed around.

**How much did tuning buy, as distinct from the upgrade?** Four points answer it, and none of
them can be dropped:

    v1 c_base        untuned + broken scheduler      (v1's control)
    v1 c_tuned       v1-tuned + broken scheduler     (v1's deliverable)
    v2 c2_base       untuned + fixed scheduler       <- the upgrade alone
    v2 c2_top1       v2-tuned + fixed scheduler      <- tuning on top of the upgrade

**Does a ranking measured on the probe survive at 24 tasks?** v1's stage B is the reason to
doubt it: heads tuned per-task on a single-task probe did not transfer to 24-task continual
training — 2 of 24 gains survived and 5 tasks got worse. So v2 promotes its top THREE probe
configurations into the real regime instead of one, which turns "the probe ranking transfers"
from an assumption into a measurement. If c2_top1 > c2_top2 > c2_top3 holds here, the probe is
predictive; if the order scrambles, then probe rank is not a deployment rank and the campaign
must say so plainly rather than quietly reporting the winner.

    python analysis/stage_c.py --arm "c2_top1=<dir>" --arm "c2_base=<dir>" \\
        --probe-ranking summary/finals_a.json -o summary/stage_c.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from common import CEILING, N_TRAIN, final_metrics

GROUPS = {
    "big": [t for t, n in N_TRAIN.items() if n >= 20000 and t != "material_type"],
    "mid": [t for t, n in N_TRAIN.items() if 3000 <= n < 20000],
    "small": [t for t, n in N_TRAIN.items() if n < 3000],
}
ACCURACY_TASKS = {"material_type"}
# 58 labels and a degenerate single-task baseline — excluded from the ranked set by the replay
# campaign for the same reason, so the two campaigns' mean R2 stay comparable.
EXCLUDED = {"magnetic_susceptibility"}


def score(metrics: dict[str, dict]) -> dict:
    r2 = {
        t: m["r2"]
        for t, m in metrics.items()
        if t not in ACCURACY_TASKS and t not in EXCLUDED and m.get("r2") is not None
    }
    result = {
        "n_tasks": len(r2),
        "mean_r2": statistics.fmean(r2.values()) if r2 else float("nan"),
        "per_task": {t: round(v, 4) for t, v in sorted(r2.items())},
        "deficit": {},
    }
    for group, tasks in GROUPS.items():
        deficits = [CEILING[t] - r2[t] for t in tasks if t in r2 and t in CEILING]
        result["deficit"][group] = statistics.fmean(deficits) if deficits else None
    all_def = [CEILING[t] - r2[t] for t in r2 if t in CEILING]
    result["deficit"]["worst_group"] = (
        max(v for v in result["deficit"].values() if v is not None)
        if any(v is not None for v in result["deficit"].values()) else None
    )
    result["deficit"]["all"] = statistics.fmean(all_def) if all_def else None
    for task in ACCURACY_TASKS:
        if task in metrics:
            result[task] = {k: metrics[task].get(k) for k in ("accuracy", "macro_f1")}
    return result


def transfer_check(arms: list[dict], probe_ranking: list[str] | None) -> dict:
    """Did the probe's ordering of top1/top2/top3 hold up at 24 tasks?

    Reported as a measurement with its own caveat: stage C' is ONE seed per arm (a 20-hour run
    cannot be repeated 25 times), so a small reordering here is not evidence of anything. Only a
    reordering larger than the probe's own resolved margins is informative, and the honest default
    when the arms land close together is "the probe could not be shown to mispredict", not "the
    probe transfers".
    """
    ranked = [a for a in arms if a["label"].startswith("c2_top")]
    if len(ranked) < 2:
        return {"checked": False, "reason": "fewer than two promoted arms"}
    deployed = [a["label"] for a in sorted(ranked, key=lambda a: a["mean_r2"], reverse=True)]
    expected = sorted(a["label"] for a in ranked)  # c2_top1 < c2_top2 < c2_top3 lexically
    spread = max(a["mean_r2"] for a in ranked) - min(a["mean_r2"] for a in ranked)
    return {
        "checked": True,
        "probe_order": expected,
        "deployed_order": deployed,
        "order_preserved": deployed == expected,
        "mean_r2_spread_across_promoted_arms": spread,
        "probe_ranking_source": probe_ranking,
        "caveat": (
            "One seed per arm. A spread smaller than the probe's own resolved margin is not "
            "evidence either way; report it as unresolved rather than as transfer."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--probe-ranking", type=Path, help="summary/finals_a.json, for the transfer check")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    arms = []
    missing = []
    for spec in args.arm:
        label, _, path = spec.partition("=")
        try:
            metrics, last_step = final_metrics(Path(path))
        except FileNotFoundError as exc:
            missing.append(f"{label}: {exc}")
            continue
        arms.append({"label": label, "dir": path, "last_step": last_step} | score(metrics))

    if not arms:
        raise SystemExit("no arms could be read: " + "; ".join(missing))

    probe_ranking = None
    if args.probe_ranking and args.probe_ranking.exists():
        probe_ranking = json.loads(args.probe_ranking.read_text()).get("ranking")

    by_label = {a["label"]: a for a in arms}

    def delta(a: str, b: str) -> float | None:
        if a in by_label and b in by_label:
            return by_label[b]["mean_r2"] - by_label[a]["mean_r2"]
        return None

    attribution = {
        "upgrade_alone": {
            "from": "c_base", "to": "c2_base",
            "delta_mean_r2": delta("c_base", "c2_base"),
            "what_it_isolates": "the LR-scheduler cadence fix (#45), with no tuning at all",
        },
        "tuning_on_top": {
            "from": "c2_base", "to": "c2_top1",
            "delta_mean_r2": delta("c2_base", "c2_top1"),
            "what_it_isolates": "v2's tuning, measured against the SAME image and scheduler",
        },
        "v1_headline_for_reference": {
            "from": "c_base", "to": "c_tuned",
            "delta_mean_r2": delta("c_base", "c_tuned"),
            "what_it_isolates": "v1's reported gain — tuning and a broken scheduler, entangled",
        },
    }

    out = {
        "stage": "stage_c",
        "arms": arms,
        "missing_arms": missing,
        "attribution": attribution,
        "transfer": transfer_check(arms, probe_ranking),
        "notes": [
            "deficit = single-task ceiling (H200, untuned arch) − final R², averaged in the group. "
            "The ceilings come from other hardware and another container, so they are a reference "
            "frame shared with REPORT_20260809, never a target.",
            "groups: big >=20k (6 tasks) · mid 3k-8.1k (14) · small <=1.2k (2). "
            "material_type is an accuracy task and is reported separately; magnetic_susceptibility "
            "(58 labels) is excluded, as in the replay campaign.",
            "read from per-step metrics JSONs, which stay complete across a --resume.",
            "ONE seed per arm — stage C' cannot be repeated at the seed counts the probe stages "
            "use, so differences here carry no seed band and small gaps are unresolved.",
        ],
    }

    width = max(len(a["label"]) for a in arms)
    print(f"{'arm':{width}s}  {'tasks':>5s}  {'mean R2':>8s}  {'big':>7s}  {'mid':>7s}  {'small':>7s}  material_type")
    for a in arms:
        clf = a.get("material_type") or {}
        clf_text = (f"acc {clf['accuracy']:.3f} / F1 {clf['macro_f1']:.3f}"
                    if clf.get("accuracy") is not None else "-")
        d = a["deficit"]
        fmt = lambda v: f"{v:7.4f}" if v is not None else "      -"  # noqa: E731
        print(f"{a['label']:{width}s}  {a['n_tasks']:5d}  {a['mean_r2']:8.4f}  "
              f"{fmt(d['big'])}  {fmt(d['mid'])}  {fmt(d['small'])}  {clf_text}")
    print()
    for name, entry in attribution.items():
        v = entry["delta_mean_r2"]
        print(f"  {name:26s} {entry['from']:9s} -> {entry['to']:9s} "
              f"{'n/a' if v is None else f'{v:+.4f}'}   {entry['what_it_isolates']}")
    t = out["transfer"]
    if t.get("checked"):
        print(f"\n  probe order {t['probe_order']} -> deployed {t['deployed_order']}: "
              f"{'PRESERVED' if t['order_preserved'] else 'SCRAMBLED'} "
              f"(spread {t['mean_r2_spread_across_promoted_arms']:.4f})")
    if missing:
        print("\n  MISSING ARMS (reported, not silently dropped):")
        for m in missing:
            print(f"    {m}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{args.output}")


if __name__ == "__main__":
    main()
