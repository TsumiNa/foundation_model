"""Combine explicitly registered reusable fits with six-run protocol completions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REUSE_MANIFEST_SHA256 = "fda38283df55e0292072cad30876023c9d08551c974b39cb222525db3a9e63b8"


def usability_counts(frame: pd.DataFrame) -> list[dict]:
    """Count task/size effects when BOTH method means reach R² >= 0.2.

    All repeats contribute to each mean; never select individual runs by test score.
    This is a descriptive selected subset, not an overall transfer success rate.
    """
    points = frame.groupby(["task", "n", "arm"]).r2.mean().unstack("arm")
    points["eligible"] = (points.scratch >= 0.2) & (points.transfer >= 0.2)
    points["gain"] = points.transfer - points.scratch
    rows = []
    for n, group in points.groupby(level="n"):
        retained = group[group.eligible]
        rows.append(
            {
                "n": int(n),
                "evaluated": len(group),
                "retained": len(retained),
                "excluded": int((~group.eligible).sum()),
                "better": int((retained.gain > 0.01).sum()),
                "similar": int((retained.gain.abs() <= 0.01).sum()),
                "worse": int((retained.gain < -0.01).sum()),
            }
        )
    return rows


def collect(root: Path, reuse_root: Path, manifest_path: Path, revision: str, output: Path) -> dict:
    """Accept two pinned identities without rewriting either campaign's provenance."""
    manifest = json.loads(manifest_path.read_text())
    old_manifest = manifest["reuse_input_manifest"]
    encoded = manifest["reuse_identity"]
    if hashlib.sha256(json.dumps(old_manifest, indent=2).encode()).hexdigest() != REUSE_MANIFEST_SHA256:
        raise ValueError("Reuse input manifest is not the pinned completed campaign")
    if encoded != {
        "manifest": REUSE_MANIFEST_SHA256,
        "revision": "630e973dfa51bde36c0ce8dbaee34919ad9f73ed",
        "image_hash": "f90adc81f1148db2fff0ae94551b751e77f82c5d52d032ea54641f746dbe2f67",
        "smoke": False,
        "script_sha256": old_manifest["script_sha256"],
    }:
        raise ValueError("Reuse runtime identity is not the pinned completed campaign")
    if old_manifest["cases"] != manifest["reuse_cases"]:
        raise ValueError("Reused case registry differs")
    new_subsets = {(s["n"], s["seed"]): s for s in manifest["subsets"]}
    if any(new_subsets.get((s["n"], s["seed"])) != s for s in old_manifest["subsets"]):
        raise ValueError("Reused subset bytes differ")
    if any(manifest["source_hashes"].get(k) != v for k, v in old_manifest["source_hashes"].items()):
        raise ValueError("Reused source data or recipe bytes differ")
    if old_manifest["script_sha256"]["run.py"] != manifest["script_sha256"]["run.py"]:
        raise ValueError("Scientific worker changed")
    if (
        old_manifest["recipes"] != manifest["recipes"]
        or old_manifest["checkpoints"] != manifest["checkpoints"][:3]
        or old_manifest["auxiliary"] != manifest["auxiliary"]
    ):
        raise ValueError("Reuse inputs no longer match the paired protocol")
    if len(revision) != 40:
        raise ValueError("Require full merged runtime revision")
    expected_new = {
        "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "revision": revision,
        "image_hash": encoded["image_hash"],
        "smoke": False,
        "script_sha256": manifest["script_sha256"],
    }
    rows = []
    for directory, registry, identity, origin in (
        (reuse_root, manifest["reuse_cases"], encoded, "reused_oct8"),
        (root, manifest["cases"], expected_new, "new_six_run"),
    ):
        by_index = {c["case"]: c for c in registry}
        for marker in sorted(directory.glob("case*/done.json")):
            lane = json.loads(marker.read_text())
            case = lane["case"]
            if by_index.get(case["case"]) != case or marker.parent.name != f"case{case['case']:03d}":
                raise ValueError("Unregistered case")
            if identity["smoke"] or lane["identity"] != identity:
                raise ValueError("Unexpected campaign identity or smoke fit")
            pair = [json.loads((marker.parent / arm / "done.json").read_text()) for arm in ("scratch", "transfer")]
            if pair[0]["head_initialization"] != pair[1]["head_initialization"]:
                raise ValueError("Unpaired target-head initialization")
            if pair[0]["metrics"]["test_hash"] != pair[1]["metrics"]["test_hash"]:
                raise ValueError("Unpaired test labels")
            for arm, record in zip(("scratch", "transfer"), pair, strict=True):
                if record["identity"] != identity or record["case"] != case or record["arm"] != arm:
                    raise ValueError("Fit identity mismatch")
                checkpoint = next(c for c in manifest["checkpoints"] if c["file"] == case["checkpoint"])
                if record["checkpoint_hash"] != checkpoint["sha256"]:
                    raise ValueError("Source checkpoint mismatch")
                metrics = record["metrics"]
                if not np.isfinite([metrics[k] for k in ("r2", "rmse", "mae")]).all() or metrics["steps"] <= 0:
                    raise ValueError("Nonfinite metric or untrained fit")
                rows.append({**case, "arm": arm, "origin": origin, **metrics, "seconds": record["seconds"]})
    frame = pd.DataFrame(rows)
    if len(frame) and frame.duplicated(["task", "n", "seed", "arm"]).any():
        raise ValueError("Duplicate scientific fit")
    expected_keys = {(c["task"], c["n"], c["seed"]) for c in manifest["all_cases"]}
    actual_keys = {(r["task"], r["n"], r["seed"]) for r in rows}
    if not actual_keys <= expected_keys or len(expected_keys) != manifest["paired_cases"]:
        raise ValueError("Unexpected scientific coverage")
    # The exact fixed test set must also be identical across sizes, seeds and campaigns.
    if len(frame) and frame.groupby("task").test_hash.nunique().gt(1).any():
        raise ValueError("Test set changed across the unified curve")
    summary = {
        "complete_pairs": len(frame) // 2,
        "expected_pairs": manifest["paired_cases"],
        "partial": actual_keys != expected_keys,
        "identities": {"new": expected_new, "reuse": encoded},
        "points": [],
        "usability_threshold": 0.2,
        "usability_rule": "Both mean test R2 >= 0.2; all repeats retained; descriptive test-score filter",
    }
    if len(frame):
        for (task, n, arm), group in frame.groupby(["task", "n", "arm"]):
            summary["points"].append(
                {
                    "task": task,
                    "n": int(n),
                    "arm": arm,
                    "seeds": sorted(group.seed.tolist()),
                    "seed_count": len(group),
                    **{f"{k}_mean": float(group[k].mean()) for k in ("r2", "rmse", "mae")},
                    **{f"{k}_std": float(group[k].std()) if len(group) > 1 else None for k in ("r2", "rmse", "mae")},
                }
            )
        complete_points = frame.groupby(["task", "n", "arm"]).seed.nunique().unstack("arm")
        complete_keys = complete_points[(complete_points.scratch == 6) & (complete_points.transfer == 6)].index
        balanced = frame.set_index(["task", "n"]).loc[lambda f: f.index.isin(complete_keys)].reset_index()
        summary["usable_task_counts"] = usability_counts(balanced) if len(balanced) else []
        summary["task_count_summary_requires_six_runs"] = True
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reuse-root", "manifest", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()
    print(json.dumps(collect(args.root, args.reuse_root, args.manifest, args.revision, args.output), indent=2))
