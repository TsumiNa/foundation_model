"""Paired exact-count regression fits through the installed production package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import foundation_model
from foundation_model.workflows import finetune, pretrain
from foundation_model.workflows.recording import RunRecorder


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe(manifest: dict, case: dict, arm: str, data: Path, smoke: bool) -> dict:
    raw = copy.deepcopy(manifest["recipes"][arm])
    subset = next(s for s in manifest["subsets"] if s["n"] == case["n"] and s["seed"] == case["seed"])
    for name, spec in raw["datasets"].items():
        spec["path"] = str(data / (subset["file"] if name == "qc" else spec["path"]))
    raw["training"].update(seed=2025 + case["seed"], max_epochs=2 if smoke else 150)
    raw["training"]["logging"] = {"csv": True, "tensorboard": False}
    if arm == "scratch":
        raw["pretrain"]["task_sequence"] = [case["task"]]
    else:
        raw["finetune"].update(tasks=[case["task"]], epochs=2 if smoke else 150, freeze_encoder=False)
    return raw


def audit_fit(root: Path, task: str, arm: str) -> dict:
    step = root / "training" / (f"step01_{task}" if arm == "scratch" else "finetune")
    frame = pd.read_parquet(step / f"{task}_pred.parquet")
    if len(frame) < 2 or frame.composition.duplicated().any() or not np.isfinite(frame[["true", "pred"]]).all().all():
        raise ValueError("Invalid held-out predictions")
    logs = list((root / "logs").glob("*/version_*/metrics.csv"))
    if len(logs) != 1:
        raise ValueError("Expected one training history")
    history = pd.read_csv(logs[0])
    loss = history["train_final_loss_epoch"].dropna()
    if history["step"].max() <= 0 or loss.empty or not np.isfinite(loss).all():
        raise ValueError("Missing actual finite training steps")
    residual = frame.pred.to_numpy() - frame.true.to_numpy()
    denominator = np.square(frame.true - frame.true.mean()).sum()
    if denominator <= 0:
        raise ValueError("Undefined R2 for constant targets")
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.square(residual).sum() / denominator),
        "test_samples": len(frame),
        "test_hash": hashlib.sha256(frame[["composition", "true"]].to_json().encode()).hexdigest(),
        "epochs": int(history.epoch.max()) + 1,
        "steps": int(history.step.max()) + 1,
        "initial_train_loss": float(loss.iloc[0]),
        "last_train_loss": float(loss.iloc[-1]),
    }


def restart_incomplete(root: Path, arm: str) -> None:
    """Preserve one failed attempt; permit at most one explicit same-identity retry."""
    dest = root / arm
    if dest.exists():
        if list(root.glob(f"failed_{arm}_*")):
            raise RuntimeError("Bounded fit recovery exhausted; inspect saved failure logs")
        dest.rename(root / f"failed_{arm}_{time.time_ns()}")


def run(data: Path, output: Path, index: int, revision: str, image_hash: str, smoke: bool) -> None:
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    case = manifest["cases"][index]
    if metadata.version("foundation-model") != manifest["package"] or not torch.cuda.is_available():
        raise RuntimeError("Requires package 0.5.0 and an allocated CUDA GPU")
    if "site-packages" not in str(foundation_model.__file__) or os.environ.get("PYTHONPATH"):
        raise RuntimeError("Requires installed image package without PYTHONPATH")
    subset = next(s for s in manifest["subsets"] if s["n"] == case["n"] and s["seed"] == case["seed"])
    checkpoint = next(s for s in manifest["checkpoints"] if s["file"] == case["checkpoint"])
    for item in (subset, checkpoint, *manifest["auxiliary"]):
        if digest(data / item["file"]) != item["sha256"]:
            raise ValueError("Input checksum mismatch")
    identity = {"manifest": digest(manifest_path), "revision": revision, "image_hash": image_hash, "smoke": smoke}
    if len(revision) != 40 or len(image_hash) != 64:
        raise ValueError("Requires full script revision and image hash")
    root = output / f"case{index:03d}"
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "identity.json"
    expected = {"identity": identity, "case": case}
    if identity_path.exists() and json.loads(identity_path.read_text()) != expected:
        raise ValueError("Existing lane has a different identity")
    identity_path.write_text(json.dumps(expected, indent=2))
    records = {}
    torch.set_num_threads(4)
    for arm in ("scratch", "transfer"):
        dest = root / arm
        marker = dest / "done.json"
        if marker.exists():
            result = json.loads(marker.read_text())
            if result["identity"] != identity or result["case"] != case:
                raise ValueError("Completed fit has a different identity")
            records[arm] = result
            continue
        restart_incomplete(root, arm)
        raw = recipe(manifest, case, arm, data, smoke)
        cfg = (
            pretrain.build_pretrain_config(raw, output_dir=dest)
            if arm == "scratch"
            else finetune.build_finetune_config(raw, output_dir=dest, checkpoint=data / case["checkpoint"])
        )
        start = time.monotonic()
        rec = RunRecorder(dest)
        try:
            rec.write_provenance(config=cfg, argv=["paired regression study"], seeds={"training": cfg.training.seed})
            (pretrain.run if arm == "scratch" else finetune.run)(cfg, rec)
        finally:
            rec.close()
        result = {
            "case": case,
            "arm": arm,
            "identity": identity,
            "metrics": audit_fit(dest, case["task"], arm),
            "seconds": time.monotonic() - start,
            "package_path": str(foundation_model.__file__),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_hash": checkpoint["sha256"],
        }
        tmp = dest / "done.tmp"
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
        tmp.replace(marker)
        records[arm] = result
    if records["scratch"]["metrics"]["test_hash"] != records["transfer"]["metrics"]["test_hash"]:
        raise ValueError("Paired test labels differ")
    tmp = root / "done.tmp"
    tmp.write_text(json.dumps({"case": case, "identity": identity}, indent=2))
    tmp.replace(root / "done.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--image-hash", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.data, args.output, args.case, args.revision, args.image_hash, args.smoke)
