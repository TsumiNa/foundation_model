"""One paired encoder/seed trajectory; isolated workflow processes for every fit."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import joblib
import pandas as pd
import torch

import foundation_model
from foundation_model.workflows.finetune import build_finetune_config
from foundation_model.workflows.pretrain import build_pretrain_config

from benchmark import (
    DATE,
    SOURCE_TASKS,
    TARGET_TASKS,
    composition_metrics,
    file_hash,
    inference_raw,
    initialize_target,
    load_protocol,
    workflow_config,
    write_toml,
)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_command(arguments: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n" + " ".join(arguments) + "\n")
        stream.flush()
        subprocess.run(arguments, stdout=stream, stderr=subprocess.STDOUT, check=True)


def train_source(raw: dict, output: Path, protocol: Path, source_count: int) -> None:
    source = deepcopy(raw)
    source["output"] = {"dir": str(output)}
    source["pretrain"] = {
        "task_sequence": list(SOURCE_TASKS[:source_count]),
        "task_order": "fixed",
        "n_runs": 1,
        "resume": True,
        "replay": {"interval": 1, "amount": 0.3, "resample": "epoch", "per_task": {}},
    }
    frame = pd.read_parquet(source["datasets"]["source"]["path"])
    if int(frame["split"].eq("train").sum()) < source["data"]["batch_size"]:
        raise ValueError("Source training rows must fill at least one batch (workflow uses drop_last)")
    for name in SOURCE_TASKS[:source_count]:
        count = int((frame["split"].eq("train") & frame[name].notna()).sum())
        source["pretrain"]["replay"]["per_task"][name] = max(1500, int(0.3 * count))
    build_pretrain_config(source)  # fail before launching if the installed schema differs
    config_path = output / "pretrain.toml"
    write_toml(source, config_path)
    started = time.monotonic()
    run_command(["fm", "pretrain", "--config", str(config_path)], output / "workflow.log")
    if not (output / "training/final_model.pt").is_file():
        raise RuntimeError("Source workflow exited without its final checkpoint")
    stages = json.loads((output / "training/experiment_records.json").read_text())
    if len(stages) != source_count or any(stage.get("epochs_run", 0) < 1 for stage in stages):
        raise RuntimeError("Source workflow completed a stage without training")
    atomic_json(
        output / "done.json",
        {
            "source_count": source_count,
            "elapsed_seconds": time.monotonic() - started,
            "protocol": asdict(load_protocol(protocol)[0]),
        },
    )


def train_target(
    raw: dict,
    source: Path | None,
    output: Path,
    target: str,
    *,
    mode: str,
    k: int,
    arm: str,
    seed: int,
    fraction: float,
    protocol: Path,
    keep_checkpoint: bool = False,
) -> None:
    if (output / "done.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    target_frame = pd.read_parquet(raw["datasets"]["target"]["path"], columns=["split"])
    if int(target_frame["split"].eq("train").sum()) < raw["data"]["batch_size"]:
        raise ValueError("Target training rows must fill at least one batch (workflow uses drop_last)")
    initialized = output / "initializer.pt"
    init_info = initialize_target(raw, target, initialized, source=source)
    if init_info["source_task_count"] != k:
        raise ValueError("Checkpoint does not contain the requested number of source tasks")
    ft = deepcopy(raw)
    ft["finetune"] = {
        "checkpoint": str(initialized),
        "tasks": [target],
        "epochs": raw["training"]["max_epochs"],
        "freeze_encoder": mode == "frozen",
    }
    ft["output"] = {"dir": str(output)}
    build_finetune_config(ft)
    config_path = output / "finetune.toml"
    write_toml(ft, config_path)
    started = time.monotonic()
    run_command(["fm", "finetune", "--config", str(config_path)], output / "workflow.log")
    checkpoint = output / "training/final_model.pt"
    entry = next(
        t
        for t in json.loads((Path(raw["datasets"]["target"]["path"]).parent / f"manifest_{DATE}.json").read_text())[
            "targets"
        ]
        if t["task"] == target and t["seed"] == seed and t["fraction"] == fraction
    )
    scaler = joblib.load(Path(raw["datasets"]["target"]["path"]).parent / entry["scaler"])
    metrics = {}
    # The normal fine-tuning workflow already writes physical-unit test predictions.
    metrics["test"] = composition_metrics(
        pd.read_parquet(output / f"training/finetune/{target}_pred.parquet"), float(scaler.scale_[0])
    )
    validation = inference_raw(raw, target, checkpoint, output / "validation", "val")
    validation_path = output / "validation.toml"
    write_toml(validation, validation_path)
    run_command(["fm", "predict", "--config", str(validation_path)], output / "workflow.log")
    metrics["val"] = composition_metrics(
        pd.read_parquet(output / f"validation/predict/{target}_pred.parquet"), float(scaler.scale_[0])
    )
    summary = json.loads((output / "training/finetune_summary.json").read_text())
    if summary["epochs_run"] < 1:
        raise RuntimeError("Target workflow returned without any training epochs")
    atomic_json(
        output / "done.json",
        {
            "arm": arm,
            "seed": seed,
            "target": target,
            "fraction": fraction,
            "source_count": k,
            "mode": mode,
            "epochs_run": summary["epochs_run"],
            "elapsed_seconds": time.monotonic() - started,
            "metrics": metrics,
            **init_info,
            "protocol": asdict(load_protocol(protocol)[0]),
        },
    )
    initialized.unlink(missing_ok=True)
    # Keep every source checkpoint and representative target checkpoints. All predictions survive.
    if not keep_checkpoint and not (seed == 0 and k == 7 and fraction == 1 and mode == "full"):
        checkpoint.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--source-count", type=int, default=7)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    protocol, _, arms = load_protocol(args.protocol)
    if args.arm not in arms or args.seed not in protocol.seeds:
        raise ValueError("Unknown encoder or seed")
    if args.source_count not in range(1, 8):
        raise ValueError("source-count must be in 1..7")
    if metadata.version("foundation-model") != protocol.package_version:
        raise RuntimeError("Package version does not match the registered image")
    data_manifest = json.loads((args.data_dir / f"manifest_{DATE}.json").read_text())
    if data_manifest.get("functional_smoke") and args.max_epochs is None:
        raise ValueError("Functional fixture requires an explicit smoke epoch cap")
    if (
        not data_manifest.get("functional_smoke")
        and file_hash(args.data_dir / f"manifest_{DATE}.json") != protocol.data_manifest_sha256
    ):
        raise ValueError("Scientific data manifest differs from the registered campaign")
    for entry in [data_manifest["source"], *data_manifest["targets"]]:
        if file_hash(args.data_dir / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {entry['file']}")
        scaler_path = args.data_dir / entry.get("scaler", f"source_scalers_{DATE}.joblib")
        if file_hash(scaler_path) != entry["scaler_sha256"]:
            raise ValueError(f"Scaler checksum mismatch: {scaler_path.name}")
    if not args.cpu_smoke:
        if platform.machine() != "aarch64" or "/site-packages/" not in str(foundation_model.__file__):
            raise RuntimeError("RIKYU requires the ARM image-installed package, without source overrides")
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one allocated CUDA GPU must be visible")
        if os.environ.get("FM_SIF_SHA256") != protocol.sif_sha256:
            raise RuntimeError("Launcher did not verify the registered SIF checksum")
    lane = args.output_root.resolve() / f"{args.arm}_s{args.seed}"
    lane.mkdir(parents=True, exist_ok=True)
    # Refuse stale completion markers if any scientific input or execution mode changes.
    identity = {
        "protocol_sha256": file_hash(args.protocol),
        "manifest_sha256": file_hash(args.data_dir / f"manifest_{DATE}.json"),
        "benchmark_revision": os.environ.get("FM_BENCHMARK_REVISION"),
        "source_count": args.source_count,
        "max_epochs_override": args.max_epochs,
        "cpu_smoke": args.cpu_smoke,
    }
    identity_path = lane / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Output lane belongs to a different protocol, dataset or execution mode")
    atomic_json(identity_path, identity)
    runtime = {
        "image_uri": protocol.image_uri,
        "image_revision": protocol.image_revision,
        "package_version": metadata.version("foundation-model"),
        "package_path": str(foundation_model.__file__),
        "architecture": platform.machine(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "benchmark_revision": os.environ.get("FM_BENCHMARK_REVISION"),
        "cpu_smoke": args.cpu_smoke,
        "functional_smoke": bool(data_manifest.get("functional_smoke")),
        "max_epochs_override": args.max_epochs,
        "sif_sha256": os.environ.get("FM_SIF_SHA256"),
    }
    atomic_json(lane / "runtime.json", runtime)
    source_raw = workflow_config(args.protocol.resolve(), args.data_dir.resolve(), args.arm, args.seed)
    if args.cpu_smoke:
        source_raw["training"]["accelerator"] = "cpu"
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError("max-epochs must be positive")
        source_raw["training"]["max_epochs"] = args.max_epochs
    source_dir = lane / f"source_k{args.source_count}"
    if not (source_dir / "done.json").exists():
        train_source(source_raw, source_dir, args.protocol, args.source_count)
    if args.source_only:
        print(f"PASS source {args.arm} seed={args.seed}")
        return
    for target in TARGET_TASKS:
        for fraction in protocol.fractions:
            for k, mode in [
                (0, "scratch"),
                *[(k, m) for k in protocol.source_counts if k <= args.source_count for m in ("frozen", "full")],
            ]:
                raw = workflow_config(
                    args.protocol.resolve(),
                    args.data_dir.resolve(),
                    args.arm,
                    args.seed,
                    target=target,
                    fraction=fraction,
                    mode=mode,
                )
                if args.cpu_smoke:
                    raw["training"]["accelerator"] = "cpu"
                if args.max_epochs is not None:
                    raw["training"]["max_epochs"] = args.max_epochs
                checkpoint = None
                if k:
                    matches = sorted((source_dir / "training").glob(f"step{k:02d}_*/checkpoint.pt"))
                    if len(matches) != 1:
                        raise RuntimeError(f"Missing or ambiguous source checkpoint for k={k}")
                    checkpoint = matches[0]
                output = lane / "targets" / f"{target}_f{round(fraction * 100):03d}_k{k}_{mode}"
                # Each workflow is already a fresh subprocess; parent retains CPU-only initializers.
                train_target(
                    raw,
                    checkpoint,
                    output,
                    target,
                    mode=mode,
                    k=k,
                    arm=args.arm,
                    seed=args.seed,
                    fraction=fraction,
                    protocol=args.protocol,
                    keep_checkpoint=(
                        args.max_epochs is not None and k == args.source_count and fraction == 1 and mode == "full"
                    ),
                )
    atomic_json(
        lane / "done.json",
        {
            "arm": args.arm,
            "seed": args.seed,
            "source_count": args.source_count,
            "max_epochs_override": args.max_epochs,
            "cpu_smoke": args.cpu_smoke,
        },
    )
    print(f"PASS lane {args.arm} seed={args.seed}")


if __name__ == "__main__":
    main()
