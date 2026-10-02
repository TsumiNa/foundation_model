"""Image-only source-training stability probe; kept separate from transfer evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import tomllib
from typing import Any

import pandas as pd
import torch

import foundation_model
from foundation_model.workflows._engine import build_model_for_checkpoint, checkpoint_task_order
from foundation_model.workflows._sections import build_model_section
from foundation_model.workflows.recording import _fold_disabled_heads, load_checkpoint_state
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

from benchmark import DATE, SOURCE_TASKS, file_hash, load_protocol, workflow_config
from run_lane import atomic_json, train_source


@dataclass
class StabilityConfig:
    source_count: int
    max_epochs: int
    seeds: list[int]
    arms: list[str]
    encoder_learning_rates: list[float]
    control_arm: str
    diagnostic_per_class: int

    def __post_init__(self) -> None:
        for name, value in (
            ("source_count", self.source_count),
            ("max_epochs", self.max_epochs),
            ("diagnostic_per_class", self.diagnostic_per_class),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.source_count > len(SOURCE_TASKS):
            raise ValueError("source_count exceeds the registered source sequence")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if not self.arms or len(set(self.arms)) != len(self.arms) or self.control_arm in self.arms:
            raise ValueError("arms must be unique and distinct from control_arm")
        if not self.encoder_learning_rates or any(
            isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0
            for lr in self.encoder_learning_rates
        ):
            raise ValueError("encoder_learning_rates must be finite and positive")
        if len(set(self.encoder_learning_rates)) != len(self.encoder_learning_rates):
            raise ValueError("encoder_learning_rates must be unique")


def probe_cases(protocol: Path, probe_config: Path) -> tuple[StabilityConfig, list[dict[str, Any]]]:
    config = StabilityConfig(**tomllib.loads(probe_config.read_text()))
    registered, _, arms = load_protocol(protocol)
    if not set(config.seeds) <= set(registered.seeds) or not {*config.arms, config.control_arm} <= set(arms):
        raise ValueError("Probe arms and seeds must belong to the main protocol")
    cases = [
        {"arm": arm, "seed": seed, "encoder_lr": lr, "control": False}
        for arm in config.arms
        for lr in config.encoder_learning_rates
        for seed in config.seeds
    ]
    cases += [
        {"arm": config.control_arm, "seed": seed, "encoder_lr": arms[config.control_arm]["encoder_lr"], "control": True}
        for seed in config.seeds
    ]
    return config, cases


def diagnostic_compositions(frame: pd.DataFrame, per_class: int) -> list[str]:
    """A fixed, label-stratified validation probe, never target-test selection."""
    selected = frame.loc[frame["split"].eq("val") & frame["material_type"].notna()]
    if selected.empty:
        raise ValueError("Representation diagnostics require labeled validation compositions")
    return [
        comp
        for _, group in selected.groupby("material_type", sort=True)
        for comp in group.sample(n=min(per_class, len(group)), random_state=20261003)["composition"].tolist()
    ]


def representation_statistics(z: torch.Tensor) -> dict[str, float]:
    if z.ndim != 2 or z.shape[0] < 2 or not torch.isfinite(z).all():
        raise ValueError("Latent diagnostics require finite embeddings of at least two compositions")
    h = z.tanh()
    return {
        "pre_tanh_abs_median": float(z.abs().median()),
        "pre_tanh_abs_max": float(z.abs().max()),
        "saturated_fraction": float((h.abs() > 0.99).float().mean()),
        "low_variance_dimension_fraction": float((h.var(dim=0, unbiased=False) < 1e-6).float().mean()),
        "mean_latent_variance": float(h.var(dim=0, unbiased=False).mean()),
    }


def checkpoint_diagnostics(
    checkpoint: Path, catalog: TaskCatalog, raw: dict[str, Any], task_names: list[str], x: torch.Tensor
) -> dict[str, Any]:
    if checkpoint.suffix == ".ckpt":
        # Only this probe's own Lightning files: their hparams contain config dataclasses.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {"model": _fold_disabled_heads(saved["state_dict"])}
        del saved
    else:
        state = load_checkpoint_state(checkpoint)
    model = build_model_for_checkpoint(catalog, build_model_section(raw["model"]), task_names)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(x.device).eval()
    with torch.inference_mode():
        result: dict[str, Any] = representation_statistics(model.encoder(x))
    result["batch_norm"] = [
        {
            "module": name,
            "running_var_min": float(module.running_var.min()),
            "running_var_median": float(module.running_var.median()),
        }
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.BatchNorm1d) and module.running_var is not None
    ]
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_sha256"] = file_hash(checkpoint)
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    args = parser.parse_args()
    config, cases = probe_cases(args.protocol, args.probe_config)
    if not 0 <= args.case < len(cases):
        raise ValueError("Case index outside the registered probe")
    case = cases[args.case]
    base, _, _ = load_protocol(args.protocol)
    if metadata.version("foundation-model") != base.package_version:
        raise RuntimeError("Package version differs from the registered image")
    if platform.machine() != "aarch64" or "/site-packages/" not in str(foundation_model.__file__):
        raise RuntimeError("Use the ARM image-installed package, without source overrides")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated CUDA GPU is required")
    if os.environ.get("FM_SIF_SHA256") != base.sif_sha256:
        raise RuntimeError("The launcher must verify the registered SIF")
    revision = os.environ.get("FM_BENCHMARK_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Record the merged experiment revision")
    manifest_path = args.data_dir / f"manifest_{DATE}.json"
    if file_hash(manifest_path) != base.data_manifest_sha256:
        raise ValueError("Data manifest differs from the registered scientific data")
    manifest = json.loads(manifest_path.read_text())
    source = manifest["source"]
    if (
        file_hash(args.data_dir / source["file"]) != source["sha256"]
        or file_hash(args.data_dir / f"source_scalers_{DATE}.joblib") != source["scaler_sha256"]
    ):
        raise ValueError("Source data or training-fitted scaler checksum differs")
    lane = args.output_root.resolve() / f"case{args.case:03d}_{case['arm']}_s{case['seed']}"
    lane.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol_sha256": file_hash(args.protocol),
        "probe_config_sha256": file_hash(args.probe_config),
        "manifest_sha256": file_hash(manifest_path),
        "benchmark_revision": revision,
        "case": case,
    }
    if (lane / "identity.json").exists() and json.loads((lane / "identity.json").read_text()) != identity:
        raise ValueError("Stale probe output identity")
    atomic_json(lane / "identity.json", identity)
    if (lane / "probe_done.json").exists():
        print(f"PASS existing probe case {args.case}")
        return
    atomic_json(
        lane / "runtime.json",
        {
            "benchmark_revision": revision,
            "image_revision": base.image_revision,
            "sif_sha256": base.sif_sha256,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "package_version": base.package_version,
            "package_path": str(foundation_model.__file__),
            "architecture": platform.machine(),
            "cpu_smoke": False,
            "functional_smoke": False,
            "max_epochs_override": config.max_epochs,
            "stability_probe": True,
        },
    )
    raw = workflow_config(args.protocol.resolve(), args.data_dir.resolve(), case["arm"], case["seed"])
    raw["training"]["encoder_lr"] = case["encoder_lr"]
    raw["training"]["max_epochs"] = config.max_epochs
    raw["training"]["checkpoint"] = {"enabled": True, "save_last": True, "save_top_k": 1, "filename": "best"}
    source_dir = lane / f"source_k{config.source_count}"
    if not (source_dir / "done.json").exists():
        train_source(raw, source_dir, args.protocol, config.source_count)
    frame = pd.read_parquet(args.data_dir / source["file"], columns=["composition", "split", "material_type"])
    compositions = diagnostic_compositions(frame, config.diagnostic_per_class)
    catalog = TaskCatalog(build_task_catalog_config({k: raw[k] for k in ("data", "descriptor", "datasets", "tasks")}))
    descriptors = catalog.descriptor_fn()(compositions)
    if list(descriptors.index) != compositions:
        raise ValueError("Probe descriptor membership/order changed")
    x = torch.tensor(descriptors.to_numpy(), dtype=torch.float32, device="cuda")
    diagnostics = []
    for k, task in enumerate(SOURCE_TASKS[: config.source_count], 1):
        step = source_dir / "training" / f"step{k:02d}_{task}"
        end = step / "checkpoint.pt"
        names = checkpoint_task_order(load_checkpoint_state(end))
        diagnostics.append({"step": k, "kind": "end_of_stage", **checkpoint_diagnostics(end, catalog, raw, names, x)})
        # Also keep callback/optimizer evidence. Versioned files can occur after an interrupted attempt.
        for path in sorted((step / "lightning").glob("*.ckpt")):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            lrs = [
                group["lr"] for optimizer in saved.get("optimizer_states", []) for group in optimizer["param_groups"]
            ]
            scores = [
                float(callback["best_model_score"])
                for callback in saved.get("callbacks", {}).values()
                if isinstance(callback, dict) and callback.get("best_model_score") is not None
            ]
            epoch = saved["epoch"]
            del saved
            diagnostics.append(
                {
                    "step": k,
                    "kind": path.stem,
                    "epoch": epoch,
                    "optimizer_learning_rates": lrs,
                    "callback_best_scores": scores,
                    **checkpoint_diagnostics(path, catalog, raw, names, x),
                }
            )
        if not any(d["step"] == k and d["kind"].startswith("best") for d in diagnostics):
            raise RuntimeError(f"Missing best-validation checkpoint for step {k}")
        if not any(d["step"] == k and d["kind"].startswith("last") for d in diagnostics):
            raise RuntimeError(f"Missing last-epoch checkpoint for step {k}")
    atomic_json(
        lane / "probe_done.json",
        {
            "case": case,
            "source_count": config.source_count,
            "max_epochs": config.max_epochs,
            "diagnostic_compositions": compositions,
            "diagnostics": diagnostics,
        },
    )
    print(f"PASS stability probe case {args.case}")


if __name__ == "__main__":
    main()
