# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Plan and execute the agreed AGIS compound-holdout experiment.

Each final fit uses the same newly initialized pressure head and fixed epoch budget.
Warm-start routes are shared by the frozen/unfrozen final fits, and exclude their target pressure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning import seed_everything
from loguru import logger

from foundation_model.models.model_config import KernelRegressionTaskConfig
from foundation_model.models.task_head.kernel_regression import KernelRegressionHead
from foundation_model.workflows._engine import build_empty_model, build_head_config
from foundation_model.workflows.finetune import FinetuneConfig, build_finetune_config, run as finetune_run
from foundation_model.workflows.pretrain import PretrainConfig, build_pretrain_config, run as pretrain_run
from foundation_model.workflows.recording import RunRecorder, load_checkpoint_state
from foundation_model.workflows.task_catalog import TaskCatalog

PRESSURES = (0, 10, 20)


class Route(StrEnum):
    DIRECT = "direct"
    WARM = "warm"
    SCRATCH = "scratch"


@dataclass(kw_only=True)
class CampaignSettings:
    date: str = "20261001"
    final_epochs: int = 1000
    warm_epochs: int = 150
    first_warm_checkpoints: int = 5
    seed: int = 20261001

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{8}", self.date):
            raise ValueError("date must have YYYYMMDD format")
        datetime.strptime(self.date, "%Y%m%d")
        for name in ("final_epochs", "warm_epochs", "first_warm_checkpoints"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.first_warm_checkpoints > 10:
            raise ValueError("first_warm_checkpoints cannot exceed the ten selected checkpoints")


@dataclass(kw_only=True)
class RunUnit:
    route: Route
    fold: int
    checkpoint_index: int | None
    pressure: int

    def __post_init__(self) -> None:
        self.route = Route(self.route)
        if self.fold not in range(1, 9) or self.pressure not in PRESSURES:
            raise ValueError("Require fold 1–8 and pressure 0/10/20 GPa")
        if self.route == Route.SCRATCH:
            if self.checkpoint_index is not None:
                raise ValueError("Scratch models have no pretrained checkpoint or ten repeats")
        elif self.checkpoint_index not in range(10):
            raise ValueError("Transfer routes require checkpoint_index 0–9")

    @property
    def name(self) -> str:
        checkpoint = "scratch" if self.checkpoint_index is None else f"c{self.checkpoint_index + 1:02d}"
        return f"{self.route}_f{self.fold:02d}_{checkpoint}_p{self.pressure}"


def build_units() -> list[RunUnit]:
    return [
        RunUnit(route=route, fold=fold, checkpoint_index=checkpoint, pressure=pressure)
        for route in Route
        for fold in range(1, 9)
        for checkpoint in ([None] if route == Route.SCRATCH else range(10))
        for pressure in PRESSURES
    ]


def plan_campaign(base_config: Path, output: Path, settings: CampaignSettings) -> Path:
    selection_path = Path(f"data/agis_pretrained_{settings.date}/selection_{settings.date}.json")
    selection = json.loads(selection_path.read_text())
    if len(selection["models"]) != 10 or len({m["sha256"] for m in selection["models"]}) != 10:
        raise ValueError("Require ten distinct randomly selected non-AGIS checkpoints")
    folds_path = Path(f"data/agis_preprocessing_{settings.date}/manifest_{settings.date}.json")
    preprocessing = json.loads(folds_path.read_text())
    if preprocessing["n_compounds"] != 8 or preprocessing["n_curves"] != 24:
        raise ValueError("Require the complete eight-compound, three-pressure dataset")
    if len(preprocessing["folds"]) != 8:
        raise ValueError("Require all eight outer folds")
    base = tomllib.loads(base_config.read_text())
    # Training architecture is the architecture of the selected checkpoint population.
    model_keys = ("latent_dim", "encoder_hidden_dims", "head_hidden_dims")
    architecture = {key: selection["models"][0]["hyperparameters"][key] for key in model_keys}
    for selected in selection["models"]:
        if any(selected["hyperparameters"][key] != architecture[key] for key in model_keys):
            raise ValueError("The selected checkpoint architectures differ")
    base["model"].update(architecture)
    first_warm = 8 * settings.first_warm_checkpoints * 3 * 2
    manifest = {
        "settings": asdict(settings),
        "base_config": base,
        "selection": selection,
        "preprocessing": preprocessing,
        "preprocessing_sha256": hashlib.sha256(folds_path.read_bytes()).hexdigest(),
        "units": [asdict(unit) for unit in build_units()],
        "expected_final_models": {
            "direct": 480,
            "warm_first": first_warm,
            "warm_rest": 480 - first_warm,
            "scratch": 24,
        },
        "head_initialization": "Identical new target-head state per fold/pressure across all arms and checkpoints",
        "holdout": "All AGIS pressures held out; no AGIS held-out label drives fit, scaler or epoch selection",
        "pretraining_overlap": "La3Ni2O7 has an original non-AGIS Tc label; this is an AGIS-label holdout",
    }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    for stage in ("direct", "scratch", "warm_first", "warm_rest"):
        indices = []
        for index, raw in enumerate(manifest["units"]):
            unit = RunUnit(**raw)
            unit_stage = unit.route.value
            if unit.route == Route.WARM:
                assert unit.checkpoint_index is not None
                unit_stage = "warm_first" if unit.checkpoint_index < settings.first_warm_checkpoints else "warm_rest"
            if stage == unit_stage:
                indices.append(str(index))
        (output / f"{stage}.txt").write_text("\n".join(indices) + "\n")
    logger.info("Planned {} units / 984 final models at {}", len(manifest["units"]), path)
    return path


def final_config(manifest: dict[str, Any], unit: RunUnit) -> dict[str, Any]:
    """Build identical final-fit settings; the initializer determines the training arm."""
    settings = CampaignSettings(**manifest["settings"])
    raw = copy.deepcopy(manifest["base_config"])
    raw.pop("pretrain", None)
    raw.pop("output", None)
    fragment = tomllib.loads(
        Path(f"data/agis_preprocessing_{settings.date}/fold_{unit.fold:02d}/tasks_{settings.date}.toml").read_text()
    )
    raw["datasets"].update(fragment["datasets"])
    raw["tasks"].extend(fragment["tasks"])
    # One full batch of seven compounds. No test/validation label enters the fit.
    raw["data"].update(batch_size=7, num_workers=0, val_split=0.0, test_split=0.0)
    raw["training"].update(
        seed=settings.seed + unit.fold * 100 + unit.pressure,
        accelerator="gpu",
        devices=1,
        encoder_lr=0.002,
        kr_lr=0.0005,
        max_epochs=settings.final_epochs,
    )
    raw["training"]["early_stopping"] = {"enabled": False}
    raw["training"]["checkpoint"] = {"enabled": False}
    raw["training"]["logging"] = {"csv": True, "tensorboard": False}
    raw["training"]["scheduler"] = {
        "enabled": True,
        "mode": "min",
        "factor": 0.5,
        "patience": 20,
        "min_lr": 1e-5,
        "monitor": "train_final_loss_epoch",
    }
    raw["finetune"] = {
        "tasks": [f"agis_rho_{unit.pressure}gpa"],
        "epochs": settings.final_epochs,
        "freeze_encoder": False,
        "add_new_tasks": False,
    }
    if unit.route == Route.SCRATCH:
        raw["tasks"] = [task for task in raw["tasks"] if task["name"] == f"agis_rho_{unit.pressure}gpa"]
        raw["datasets"] = fragment["datasets"]
    return raw


def warm_config(manifest: dict[str, Any], unit: RunUnit) -> dict[str, Any]:
    if unit.route != Route.WARM:
        raise ValueError("Only warm routes have a continuation stage")
    settings = CampaignSettings(**manifest["settings"])
    raw = final_config(manifest, unit)
    raw.pop("finetune")
    raw["data"].update(batch_size=256, val_split=0.1, test_split=0.1)
    raw["training"].update(max_epochs=settings.warm_epochs)
    # Validation comes exclusively from the original non-AGIS tasks, never the heldout curve.
    raw["training"]["early_stopping"] = {
        "enabled": True,
        "monitor": "val_final_loss",
        "mode": "min",
        "patience": 24,
        "min_delta": 1e-4,
    }
    raw["training"]["scheduler"]["patience"] = 5
    replay = copy.deepcopy(manifest["base_config"]["pretrain"]["replay"])
    replay.setdefault("per_task", {}).update({f"agis_rho_{p}gpa": 7 for p in PRESSURES})
    raw["pretrain"] = {
        "task_sequence": [f"agis_rho_{p}gpa" for p in PRESSURES if p != unit.pressure],
        "n_runs": 1,
        "task_order": "fixed",
        "replay": replay,
    }
    return raw


def initialize_target(raw: dict[str, Any], source: Path | None, destination: Path) -> Path:
    """Reuse a precisely identical fresh target head without changing source encoder weights."""
    cfg = build_finetune_config(raw, checkpoint=source or destination, output_dir=destination.parent)
    catalog = TaskCatalog(cfg.catalog)
    seed_everything(cfg.training.seed, workers=True)
    if source is None:
        model = build_empty_model(catalog, cfg.model, cfg.training)
        state: dict[str, Any] = {"model": model.state_dict(), "task_sequence": []}
    else:
        state = load_checkpoint_state(source)
    target = cfg.tasks[0]
    if target in state["task_sequence"]:
        raise ValueError("The target pressure must be absent before its final fit")
    seed_everything(cfg.training.seed + 10_000, workers=True)
    config = build_head_config(catalog, cfg.model, cfg.training, target)
    if not isinstance(config, KernelRegressionTaskConfig):
        raise ValueError("AGIS final task must be kernel regression")
    head = KernelRegressionHead(config)
    state = {**state, "model": dict(state["model"]), "task_sequence": [*state["task_sequence"], target]}
    state["model"].update({f"task_heads.{target}.{name}": value for name, value in head.state_dict().items()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return destination


def verify_final(output: Path, target: str, heldout: str, epochs: int) -> None:
    frame = pd.read_parquet(output / "training/finetune" / f"{target}_pred.parquet")
    if len(frame) != 300 or set(frame.composition) != {heldout}:
        raise ValueError("Final prediction must contain exactly 300 points of the heldout compound")
    if not np.isfinite(frame[["true", "pred", "t"]].to_numpy()).all():
        raise ValueError("Nonfinite final predictions or targets")
    if not np.allclose(frame.t, np.linspace(6, 290, 300)):
        raise ValueError("The final prediction grid differs from the shared 6–290 K grid")
    summary = json.loads((output / "training/finetune_summary.json").read_text())
    if summary["epochs_run"] != epochs:
        raise ValueError("The fixed final-fit epoch budget was not completed")
    if not (output / "training/final_model.pt").is_file():
        raise FileNotFoundError("Missing final checkpoint")
    if summary["freeze_encoder"]:
        initial = load_checkpoint_state(output / "initial_model.pt")["model"]
        final = load_checkpoint_state(output / "training/final_model.pt")["model"]
        if any(not torch.equal(value, final[key]) for key, value in initial.items() if key.startswith("encoder.")):
            raise ValueError("Frozen encoder parameters or BatchNorm buffers changed")


def fit_spec(path: Path) -> None:
    spec = json.loads(path.read_text())
    raw = spec["raw"]
    output = Path(spec["output"])
    source = Path(spec["source"]) if spec["source"] else None
    cfg: FinetuneConfig | PretrainConfig
    if spec["mode"] == "finetune":
        initial = initialize_target(raw, source, output / "initial_model.pt")
        cfg = build_finetune_config(raw, checkpoint=initial, output_dir=output)
    else:
        cfg = build_pretrain_config(raw, checkpoint=source, output_dir=output, resume=True)
    recorder = RunRecorder(output)
    try:
        recorder.write_provenance(config=cfg, argv=sys.argv, seeds={"training": cfg.training.seed})
        if isinstance(cfg, FinetuneConfig):
            finetune_run(cfg, recorder)
        else:
            pretrain_run(cfg, recorder)
    finally:
        recorder.close()
    if isinstance(cfg, FinetuneConfig):
        verify_final(output, cfg.tasks[0], spec["heldout"], cfg.epochs)
    (output / "DONE").write_text("completed\n")


def execute_unit(manifest_path: Path, index: int, output_root: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    unit = RunUnit(**manifest["units"][index])
    settings = CampaignSettings(**manifest["settings"])
    root = output_root / unit.name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "DONE").exists():
        logger.info("Already complete: {}", unit.name)
        return
    heldout = manifest["preprocessing"]["folds"][unit.fold - 1]["heldout_composition"]
    source: Path | None = None
    if unit.checkpoint_index is not None:
        selected = manifest["selection"]["models"][unit.checkpoint_index]
        source = Path(f"data/agis_pretrained_{settings.date}") / f"{selected['run']}.pt"
        if hashlib.sha256(source.read_bytes()).hexdigest() != selected["sha256"]:
            raise ValueError("Pretrained source checkpoint hash mismatch")
        state = load_checkpoint_state(source)
        if len(state["task_sequence"]) != 24 or any(name.startswith("agis_") for name in state["task_sequence"]):
            raise ValueError("Source must have exactly 24 non-AGIS tasks")
    started = time.monotonic()
    fits: list[tuple[str, dict[str, Any], bool | None]] = []
    if unit.route == Route.WARM:
        fits.append(("warm", warm_config(manifest, unit), None))
    for freeze_setting in [False] if unit.route == Route.SCRATCH else [True, False]:
        raw = final_config(manifest, unit)
        raw["finetune"]["freeze_encoder"] = freeze_setting
        fits.append(("frozen" if freeze_setting else "unfrozen", raw, freeze_setting))
    for label, raw, frozen in fits:
        out = root / label
        out.mkdir(exist_ok=True)
        if not (out / "DONE").exists():
            spec = {
                "mode": "pretrain" if frozen is None else "finetune",
                "raw": raw,
                "source": str(source) if source is not None else None,
                "output": str(out),
                "heldout": heldout,
            }
            spec_path = out / "fit_spec.json"
            spec_path.write_text(json.dumps(spec, indent=2))
            with (out / "console.log").open("a") as log:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "fit", "--spec", str(spec_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if frozen is None:
            source = out / "training/final_model.pt"
            heads = load_checkpoint_state(source)["task_sequence"]
            expected = {f"agis_rho_{p}gpa" for p in PRESSURES if p != unit.pressure}
            if {name for name in heads if name.startswith("agis_")} != expected:
                raise ValueError("Warm-start learned the wrong pressure heads")
        else:
            verify_final(out, f"agis_rho_{unit.pressure}gpa", heldout, settings.final_epochs)
    (root / "result.json").write_text(
        json.dumps(
            {
                **asdict(unit),
                "heldout": heldout,
                "elapsed_seconds": time.monotonic() - started,
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
            },
            indent=2,
        )
    )
    (root / "DONE").write_text("completed\n")
    logger.info("Completed {} in {:.1f} seconds", unit.name, time.monotonic() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--base-config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--final-epochs", type=int, default=1000)
    plan.add_argument("--warm-epochs", type=int, default=150)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan_campaign(
            args.base_config,
            args.output,
            CampaignSettings(final_epochs=args.final_epochs, warm_epochs=args.warm_epochs),
        )
    elif args.command == "run":
        execute_unit(args.manifest, args.index, args.output_root)
    else:
        fit_spec(args.spec)


if __name__ == "__main__":
    main()
