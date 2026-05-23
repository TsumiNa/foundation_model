# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Targeted fine-tune of the three heads used by inverse design.

Loads a ``final_model.pt`` checkpoint produced by ``continual_rehearsal_demo``, freezes the
encoder and every other task head (including the autoencoder), and runs a short fine-tune on
just the three inverse-design heads — by default ``formation_energy``, ``klat`` and
``material_type`` — so they are as sharp as possible before we compare inverse-design methods
(latent-with-cycle-consistency vs differentiable KMD).

The script is **independent of the rehearsal demo** (its own CLI, output dir, and checkpoint).
It reuses the demo runner only for data loading + model reconstruction; no rehearsal loop is run.

    python -m foundation_model.scripts.finetune_inverse_heads \\
        --config-file samples/continual_rehearsal_demo_config_inverse_baseline.toml \\
        --checkpoint artifacts/continual_rehearsal_inverse_baseline/final_model.pt \\
        --output-dir artifacts/inverse_heads_finetuned \\
        --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from lightning import Trainer, seed_everything
from loguru import logger

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.scripts.continual_rehearsal_demo import (
    ContinualRehearsalConfig,
    ContinualRehearsalRunner,
    _parse_args as _demo_parse_args,  # noqa: F401  (kept for documentation; we parse our own args)
)

DEFAULT_INVERSE_HEADS = ("formation_energy", "klat", "material_type")


def freeze_except(model, keep_heads: Iterable[str]) -> dict[str, bool]:
    """Freeze encoder + every head NOT in ``keep_heads``; return the prior requires_grad state."""
    keep = set(keep_heads)
    saved: dict[str, bool] = {}
    for name, p in model.named_parameters():
        saved[name] = p.requires_grad
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for head_name, head in model.task_heads.items():
        train = head_name in keep
        for p in head.parameters():
            p.requires_grad_(train)
    return saved


def _restore_requires_grad(model, saved: dict[str, bool]) -> None:
    for name, p in model.named_parameters():
        if name in saved:
            p.requires_grad_(saved[name])


def finetune(config: ContinualRehearsalConfig, ckpt_path: Path, inverse_heads: tuple[str, ...], epochs: int) -> Path:
    seed_everything(config.random_seed, workers=True)
    runner = ContinualRehearsalRunner(config)  # loads data + builds KMD cache (same as demo)

    logger.info(f"Loading model checkpoint {ckpt_path}")
    model = runner._build_full_model()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)

    missing = [t for t in inverse_heads if t not in model.task_heads]
    if missing:
        raise ValueError(
            f"Heads {missing} not found in the loaded model (have {list(model.task_heads.keys())}). "
            "Check that the checkpoint was produced with the same task_sequence."
        )

    logger.info(f"Freezing everything except heads: {sorted(inverse_heads)}")
    freeze_except(model, inverse_heads)

    # Use the same task configs as training (built by the runner), but restrict the DataModule to
    # the inverse-head tasks and disable masking (we want all available labels for these heads).
    task_configs = {name: runner._build_task_config(name) for name in inverse_heads}
    for cfg in task_configs.values():
        cfg.task_masking_ratio = 1.0  # no rehearsal-style dropout — we want every label

    datamodule = CompoundDataModule(
        task_configs=list(task_configs.values()),
        descriptor_fn=runner.descriptor_fn,
        task_frames={name: runner.task_frames[name] for name in inverse_heads},
        composition_column="composition",
        random_seed=config.datamodule_random_seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    trainer = Trainer(
        max_epochs=epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=datamodule)

    out_path = Path(config.output_dir) / "final_model.pt"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "task_sequence": list(config.task_sequence),
            "finetuned_heads": list(inverse_heads),
            "finetune_epochs": int(epochs),
            "from_checkpoint": str(ckpt_path),
        },
        out_path,
    )
    (Path(config.output_dir) / "finetune_summary.json").write_text(
        json.dumps(
            {
                "from_checkpoint": str(ckpt_path),
                "finetuned_heads": list(inverse_heads),
                "epochs": int(epochs),
                "task_sequence": list(config.task_sequence),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Saved fine-tuned checkpoint to {out_path}")
    return out_path


def _parse_args(argv: list[str] | None = None) -> tuple[ContinualRehearsalConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Targeted fine-tune of inverse-design heads.")
    parser.add_argument("--config-file", type=Path, required=True, help="Demo config (paths + task_sequence).")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="final_model.pt produced by continual_rehearsal_demo."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Where to write the fine-tuned checkpoint + summary."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Fine-tune epochs (default 20).")
    parser.add_argument(
        "--inverse-heads",
        type=str,
        default=",".join(DEFAULT_INVERSE_HEADS),
        help=f"Comma-separated head names to fine-tune. Default: {','.join(DEFAULT_INVERSE_HEADS)}.",
    )
    args = parser.parse_args(argv)

    # Build the demo config (reuses the same TOML schema), overriding output_dir.
    import tomllib

    data = tomllib.loads(args.config_file.read_text(encoding="utf-8"))
    data["output_dir"] = str(args.output_dir)
    field_names = set(ContinualRehearsalConfig.__dataclass_fields__)
    path_fields = {
        "qc_data_path",
        "qc_preprocessing_path",
        "superconductor_path",
        "magnetic_path",
        "phonix_path",
        "output_dir",
    }
    kwargs: dict[str, object] = {}
    for key, value in data.items():
        if key not in field_names:
            continue
        kwargs[key] = Path(value) if key in path_fields and value is not None else value
    config = ContinualRehearsalConfig(**kwargs)
    return config, args


def main(argv: list[str] | None = None) -> None:
    config, args = _parse_args(argv)
    heads = tuple(h.strip() for h in args.inverse_heads.split(",") if h.strip())
    finetune(config, args.checkpoint, heads, args.epochs)


if __name__ == "__main__":
    main()
