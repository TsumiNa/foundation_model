#!/usr/bin/env python
"""
Reproduces the latent-space optimization workflow from
notebooks/advanced_optimization_from_latent.ipynb using the new
enable_autoencoder interface, with proper Trainer + CompoundDataModule training.

Usage:
    uv run python scripts/optimize_latent_demo.py [--epochs N] [--fast]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import (
    MLPEncoderConfig,
    OptimizerConfig,
    RegressionTaskConfig,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
DESCRIPTOR_PATH = DATA_DIR / "qc_ac_te_mp_dos_kmd1d_desc_20250615.pd.parquet"
PROPERTY_PATH = DATA_DIR / "qc_ac_te_mp_dos_reformat_20250615.pd.parquet"

# ---------------------------------------------------------------------------
# Task / model config
# ---------------------------------------------------------------------------
TARGET_NAME = "Dielectric total (normalized)"
TARGET_RAW_NAME = "Dielectric total"
MAX_RAW_VALUE = 1000      # filter outliers (same as notebook)
LATENT_DIM = 128

# ---------------------------------------------------------------------------
# Optimization config
# ---------------------------------------------------------------------------
OPT_TARGET = 3.0
OPT_STEPS = 300
OPT_LR = 0.005
OPT_RESTARTS = 6
OPT_PERTURB = 0.3
HIGH_VALUE_THRESH = 2.6   # pick seeds with normalized value > this

SEED = 42


# ---------------------------------------------------------------------------
# Descriptor function (looks up precomputed KMD1D rows by mp-id)
# ---------------------------------------------------------------------------
def make_descriptor_fn(desc_df: pd.DataFrame):
    idx_map = {str(k): k for k in desc_df.index}

    def descriptor_fn(compositions: list[str]) -> pd.DataFrame:
        labels = [idx_map[c] for c in compositions if c in idx_map]
        return desc_df.loc[labels]

    return descriptor_fn


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(max_epochs: int = 500, fast: bool = False) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data …")
    desc_df = pd.read_parquet(DESCRIPTOR_PATH)
    props_df = pd.read_parquet(PROPERTY_PATH)[[TARGET_NAME, TARGET_RAW_NAME, "split"]]
    # Filter outliers (same as notebook)
    props_df = props_df[props_df[TARGET_RAW_NAME] < MAX_RAW_VALUE]
    # Keep only compositions with descriptors
    props_df = props_df.loc[props_df.index.intersection(desc_df.index)]

    input_dim = desc_df.shape[1]
    logger.info(
        f"Descriptors: {desc_df.shape}  |  "
        f"Properties after filter: {props_df.shape}  |  "
        f"split: {props_df['split'].value_counts().to_dict()}"
    )

    # ------------------------------------------------------------------
    # 2. Task config  (AE task is owned by the model, not the DataModule)
    # ------------------------------------------------------------------
    reg_task = RegressionTaskConfig(
        name=TARGET_NAME,
        data_column=TARGET_NAME,
        dims=[LATENT_DIM, 64, 32, 1],
        norm=True,
        split_column="split",          # use the pre-existing train/val/test split
    )

    # ------------------------------------------------------------------
    # 3. DataModule
    # ------------------------------------------------------------------
    dm = CompoundDataModule(
        task_configs=[reg_task],
        descriptor_fn=make_descriptor_fn(desc_df),
        task_frames={TARGET_NAME: props_df[[TARGET_NAME, "split"]]},
        batch_size=256 if not fast else 512,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 4. Model  (enable_autoencoder=True → decoder auto-derived as mirror)
    # ------------------------------------------------------------------
    model = FlexibleMultiTaskModel(
        encoder_config=MLPEncoderConfig(hidden_dims=[input_dim, 256, LATENT_DIM], norm=True),
        task_configs=[reg_task],
        enable_autoencoder=True,
        autoencoder_nonnegative=False,
        shared_block_optimizer=OptimizerConfig(lr=5e-3),
    )
    ae_dims = model.task_configs_map["__reconstruction__"].dims
    logger.info(f"AE decoder dims (auto-derived): {ae_dims}")

    # ------------------------------------------------------------------
    # 5. Trainer
    # ------------------------------------------------------------------
    output_dir = Path("outputs/optimize_latent_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            monitor="val_final_loss",
            mode="min",
            save_top_k=1,
            filename="best",
        ),
        EarlyStopping(monitor="val_final_loss", patience=30, mode="min"),
    ]
    trainer = L.Trainer(
        max_epochs=max_epochs if not fast else 30,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    logger.info("Starting training …")
    trainer.fit(model, datamodule=dm)

    # ------------------------------------------------------------------
    # 6. Test metrics
    # ------------------------------------------------------------------
    logger.info("Running test evaluation …")
    trainer.test(model, datamodule=dm, ckpt_path="best")

    # ------------------------------------------------------------------
    # 7. Latent-space optimization on high-value seeds
    # ------------------------------------------------------------------
    logger.info(f"Picking seed compositions with {TARGET_NAME} > {HIGH_VALUE_THRESH} …")
    model.eval()

    candidates = props_df[props_df[TARGET_NAME] > HIGH_VALUE_THRESH]
    logger.info(f"  {len(candidates)} candidates found")
    if candidates.empty:
        logger.warning("No high-value candidates found; skipping optimization.")
        return

    random.seed(SEED)
    seed_id = random.choice(candidates.index.tolist())
    seed_desc = torch.tensor(
        desc_df.loc[seed_id].values.astype(np.float32), dtype=torch.float32
    ).unsqueeze(0)

    logger.info(
        f"Seed: {seed_id}  |  "
        f"{TARGET_NAME}={props_df.loc[seed_id, TARGET_NAME]:.4f}  "
        f"(raw {TARGET_RAW_NAME}={props_df.loc[seed_id, TARGET_RAW_NAME]:.2f})"
    )

    result = model.optimize_latent(
        task_name=TARGET_NAME,
        initial_input=seed_desc,
        target_value=OPT_TARGET,
        steps=OPT_STEPS,
        lr=OPT_LR,
        num_restarts=OPT_RESTARTS,
        perturbation_std=OPT_PERTURB,
        optimize_space="latent",      # uses __reconstruction__ head internally
    )

    # result shapes: optimized_input (1, R, D), optimized_target (1, R, 1)
    opt_inputs = result.optimized_input[0].detach().cpu()    # (R, D)
    opt_scores = result.optimized_target[0, :, 0].detach().cpu().numpy()  # (R,)

    # Feed reconstructed descriptors back through model to check consistency
    with torch.no_grad():
        init_pred = model(seed_desc)[TARGET_NAME].item()
        recon_preds = model(opt_inputs)[TARGET_NAME].cpu().numpy().flatten()

    header = f"{'Restart':<8} {'Opt score':>12} {'Recon pred':>12} {'Gap':>10}"
    sep = "-" * 46
    logger.info(f"\nSeed initial prediction: {init_pred:.4f}\n{header}\n{sep}")
    for i in range(OPT_RESTARTS):
        gap = recon_preds[i] - opt_scores[i]
        logger.info(f"{i:<8} {opt_scores[i]:>12.4f} {recon_preds[i]:>12.4f} {gap:>10.4f}")

    logger.info(
        "\nOpt score    = property value achieved during latent-space gradient descent.\n"
        "Recon pred   = model prediction on the reconstructed descriptor (AE decode).\n"
        "Gap          = decoder reconstruction error seen by the property head.\n"
        "Next step    : pass opt_inputs through kmd_1d.inverse() to get candidate compositions."
    )

    # Save optimized descriptors for downstream KMD inverse transform
    out_path = output_dir / "optimized_descriptors.parquet"
    pd.DataFrame(
        opt_inputs.numpy(),
        index=[f"restart_{i}" for i in range(OPT_RESTARTS)],
        columns=desc_df.columns,
    ).to_parquet(out_path)
    logger.info(f"Optimized descriptors saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--fast", action="store_true", help="30 epochs, larger batch — quick smoke test")
    args = parser.parse_args()
    main(max_epochs=args.epochs, fast=args.fast)
