"""Frozen encoder/readout factorial study; uses image-installed encoders unchanged."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import time
import tomllib
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

import foundation_model
from foundation_model.models.components.fc_layers import LinearBlock
from foundation_model.workflows._engine import build_model_for_checkpoint, checkpoint_task_order
from foundation_model.workflows._sections import build_model_section
from foundation_model.workflows.recording import load_checkpoint_state
from foundation_model.workflows.task_catalog import TaskCatalog, build_task_catalog_config

from benchmark import DATE, SOURCE_TASKS, file_hash, load_protocol, workflow_config
from run_lane import atomic_json


class Readout(str, Enum):
    LEGACY = "legacy"
    LINEAR_OUTPUT = "linear_output"
    WIDE = "wide"


@dataclass(kw_only=True)
class ProbeConfig:
    arms: list[str]
    seeds: list[int]
    source_counts: list[int]
    fractions: list[float]
    learning_rates: list[float]
    ridge_alphas: list[float]
    max_epochs: int
    patience: int
    batch_size: int
    source_revision: str

    def __post_init__(self) -> None:
        for values, allowed in (
            (self.arms, {"mlp_tuned", "grouped_mean"}),
            (self.seeds, {0, 1, 2}),
            (self.source_counts, {1, 3, 7}),
            (self.fractions, {0.1, 1.0}),
        ):
            if (
                not values
                or any(isinstance(v, bool) for v in values)
                or len(set(values)) != len(values)
                or not set(values) <= allowed
            ):
                raise ValueError("Invalid or duplicate registered cases")
        for values in (self.learning_rates, self.ridge_alphas):
            if not values or len(set(values)) != len(values) or any(not np.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Search values must be unique, positive and finite")
        if any(type(v) is not int or v < 1 for v in (self.max_epochs, self.patience, self.batch_size)):
            raise ValueError("Budgets must be positive integers")
        if 0.002 not in self.learning_rates:
            raise ValueError("The paired activation control requires LR 0.002")
        if self.batch_size < 2 or not re.fullmatch(r"[0-9a-f]{40}", self.source_revision):
            raise ValueError("BatchNorm requires batches >=2; record the source campaign revision")

    def cases(self) -> list[tuple[str, int, int]]:
        return [(a, s, k) for a in self.arms for s in self.seeds for k in self.source_counts]


def make_head(width: int, readout: Readout, seed: int) -> nn.Module:
    """None reproduces 0.5.0; explicit Identity tests the alternative without patching it."""
    torch.manual_seed(seed)
    hidden = [512, 256] if readout == Readout.WIDE else [128, 64]
    return LinearBlock(
        [width, *hidden],
        normalization=True,
        residual=False,
        dim_output_layer=1,
        output_active=None if readout == Readout.LEGACY else nn.Identity(),
    )


def validate_frame(frame: pd.DataFrame) -> None:
    if frame.empty or frame["composition"].duplicated().any():
        raise ValueError("Target rows must have unique composition identities")
    if set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError("Require nonempty train/val/test partitions")
    if not np.isfinite(frame["dielectric_total"].to_numpy(dtype=float)).all():
        raise ValueError("Target contains nonfinite values")


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    if pred.shape != true.shape or pred.size == 0 or not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Metrics require aligned finite predictions and targets")
    error = pred - true
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs(error))),
        "bias": float(error.mean()),
        "absolute_error_q50": float(np.quantile(abs(error), 0.5)),
        "absolute_error_q90": float(np.quantile(abs(error), 0.9)),
    }


def fit_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    readout: Readout,
    seed: int,
    lr: float,
    config: ProbeConfig,
    output: Path,
    device: str,
) -> dict[str, Any]:
    """No test inputs enter training or checkpoint selection. Resume only completed trials."""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "done.json").exists():
        return json.loads((output / "done.json").read_text())
    if len(x_train) < config.batch_size:
        raise ValueError("Training data must fill a batch")
    model = make_head(x_train.shape[1], readout, seed).to(device)
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device).reshape(-1, 1)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device).reshape(-1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    generator = torch.Generator().manual_seed(seed)
    best = float("inf")
    stopping_best = float("inf")
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(config.max_epochs):
        model.train()
        indices = torch.randperm(len(x), generator=generator).to(device)
        losses = []
        for start in range(0, len(x) - config.batch_size + 1, config.batch_size):
            idx = indices[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[idx]) - y[idx]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_loss = float((model(xv) - yv).square().mean())
        if not np.isfinite(val_loss):
            raise ValueError("Nonfinite validation loss")
        history.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "val_mse": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), output / "best.pt")
        if val_loss < stopping_best - 1e-4:
            stopping_best, stale = val_loss, 0
        else:
            stale += 1
        scheduler.step(val_loss)
        if stale >= config.patience:
            break
    torch.save(model.state_dict(), output / "last.pt")
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    result = {
        "best_val_mse": best,
        "best_epoch": best_epoch,
        "last_val_mse": val_loss,
        "epochs": epoch + 1,
        "lr": lr,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "done.json", result)
    return result


def ridge_search(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, alphas: list[float]
) -> tuple[StandardScaler, Ridge, list[dict[str, float]]]:
    scaler = StandardScaler().fit(x_train)
    xt, xv = scaler.transform(x_train), scaler.transform(x_val)
    trials = []
    models = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="svd").fit(xt, y_train)
        score = float(np.mean((model.predict(xv) - y_val) ** 2))
        if not np.isfinite(score):
            raise ValueError("Nonfinite ridge validation score")
        trials.append({"alpha": alpha, "val_mse": score})
        models.append(model)
    return scaler, models[int(np.argmin([t["val_mse"] for t in trials]))], trials


def representation_stats(z: np.ndarray) -> dict[str, float]:
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("Require finite validation embeddings")
    h = np.tanh(z)
    singular = np.linalg.svd(h - h.mean(0), compute_uv=False)
    energy = singular**2
    p = energy / max(float(energy.sum()), 1e-300)
    p = p[p > 0]
    return {
        "saturated_fraction": float(np.mean(abs(h) > 0.99)),
        "low_variance_fraction": float(np.mean(h.var(0) < 1e-6)),
        "effective_rank": float(np.exp(-np.sum(p * np.log(p)))) if len(p) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "config", "data-dir", "source-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = ProbeConfig(**tomllib.loads(args.config.read_text()))
    base, _, _ = load_protocol(args.protocol)
    if not 0 <= args.case < len(config.cases()):
        raise ValueError("Case index outside the registered matrix")
    arm, seed, k = config.cases()[args.case]
    if (
        metadata.version("foundation-model") != base.package_version
        or platform.machine() != "aarch64"
        or "/site-packages/" not in str(foundation_model.__file__)
        or not os.environ.get("SLURM_JOB_ID")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or os.environ.get("FM_SIF_SHA256") != base.sif_sha256
    ):
        raise RuntimeError("Requires the verified official ARM image and one allocated GPU")
    revision = os.environ.get("FM_BENCHMARK_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Missing merged experiment revision")
    manifest_path = args.data_dir / f"manifest_{DATE}.json"
    if file_hash(manifest_path) != base.data_manifest_sha256:
        raise ValueError("Unexpected data manifest")
    manifest = json.loads(manifest_path.read_text())
    source_lane = args.source_root / f"{arm}_s{seed}"
    source_identity = json.loads((source_lane / "identity.json").read_text())
    runtime = json.loads((source_lane / "runtime.json").read_text())
    if (
        source_identity["benchmark_revision"] != config.source_revision
        or source_identity["protocol_sha256"] != file_hash(args.protocol)
        or source_identity["manifest_sha256"] != base.data_manifest_sha256
        or source_identity["cpu_smoke"]
        or source_identity["max_epochs_override"] is not None
        or runtime["sif_sha256"] != base.sif_sha256
        or runtime["functional_smoke"]
    ):
        raise ValueError("Source checkpoint is not from the registered full-budget campaign")
    checkpoint = source_lane / "source_k7/training" / f"step{k:02d}_{SOURCE_TASKS[k - 1]}/checkpoint.pt"
    state = load_checkpoint_state(checkpoint)
    names = checkpoint_task_order(state)
    if set(names) != set(SOURCE_TASKS[:k]) or state["step"] != k:
        raise ValueError("Wrong source stage")
    output = args.output_root / f"{arm}_s{seed}_k{k}"
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "arm": arm,
        "seed": seed,
        "source_count": k,
        "smoke": args.smoke,
        "config_sha256": file_hash(args.config),
        "protocol_sha256": file_hash(args.protocol),
        "checkpoint_sha256": file_hash(checkpoint),
        "manifest_sha256": file_hash(manifest_path),
        "revision": revision,
        "sif_sha256": base.sif_sha256,
    }
    if (output / "identity.json").exists() and json.loads((output / "identity.json").read_text()) != identity:
        raise ValueError("Stale output identity")
    atomic_json(output / "identity.json", identity)
    if (output / "done.json").exists():
        print(f"PASS existing {output.name}")
        return
    started = time.monotonic()
    atomic_json(
        output / "run_provenance.json",
        {
            **identity,
            "package_path": str(foundation_model.__file__),
            "package_version": base.package_version,
            "image_revision": base.image_revision,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "checkpoint_input": str(checkpoint),
        },
    )
    source_entry = manifest["source"]
    for path, digest in (
        (args.data_dir / source_entry["file"], source_entry["sha256"]),
        (args.data_dir / f"source_scalers_{DATE}.joblib", source_entry["scaler_sha256"]),
    ):
        if file_hash(path) != digest:
            raise ValueError("Source data/scaler mismatch")
    raw = workflow_config(args.protocol.resolve(), args.data_dir.resolve(), arm, seed)
    catalog = TaskCatalog(
        build_task_catalog_config({key: raw[key] for key in ("data", "descriptor", "datasets", "tasks")})
    )
    model = build_model_for_checkpoint(catalog, build_model_section(raw["model"]), names)
    model.load_state_dict(state["model"], strict=True)
    encoder = model.encoder.to("cuda").eval().requires_grad_(False)
    before = deepcopy(encoder.state_dict())
    if args.smoke:
        config.max_epochs = 2
    selected_rows = []
    for fraction in config.fractions:
        entry = next(
            t
            for t in manifest["targets"]
            if t["task"] == "dielectric_total" and t["seed"] == seed and t["fraction"] == fraction
        )
        for field, digest in (("file", "sha256"), ("scaler", "scaler_sha256")):
            if file_hash(args.data_dir / entry[field]) != entry[digest]:
                raise ValueError("Target data/scaler mismatch")
        frame = pd.read_parquet(args.data_dir / entry["file"])
        validate_frame(frame)
        if frame["split"].value_counts().to_dict() != entry["counts"]:
            raise ValueError("Unexpected training/validation/test sizes")
        if args.smoke:
            frame = frame.groupby("split", sort=False).head(256).reset_index(drop=True)
        folder = output / f"f{round(100 * fraction):03d}"
        folder.mkdir(exist_ok=True)
        descriptor = catalog.descriptor_fn()(frame["composition"].tolist())
        if list(descriptor.index) != frame["composition"].tolist():
            raise ValueError("Descriptor reordered composition identities")
        x = descriptor.to_numpy(dtype=np.float32)
        with torch.inference_mode():
            z = np.concatenate(
                [
                    encoder(torch.tensor(part, device="cuda")).cpu().numpy()
                    for part in np.array_split(x, max(1, (len(x) + 255) // 256))
                ]
            )
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite encoder features")
        masks = {split: frame["split"].eq(split).to_numpy() for split in ("train", "val", "test")}
        y = frame["dielectric_total"].to_numpy(dtype=np.float64)
        scale = joblib.load(args.data_dir / entry["scaler"])
        true = scale.inverse_transform(y[masks["test"]].reshape(-1, 1)).ravel()
        np.savez_compressed(
            folder / "features.npz",
            pre=z,
            post=np.tanh(z),
            descriptor=x,
            composition=frame["composition"].to_numpy(dtype=str),
            split=frame["split"].to_numpy(dtype=str),
        )
        atomic_json(folder / "representation.json", representation_stats(z[masks["val"]]))
        pd.DataFrame({"composition": frame.loc[masks["test"], "composition"], "true": true}).to_parquet(
            folder / "reference.parquet", index=False
        )
        h = np.tanh(z)
        head_seed = 20271002 + seed
        readouts = [Readout.LINEAR_OUTPUT, Readout.WIDE] + ([Readout.LEGACY] if k == 7 else [])
        for readout in readouts:
            trials = []
            for lr in config.learning_rates:
                trial_dir = folder / readout.value / f"lr{lr:g}"
                trials.append(
                    fit_head(
                        h[masks["train"]],
                        y[masks["train"]],
                        h[masks["val"]],
                        y[masks["val"]],
                        readout,
                        head_seed,
                        lr,
                        config,
                        trial_dir,
                        "cuda",
                    )
                )
            selected = min(trials, key=lambda t: t["best_val_mse"])
            # Test access happens only after validation selection. Keep all trial losses, not trial test rankings.
            head = make_head(h.shape[1], readout, head_seed).to("cuda").eval()
            trial_dir = folder / readout.value / f"lr{selected['lr']:g}"
            head.load_state_dict(torch.load(trial_dir / "best.pt", map_location="cuda", weights_only=True))
            with torch.inference_mode():
                pred = head(torch.tensor(h[masks["test"]], device="cuda")).cpu().numpy()
            pred = scale.inverse_transform(pred).ravel().astype(float)
            pd.DataFrame(
                {"composition": frame.loc[masks["test"], "composition"], "true": true, "pred": pred}
            ).to_parquet(folder / f"{readout.value}_pred.parquet", index=False)
            row = {
                "arm": arm,
                "seed": seed,
                "k": k,
                "fraction": fraction,
                "readout": readout.value,
                **metrics(pred, true),
                **selected,
                "trials": trials,
            }
            atomic_json(folder / f"{readout.value}_selected.json", row)
            selected_rows.append(row)
            # Predetermined activation comparison at the original head LR; do not select it on test.
            evaluations = [("selected_last", selected["lr"], "last.pt")]
            if k == 7 and readout in {Readout.LEGACY, Readout.LINEAR_OUTPUT}:
                evaluations.append(("fixed_lr", 0.002, "best.pt"))
            for label, rate, filename in evaluations:
                head.load_state_dict(
                    torch.load(
                        folder / readout.value / f"lr{rate:g}" / filename, map_location="cuda", weights_only=True
                    )
                )
                with torch.inference_mode():
                    aux = head(torch.tensor(h[masks["test"]], device="cuda")).cpu().numpy()
                aux = scale.inverse_transform(aux).ravel().astype(float)
                pd.DataFrame(
                    {"composition": frame.loc[masks["test"], "composition"], "true": true, "pred": aux}
                ).to_parquet(folder / f"{readout.value}_{label}_pred.parquet", index=False)
                atomic_json(folder / f"{readout.value}_{label}.json", {"lr": rate, **metrics(aux, true)})
        for feature, values in [("post", h), ("pre", z)] + (
            [("descriptor", x)] if arm == config.arms[0] and k == config.source_counts[0] else []
        ):
            scaler, ridge, trials = ridge_search(
                values[masks["train"]].astype(float),
                y[masks["train"]],
                values[masks["val"]].astype(float),
                y[masks["val"]],
                config.ridge_alphas,
            )
            pred = scale.inverse_transform(
                ridge.predict(scaler.transform(values[masks["test"]])).reshape(-1, 1)
            ).ravel()
            joblib.dump({"feature_scaler": scaler, "ridge": ridge}, folder / f"ridge_{feature}.joblib")
            pd.DataFrame(
                {"composition": frame.loc[masks["test"], "composition"], "true": true, "pred": pred}
            ).to_parquet(folder / f"ridge_{feature}_pred.parquet", index=False)
            row = {
                "arm": arm,
                "seed": seed,
                "k": k,
                "fraction": fraction,
                "readout": f"ridge_{feature}",
                **metrics(pred, true),
                "alpha": ridge.alpha,
                "best_val_mse": min(t["val_mse"] for t in trials),
                "trials": trials,
            }
            atomic_json(folder / f"ridge_{feature}_selected.json", row)
            selected_rows.append(row)
    if any(not torch.equal(value, encoder.state_dict()[name]) for name, value in before.items()):
        raise RuntimeError("Encoder parameters or buffers changed")
    atomic_json(
        output / "done.json",
        {
            "selected": selected_rows,
            "encoder_unchanged": True,
            "elapsed_seconds": time.monotonic() - started,
            "identity": identity,
        },
    )
    print(f"PASS head probe {output.name}")


if __name__ == "__main__":
    main()
