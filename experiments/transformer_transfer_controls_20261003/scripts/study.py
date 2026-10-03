"""Fixed-update source controls and validation-selected downstream probes, image-only."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import tomllib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

import foundation_model
from foundation_model.models.components.foundation_encoder import FoundationEncoder
from foundation_model.models.components.fc_layers import LinearBlock
from foundation_model.models.model_config import MLPEncoderConfig, TransformerEncoderConfig
from prepare import DATE, SOURCE, TARGET, digest

CONDITIONS = ("random", "real1", "real3", "real7", "shuffled7")


@dataclass(kw_only=True)
class StudyConfig:
    pilot_seeds: list[int]
    seeds: list[int]
    fractions: list[float]
    source_steps: int
    pilot_steps: int
    checkpoint_steps: list[int]
    batch_size: int
    validation_interval: int
    head_epochs: int
    head_patience: int
    head_lrs: list[float]
    full_lr_multipliers: list[float]
    ridge_alphas: list[float]
    latent_dim: int
    package_version: str
    image_revision: str
    sif_sha256: str
    data_manifest_sha256: str

    def __post_init__(self) -> None:
        for seeds in [self.seeds, self.pilot_seeds]:
            if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
                raise ValueError("Unique nonnegative seeds required")
        if set(self.seeds) & set(self.pilot_seeds):
            raise ValueError("Pilot and formal seeds must be disjoint")
        for values in [self.fractions, self.head_lrs, self.full_lr_multipliers, self.ridge_alphas]:
            if not values or len(set(values)) != len(values) or any(not np.isfinite(x) or x <= 0 for x in values):
                raise ValueError("Finite positive unique grid required")
        if self.fractions != [0.1, 1.0]:
            raise ValueError("Registered fixed fractions are 10% and 100%")
        for v in [
            self.source_steps,
            self.pilot_steps,
            self.batch_size,
            self.validation_interval,
            self.head_epochs,
            self.head_patience,
            self.latent_dim,
        ]:
            if type(v) is not int or v < 1:
                raise ValueError("Positive integer budgets required")
        if (
            self.batch_size < 2
            or self.pilot_steps > self.source_steps
            or any(type(k) is not int or not 0 < k <= self.source_steps for k in self.checkpoint_steps)
        ):
            raise ValueError("Invalid batch/checkpoint budget")


def atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cases(raw: dict, phase: str, selection: dict | None = None) -> list[dict]:
    cfg = StudyConfig(**raw["study"])
    if phase == "pilot":
        return [
            dict(arm=a, seed=s, condition="real7", lr=lr)
            for a, v in raw["arms"].items()
            for lr in v["learning_rates"]
            for s in cfg.pilot_seeds
        ]
    if phase != "formal" or selection is None or set(selection) != set(raw["arms"]):
        raise ValueError("Formal work requires complete pilot selection")
    if any(selection[a] not in v["learning_rates"] for a, v in raw["arms"].items()):
        raise ValueError("Unregistered selected learning rate")
    return [
        dict(arm=a, seed=s, condition=c, lr=selection[a]) for a in raw["arms"] for s in cfg.seeds for c in CONDITIONS
    ]


def make_encoder(arm: dict, width: int, latent: int, seed: int) -> nn.Module:
    seed_all(seed)
    if arm["kind"] == "mlp":
        config = MLPEncoderConfig(hidden_dims=[width, *arm["hidden"], latent])
    elif arm["kind"] == "transformer":
        config = TransformerEncoderConfig(
            input_dim=width,
            output_dim=latent,
            d_model=192,
            num_layers=4,
            nhead=6,
            tokenization="grouped",
            group_size=8,
            pooling="mean",
            use_attention=arm["attention"],
        )
    else:
        raise ValueError("Unknown encoder")
    return FoundationEncoder(config)


def head(latent: int, seed: int) -> nn.Module:
    seed_all(seed)
    return LinearBlock(
        [latent, 128, 64], normalization=True, residual=False, dim_output_layer=1, output_active=nn.Identity()
    )


def normalization(y: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = y[train]
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("Insufficient finite training labels")
    mean = float(values.mean())
    scale = float(values.std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate training scale")
    return (y - mean) / scale, mean, scale


def shuffle_labels(y: np.ndarray, splits: np.ndarray, seed: int) -> np.ndarray:
    out = y.copy()
    rng = np.random.default_rng(seed)
    for col in range(y.shape[1]):
        for split in ["train", "val"]:
            idx = np.flatnonzero((splits == split) & np.isfinite(y[:, col]))
            out[idx, col] = rng.permutation(y[idx, col])
    return out


@torch.inference_mode()
def encode(model: nn.Module, x: torch.Tensor, batch: int = 512) -> np.ndarray:
    model.eval()
    return torch.cat([model(v).tanh() for v in x.split(batch)]).cpu().numpy()


def ridge_probe(
    z: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, alphas: list[float]
) -> tuple[StandardScaler, Ridge, list[dict]]:
    scaler = StandardScaler().fit(z[tr])
    a = scaler.transform(z[tr])
    b = scaler.transform(z[va])
    models = []
    scores = []
    for alpha in alphas:
        m = Ridge(alpha=alpha, solver="svd").fit(a, y[tr])
        score = float(np.mean((m.predict(b) - y[va]) ** 2))
        if not np.isfinite(score):
            raise ValueError("Nonfinite ridge score")
        models.append(m)
        scores.append(dict(alpha=alpha, val_mse=score))
    return scaler, models[int(np.argmin([s["val_mse"] for s in scores]))], scores


def source_fit(
    encoder: nn.Module, x: torch.Tensor, frame: pd.DataFrame, case: dict, cfg: StudyConfig, out: Path, steps: int
) -> dict:
    final = out / "source_final.pt"
    if (out / "source_done.json").exists():
        encoder.load_state_dict(torch.load(final, map_location=x.device, weights_only=True))
        return json.loads((out / "source_done.json").read_text())
    if case["condition"] == "random":
        torch.save(encoder.state_dict(), final)
        info = {"steps": 0, "seconds": 0.0}
        atomic(out / "source_done.json", info)
        return info
    k = int(case["condition"][-1])
    y = frame[SOURCE].to_numpy(dtype=float)
    splits = frame.split.to_numpy()
    for j in range(7):
        y[:, j], _, _ = normalization(y[:, j], splits == "train")
    if case["condition"] == "shuffled7":
        y = shuffle_labels(y, splits, case["seed"] + 40000)
    targets = torch.tensor(y, dtype=torch.float32, device=x.device)
    heads = nn.ModuleList([head(cfg.latent_dim, case["seed"] + 1000 + j) for j in range(k)]).to(x.device)
    optimizer = torch.optim.AdamW(
        [{"params": encoder.parameters(), "lr": case["lr"]}, {"params": heads.parameters(), "lr": 0.002}],
        weight_decay=0.001,
        eps=1e-6,
    )
    indices = [np.flatnonzero((splits == "train") & np.isfinite(y[:, j])) for j in range(k)]
    validation = [np.flatnonzero((splits == "val") & np.isfinite(y[:, j])) for j in range(k)]
    if any(len(i) < 2 for i in indices + validation):
        raise ValueError("Source task lacks training/validation labels")
    rng = np.random.default_rng(case["seed"] + 70000)
    history = []
    counts = np.zeros(k, dtype=int)
    start = time.monotonic()
    seed_all(case["seed"] + 80000)
    for step in range(1, steps + 1):
        warm = max(1, int(0.05 * steps))
        factor = (
            min(step / warm, 1.0)
            if step <= warm
            else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
        )
        for group, base in zip(optimizer.param_groups, [case["lr"], 0.002], strict=True):
            group["lr"] = base * factor
        j = int(rng.integers(k))
        idx = rng.choice(indices[j], size=cfg.batch_size, replace=len(indices[j]) < cfg.batch_size)
        counts[j] += len(idx)
        encoder.train()
        heads.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (heads[j](encoder(x[idx]).tanh()).flatten() - targets[idx, j]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite source loss")
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *heads.parameters()], 1.0)
        optimizer.step()
        if not torch.isfinite(gn):
            raise ValueError("Nonfinite source gradient")
        if step % cfg.validation_interval == 0 or step == steps:
            encoder.eval()
            heads.eval()
            vals = []
            with torch.inference_mode():
                for task, ids in enumerate(validation):
                    errors = []
                    for v in np.array_split(ids, max(1, int(np.ceil(len(ids) / 512)))):
                        errors.append((heads[task](encoder(x[v]).tanh()).flatten() - targets[v, task]).square())
                    vals.append(float(torch.cat(errors).mean()))
                probe = encoder(x[validation[0][:512]]).tanh()
                row = dict(
                    step=step,
                    train_mse=float(loss.detach()),
                    validation_mse=float(np.mean(vals)),
                    gradient_norm=float(gn),
                    lr=optimizer.param_groups[0]["lr"],
                    saturation=float((abs(probe) > 0.99).float().mean()),
                    low_variance=float((probe.var(0) < 1e-6).float().mean()),
                )
                row.update({f"val_task{j}": v for j, v in enumerate(vals)})
            history.append(row)
            if not all(np.isfinite(value) for value in row.values()):
                raise ValueError("Nonfinite source diagnostics")
            pd.DataFrame(history).to_csv(out / "source_history.csv", index=False)
        if step in cfg.checkpoint_steps or step == steps:
            torch.save(encoder.state_dict(), out / f"source_step{step}.pt")
    torch.save(encoder.state_dict(), final)
    result = dict(
        steps=steps,
        seconds=time.monotonic() - start,
        exposures=counts.tolist(),
        encoder_parameters=sum(p.numel() for p in encoder.parameters()),
    )
    atomic(out / "source_done.json", result)
    return result


@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.cat([model(v).flatten() for v in x.split(512)]).cpu().numpy()


def fit_target(
    encoder: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    mode: str,
    rate: float,
    seed: int,
    cfg: StudyConfig,
    out: Path,
) -> dict:
    if (out / "done.json").exists():
        return json.loads((out / "done.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    h = head(cfg.latent_dim, seed + 90000).to(x.device)
    if mode == "frozen":
        model = h
        features = z
        groups = [{"params": h.parameters(), "lr": rate}]
    else:
        enc = deepcopy(encoder)
        model = nn.Sequential(enc, nn.Tanh(), h)
        features = x
        groups = [{"params": enc.parameters(), "lr": rate}, {"params": h.parameters(), "lr": 0.002}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.001, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-8)
    rng = np.random.default_rng(seed + 91000)
    target = torch.tensor(y, dtype=torch.float32, device=x.device)
    ids = np.flatnonzero(tr)
    batch = min(cfg.batch_size, len(ids))
    best = float("inf")
    stop_best = float("inf")
    stale = 0
    hist = []
    start = time.monotonic()
    for epoch in range(cfg.head_epochs):
        model.train()
        order = rng.permutation(ids)
        losses = []
        for idx in np.array_split(order, max(1, int(np.ceil(len(order) / batch)))):
            if len(idx) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = (model(features[idx]).flatten() - target[idx]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite target loss")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if not torch.isfinite(gn):
                raise ValueError("Nonfinite target gradient")
            losses.append(float(loss.detach()))
        val = float(np.mean((predict(model, features[va]) - y[va]) ** 2))
        if not np.isfinite(val):
            raise ValueError("Nonfinite target validation")
        hist.append(
            dict(epoch=epoch, train_mse=float(np.mean(losses)), val_mse=val, lr=optimizer.param_groups[0]["lr"])
        )
        if val < best:
            best = val
            best_epoch = epoch
            torch.save(model.state_dict(), out / "best.pt")
        if val < stop_best - 1e-4:
            stop_best = val
            stale = 0
        else:
            stale += 1
        scheduler.step(val)
        if stale >= cfg.head_patience:
            break
    torch.save(model.state_dict(), out / "last.pt")
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    info = dict(
        val_mse=best,
        best_epoch=best_epoch,
        epochs=epoch + 1,
        last_val_mse=val,
        rate=rate,
        seconds=time.monotonic() - start,
    )
    atomic(out / "done.json", info)
    return info


def target_data(
    frame: pd.DataFrame, j: int, fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    y = frame[TARGET[j]].to_numpy(dtype=float)
    ok = np.isfinite(y)
    tr = (frame.split.to_numpy() == "train") & ok
    if fraction == 0.1:
        tr &= frame[f"target{j}_low_train"].to_numpy(dtype=bool)
    va = (frame.split.to_numpy() == "val") & ok
    te = (frame.split.to_numpy() == "test") & ok
    normalized, mean, scale = normalization(y, tr)
    return normalized, tr, va, te, mean, scale


def evaluate(
    encoder: nn.Module, x: torch.Tensor, frame: pd.DataFrame, case: dict, cfg: StudyConfig, out: Path, pilot: bool
) -> list[dict]:
    before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    z = encode(encoder, x)
    zt = torch.tensor(z, device=x.device)
    rows = []
    for j, name in enumerate(TARGET):
        for fraction in [1.0] if pilot else cfg.fractions:
            dest = out / f"target{j}_f{round(fraction * 100):03d}"
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "done.json").exists():
                rows.extend(json.loads((dest / "done.json").read_text())["metrics"])
                continue
            y, tr, va, te, mean, scale = target_data(frame, j, fraction)
            scaler, ridge, search = ridge_probe(z, y, tr, va, cfg.ridge_alphas)
            metrics = [dict(target=name, fraction=fraction, mode="ridge", val_mse=min(t["val_mse"] for t in search))]
            atomic(dest / "ridge_search.json", {"trials": search, "mean": mean, "scale": scale})
            if pilot:
                atomic(dest / "done.json", {"metrics": metrics})
                rows.extend(metrics)
                continue
            outputs = {"ridge": (ridge.predict(scaler.transform(z[te])), None)}
            for mode, rates in [("frozen", cfg.head_lrs), ("full", [case["lr"] * v for v in cfg.full_lr_multipliers])]:
                trials = [
                    fit_target(
                        encoder, x, zt, y, tr, va, mode, lr, case["seed"] + j * 100, cfg, dest / mode / f"lr{lr:g}"
                    )
                    for lr in rates
                ]
                selected = min(trials, key=lambda v: v["val_mse"])
                lr = selected["rate"]
                h = head(cfg.latent_dim, case["seed"] + j * 100 + 90000).to(x.device)
                model = h if mode == "frozen" else nn.Sequential(deepcopy(encoder), nn.Tanh(), h)
                values = []
                for ckpt in ["best", "last"]:
                    model.load_state_dict(
                        torch.load(dest / mode / f"lr{lr:g}" / f"{ckpt}.pt", map_location=x.device, weights_only=True)
                    )
                    values.append(predict(model, zt[te] if mode == "frozen" else x[te]))
                outputs[mode] = (values[0], values[1])
                metrics.append(dict(target=name, fraction=fraction, mode=mode, **selected))
            for row in metrics:
                pred, last = outputs[row["mode"]]
                truth = y[te] * scale + mean
                pred = pred * scale + mean
                if not np.isfinite(pred).all():
                    raise ValueError("Nonfinite predictions")
                error = pred - truth
                row.update(
                    rmse=float(np.sqrt(np.mean(error**2))),
                    mae=float(np.mean(abs(error))),
                    standardized_rmse=float(np.sqrt(np.mean(error**2)) / scale),
                    n_train=int(tr.sum()),
                    n_test=int(te.sum()),
                )
                data = {"composition": frame.loc[te, "composition"].to_numpy(), "true": truth, "pred": pred}
                if last is not None:
                    data["pred_last"] = last * scale + mean
                    row["last_rmse"] = float(np.sqrt(np.mean((data["pred_last"] - truth) ** 2)))
                pd.DataFrame(data).to_parquet(dest / f"{row['mode']}_pred.parquet", index=False)
            for row in metrics:
                if row["mode"] == "ridge":
                    continue
                selected_dir = dest / row["mode"] / f"lr{row['rate']:g}"
                for trial_dir in (dest / row["mode"]).glob("lr*"):
                    if trial_dir != selected_dir:
                        for filename in ("best.pt", "last.pt"):
                            (trial_dir / filename).unlink(missing_ok=True)
            atomic(dest / "done.json", {"metrics": metrics})
            rows.extend(metrics)
    if any(not torch.equal(v, encoder.state_dict()[k]) for k, v in before.items()):
        raise RuntimeError("Source encoder or buffers changed during downstream probes")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["config", "data", "output"]:
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--phase", choices=["pilot", "formal"], required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cpu-test", action="store_true")
    a = p.parse_args()
    raw = tomllib.loads(a.config.read_text())
    cfg = StudyConfig(**raw["study"])
    chosen = json.loads(a.selection.read_text())["learning_rates"] if a.selection else None
    all_cases = cases(raw, a.phase, chosen)
    if not 0 <= a.case < len(all_cases):
        raise ValueError("Case out of range")
    case = all_cases[a.case]
    manifest = json.loads((a.data / "manifest.json").read_text())
    if digest(a.data / "manifest.json") != cfg.data_manifest_sha256:
        raise ValueError("Unregistered data manifest")
    if a.selection:
        selection_record = json.loads(a.selection.read_text())
        if (
            selection_record["config_sha256"] != digest(a.config)
            or selection_record["manifest_sha256"] != cfg.data_manifest_sha256
        ):
            raise ValueError("Pilot selection belongs to another dataset or protocol")
    for name, value in manifest["files"].items():
        if digest(a.data / name) != value:
            raise ValueError("Data checksum mismatch")
    if metadata.version("foundation-model") != cfg.package_version:
        raise ValueError("Package mismatch")
    if not a.cpu_test:
        if (
            platform.machine() != "aarch64"
            or "/site-packages/" not in str(foundation_model.__file__)
            or not os.environ.get("SLURM_JOB_ID")
            or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or os.environ.get("FM_SIF_SHA256") != cfg.sif_sha256
        ):
            raise RuntimeError("Require verified ARM image and one allocated GPU")
    device = "cpu" if a.cpu_test else "cuda"
    torch.set_num_threads(min(4, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    out = a.output / f"case{a.case:03d}"
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "case": case,
        "phase": a.phase,
        "config": digest(a.config),
        "manifest": digest(a.data / "manifest.json"),
        "script": digest(Path(__file__)),
        "preparation_script": digest(Path(__file__).with_name("prepare.py")),
        "selection": digest(a.selection) if a.selection else None,
        "revision": os.environ.get("FM_BENCHMARK_REVISION"),
        "smoke": a.smoke,
        "cpu_test": a.cpu_test,
        "package": metadata.version("foundation-model"),
    }
    if (out / "identity.json").exists() and json.loads((out / "identity.json").read_text()) != identity:
        raise ValueError("Stale lane identity")
    atomic(out / "identity.json", identity)
    if (out / "done.json").exists():
        print("PASS already complete")
        return
    frame = pd.read_parquet(a.data / f"matrix_{DATE}.parquet")
    features = np.load(a.data / f"descriptors_{DATE}.npz")["x"]
    if len(frame) != len(features) or not np.isfinite(features).all() or frame.identity.duplicated().any():
        raise ValueError("Malformed paired matrix")
    if a.smoke:
        keep = []
        for name in SOURCE + TARGET:
            for split in ["train", "val", "test"]:
                keep.extend(frame.index[frame.split.eq(split) & frame[name].notna()][:32])
        ids = sorted(set(keep))
        frame = frame.iloc[ids].reset_index(drop=True)
        features = features[ids]
        for j in range(len(TARGET)):
            frame[f"target{j}_low_train"] = frame.split.eq("train")
        cfg.head_epochs = 2
        cfg.source_steps = 2
        cfg.pilot_steps = 2
        cfg.batch_size = 16
        cfg.checkpoint_steps = [1, 2]
    x = torch.tensor(features, dtype=torch.float32, device=device)
    encoder = make_encoder(raw["arms"][case["arm"]], x.shape[1], cfg.latent_dim, case["seed"]).to(device)
    started = time.monotonic()
    source = source_fit(encoder, x, frame, case, cfg, out, cfg.pilot_steps if a.phase == "pilot" else cfg.source_steps)
    rows = evaluate(encoder, x, frame, case, cfg, out, a.phase == "pilot" and not a.smoke)
    atomic(
        out / "done.json",
        {
            "case": case,
            "source": source,
            "metrics": rows,
            "seconds": time.monotonic() - started,
            "runtime": {
                "package_path": str(foundation_model.__file__),
                "sif_sha256": os.environ.get("FM_SIF_SHA256"),
                "job": os.environ.get("SLURM_JOB_ID"),
            },
        },
    )
    print(f"PASS {a.phase} case {a.case}")


if __name__ == "__main__":
    main()
