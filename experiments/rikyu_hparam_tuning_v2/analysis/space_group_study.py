#!/usr/bin/env python3
"""Why does space_group score 0.24 top-1 here when the ShotgunCSP classifier scores 0.60?

A controlled factorial on the SAME rows, label (151 classes) and split as the pipeline's
stN_space_group runs. Every arm varies one or more of four factors against the pipeline replica:

  descriptor  kmd     the pipeline's KMD (n_grids = 8, sigma auto, scale = True; 464 columns)
              classic the notebook's XenonPy classic table (290 columns, StandardScaler + Yeo-Johnson)
              nosum   the same table without the WeightedSum block (232 columns) — the feature set the
                      2026 ShotgunCSP refits use ("intensive")
  scaling     none    features as the pipeline feeds them
              shotgun min-max → Yeo-Johnson → standard, fitted on the train split (ShotgunCSP's Scaler chain)
  model       fm      FoundationEncoder([in, 256, 384]) + ClassificationHead([384, 64] → 151), the project's
                      own classes (BatchNorm + LeakyReLU(0.1) blocks), i.e. the adopted stage_single shape
              paper   ShotgunCSP's persisted MP refit: 4 Linear→Dropout(0.232)→GELU layers, widths
                      in×1.486 then ×0.859 per layer, no batch-norm, Linear output
  loss        balanced sklearn-"balanced" class weights in the cross-entropy (the pipeline's unconditional default)
              plain    unweighted cross-entropy (the paper)
  recipe      fm      AdamW (encoder lr 2e-3 wd 1e-2, head lr 5e-3 wd 1e-5, eps 1e-6), batch 256,
                      ReduceLROnPlateau(0.5, patience 5, min_lr 1e-5) on the train loss, early stopping on
                      the validation loss (patience 24, min_delta 1e-4), max 150 epochs
              paper   Adam lr 2.99e-3 wd 2.63e-7, grad clip 0.052, batch 1024, early stopping patience 30
                      on the validation loss, max 150 epochs

Every run evaluates the test split twice: with the LAST epoch's weights (what the pipeline reports —
checkpointing is off) and with the weights of the epoch whose validation loss was lowest. Metrics:
top-1 accuracy, macro-F1, top-5 / top-10 / top-30 recall.

    uv run python analysis/space_group_study.py --cache <sg_cache.npz> --arms all --seeds 3 -o summary/space_group_study.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from torch import nn

from foundation_model.models.components.foundation_encoder import FoundationEncoder
from foundation_model.models.model_config import ClassificationTaskConfig, MLPEncoderConfig
from foundation_model.models.task_head.classification import ClassificationHead

# ---------------------------------------------------------------- arms
ARMS = {
    # name:                      (descriptor, scaling, model,  loss,       recipe)
    "fm_replica":               ("kmd",     "none",    "fm",    "balanced", "fm"),
    "fm_plain":                 ("kmd",     "none",    "fm",    "plain",    "fm"),
    "fm_plain_scaled":          ("kmd",     "shotgun", "fm",    "plain",    "fm"),
    "fm_model_paper_recipe":    ("kmd",     "shotgun", "fm",    "plain",    "paper"),
    "paper_model_fm_recipe_bal": ("kmd",    "shotgun", "paper", "balanced", "fm"),
    "paper_model_fm_recipe":    ("kmd",     "shotgun", "paper", "plain",    "fm"),
    "paper_full_kmd":           ("kmd",     "shotgun", "paper", "plain",    "paper"),
    "paper_full_kmd_bal":       ("kmd",     "shotgun", "paper", "balanced", "paper"),
    "paper_full_classic":       ("classic", "none",    "paper", "plain",    "paper"),
    "paper_full_nosum":         ("nosum",   "none",    "paper", "plain",    "paper"),
    "fm_replica_classic":       ("classic", "none",    "fm",    "balanced", "fm"),
    "fm_plain_classic":         ("classic", "none",    "fm",    "plain",    "fm"),
    "fm_replica_scaled":        ("kmd",     "shotgun", "fm",    "balanced", "fm"),
    # the paper's protocol is a random split; "_rsplit" arms re-split the same rows 80/10/10 at random (seeded)
    "paper_full_kmd_rsplit":    ("kmd",     "shotgun", "paper", "plain",    "paper", "random"),
    "paper_full_nosum_rsplit":  ("nosum",   "none",    "paper", "plain",    "paper", "random"),
    "fm_replica_rsplit":        ("kmd",     "none",    "fm",    "balanced", "fm",    "random"),
}
PAPER = dict(n_layers=4, first_ratio=1.4862879010402992, decay=0.8591666427029987, dropout=0.23232204124974365,
             lr=0.0029910748103204056, weight_decay=2.631431874473181e-07, clip_value=0.05225454005649297,
             batch_size=1024, patience=30)          # notebooks/prediction_models/space_group/describe.pkl.z
FM = dict(encoder_lr=2e-3, encoder_wd=1e-2, head_lr=5e-3, head_wd=1e-5, eps=1e-6, batch_size=256,
          sched_factor=0.5, sched_patience=5, min_lr=1e-5, es_patience=24, es_min_delta=1e-4)
MAX_EPOCHS = 150
NUM_CLASSES = 151
TOPK = (1, 5, 10, 30)


class PaperNet(nn.Module):
    """ShotgunCSP SequentialLinear as persisted: Linear → Dropout → GELU per layer, Linear output."""

    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        widths, w = [], d_in
        for i in range(PAPER["n_layers"]):
            w = round(w * (PAPER["first_ratio"] if i == 0 else PAPER["decay"]))
            widths.append(w)
        layers, prev = [], d_in
        for w in widths:
            layers += [nn.Linear(prev, w), nn.Dropout(PAPER["dropout"]), nn.GELU()]
            prev = w
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(prev, n_out)
        self.widths = widths

    def forward(self, x):
        return self.out(self.body(x))


class FmNet(nn.Module):
    """The pipeline's own encoder + classification head, at the adopted stage_single shape."""

    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.encoder = FoundationEncoder(MLPEncoderConfig(hidden_dims=[d_in, 256, 384]))
        self.head = ClassificationHead(ClassificationTaskConfig(name="space_group", dims=[384, 64], num_classes=n_out))

    def forward(self, x):
        h = self.encoder(x)
        if isinstance(h, (tuple, list)):
            h = h[0]
        return self.head(h)


def topk_recall(logits: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-logits, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def evaluate(model, X, y, device) -> dict:
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 4096].to(device)).float().cpu() for i in range(0, len(X), 4096)]).numpy()
    pred = logits.argmax(1)
    out = {"accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}
    for k in TOPK:
        out[f"top{k}"] = topk_recall(logits, y, k)
    return out


def run(arm: str, seed: int, cache: dict, device: torch.device) -> dict:
    desc, scaling, model_kind, loss_kind, recipe, *rest = ARMS[arm]
    split_kind = rest[0] if rest else "dataset"
    X = cache[{"kmd": "Xk", "classic": "Xc", "nosum": "Xn"}[desc]].astype(np.float32)
    y, split = cache["y"].astype(int), cache["split"]
    if split_kind == "random":  # 80 / 10 / 10 at random, seeded; the paper's 4:1 protocol plus a val slice for early stopping
        rng = np.random.default_rng(seed); order = rng.permutation(len(y)); n = len(y)
        split = np.full(n, "train", dtype=object); split[order[int(0.8 * n):int(0.9 * n)]] = "val"; split[order[int(0.9 * n):]] = "test"
    tr, va, te = split == "train", split == "val", split == "test"
    if scaling == "shotgun":  # ShotgunCSP: Scaler().min_max().power_transformer().standard(), fitted on train
        mm = MinMaxScaler().fit(X[tr]); X = mm.transform(X)
        pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(X[tr]); X = pt.transform(X)
        ss = StandardScaler().fit(X[tr]); X = ss.transform(X).astype(np.float32)
    Xt = torch.as_tensor(X)
    yt = torch.as_tensor(y)
    torch.manual_seed(seed); np.random.seed(seed)
    model = (FmNet if model_kind == "fm" else PaperNet)(X.shape[1], NUM_CLASSES).to(device)
    if loss_kind == "balanced":  # task_catalog._class_weights: counts.sum() / (num_classes * counts), train rows
        counts = np.bincount(y[tr], minlength=NUM_CLASSES).astype(float); counts[counts == 0] = 1.0
        weight = torch.as_tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float, device=device)
    else:
        weight = None
    if recipe == "fm":
        if model_kind == "fm":
            groups = [{"params": model.encoder.parameters(), "lr": FM["encoder_lr"], "weight_decay": FM["encoder_wd"]},
                      {"params": model.head.parameters(), "lr": FM["head_lr"], "weight_decay": FM["head_wd"]}]
        else:  # the paper net has no encoder/head split: body as encoder, output as head
            groups = [{"params": model.body.parameters(), "lr": FM["encoder_lr"], "weight_decay": FM["encoder_wd"]},
                      {"params": model.out.parameters(), "lr": FM["head_lr"], "weight_decay": FM["head_wd"]}]
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=FM["eps"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=FM["sched_factor"], patience=FM["sched_patience"], min_lr=FM["min_lr"])
        batch, patience, min_delta, clip = FM["batch_size"], FM["es_patience"], FM["es_min_delta"], None
    else:
        opt = torch.optim.Adam(model.parameters(), lr=PAPER["lr"], weight_decay=PAPER["weight_decay"])
        sched = None
        batch, patience, min_delta, clip = PAPER["batch_size"], PAPER["patience"], 0.0, PAPER["clip_value"]
    tr_idx = np.flatnonzero(tr); Xva, yva = Xt[va].to(device), yt[va].to(device)
    best_val, best_epoch, best_state, wait = math.inf, -1, None, 0
    curve = []
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train(); perm = np.random.permutation(tr_idx); tot, n = 0.0, 0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            loss = F.cross_entropy(model(xb), yb, weight=weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            if clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip)
            opt.step(); tot += loss.item() * len(idx); n += len(idx)
        train_loss = tot / n
        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(Xva), yva, weight=weight).item()
        curve.append((epoch, round(train_loss, 4), round(val_loss, 4)))
        if sched is not None:
            sched.step(train_loss)          # the pipeline's scheduler watches train_final_loss_epoch
        if val_loss < best_val - min_delta:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    last = evaluate(model, Xt[te], y[te], device)
    model.load_state_dict(best_state)
    best = evaluate(model, Xt[te], y[te], device)
    return {"arm": arm, "seed": seed, "factors": dict(descriptor=desc, scaling=scaling, model=model_kind, loss=loss_kind, recipe=recipe, split=split_kind),
            "d_in": int(X.shape[1]), "n_params": int(sum(p.numel() for p in model.parameters())),
            "epochs_run": len(curve), "best_epoch": best_epoch, "best_val_loss": round(best_val, 4),
            "last_val_loss": curve[-1][2], "last_train_loss": curve[-1][1], "seconds": round(time.time() - t0, 1),
            "last": last, "best": best, "curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    global MAX_EPOCHS
    if args.max_epochs:
        MAX_EPOCHS = args.max_epochs
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    cache = dict(np.load(args.cache, allow_pickle=True))
    arms = list(ARMS) if args.arms == "all" else args.arms.split(",")
    results = json.loads(args.out.read_text()) if args.out.exists() else {"runs": []}
    done = {(r["arm"], r["seed"]) for r in results["runs"]}
    for arm in arms:
        for s in range(args.seeds):
            seed = 2025 + s
            if (arm, seed) in done:
                continue
            r = run(arm, seed, cache, device)
            results["runs"].append(r)
            args.out.write_text(json.dumps(results))
            print(f"{arm:28s} s{seed}  ep {r['epochs_run']:3d} best {r['best_epoch']:3d}  "
                  f"LAST acc {r['last']['accuracy']:.4f} F1 {r['last']['macro_f1']:.4f} top5 {r['last']['top5']:.3f} | "
                  f"BEST acc {r['best']['accuracy']:.4f} F1 {r['best']['macro_f1']:.4f} top5 {r['best']['top5']:.3f} top30 {r['best']['top30']:.3f}  {r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
