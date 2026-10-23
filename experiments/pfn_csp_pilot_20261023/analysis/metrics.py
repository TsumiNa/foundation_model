"""Metrics and seed-level confidence intervals (plan section 4.1)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

EPS = 1e-6
N_BOOT = 10_000


def correct_top1(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (prob.argmax(axis=1) == y).astype(float)


def correct_topk(prob: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    """A query counts as correct only when its true class has non-zero probability and fewer
    than k classes have strictly higher probability (true class absent from context -> wrong)."""
    pt = prob[np.arange(len(y)), y]
    higher = (prob > pt[:, None]).sum(axis=1)
    return ((pt > 0) & (higher < k)).astype(float)


def log_loss(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(prob, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def macro_f1(prob: np.ndarray, y: np.ndarray) -> float:
    """Macro F1 over the true classes present in this test set."""
    return float(f1_score(y, prob.argmax(axis=1), labels=np.unique(y), average="macro", zero_division=0))


def confidence_auroc(prob: np.ndarray, y: np.ndarray) -> float:
    c = correct_top1(prob, y)
    if c.min() == c.max():
        return float("nan")
    return float(roc_auc_score(c, prob.max(axis=1)))


def all_metrics(prob: np.ndarray, y: np.ndarray, n_valid: np.ndarray, n_cov: np.ndarray) -> dict:
    return {
        "top1": float(correct_top1(prob, y).mean()),
        "top3": float(correct_topk(prob, y, 3).mean()),
        "log_loss": log_loss(prob, y),
        "macro_f1": macro_f1(prob, y),
        "auroc": confidence_auroc(prob, y),
        "coverage": float((n_cov / np.maximum(n_valid, 1)).mean()),
        "coverage_any": float((n_cov > 0).mean()),
        "n_query": int(len(y)),
    }


def seed_ci(values, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap 95% CI over seed-level values (NaNs dropped)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)
