#!/usr/bin/env python3
"""material_type under cross-validation — is 0.83 macro-F1 real, or a split that leaks systems?

Same rows as the pipeline (the qc dataset's 49,034 compositions, minus the 36 "others" rows the
XenonPy table lacks), two descriptors and two models, under three protocols:

  dataset   the dataset's own train / val / test split (what every pipeline number uses)
  random    5-fold stratified cross-validation over rows (each row's composition can have a near-twin
            from the same system in the training folds)
  system    5-fold stratified GROUP cross-validation with the element set as the group: a test
            composition's element combination never appears in training — the "new system" case

  descriptor  kmd (the pipeline's, 464 columns) | classic (the notebook's XenonPy classic table, 290)
  model       nn (the pipeline's FoundationEncoder + ClassificationHead, plain cross-entropy, pipeline
              recipe; from analysis/space_group_study.py) | rf (RandomForest, 500 trees, no class weights)

Metrics per fold: 5-class macro-F1 and accuracy, per-class precision / recall, and the 3-class
collapse used by the 2021 paper (QC = DQC + IQC, AC = DAC + IAC, others). Plus a leakage measure:
for every minority test row, the L1 distance in atomic fractions to the nearest training row.

    uv run python analysis/material_type_cv.py -o summary/material_type_cv.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "analysis"))
import space_group_study as M  # noqa: E402

from foundation_model.data.composition_sources import PrecomputedDescriptorSource  # noqa: E402
from foundation_model.utils.kmd_plus import KMD, element_features, formula_to_composition  # noqa: E402

CLASSES = ["DAC", "DQC", "IAC", "IQC", "others"]
COLLAPSE = {0: 1, 1: 0, 2: 1, 3: 0, 4: 2}  # 3-class: 0 = QC, 1 = AC, 2 = others
CLASSES3 = ["QC", "AC", "others"]


def per_class(t, p, k):
    out = {}
    for c in range(k):
        m = t == c; tp = int(((p == c) & m).sum()); npred = int((p == c).sum())
        out[c] = {"n": int(m.sum()), "recall": (tp / m.sum()) if m.sum() else None, "precision": (tp / npred) if npred else None}
    return out


def score(t, p):
    t3 = np.vectorize(COLLAPSE.get)(t); p3 = np.vectorize(COLLAPSE.get)(p)
    return {"macro_f1": float(f1_score(t, p, average="macro", zero_division=0)), "accuracy": float((t == p).mean()),
            "macro_f1_3class": float(f1_score(t3, p3, average="macro", zero_division=0)), "accuracy_3class": float((t3 == p3).mean()),
            "per_class": {CLASSES[c]: v for c, v in per_class(t, p, 5).items()}, "per_class_3": {CLASSES3[c]: v for c, v in per_class(t3, p3, 3).items()}}


def nearest_train_distance(W, y, train_idx, test_idx):
    """For each minority test row: L1 distance (atomic fractions) to the nearest training row, and to the nearest same-class training row."""
    out = []
    Wtr = W[train_idx]; ytr = y[train_idx]
    for i in test_idx:
        if y[i] == 4:
            continue
        d = np.abs(Wtr - W[i]).sum(1)
        same = d[ytr == y[i]]
        out.append({"class": CLASSES[int(y[i])], "d_any": float(d.min()), "d_same_class": float(same.min()) if len(same) else None})
    return out


def run_nn(X, y, tr, va, te, device):
    cache = {"Xk": X, "Xc": X, "Xn": X, "y": y, "split": np.array(["other"] * len(y), dtype=object)}
    cache["split"][tr] = "train"; cache["split"][va] = "val"; cache["split"][te] = "test"
    captured = {}
    orig = M.evaluate

    def ev(model, Xt, yy, dev):
        out = orig(model, Xt, yy, dev); model.eval()
        with torch.no_grad():
            captured["pred"] = model(Xt.to(dev)).argmax(1).cpu().numpy()
        return out

    M.evaluate = ev
    try:
        r = M.run("fm_plain", 2025, cache, device)
    finally:
        M.evaluate = orig
    return captured["pred"], r["epochs_run"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--protocols", default="dataset,random,system")
    ap.add_argument("--models", default="nn,rf")
    ap.add_argument("--descriptors", default="classic,kmd")
    a = ap.parse_args()
    M.NUM_CLASSES = 5; M.TOPK = (1, 2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    F = ROOT / "data" / "qc_ac_te_mp_dos_reformat_20260912.pd.parquet"
    df = pd.read_parquet(F, columns=["composition", "split", "Material type (label)"])
    src = PrecomputedDescriptorSource(str(ROOT / "data" / "desc_xenonpy_classic_trans.parquet"), composition_column="composition")
    d = src(list(df.composition)); df = df[df.composition.isin(d.index)].reset_index(drop=True)
    Xc = d.reindex(df.composition).to_numpy(np.float32)
    W = np.stack(df.composition.map(formula_to_composition).values)
    Xk = KMD(element_features.values, method="1d", n_grids=8, sigma="auto", scale=True).transform(W).astype(np.float32)
    y = df["Material type (label)"].to_numpy(int); split = df.split.to_numpy()
    groups = df.composition.map(lambda s: "|".join(sorted(set(re.findall(r"[A-Z][a-z]?", s))))).to_numpy()
    print(f"rows {len(df)}  minority {int((y != 4).sum())}  element-set groups {len(set(groups))}", flush=True)
    X = {"classic": Xc, "kmd": Xk}
    results = json.loads(a.out.read_text()) if a.out.exists() else {"runs": [], "leakage": {}}
    done = {(r["protocol"], r["fold"], r["descriptor"], r["model"]) for r in results["runs"]}

    def folds(protocol):
        if protocol == "dataset":
            tr = np.flatnonzero(split == "train"); va = np.flatnonzero(split == "val"); te = np.flatnonzero(split == "test")
            yield 0, tr, va, te
        else:
            kf = StratifiedKFold(a.folds, shuffle=True, random_state=0) if protocol == "random" else StratifiedGroupKFold(a.folds, shuffle=True, random_state=0)
            for k, (trv, te) in enumerate(kf.split(np.zeros(len(y)), y, groups)):
                tr, va = train_test_split(trv, test_size=0.1, random_state=k, stratify=y[trv])
                yield k, tr, va, te

    for protocol in a.protocols.split(","):
        for k, tr, va, te in folds(protocol):
            key = f"{protocol}/{k}"
            if key not in results["leakage"]:
                results["leakage"][key] = nearest_train_distance(W, y, np.r_[tr, va], te)
            for desc in a.descriptors.split(","):
                for model in a.models.split(","):
                    if (protocol, k, desc, model) in done:
                        continue
                    t0 = time.time()
                    if model == "nn":
                        pred, ep = run_nn(X[desc], y, tr, va, te, device)
                    else:
                        rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=k).fit(X[desc][np.r_[tr, va]], y[np.r_[tr, va]])
                        pred, ep = rf.predict(X[desc][te]), None
                    sc = score(y[te], pred)
                    rec = {"protocol": protocol, "fold": k, "descriptor": desc, "model": model, "n_train": int(len(tr) + len(va)), "n_test": int(len(te)),
                           "n_test_minority": int((y[te] != 4).sum()), "epochs": ep, "seconds": round(time.time() - t0, 1), **sc}
                    results["runs"].append(rec); a.out.write_text(json.dumps(results))
                    pc = sc["per_class"]
                    print(f"{protocol:8s} fold {k} {desc:8s} {model:3s}  macro-F1 {sc['macro_f1']:.3f}  acc {sc['accuracy']:.4f}  3-class F1 {sc['macro_f1_3class']:.3f} "
                          f"| recall IAC {pc['IAC']['recall'] or 0:.2f} IQC {pc['IQC']['recall'] or 0:.2f} | precision IAC {pc['IAC']['precision'] or 0:.2f} IQC {pc['IQC']['precision'] or 0:.2f}  {rec['seconds']}s", flush=True)
    # summary
    by = {}
    for r in results["runs"]:
        by.setdefault((r["protocol"], r["descriptor"], r["model"]), []).append(r)
    print("\nprotocol   descriptor model  n   macro-F1 (5 classes)   accuracy   macro-F1 (3 classes: QC / AC / others)")
    for (p, dsc, mdl), rs in sorted(by.items()):
        f = [r["macro_f1"] for r in rs]; acc = [r["accuracy"] for r in rs]; f3 = [r["macro_f1_3class"] for r in rs]
        sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
        print(f"{p:10s} {dsc:10s} {mdl:5s} {len(rs)}   {st.fmean(f):.3f} ± {sd(f):.3f}        {st.fmean(acc):.4f}     {st.fmean(f3):.3f} ± {sd(f3):.3f}")
    for key, L in results["leakage"].items():
        d = [x["d_any"] for x in L]
        print(f"leakage {key:10s} minority test rows {len(L):3d}: nearest-train L1 distance median {np.median(d):.3f}, share < 0.05: {np.mean(np.array(d) < 0.05):.2f}, share < 0.10: {np.mean(np.array(d) < 0.10):.2f}")


if __name__ == "__main__":
    main()
