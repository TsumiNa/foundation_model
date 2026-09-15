#!/usr/bin/env python3
"""Does a pretrained encoder help material_type on NEW systems? Warm-start vs alone under grouped CV.

Same folds as analysis/material_type_cv.py ("system" = element-set-grouped StratifiedGroupKFold,
"random" = row-wise StratifiedKFold), the pipeline's KMD descriptor and network, unweighted
cross-entropy, pipeline recipe. Two arms per fold:

  alone       fresh encoder + head, 3 seeds
  warm-start  encoder loaded from a 23-task pretrained checkpoint that NEVER saw material_type
              (stage_xu o0 / o1 / o2, the same encoders as the never-seen arm on RIKYU), fresh head;
              encoder + head trained

    uv run python analysis/material_type_transfer_cv.py --ckpt <dir with o0.pt o1.pt o2.pt> -o summary/material_type_transfer_cv.json
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "analysis"))
import material_type_cv as C  # noqa: E402
import space_group_study as M  # noqa: E402

from foundation_model.data.composition_sources import PrecomputedDescriptorSource  # noqa: E402
from foundation_model.utils.kmd_plus import KMD, element_features, formula_to_composition  # noqa: E402

PRETRAINED: dict = {"path": None}


class WarmFmNet(M.FmNet):
    """The pipeline's encoder + head, with the encoder weights loaded from a pretraining checkpoint."""

    def __init__(self, d_in, n_out):
        super().__init__(d_in, n_out)
        if PRETRAINED["path"] is not None:
            sd = torch.load(PRETRAINED["path"], map_location="cpu", weights_only=False)["model"]
            enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
            missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
            assert not [k for k in missing if "num_batches" not in k], missing
            PRETRAINED["loaded"] = len(enc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--protocols", default="system,random")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    M.NUM_CLASSES = 5; M.TOPK = (1, 2); M.FmNet = WarmFmNet
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    F = ROOT / "data" / "qc_ac_te_mp_dos_reformat_20260912.pd.parquet"
    df = pd.read_parquet(F, columns=["composition", "split", "Material type (label)"])
    src = PrecomputedDescriptorSource(str(ROOT / "data" / "desc_xenonpy_classic_trans.parquet"), composition_column="composition")
    d = src(list(df.composition)); df = df[df.composition.isin(d.index)].reset_index(drop=True)  # the same 48,998 rows as material_type_cv
    W = np.stack(df.composition.map(formula_to_composition).values)
    Xk = KMD(element_features.values, method="1d", n_grids=8, sigma="auto", scale=True).transform(W).astype(np.float32)
    y = df["Material type (label)"].to_numpy(int)
    groups = df.composition.map(lambda s: "|".join(sorted(set(re.findall(r"[A-Z][a-z]?", s))))).to_numpy()
    results = json.loads(a.out.read_text()) if a.out.exists() else {"runs": []}
    done = {(r["protocol"], r["fold"], r["arm"], r["init"]) for r in results["runs"]}
    for protocol in a.protocols.split(","):
        kf = StratifiedKFold(a.folds, shuffle=True, random_state=0) if protocol == "random" else StratifiedGroupKFold(a.folds, shuffle=True, random_state=0)
        for k, (trv, te) in enumerate(kf.split(np.zeros(len(y)), y, groups)):
            tr, va = train_test_split(trv, test_size=0.1, random_state=k, stratify=y[trv])
            arms = [("alone", None, s) for s in (2025, 2026, 2027)] + [("warm", str(a.ckpt / f"o{i}.pt"), 2025) for i in range(3)]
            for arm, ck, seed in arms:
                init = Path(ck).stem if ck else f"seed{seed}"
                if (protocol, k, arm, init) in done:
                    continue
                PRETRAINED["path"] = ck; t0 = time.time()
                cache = {"Xk": Xk, "Xc": Xk, "Xn": Xk, "y": y, "split": np.array(["other"] * len(y), dtype=object)}
                cache["split"][tr] = "train"; cache["split"][va] = "val"; cache["split"][te] = "test"
                captured = {}; orig = M.evaluate

                def ev(model, Xt, yy, dev):
                    out = orig(model, Xt, yy, dev); model.eval()
                    with torch.no_grad():
                        captured["pred"] = model(Xt.to(dev)).argmax(1).cpu().numpy()
                    return out

                M.evaluate = ev
                try:
                    r = M.run("fm_plain", seed, cache, device)
                finally:
                    M.evaluate = orig
                sc = C.score(y[te], captured["pred"])
                rec = {"protocol": protocol, "fold": k, "arm": arm, "init": init, "epochs": r["epochs_run"], "best_epoch": r["best_epoch"], "seconds": round(time.time() - t0, 1), **sc}
                results["runs"].append(rec); a.out.write_text(json.dumps(results))
                p3 = sc["per_class_3"]
                print(f"{protocol:7s} fold {k} {arm:5s} {init:8s} ep {r['epochs_run']:3d}  3-class F1 {sc['macro_f1_3class']:.3f} (QC R {p3['QC']['recall'] or 0:.2f} P {p3['QC']['precision'] or 0:.2f} · AC R {p3['AC']['recall'] or 0:.2f} P {p3['AC']['precision'] or 0:.2f})  5-class {sc['macro_f1']:.3f}  {rec['seconds']}s", flush=True)
    print("\nprotocol arm    n   3-class macro-F1     5-class macro-F1    epochs")
    by = {}
    for r in results["runs"]:
        by.setdefault((r["protocol"], r["arm"]), []).append(r)
    for (p, arm), rs in sorted(by.items()):
        f3 = [r["macro_f1_3class"] for r in rs]; f5 = [r["macro_f1"] for r in rs]; ep = [r["epochs"] for r in rs]
        print(f"{p:8s} {arm:5s} {len(rs):2d}   {st.fmean(f3):.3f} ± {st.stdev(f3):.3f}       {st.fmean(f5):.3f} ± {st.stdev(f5):.3f}     {st.fmean(ep):.0f}")
    # paired per fold: warm mean − alone mean
    for p in a.protocols.split(","):
        diffs = []
        for k in range(a.folds):
            al = [r["macro_f1_3class"] for r in results["runs"] if r["protocol"] == p and r["fold"] == k and r["arm"] == "alone"]
            wm = [r["macro_f1_3class"] for r in results["runs"] if r["protocol"] == p and r["fold"] == k and r["arm"] == "warm"]
            if al and wm:
                diffs.append(st.fmean(wm) - st.fmean(al))
        if diffs:
            print(f"{p}: warm − alone per fold (3-class F1): {[round(x, 3) for x in diffs]}  mean {st.fmean(diffs):+.3f}  2×SE {2 * st.stdev(diffs) / len(diffs) ** 0.5:.3f}")


if __name__ == "__main__":
    main()
