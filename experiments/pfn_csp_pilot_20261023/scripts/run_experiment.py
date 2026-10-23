"""Run one (stoichiometry x label x partition x engine x protocol x seed) unit (plan section 3).

    uv run python scripts/run_experiment.py --cls AB --label L2 --part P1 --engine tabpfn \
        --protocol A --seed 0

Writes artifacts/runs/<cls>_<label>_<part>/<engine>/prot<protocol>_s<seed>.pkl holding, per
(condition, k), the query indices, the draw-averaged mapped probabilities and the coverage
counts. Metrics are computed afterwards by analysis/summarize.py. Existing outputs are skipped.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csp_features import zscore_imputed  # noqa: E402
from engines import Engine  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
K_A = (5, 10, 20, 50)
K_B = (5, 10, 20)
N_DRAWS = 5
MIN_VALID_B = 3
MAX_DRAWS_B = 2000

# Stage tags for SeedSequence so every random stream derives from the experiment seed.
TAG_SPLIT, TAG_C2, TAG_C3, TAG_B = 1, 2, 3, 4


def rng_for(seed: int, *tags: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *tags]))


def load_table(cls: str, label: str, part: str):
    df = pd.read_parquet(EXP / "artifacts" / "data" / f"table_{cls}.parquet")
    df = df[df[part].notna()].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c.startswith(("magpie::", "site::"))]
    x = df[feat_cols].to_numpy(dtype=np.float32)
    labels = sorted(df[label].unique())
    y = df[label].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
    fams = sorted(df[part].unique())
    fam = df[part].map({f: i for i, f in enumerate(fams)}).to_numpy()
    return df, x, y, fam, labels, fams


def split_protocol_a(y: np.ndarray, fam: np.ndarray, seed: int) -> np.ndarray:
    """Per family, per class: n_test = round-half-up(0.2 n), at least 1 when n >= 2, 0 when n = 1."""
    rng = rng_for(seed, TAG_SPLIT)
    is_test = np.zeros(len(y), dtype=bool)
    for f in np.unique(fam):
        for c in np.unique(y[fam == f]):
            idx = np.flatnonzero((fam == f) & (y == c))
            n = len(idx)
            if n < 2:
                continue
            n_test = max(1, int(np.floor(0.2 * n + 0.5)))
            is_test[rng.choice(idx, n_test, replace=False)] = True
    return is_test


class Accum:
    """Sum of mapped probabilities and coverage counts per query for one (condition, k)."""

    def __init__(self, queries: np.ndarray, n_classes: int):
        self.q = np.asarray(queries)
        self.pos = {int(i): j for j, i in enumerate(self.q)}
        self.p = np.zeros((len(self.q), n_classes))
        self.valid = np.zeros(len(self.q), dtype=int)
        self.covered = np.zeros(len(self.q), dtype=int)
        self.same_fam = np.full(len(self.q), np.nan)

    def add(self, qidx: np.ndarray, probs: np.ndarray, ytrue: np.ndarray, ctx_classes) -> None:
        rows = np.array([self.pos[int(i)] for i in qidx], dtype=int)
        self.p[rows] += probs
        self.valid[rows] += 1
        self.covered[rows] += np.isin(ytrue, list(ctx_classes)).astype(int)

    def result(self) -> dict:
        p = self.p / np.maximum(self.valid, 1)[:, None]
        return {
            "query": self.q,
            "prob": p.astype(np.float32),
            "n_valid": self.valid,
            "n_covered": self.covered,
            "same_family_share": self.same_fam,
        }


def protocol_a(eng: Engine, x, y, fam, seed: int, conditions: list[str], log) -> dict:
    is_test = split_protocol_a(y, fam, seed)
    train = np.flatnonzero(~is_test)
    test = np.flatnonzero(is_test)
    n_cls = eng.n_classes
    out: dict = {"is_test": is_test}

    if "C1" in conditions:
        t = time.time()
        acc = Accum(test, n_cls)
        acc.add(test, eng.fit_predict(x[train], y[train], x[test], seed), y[test], set(y[train]))
        out[("C1", 0)] = acc.result()
        log(f"C1 fit on {len(train)} rows: {time.time() - t:.1f}s")
    if eng.name == "catboost":
        return out

    ntrain_f = {f: int(((fam == f) & ~is_test).sum()) for f in np.unique(fam)}
    for k in K_A:
        elig_f = [f for f in np.unique(fam) if ntrain_f[f] >= k]
        q_k = test[np.isin(fam[test], elig_f)]
        out[("eligible", k)] = q_k
        for cond, tag in (("C2", TAG_C2), ("C3", TAG_C3)):
            if cond not in conditions:
                continue
            t = time.time()
            acc = Accum(q_k, n_cls)
            for f in elig_f:
                qf = test[fam[test] == f]
                pool = train[fam[train] == f] if cond == "C2" else train[fam[train] != f]
                rng = rng_for(seed, tag, int(f), k)
                for _ in range(N_DRAWS):
                    ctx = rng.choice(pool, k, replace=False)
                    acc.add(qf, eng.fit_predict(x[ctx], y[ctx], x[qf], seed), y[qf], set(y[ctx]))
            out[(cond, k)] = acc.result()
            log(f"{cond} k={k}: {len(elig_f)} families x {N_DRAWS} draws, {time.time() - t:.1f}s")
        if "C4" in conditions:
            t = time.time()
            ztr = zscore_imputed(x[train].astype(float))
            zq = zscore_imputed(x[q_k].astype(float), ref=x[train].astype(float))
            acc = Accum(q_k, n_cls)
            for j, qi in enumerate(q_k):
                d = np.linalg.norm(ztr - zq[j], axis=1)
                ctx = train[np.argsort(d, kind="stable")[:k]]
                acc.add(np.array([qi]), eng.fit_predict(x[ctx], y[ctx], x[[qi]], seed), y[[qi]], set(y[ctx]))
                acc.same_fam[j] = float(np.mean(fam[ctx] == fam[qi]))
            out[("C4", k)] = acc.result()
            log(f"C4 k={k}: {len(q_k)} queries, {time.time() - t:.1f}s")
    return out


def protocol_b(eng: Engine, x, y, fam, seed: int, log) -> dict:
    n_cls = eng.n_classes
    out: dict = {}
    for f in np.unique(fam):
        held = np.flatnonzero(fam == f)
        rest = np.flatnonzero(fam != f)
        t = time.time()
        acc = Accum(held, n_cls)
        acc.add(held, eng.fit_predict(x[rest], y[rest], x[held], seed), y[held], set(y[rest]))
        out[("C1", int(f), 0)] = acc.result()
        if eng.name == "catboost":
            log(f"F={f} C1 {time.time() - t:.1f}s")
            continue
        for k in K_B:
            if len(held) < k + 3:
                continue
            rng = rng_for(seed, TAG_B, int(f), k)
            acc = Accum(held, n_cls)
            draws = 0
            while draws < N_DRAWS or acc.valid.min() < MIN_VALID_B:
                if draws >= MAX_DRAWS_B:
                    raise RuntimeError(f"family {f} k={k}: draw cap reached")
                ctx = rng.choice(held, k, replace=False)
                q = np.setdiff1d(held, ctx)
                acc.add(q, eng.fit_predict(x[ctx], y[ctx], x[q], seed), y[q], set(y[ctx]))
                draws += 1
            res = acc.result()
            res["n_draws"] = draws
            out[("C2", int(f), k)] = res
        log(f"F={f} (|F|={len(held)}) done {time.time() - t:.1f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, choices=["AB", "ABX3"])
    ap.add_argument("--label", required=True, choices=["L1", "L2"])
    ap.add_argument("--part", required=True, choices=["P1", "P2"])
    ap.add_argument("--engine", required=True, choices=["tabpfn", "tabicl", "catboost"])
    ap.add_argument("--protocol", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conditions", default="C1,C2,C3,C4")
    ap.add_argument("--out-root", default=str(EXP / "artifacts" / "runs"))
    a = ap.parse_args()

    combo = f"{a.cls}_{a.label}_{a.part}"
    out_dir = Path(a.out_root) / combo / a.engine
    out_file = out_dir / f"prot{a.protocol}_s{a.seed}.pkl"
    if out_file.exists():
        print(f"exists: {out_file}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df, x, y, fam, labels, fams = load_table(a.cls, a.label, a.part)
    eng = Engine(a.engine, len(labels), a.device)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{combo} {a.engine} {a.protocol} s{a.seed} {time.time() - t0:7.0f}s] {msg}", flush=True)

    if a.protocol == "A":
        res = protocol_a(eng, x, y, fam, a.seed, a.conditions.split(","), log)
    else:
        res = protocol_b(eng, x, y, fam, a.seed, log)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=EXP).stdout.strip()
    meta = {
        "args": vars(a),
        "labels": labels,
        "families": fams,
        "y": y,
        "fam": fam,
        "material_id": df["material_id"].to_numpy(),
        "elapsed_s": time.time() - t0,
        "git_commit": commit,
    }
    tmp = out_file.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"meta": meta, "res": res}, fh)
    tmp.rename(out_file)
    prov = out_dir / f"prot{a.protocol}_s{a.seed}.provenance.json"
    prov.write_text(
        json.dumps({**{k: v for k, v in meta.items() if k in ("args", "elapsed_s", "git_commit")}}, indent=1)
    )
    log(f"wrote {out_file}")


if __name__ == "__main__":
    main()
