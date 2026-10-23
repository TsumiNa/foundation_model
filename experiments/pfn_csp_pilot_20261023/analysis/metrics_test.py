import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from metrics import all_metrics, correct_top1, correct_topk, log_loss, seed_ci  # noqa: E402


def test_topk_counts_absent_true_class_as_wrong():
    # two classes seen in context, true class 2 never seen -> wrong at top-1 and top-3
    prob = np.array([[0.6, 0.4, 0.0, 0.0]])
    y = np.array([2])
    assert correct_top1(prob, y)[0] == 0
    assert correct_topk(prob, y, 3)[0] == 0


def test_topk_rank():
    prob = np.array([[0.5, 0.3, 0.15, 0.05], [0.5, 0.3, 0.15, 0.05]])
    assert correct_topk(prob, np.array([2, 3]), 3).tolist() == [1.0, 0.0]


def test_log_loss_clips_and_renormalizes():
    prob = np.array([[1.0, 0.0]])
    ll = log_loss(prob, np.array([1]))
    p = 1e-6 / (1 + 1e-6)
    assert ll == pytest.approx(-np.log(p))


def test_all_metrics_coverage():
    prob = np.array([[0.9, 0.1], [0.2, 0.8]])
    m = all_metrics(prob, np.array([0, 0]), np.array([5, 5]), np.array([5, 0]))
    assert m["top1"] == 0.5 and m["coverage"] == 0.5 and m["coverage_any"] == 0.5


def test_seed_ci_constant_and_nan():
    assert seed_ci([0.1] * 5) == pytest.approx((0.1, 0.1, 0.1))
    mean, lo, hi = seed_ci([0.0, 0.1, 0.2, 0.3, np.nan])
    assert mean == pytest.approx(0.15) and lo <= mean <= hi


def test_split_protocol_a_rules():
    from run_experiment import split_protocol_a

    y = np.array([0] * 10 + [1] * 2 + [2] * 1 + [3] * 7)
    fam = np.zeros(len(y), dtype=int)
    t = split_protocol_a(y, fam, seed=0)
    assert t[y == 0].sum() == 2  # round(2.0)
    assert t[y == 1].sum() == 1  # round(0.4)=0 -> at least 1
    assert t[y == 2].sum() == 0  # n = 1 -> all train
    assert t[y == 3].sum() == 1  # round-half-up(1.4) = 1
    assert (split_protocol_a(y, fam, seed=0) == t).all()


def test_engine_single_class_context_and_mapping():
    from engines import Engine

    eng = Engine("catboost", n_classes=5)
    p = eng.fit_predict(np.zeros((3, 2)), np.array([4, 4, 4]), np.zeros((2, 2)), seed=0)
    assert p.tolist() == [[0, 0, 0, 0, 1.0]] * 2


def test_accum_averages_over_valid_draws():
    from run_experiment import Accum

    acc = Accum(np.array([10, 11]), n_classes=2)
    acc.add(np.array([10, 11]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), {0})
    acc.add(np.array([10]), np.array([[0.0, 1.0]]), np.array([0]), {0, 1})
    r = acc.result()
    assert r["prob"].tolist() == [[0.5, 0.5], [0.0, 1.0]]
    assert r["n_valid"].tolist() == [2, 1] and r["n_covered"].tolist() == [2, 0]


def test_skeleton_strip_and_topk_growth():
    from csp_labels import strip_orientation, top_k_labels

    assert strip_orientation("4/mm.m") == "4/mmm" and strip_orientation("-3m.") == "-3m"
    labels = [f"c{i}" for i in range(10) for _ in range(i + 1)]  # 55 labels, long tail
    out, k, other = top_k_labels(labels, k0=2, other_cap=0.30, k_max=6)
    assert k == 6 and other == pytest.approx(sum(range(1, 5)) / 55)
    assert set(out) == {"other", "c9", "c8", "c7", "c6", "c5", "c4"}


def test_roles_accept_unreduced_peroxide_formula():
    from csp_labels import RoleError, assign_roles
    from pymatgen.core import Composition

    assert assign_roles(Composition("K2O2"), "AB") == {"A": "K", "B": "O"}
    assert assign_roles(Composition("CaCO3"), "ABX3") == {"A": "Ca", "B": "C", "X": "O"}
    with pytest.raises(RoleError):
        assign_roles(Composition("Fe3N"), "AB")
    with pytest.raises(RoleError):  # coefficient-3 element (Fe) is not the most electronegative
        assign_roles(Composition("Fe3ZnN"), "ABX3")


class _FakeEngine:
    """Uniform probabilities over the context classes; records every context."""

    name = "fake"

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.contexts = []

    def fit_predict(self, xc, yc, xq, seed):
        self.contexts.append(np.sort(xc[:, 0]).tolist())
        out = np.zeros((len(xq), self.n_classes))
        out[:, np.unique(yc)] = 1.0 / len(np.unique(yc))
        return out


def _toy(n=60, n_fam=3, n_cls=4, seed=0):
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.arange(n, dtype=float), rng.normal(size=(n, 3))]).astype(np.float32)
    return x, rng.integers(0, n_cls, n), np.repeat(np.arange(n_fam), n // n_fam)


def test_protocol_m_uses_protocol_b_draws_plus_other_families():
    from run_experiment import protocol_b, protocol_m

    x, y, fam = _toy()
    eb, em = _FakeEngine(4), _FakeEngine(4)
    rb = protocol_b(eb, x, y, fam, 0, lambda m: None)
    rm = protocol_m(em, x, y, fam, 0, lambda m: None)
    c2 = [c for c in eb.contexts if len(c) in (5, 10, 20)]
    assert len(c2) == len(em.contexts)
    rest_sizes = {len(np.flatnonzero(fam != f)) for f in range(3)}
    for pure, mixed in zip(c2, em.contexts):
        assert set(pure) <= set(mixed) and len(mixed) - len(pure) in rest_sizes
    for key, res in rm.items():
        assert res["n_draws"] == rb[("C2", key[1], key[2])]["n_draws"]
        assert res["n_valid"].min() >= 3


def test_protocol_r_conditions_and_neighbours():
    from run_experiment import K_R, protocol_r

    x, y, fam = _toy(n=90)
    eng = _FakeEngine(4)
    r = protocol_r(eng, x, y, fam, 0, lambda m: None)
    test = r[("C1", 0)]["query"]
    n_train = 90 - len(test)
    for k in K_R:
        if k > n_train:
            assert ("C4", k) not in r
            continue
        assert len(r[("C4", k)]["query"]) == len(test) and (r[("C5", k)]["n_valid"] == 5).all()
