# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the kernel mean descriptor utilities."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core.composition import Composition

from foundation_model.utils.kmd_plus import (
    KMD,
    elements,
    formula_to_composition,
    stats_descriptor,
)


@pytest.fixture
def component_features() -> np.ndarray:
    """A small, well-separated feature matrix (8 components, 3 features)."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(8, 3))


@pytest.fixture
def weight() -> np.ndarray:
    """Row-normalized, sparse-ish mixing weights for 5 samples over 8 components."""
    rng = np.random.default_rng(1)
    w = rng.random((5, 8))
    w[w < 0.4] = 0.0  # introduce zeros so max/min pooling is exercised
    w[0] = 0.0
    w[0, :2] = [0.5, 0.5]  # guard against an all-zero row
    return w / w.sum(axis=1, keepdims=True)


# --- formula_to_composition --------------------------------------------------


def test_formula_to_composition_basic():
    vec = formula_to_composition("SiO2")
    assert vec.shape == (len(elements),)
    assert vec[elements.index("Si")] == pytest.approx(1 / 3)
    assert vec[elements.index("O")] == pytest.approx(2 / 3)
    assert vec.sum() == pytest.approx(1.0)


def test_formula_to_composition_accepts_dict_and_composition():
    from_dict = formula_to_composition({"Si": 1.0, "O": 2.0})
    from_comp = formula_to_composition(Composition("SiO2"))
    np.testing.assert_allclose(from_dict, from_comp)


def test_formula_to_composition_rejects_unknown_type():
    with pytest.raises(TypeError):
        formula_to_composition(42)  # type: ignore[arg-type]


# --- KMD construction --------------------------------------------------------


def test_invalid_method_raises(component_features):
    with pytest.raises(ValueError, match="method"):
        KMD(component_features, method="nope")  # type: ignore[arg-type]


def test_1d_requires_n_grids(component_features):
    with pytest.raises(ValueError, match="n_grids"):
        KMD(component_features, method="1d")


@pytest.mark.parametrize("method,kwargs", [("1d", {"n_grids": 10}), ("md", {})])
def test_transform_shape(method, kwargs, weight, component_features):
    kmd = KMD(component_features, method=method, **kwargs)
    out = kmd.transform(weight)
    n_features = component_features.shape[1]
    expected_cols = n_features * 10 if method == "1d" else component_features.shape[0]
    assert out.shape == (weight.shape[0], expected_cols)


@pytest.mark.parametrize("method,kwargs", [("1d", {"n_grids": 10}), ("md", {})])
def test_call_matches_transform(method, kwargs, weight, component_features):
    kmd = KMD(component_features, method=method, **kwargs)
    np.testing.assert_array_equal(kmd(weight), kmd.transform(weight))


# --- roundtrip ---------------------------------------------------------------


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("1d", {"n_grids": 12}),
        ("md", {}),
        ("md", {"sigma": 0.5}),
    ],
)
def test_inverse_reconstructs_weights(method, kwargs, weight, component_features):
    kmd = KMD(component_features, method=method, **kwargs)
    recovered = kmd.inverse(kmd.transform(weight))
    assert recovered.shape == weight.shape
    np.testing.assert_allclose(recovered.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(recovered, weight, atol=1e-4)


def test_inverse_caches_gram(component_features, weight):
    kmd = KMD(component_features, method="md")
    assert kmd._gram is None
    kmd.inverse(kmd.transform(weight))
    assert kmd._gram is not None


# --- stats_descriptor --------------------------------------------------------


def test_stats_descriptor_shape_and_mean(weight, component_features):
    sd = stats_descriptor(weight, component_features)
    n_features = component_features.shape[1]
    assert sd.shape == (weight.shape[0], n_features * 4)
    # First block is the weighted mean.
    np.testing.assert_allclose(sd[:, :n_features], weight @ component_features)


def test_stats_descriptor_rejects_unknown_stat(weight, component_features):
    with pytest.raises(ValueError, match="unsupported stat"):
        stats_descriptor(weight, component_features, stats=["mean", "bogus"])
