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


@pytest.mark.parametrize("n_grids", [None, 1])
def test_1d_requires_valid_n_grids(component_features, n_grids):
    with pytest.raises(ValueError, match="n_grids"):
        KMD(component_features, method="1d", n_grids=n_grids)


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0])
def test_rejects_non_positive_sigma(component_features, bad_sigma):
    with pytest.raises(ValueError, match="sigma"):
        KMD(component_features, method="md", sigma=bad_sigma)


def test_requires_at_least_two_components():
    with pytest.raises(ValueError, match="at least 2 components"):
        KMD(np.ones((1, 3)), method="md")


def test_md_auto_sigma_rejects_duplicate_components():
    cf = np.ones((4, 3))  # identical rows -> median nearest distance is 0
    with pytest.raises(ValueError, match="auto sigma is undefined"):
        KMD(cf, method="md", scale=False)


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


# --- differentiable torch transform -----------------------------------------


@pytest.mark.parametrize("method,kwargs", [("1d", {"n_grids": 10}), ("md", {})])
def test_transform_torch_matches_numpy(method, kwargs, weight, component_features):
    import torch

    kmd = KMD(component_features, method=method, **kwargs)
    np_out = kmd.transform(weight)
    torch_out = kmd.transform_torch(torch.as_tensor(weight, dtype=torch.float64))
    assert torch.allclose(torch_out, torch.as_tensor(np_out), atol=1e-10)


def test_transform_torch_is_differentiable(component_features):
    """Gradients flow through transform_torch back to the weight tensor."""
    import torch

    kmd = KMD(component_features, method="1d", n_grids=10)
    w = torch.rand(3, component_features.shape[0], dtype=torch.float64, requires_grad=True)
    desc = kmd.transform_torch(w)
    loss = desc.pow(2).sum()
    loss.backward()
    assert w.grad is not None
    assert w.grad.shape == w.shape
    assert torch.any(w.grad != 0)


def test_kernel_torch_mutation_does_not_affect_numpy_kmd(component_features, weight):
    """Backwards-compatibility lock: torch additions never affect the numpy KMD path.

    The legacy ``transform`` / ``inverse`` callers must keep working unchanged even after we
    hand out torch views of the kernel — including the worst case where downstream code
    accidentally mutates the returned tensor in place.
    """

    kmd = KMD(component_features, method="1d", n_grids=10)
    before = kmd.transform(weight)
    # Try to mutate the returned tensor in place (worst-case adversarial use).
    t = kmd.kernel_torch()
    t.zero_()
    t2 = kmd.kernel_torch()
    t2.add_(1000.0)
    # The numpy transform must still produce the original output.
    after = kmd.transform(weight)
    np.testing.assert_array_equal(before, after)
    # And a fresh kernel_torch call must still match the numpy kernel.
    fresh = kmd.kernel_torch()
    np.testing.assert_allclose(fresh.numpy(), kmd.transform(np.eye(component_features.shape[0])))


def test_transform_torch_reuses_internal_cache(component_features):
    """transform_torch must not deep-copy the kernel on every call (perf-critical for loops)."""
    import torch

    kmd = KMD(component_features, method="1d", n_grids=10)
    w = torch.rand(2, component_features.shape[0])
    _ = kmd.transform_torch(w)  # first call seeds the internal cache
    cached = kmd._internal_kernel_torch(device=w.device, dtype=w.dtype)
    again = kmd._internal_kernel_torch(device=w.device, dtype=w.dtype)
    assert cached is again  # same (device, dtype) key returns the cached tensor


def test_transform_torch_preserves_input_dtype():
    """The torch transform should respect the caller's dtype (e.g. fp64 stays fp64)."""
    import torch

    cf = np.random.default_rng(0).normal(size=(6, 3))
    kmd = KMD(cf, method="1d", n_grids=8)
    w_fp64 = torch.rand(2, 6, dtype=torch.float64)
    assert kmd.transform_torch(w_fp64).dtype == torch.float64
    w_fp32 = w_fp64.to(torch.float32)
    assert kmd.transform_torch(w_fp32).dtype == torch.float32


# --- roundtrip ---------------------------------------------------------------


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("1d", {"n_grids": 12}),
        ("1d", {"n_grids": 12, "sigma": 0.1}),
        ("1d", {"n_grids": 12, "scale": False}),
        ("md", {}),
        ("md", {"sigma": 0.5}),
        ("md", {"scale": False}),
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


def test_inverse_rejects_non_invertible_kernel(weight):
    # Duplicate components make the Gram matrix singular -> inverse must error.
    cf = np.zeros((6, 3))
    cf[:3] = np.eye(3)
    cf[3:] = np.eye(3)  # rows 3-5 duplicate rows 0-2
    kmd = KMD(cf, method="1d", n_grids=4, scale=False)
    with pytest.raises(ValueError, match="not invertible"):
        kmd.inverse(np.zeros((1, kmd._kernel.shape[1])))


# --- stats_descriptor --------------------------------------------------------


def test_stats_descriptor_blocks(weight, component_features):
    sd = stats_descriptor(weight, component_features)
    nf = component_features.shape[1]
    assert sd.shape == (weight.shape[0], nf * 4)

    cf = np.asarray(component_features)
    mean = weight @ cf
    np.testing.assert_allclose(sd[:, :nf], mean)
    # var block: weighted variance about the weighted mean.
    var = np.array([weight[i] @ (cf - mean[i]) ** 2 for i in range(weight.shape[0])])
    np.testing.assert_allclose(sd[:, nf : 2 * nf], var)
    # max/min pooling over the present (non-zero-weight) components.
    for i in range(weight.shape[0]):
        present = cf[weight[i] != 0]
        np.testing.assert_allclose(sd[i, 2 * nf : 3 * nf], present.max(axis=0))
        np.testing.assert_allclose(sd[i, 3 * nf : 4 * nf], present.min(axis=0))


def test_stats_descriptor_respects_stats_order(weight, component_features):
    nf = component_features.shape[1]
    sd = stats_descriptor(weight, component_features, stats=("min", "mean"))
    assert sd.shape == (weight.shape[0], nf * 2)
    np.testing.assert_allclose(sd[:, nf:], weight @ np.asarray(component_features))


def test_stats_descriptor_rejects_unknown_stat(weight, component_features):
    with pytest.raises(ValueError, match="unsupported stat"):
        stats_descriptor(weight, component_features, stats=["mean", "bogus"])


@pytest.mark.parametrize("stat", ["max", "min"])
def test_stats_descriptor_rejects_all_zero_weight_row(component_features, stat):
    w = np.zeros((2, component_features.shape[0]))
    w[0, 0] = 1.0  # row 1 stays all-zero
    with pytest.raises(ValueError, match="nonzero weight"):
        stats_descriptor(w, component_features, stats=[stat])
