# Copyright 2027 TsumiNa.
# SPDX-License-Identifier: Apache-2.0
#
# Kernel mean descriptor (KMD) originally authored by Minoru Kusaba
# (ISM, kusaba@ism.ac.jp). Refactored into a stateful class with a
# precomputed kernel basis.

"""Kernel mean and summary-statistics descriptors for mixture compositions.

The module represents a mixture by two aligned arrays:

``component_features``
        A fixed matrix of shape ``(n_components, n_features)``. Row ``j`` contains
        the features of component ``j``. Components may be chemical elements,
        monomers, solvents, alloy endmembers, or any other constituents.
``weight``
        A sample matrix of shape ``(n_samples, n_components)``. Column ``j`` is the
        amount of the same component represented by row ``j`` of
        ``component_features``. KMD conventionally expects non-negative rows that
        sum to one, although :meth:`KMD.transform` deliberately does not enforce or
        normalize this constraint.

A :class:`KMD` instance converts the fixed component feature matrix into a
kernel basis ``K`` once, then reuses it for every sample. The forward operation
is simply ``weight @ K``. :meth:`KMD.inverse` solves the corresponding
non-negative, sum-to-one quadratic program, and :meth:`KMD.transform_torch`
provides the same forward operation with gradients through ``weight``.

The bundled :data:`element_features` table contains 58 features for 94 elements
(``H`` through ``Pu``). Together with :func:`formula_to_composition`, it
provides an element-composition descriptor out of the box.

Examples
--------
Create an 8-dimensional descriptor from two constituent features and four grid
points per feature:

>>> component_features = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 2.0]])
>>> weight = np.array([[0.25, 0.75, 0.0], [0.0, 0.4, 0.6]])
>>> kmd = KMD(component_features, method="1d", n_grids=4)
>>> kmd.transform(weight).shape
(2, 8)

For the complete migration checklist and element-based examples, see
``docs/kmd_descriptor_guide.md``.
"""

from __future__ import annotations

import os
from statistics import median
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from pymatgen.core.composition import Composition
from qpsolvers import solve_qp
from scipy.spatial import distance_matrix

# Element-level descriptors of shape (94, 58), indexed "H" ~ "Pu".
_ELEMENT_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "element_features.csv")
element_features: pd.DataFrame = pd.read_csv(_ELEMENT_FEATURES_PATH, index_col=0)
#: Default element layout ("H" ~ "Pu") used when none is supplied.
DEFAULT_ELEMENTS: list[str] = list(element_features.index)
elements = DEFAULT_ELEMENTS  # backwards-compatible alias

Method = Literal["md", "1d"]
Sigma = float | Literal["auto"]


def formula_to_composition(
    formula: str | dict[str, float] | Composition,
    elements: Sequence[str] | None = None,
) -> npt.NDArray[np.float64]:
    """Convert one chemical formula to an element-aligned atomic-fraction vector.

    The output index is defined exclusively by ``elements``. Entry ``j`` is the
    atomic fraction of ``elements[j]`` in the full composition. Consequently,
    the vector can be used as one row of ``weight`` only when ``elements`` has
    the same order as the rows of the KMD ``component_features`` matrix.

    Parameters
    ----------
    formula:
        Chemical formula such as ``"SiO2"``; a mapping from element symbol to
        amount, such as ``{"Si": 1, "O": 2}``; or an existing
        :class:`~pymatgen.core.composition.Composition`. Mapping values need not
        be normalized because pymatgen computes atomic fractions.
    elements:
        Ordered element symbols defining the output layout. The default is
        :data:`DEFAULT_ELEMENTS`, which matches the row order of the bundled
        :data:`element_features` table (94 elements from ``H`` through ``Pu``).

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` array of shape ``(len(elements),)``.
        Entries for absent elements are zero.

    Raises
    ------
    TypeError
        If ``formula`` is not a string, dictionary, or pymatgen
        :class:`~pymatgen.core.composition.Composition`.
    ValueError
        If pymatgen cannot parse the formula or its amounts.

    Notes
    -----
    Elements omitted from ``elements`` are not represented, and the remaining
    entries are not renormalized. Therefore, the result sums to one only when
    ``elements`` covers every element present in ``formula``.

    Examples
    --------
    >>> vector = formula_to_composition("SiO2", elements=["O", "Si", "Al"])
    >>> np.allclose(vector, [2 / 3, 1 / 3, 0.0])
    True
    >>> bool(np.isclose(formula_to_composition({"Li": 1, "Fe": 1, "P": 1, "O": 4}).sum(), 1.0))
    True
    """
    if elements is None:
        elements = DEFAULT_ELEMENTS

    if isinstance(formula, str):
        comp = Composition(formula)
    elif isinstance(formula, Composition):
        comp = formula
    elif isinstance(formula, dict):
        comp = Composition.from_dict(formula)
    else:
        raise TypeError("formula must be a str, dict, or pymatgen Composition.")

    return np.array([comp.get_atomic_fraction(el) if el in comp else 0.0 for el in elements])


class KMD:
    """Precomputed kernel mean descriptor for fixed mixture components.

    Construction fixes the component vocabulary, component row order, feature
    columns, scaling rule, and kernel hyperparameters. The resulting kernel is
    reused by :meth:`transform`, :meth:`inverse`, and :meth:`transform_torch`.
    Create one instance per descriptor definition and reuse it across batches.

    Parameters
    ----------
    component_features:
        Two-dimensional numeric array-like of shape
        ``(n_components, n_features)``. Each row describes one constituent and
        each column is one constituent-level feature. Row ``j`` must correspond
        to column ``j`` of every ``weight`` passed to :meth:`transform`. At least
        two rows are required. Values are converted to ``float64``.

        With ``scale=True``, every feature column must vary across components:
        ``method="md"`` requires a non-zero sample standard deviation and
        ``method="1d"`` requires a non-zero min-max range. Constant columns can
        otherwise produce non-finite kernels.
    method:
        Kernel construction strategy. ``"md"`` computes an RBF kernel between
        components in the full feature space, producing ``n_components``
        descriptor columns. ``"1d"`` computes a separate RBF response over an
        equally spaced grid for each feature and concatenates the blocks,
        producing ``n_features * n_grids`` descriptor columns.
    n_grids:
        Number of equally spaced grid points per feature. Required and at least
        2 for ``method="1d"``; ignored for ``method="md"``. Increasing it
        raises descriptor resolution, memory use, and output width linearly.
    sigma:
        Positive RBF width or ``"auto"``. A numeric value uses
        ``exp(-d**2 / (2 * sigma**2))``. For ``method="md"``, ``"auto"`` sets
        the RBF coefficient to the inverse median nearest-neighbour squared
        distance. For ``method="1d"``, it uses the inverse squared grid spacing
        independently for each feature.
    scale:
        Whether to rescale feature columns before kernel construction.
        ``method="md"`` applies z-score scaling with sample standard deviation
        (``ddof=1``); ``method="1d"`` applies min-max scaling. Scaling parameters
        are derived from ``component_features`` and are not exposed separately.

    Raises
    ------
    ValueError
        If ``method`` is unknown, ``n_grids`` is invalid for ``"1d"``, ``sigma``
        is non-positive, fewer than two components are supplied, or automatic
        multidimensional scaling is undefined for duplicate components.

    Notes
    -----
    Descriptor compatibility requires the exact same component row order,
    feature columns, feature values, ``method``, ``n_grids``, ``sigma``, and
    ``scale`` at training and inference time. This class stores the resulting
    kernel, not a public copy of the scaled component feature matrix.

    Examples
    --------
    Build and reuse a one-dimensional-grid KMD:

    >>> features = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 2.0]])
    >>> weights = np.array([[0.2, 0.3, 0.5]])
    >>> descriptor = KMD(features, method="1d", n_grids=4)
    >>> descriptor(weights).shape
    (1, 8)

    The multidimensional method instead returns one value per component:

    >>> KMD(features, method="md").transform(weights).shape
    (1, 3)
    """

    def __init__(
        self,
        component_features: npt.ArrayLike,
        *,
        method: Method = "1d",
        n_grids: int | None = None,
        sigma: Sigma = "auto",
        scale: bool = True,
    ) -> None:
        if method not in ("md", "1d"):
            raise ValueError(f'method must be "md" or "1d", got {method!r}.')
        if method == "1d" and (n_grids is None or n_grids < 2):
            raise ValueError('n_grids must be an integer >= 2 when method="1d".')
        if sigma != "auto" and (not isinstance(sigma, (int, float)) or sigma <= 0):
            raise ValueError(f'sigma must be "auto" or a positive float, got {sigma!r}.')

        self.method: Method = method
        self.n_grids = n_grids
        self.sigma = sigma
        self.scale = scale

        cf = np.asarray(component_features, dtype=float)
        if cf.shape[0] < 2:
            raise ValueError(f"need at least 2 components, got {cf.shape[0]}.")
        self.n_components: int = cf.shape[0]
        self._kernel: npt.NDArray[np.float64] = self._build_kernel(cf)
        self._gram: npt.NDArray[np.float64] | None = None

    def transform(self, weight: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map a batch of component weights to kernel mean descriptors.

            Parameters
            ----------
            weight:
                Two-dimensional numeric array-like of shape
                ``(n_samples, n_components)``. Column ``j`` weights row ``j`` of the
                ``component_features`` passed at construction. Values are converted
                to ``float64`` before multiplication.

                KMD mixtures conventionally use non-negative rows summing to one,
                but this method intentionally performs no sign, finiteness, or
                normalization checks. Normalize and validate upstream when those
                constraints are required.

            Returns
            -------
            numpy.ndarray
                ``float64`` descriptor matrix. Its shape is
                ``(n_samples, n_components)`` for ``method="md"`` and
                ``(n_samples, n_features * n_grids)`` for ``method="1d"``.

        Raises
        ------
        ValueError
            If the final dimension of ``weight`` does not equal ``n_components`` or
            the input is otherwise incompatible with matrix multiplication.

        Examples
        --------
        >>> features = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 2.0]])
        >>> kmd = KMD(features, method="1d", n_grids=3)
        >>> weights = np.array([[1.0, 0.0, 0.0], [0.25, 0.25, 0.5]])
        >>> kmd.transform(weights).shape
        (2, 6)
        """
        return np.asarray(weight, dtype=float) @ self._kernel

    def __call__(self, weight: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Call :meth:`transform` with ``weight``.

        This alias has the same inputs, outputs, and constraints as
        :meth:`transform` and allows a configured :class:`KMD` instance to be
        used as a descriptor callable.
        """
        return self.transform(weight)

    def _internal_kernel_torch(self, *, device=None, dtype=None):  # type: ignore[no-untyped-def]
        """Return the cached internal torch kernel for one device and dtype.

        The returned tensor is internal mutable state and must never be exposed
        to callers. Public code should use :meth:`kernel_torch`, which clones
        this value, or :meth:`transform_torch`, which only reads it.
        """
        import torch

        if not hasattr(self, "_kernel_torch_cache"):
            self._kernel_torch_cache: dict[tuple[str, str], "torch.Tensor"] = {}
        key = (str(device) if device is not None else "default", str(dtype) if dtype is not None else "default")
        cached = self._kernel_torch_cache.get(key)
        if cached is None:
            # First time for this (device, dtype): build an independent copy of the numpy kernel.
            base = torch.from_numpy(np.array(self._kernel, copy=True))
            if device is not None or dtype is not None:
                base = base.to(device=device, dtype=dtype)
            self._kernel_torch_cache[key] = base
            cached = base
        return cached

    def kernel_torch(self, *, device=None, dtype=None):  # type: ignore[no-untyped-def]
        """Copy the precomputed kernel into an independently owned torch tensor.

        Every call returns a clone that shares no storage with the NumPy kernel,
        the internal torch cache, or earlier return values. In-place mutation of
        the result is therefore safe. For repeated forward operations, prefer
        :meth:`transform_torch` to avoid cloning on every iteration.

        Parameters
        ----------
        device:
            Optional device accepted by :meth:`torch.Tensor.to`, for example
            ``"cpu"``, ``"cuda"``, or a :class:`torch.device`. Defaults to CPU
            when the cache has not already established another explicit device.
        dtype:
            Optional torch dtype such as :data:`torch.float32`. Without an
            override, the kernel retains its NumPy-derived ``float64`` dtype.

        Returns
        -------
        torch.Tensor
            Kernel of shape ``(n_components, n_descriptor_dims)``. The second
            dimension is ``n_components`` for ``"md"`` and
            ``n_features * n_grids`` for ``"1d"``.

        Notes
        -----
        PyTorch is imported lazily, so it is required only when a torch-specific
        method is called.
        """
        return self._internal_kernel_torch(device=device, dtype=dtype).clone()

    def transform_torch(self, weight, *, device=None, dtype=None):  # type: ignore[no-untyped-def]
        """Differentiably map a batch of torch component weights to descriptors.

        This is the torch equivalent of :meth:`transform`: it computes
        ``weight @ K`` while treating ``K`` as constant. Autograd therefore
        propagates gradients to ``weight`` but not to the component features or
        KMD hyperparameters. A kernel copy is cached per ``(device, dtype)``.

        Parameters
        ----------
        weight:
            :class:`torch.Tensor` of shape
            ``(n_samples, n_components)``. Column ``j`` weights row ``j`` of
            the construction-time ``component_features``. Gradients flow through
            this tensor. As in :meth:`transform`, values are not normalized or
            constrained automatically; use an operation such as ``softmax`` if
            a differentiable simplex parameterization is required.
        device:
            Optional device for the cached kernel. Defaults to ``weight.device``.
            It normally should not differ from the input device because torch
            matrix multiplication requires both operands on the same device.
        dtype:
            Optional dtype for the cached kernel. Defaults to ``weight.dtype``.
            It normally should match the input dtype.

        Returns
        -------
        torch.Tensor
            Descriptor tensor on the selected device and dtype, with the same
            output shape as :meth:`transform`.

        Examples
        --------
        Parameterize valid mixture weights and backpropagate through KMD:

        >>> import torch
        >>> features = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 2.0]])
        >>> kmd = KMD(features, method="1d", n_grids=4)
        >>> logits = torch.zeros(2, 3, requires_grad=True)
        >>> descriptors = kmd.transform_torch(torch.softmax(logits, dim=1))
        >>> descriptors.square().mean().backward()
        >>> logits.grad.shape
        torch.Size([2, 3])
        """
        kernel = self._internal_kernel_torch(device=device or weight.device, dtype=dtype or weight.dtype)
        return weight @ kernel

    def inverse(self, kmd: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Recover non-negative, sum-to-one weights from KMD descriptors.

        For each descriptor row ``z``, solve the quadratic program whose forward
        model is ``z = weight @ K``, subject to ``weight >= 0`` and
        ``weight.sum() == 1``. The Gram matrix ``K @ K.T`` is built and checked
        once, then cached for subsequent calls.

        Parameters
        ----------
        kmd:
            Two-dimensional numeric array-like of shape
            ``(n_samples, n_descriptor_dims)``, normally produced by this exact
            instance's :meth:`transform` method. ``n_descriptor_dims`` is
            ``n_components`` for ``"md"`` and ``n_features * n_grids`` for
            ``"1d"``.

        Returns
        -------
        numpy.ndarray
            ``float64`` array of shape ``(n_samples, n_components)``. Values are
            non-negative up to solver precision and each row is normalized to
            sum to one.

        Raises
        ------
        ValueError
            If ``K @ K.T`` is not positive definite, the descriptor shape is
            incompatible, or the ``quadprog`` solver fails to return a solution.

        Notes
        -----
        Inversion uses :func:`qpsolvers.solve_qp` with the ``quadprog`` backend.
        It requires the optional runtime packages ``qpsolvers`` and ``quadprog``
        and is not differentiable. A forward descriptor may be computable even
        when the kernel is not uniquely invertible.

        Examples
        --------
        >>> features = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 2.0]])
        >>> weights = np.array([[0.2, 0.3, 0.5]])
        >>> descriptor = KMD(features, method="1d", n_grids=6)
        >>> recovered = descriptor.inverse(descriptor.transform(weights))
        >>> np.allclose(recovered, weights, atol=1e-4)
        True
        """
        kmd = np.asarray(kmd, dtype=float)
        kernel = self._kernel
        gram = self._gram_matrix()

        n = gram.shape[0]
        # Equality constraint: weights sum to 1; inequality: weights >= 0.
        a_eq = np.ones(n)
        b_eq = np.array([1.0])
        g_ineq = np.diag(-a_eq)
        h_ineq = np.zeros(n)

        solutions = []
        for i in range(kmd.shape[0]):
            sol = solve_qp(gram, -(kernel @ kmd[i]), g_ineq, h_ineq, a_eq, b_eq, solver="quadprog")
            if sol is None:
                raise ValueError(f"quadprog failed to reconstruct weights for sample {i}.")
            solutions.append(sol)

        w = np.round(np.abs(np.array(solutions)), 12)
        return w / w.sum(axis=1)[:, None]

    # -- kernel construction -------------------------------------------------

    def _build_kernel(self, cf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Build the configured kernel from a validated component feature matrix."""
        if self.method == "md":
            return self._md_kernel(cf)
        return self._1d_kernel(cf)

    def _md_kernel(self, cf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Build the square multidimensional RBF kernel between components."""
        if self.scale:
            cf = (cf - cf.mean(axis=0)) / cf.std(axis=0, ddof=1)
        d2 = distance_matrix(cf, cf) ** 2
        if self.sigma == "auto":
            nearest = [np.sort(d2[i])[1] for i in range(d2.shape[0])]  # skip the self-distance 0
            med = median(nearest)
            if med <= 0:
                raise ValueError(
                    "auto sigma is undefined when components share identical features; pass an explicit sigma."
                )
            gamma = 1.0 / med
        else:
            gamma = 1.0 / (2 * self.sigma**2)
        return np.exp(-d2 * gamma)

    def _1d_kernel(self, cf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Build and concatenate one grid-based RBF response block per feature."""
        assert self.n_grids is not None  # guaranteed by __init__
        if self.scale:
            cf = (cf - cf.min(axis=0)) / (cf.max(axis=0) - cf.min(axis=0))

        mins, maxs = cf.min(axis=0), cf.max(axis=0)
        blocks = []
        for i in range(cf.shape[1]):
            grid = np.linspace(mins[i], maxs[i], self.n_grids)
            gamma = 1.0 / (grid[1] - grid[0]) ** 2 if self.sigma == "auto" else 1.0 / (2 * self.sigma**2)
            d2 = (cf[:, i][:, None] - grid[None, :]) ** 2
            blocks.append(np.exp(-d2 * gamma))
        return np.concatenate(blocks, axis=1)

    def _gram_matrix(self) -> npt.NDArray[np.float64]:
        """Return and cache ``K @ K.T`` after checking positive definiteness.

        Raises
        ------
        ValueError
            If the smallest eigenvalue is non-positive, so the descriptor does
            not uniquely determine component weights under the current kernel.
        """
        if self._gram is None:
            gram = self._kernel @ self._kernel.T
            if np.linalg.eigvalsh(gram)[0] <= 0:
                hint = "increasing the number of grids (n_grids)" if self.method == "1d" else "using a smaller sigma"
                raise ValueError(f"KMD is not invertible for these settings; try {hint}.")
            self._gram = gram
        return self._gram


def stats_descriptor(
    weight: npt.ArrayLike,
    component_features: npt.ArrayLike,
    stats: Sequence[str] = ("mean", "var", "max", "min"),
) -> npt.NDArray[np.float64]:
    """Compute weighted summary-statistics descriptors for mixture samples.

    Unlike :class:`KMD`, this function does not construct or cache a kernel.
    Each requested statistic contributes one block of ``n_features`` columns,
    and blocks appear in the same order as ``stats``.

    Parameters
    ----------
    weight:
        Two-dimensional numeric array-like of shape
        ``(n_samples, n_components)``. Column ``j`` corresponds to row ``j`` of
        ``component_features``. Weighted mean and variance conventionally assume
        each row sums to one, but the function does not normalize or validate it.
    component_features:
        Two-dimensional numeric array-like of shape
        ``(n_components, n_features)``. Each row describes one constituent.
    stats:
        Ordered statistic names. Supported values are ``"mean"`` for the
        weighted mean, ``"var"`` for weighted variance around that mean, and
        ``"max"``/``"min"`` for feature-wise extrema over components whose
        weight is exactly non-zero in each sample. Repeated names produce
        repeated output blocks.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape
        ``(n_samples, n_features * len(stats))``.

    Raises
    ------
    ValueError
        If ``stats`` contains an unsupported name, a sample has no non-zero
        component when ``"max"`` or ``"min"`` is requested, or input shapes are
        incompatible.

    Examples
    --------
    >>> features = np.array([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]])
    >>> weights = np.array([[0.25, 0.75, 0.0]])
    >>> result = stats_descriptor(weights, features, stats=("mean", "max"))
    >>> result.shape
    (1, 4)
    >>> np.allclose(result, [[2.5, 2.5, 3.0, 4.0]])
    True
    """
    w = np.asarray(weight, dtype=float)
    cf = np.asarray(component_features, dtype=float)
    n_samples = w.shape[0]

    blocks = []
    for stat in stats:
        if stat == "mean":
            blocks.append(w @ cf)
        elif stat == "var":
            wm = w @ cf
            blocks.append(np.array([w[i] @ (cf - wm[i]) ** 2 for i in range(n_samples)]))
        elif stat in ("max", "min"):
            nonzero = w != 0
            if not nonzero.any(axis=1).all():
                raise ValueError(f"{stat!r} pooling requires every sample to have at least one nonzero weight.")
            reduce = (lambda a: a.max(axis=0)) if stat == "max" else (lambda a: a.min(axis=0))
            blocks.append(np.array([reduce(cf[nonzero[i]]) for i in range(n_samples)]))
        else:
            raise ValueError(f'unsupported stat {stat!r}; choose from "mean", "var", "max", "min".')

    return np.concatenate(blocks, axis=1)
