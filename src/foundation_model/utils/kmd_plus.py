# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0
#
# Kernel mean descriptor (KMD) originally authored by Minoru Kusaba
# (ISM, kusaba@ism.ac.jp). Refactored into a stateful class with a
# precomputed kernel basis.

"""Kernel mean descriptor (KMD) and summary-statistics descriptors for mixtures.

A :class:`KMD` instance is built once around a fixed ``component_features``
matrix and the chosen kernel hyper-parameters. It precomputes the kernel basis
shared by both directions, so:

* :meth:`KMD.transform` (also available via ``__call__``) maps mixing weights to
  descriptors (materials → descriptors), and
* :meth:`KMD.inverse` recovers the weights from descriptors by quadratic
  programming (descriptors → materials).
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
    """Convert a formula to a composition vector over ``elements``.

    Parameters
    ----------
    formula:
        Chemical formula (e.g. ``"SiO2"``), a dict of element fractions, or a
        :class:`~pymatgen.core.composition.Composition`.
    elements:
        Element symbols defining the vector layout. Defaults to the elements of
        the bundled ``element_features.csv``.

    Returns
    -------
    numpy.ndarray of shape ``(len(elements),)``
        Atomic fractions aligned to ``elements``.
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
    """Kernel mean descriptor for mixture systems.

    The kernel basis is derived once from ``component_features`` at construction
    time and reused for both :meth:`transform` and :meth:`inverse`.

    Parameters
    ----------
    component_features:
        Features for each constituent, shape ``(n_components, n_features)``.
    method:
        ``"md"`` builds the descriptor on the multidimensional feature space;
        ``"1d"`` builds a per-feature descriptor on a discretized grid and
        concatenates the results.
    n_grids:
        Number of equally spaced grid points per feature. Required for
        ``method="1d"`` and ignored for ``"md"``.
    sigma:
        Kernel width. With ``"auto"`` it is the inverse median nearest-neighbour
        distance for ``"md"`` and the inverse grid width for ``"1d"``. A float
        sets the width manually as ``exp(-d^2 / (2 * sigma^2))``.
    scale:
        Whether to rescale ``component_features`` before building the kernel
        (standardization for ``"md"``, min-max normalization for ``"1d"``).
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

        self.method: Method = method
        self.n_grids = n_grids
        self.sigma = sigma
        self.scale = scale

        cf = np.asarray(component_features, dtype=float)
        self.n_components: int = cf.shape[0]
        self._kernel: npt.NDArray[np.float64] = self._build_kernel(cf)
        self._gram: npt.NDArray[np.float64] | None = None

    def transform(self, weight: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map mixing weights to kernel mean descriptors (materials → descriptors).

        Parameters
        ----------
        weight:
            Mixing ratios, shape ``(n_samples, n_components)``.

        Returns
        -------
        numpy.ndarray
            Descriptors of shape ``(n_samples, n_components)`` for ``"md"`` and
            ``(n_samples, n_features * n_grids)`` for ``"1d"``.
        """
        return np.asarray(weight, dtype=float) @ self._kernel

    def __call__(self, weight: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Alias for :meth:`transform`."""
        return self.transform(weight)

    def inverse(self, kmd: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Recover mixing weights from descriptors (descriptors → materials).

        Solves, per sample, a non-negative simplex-constrained quadratic program
        whose minimiser is the weight vector reproducing the given descriptor.

        Parameters
        ----------
        kmd:
            Descriptors as returned by :meth:`transform`, shape
            ``(n_samples, n_descriptor_dims)``.

        Returns
        -------
        numpy.ndarray of shape ``(n_samples, n_components)``
            Reconstructed mixing weights, each row summing to 1.
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

        w_raw = np.array(
            [
                solve_qp(gram, -(kernel @ kmd[i]), g_ineq, h_ineq, a_eq, b_eq, solver="quadprog")
                for i in range(kmd.shape[0])
            ]
        )
        w = np.round(np.abs(w_raw), 12)
        return w / w.sum(axis=1)[:, None]

    # -- kernel construction -------------------------------------------------

    def _build_kernel(self, cf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.method == "md":
            return self._md_kernel(cf)
        return self._1d_kernel(cf)

    def _md_kernel(self, cf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.scale:
            cf = (cf - cf.mean(axis=0)) / cf.std(axis=0, ddof=1)
        d2 = distance_matrix(cf, cf) ** 2
        if self.sigma == "auto":
            nearest = [np.sort(d2[i])[1] for i in range(d2.shape[0])]  # skip the self-distance 0
            gamma = 1.0 / median(nearest)
        else:
            gamma = 1.0 / (2 * self.sigma**2)
        return np.exp(-d2 * gamma)

    def _1d_kernel(self, cf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
        """Return (and cache) the Gram matrix ``K Kᵀ``, validating invertibility."""
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
    """Generate summary-statistics descriptors for mixture systems.

    Parameters
    ----------
    weight:
        Mixing ratios, shape ``(n_samples, n_components)``.
    component_features:
        Features for each constituent, shape ``(n_components, n_features)``.
    stats:
        Summary statistics to compute. Supported: ``"mean"``, ``"var"``,
        ``"max"``, ``"min"``.

    Returns
    -------
    numpy.ndarray of shape ``(n_samples, n_features * len(stats))``
        Concatenated descriptors in the order given by ``stats``.
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
        elif stat == "max":
            nonzero = w != 0
            blocks.append(np.array([cf[nonzero[i]].max(axis=0) for i in range(n_samples)]))
        elif stat == "min":
            nonzero = w != 0
            blocks.append(np.array([cf[nonzero[i]].min(axis=0) for i in range(n_samples)]))
        else:
            raise ValueError(f'unsupported stat {stat!r}; choose from "mean", "var", "max", "min".')

    return np.concatenate(blocks, axis=1)
