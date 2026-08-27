# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Inverse design: search for inputs whose predicted properties hit a set of targets.

This is the *use* of a trained model, not part of training it. Two search spaces, and the
difference between them is the point:

* :meth:`~.mixin.InverseDesignMixin.optimize_latent` descends in the encoder's latent space and
  decodes back through the autoencoder, so what it returns is a *descriptor* that still has to be
  inverted to a recipe.
* :meth:`~.mixin.InverseDesignMixin.optimize_composition` descends directly over element weights
  on the simplex, so the optimised variable IS the recipe and no decode round-trip is involved.
  This is the KMD differentiable path.

WHAT LIVES WHERE

``optimize_composition`` was one 975-line method, 709 of which was code, and nine nested closures
— eight of which never referenced ``self``. Those eight, plus the ~340 lines of parameter
validation ahead of them, were a constraint system with no way to reach it except through a
twenty-keyword method, which is why none of it had a test. They are modules now:

* :mod:`.targets`     — what to optimise *toward* (the objective terms and result containers)
* :mod:`.constraints` — what the recipe is allowed to *be* (element whitelist, per-element step
  scale, pinned amounts, weight floor, cardinality) resolved once into one frozen object
* :mod:`.annealing`   — how the cardinality constraint *hardens over time* (the τ schedule)
* :mod:`.simplex`     — how a logit vector *becomes* a legal recipe (top-K, lock paste, floor)
* :mod:`.composition` / :mod:`.latent` — the two search loops
* :mod:`.mixin`       — the model-facing surface: four declared members and two public methods

The public names below are unchanged, so ``model.optimize_composition(...)`` and every import
site keep working exactly as before.
"""

from .mixin import (
    CompositionOptimizationResult,
    InverseDesignMixin,
    OptimizationResult,
    OptimizationTarget,
)

__all__ = [
    "CompositionOptimizationResult",
    "InverseDesignMixin",
    "OptimizationResult",
    "OptimizationTarget",
]
