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

``optimize_composition`` was one 975-line method — 709 of it code — with nine nested closures,
eight of which never referenced ``self``, and ~340 lines of parameter validation ahead of them.
That was a constraint system with no way to reach it except through a twenty-keyword method, which
is why none of it had a test. Each module below answers one question:

* :mod:`.targets`     — what is the search optimising *toward*?  (objective terms, results)
* :mod:`.constraints` — what is the recipe allowed to *be*?  (whitelist, pins, floor, cardinality)
* :mod:`.annealing`   — how fast does the cardinality limit *commit*?  (the τ schedule)
* :mod:`.simplex`     — how does a logit vector *become* a legal recipe?  (top-K, paste, floor)
* :mod:`.latent`      — the search over descriptors, via the autoencoder
* :mod:`.composition` — the search over element weights, straight through the KMD transform
* :mod:`.mixin`       — what does a search need *from the model*?  (four members, and the two
  steps that turn a target into a loss term against real heads)

Dependencies run one way: ``targets`` and ``annealing`` and ``simplex`` are leaves, ``constraints``
composes the latter two, the two searches use all of them, and ``mixin`` binds the searches.

The public names below are unchanged, so ``model.optimize_composition(...)`` and every import
site keep working exactly as before.
"""

from .mixin import InverseDesignMixin
from .targets import CompositionOptimizationResult, OptimizationResult, OptimizationTarget

__all__ = [
    "CompositionOptimizationResult",
    "InverseDesignMixin",
    "OptimizationResult",
    "OptimizationTarget",
]
