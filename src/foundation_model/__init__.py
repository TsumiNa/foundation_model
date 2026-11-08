# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated alias: the package is ``yammi`` from 2.0.

``import foundation_model`` and ``foundation_model.<module>`` resolve to the same module objects as
``yammi`` and ``yammi.<module>``, so code written for 1.x and checkpoints pickled under the old module
paths keep loading. New code should import ``yammi``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from types import ModuleType

_OLD, _NEW = __name__, "yammi"


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``foundation_model.x`` to the already-importable ``yammi.x`` (one module object, not a copy)."""

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_OLD + "."):
            return None
        new = _NEW + fullname[len(_OLD) :]
        if importlib.util.find_spec(new) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return importlib.import_module(_NEW + spec.name[len(_OLD) :])

    def exec_module(self, module: ModuleType) -> None:
        pass  # already executed under its yammi name


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn("foundation_model was renamed to yammi in 2.0; import yammi instead.", DeprecationWarning, stacklevel=2)

_yammi = importlib.import_module(_NEW)
sys.modules[_OLD] = _yammi
