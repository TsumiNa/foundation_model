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
from types import CodeType, ModuleType

_OLD, _NEW = __name__, "yammi"


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``foundation_model.x`` to the already-importable ``yammi.x`` (one module object, not a copy)."""

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_OLD + "."):
            return None
        new = _NEW + fullname[len(_OLD) :]
        target_spec = importlib.util.find_spec(new)
        if target_spec is None:
            return None
        # origin: runpy puts it in sys.argv[0] for ``python -m``, where click reads the program name.
        return importlib.util.spec_from_loader(
            fullname,
            self,
            origin=target_spec.origin,
            is_package=target_spec.submodule_search_locations is not None,
        )

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return importlib.import_module(_NEW + spec.name[len(_OLD) :])

    def exec_module(self, module: ModuleType) -> None:
        pass  # already executed under its yammi name

    def get_code(self, fullname: str) -> CodeType | None:
        """Code of the yammi module, for ``python -m foundation_model.<module>`` (runpy asks for it)."""
        new = _NEW + fullname[len(_OLD) :]
        spec = importlib.util.find_spec(new)
        loader = spec.loader if spec is not None else None
        return loader.get_code(new) if isinstance(loader, importlib.abc.InspectLoader) else None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn("foundation_model was renamed to yammi in 2.0; import yammi instead.", DeprecationWarning, stacklevel=2)

_yammi = importlib.import_module(_NEW)
sys.modules[_OLD] = _yammi
