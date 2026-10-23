"""Inference engines with probabilities mapped back to the full label set (plan section 0)."""

from __future__ import annotations

import os

import numpy as np

ENGINES = ("tabpfn", "tabicl", "catboost")


class Engine:
    def __init__(self, name: str, n_classes: int, device: str | None = None):
        if name not in ENGINES:
            raise ValueError(name)
        self.name = name
        self.n_classes = n_classes
        self.device = device

    def _make(self, seed: int):
        if self.name == "tabpfn":
            from tabpfn import TabPFNClassifier

            kw = {"device": self.device} if self.device else {}
            return TabPFNClassifier(n_estimators=8, random_state=seed, **kw)
        if self.name == "tabicl":
            from tabicl import TabICLClassifier

            kw = {"device": self.device} if self.device else {}
            return TabICLClassifier(n_estimators=8, random_state=seed, **kw)
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            random_seed=seed, verbose=0, thread_count=int(os.environ.get("CATBOOST_THREADS", "4"))
        )

    def fit_predict(self, xc: np.ndarray, yc: np.ndarray, xq: np.ndarray, seed: int) -> np.ndarray:
        """Fit on the context, predict the queries; returns (n_query, n_classes) probabilities.

        Classes absent from the context get probability 0. A single-class context is answered
        with probability 1 for that class without calling the engine: predict_proba of any of
        the engines can only return context classes, so this is the output they would give.
        """
        out = np.zeros((len(xq), self.n_classes))
        classes = np.unique(yc)
        if len(classes) == 1:
            out[:, classes[0]] = 1.0
            return out
        clf = self._make(seed)
        clf.fit(xc, yc)
        p = np.asarray(clf.predict_proba(xq), dtype=float)
        cls = np.asarray(clf.classes_).astype(int).ravel()
        out[:, cls] = p
        return out
