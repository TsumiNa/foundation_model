# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""Target transforms as plain data: fit, apply, invert and serialise without pickle.

A training target is stored on a normalised scale; a :class:`TargetTransform` maps physical
values onto that scale (:meth:`~TargetTransform.transform`) and back
(:meth:`~TargetTransform.inverse`). It is an ordered list of steps whose parameters are numbers,
so it round-trips through JSON and can be shipped inside a released model bundle without
executing code at load time.

The step set covers every transform the project's datasets use:

* ``standard_scaler`` — ``(x - mean) / scale`` (sklearn ``StandardScaler``, incl. ``with_mean=False``);
* ``power_transformer`` — Yeo-Johnson or Box-Cox, optionally standardised (sklearn ``PowerTransformer``);
* ``log1p`` — ``log1p(max(x, clip_lower))``;
* ``arcsinh`` — ``arcsinh(x)``;
* ``clip`` — bounds on the transformed value. The inverse clamps to the same bounds, so values at or
  beyond a bound come back as the bound's pre-image (:attr:`TargetTransform.saturates`);
* ``identity``.

A Yeo-Johnson step with ``lambda < 0`` (or ``> 2``), and any Box-Cox step, maps the real line onto a
half-bounded interval, so its inverse is undefined past that bound and returns NaN. A model can predict
past it. :func:`guard_inverse_domain` appends a ``clip`` at the training range on each such side, so the
inverse saturates at the most extreme training value instead.

:func:`from_sklearn` converts the fitted sklearn objects found in ``data/*preprocessing*.pkl.z``
and rejects anything else by name. :func:`fit_default` fits the project's default recipe
(``StandardScaler → PowerTransformer(yeo-johnson)``) on training values. Classification labels
are :class:`ClassLabels`; per-row max-abs normalised curves (the DOS head) use
:func:`normalize_max_abs` and carry no transform at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Sequence

import numpy as np

__all__ = [
    "ArcsinhStep",
    "ClassLabels",
    "ClipStep",
    "IdentityStep",
    "Log1pStep",
    "PowerTransformerStep",
    "StandardScalerStep",
    "Step",
    "TargetTransform",
    "UnsupportedTransformError",
    "fit_default",
    "from_sklearn",
    "guard_inverse_domain",
    "strip_inverse_guard",
    "normalize_max_abs",
]

_EPS = np.spacing(1.0)  # sklearn's own threshold for "lambda == 0" / "lambda == 2"


class UnsupportedTransformError(TypeError):
    """A fitted transform (or one of its steps) has no plain-data equivalent."""


def _as_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim > 1:
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0]
        raise ValueError(f"a target transform acts on one column; got shape {arr.shape}.")
    return arr


# --- steps ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardScalerStep:
    """``(x - mean) / scale``; ``mean = 0`` / ``scale = 1`` encode ``with_mean`` / ``with_std`` off."""

    kind: ClassVar[str] = "standard_scaler"
    mean: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale == 0:
            raise ValueError(f"standard_scaler.scale must be finite and non-zero, got {self.scale}.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def backward(self, y: np.ndarray) -> np.ndarray:
        return y * self.scale + self.mean

    def params(self) -> dict[str, Any]:
        return {"mean": self.mean, "scale": self.scale}


@dataclass(frozen=True)
class PowerTransformerStep:
    """Yeo-Johnson / Box-Cox with one ``lmbda``, then ``(z - mean) / scale`` when ``standardize``."""

    kind: ClassVar[str] = "power_transformer"
    method: Literal["yeo-johnson", "box-cox"]
    lmbda: float
    standardize: bool = True
    mean: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.method not in ("yeo-johnson", "box-cox"):
            raise ValueError(f"power_transformer.method must be 'yeo-johnson' or 'box-cox', got {self.method!r}.")
        if self.standardize and (not np.isfinite(self.scale) or self.scale == 0):
            raise ValueError(f"power_transformer.scale must be finite and non-zero, got {self.scale}.")

    def _yj(self, x: np.ndarray) -> np.ndarray:
        lam = self.lmbda
        out = np.full_like(x, np.nan)
        pos = x >= 0
        neg = x < 0
        if abs(lam) < _EPS:
            out[pos] = np.log1p(x[pos])
        else:
            out[pos] = np.expm1(lam * np.log1p(x[pos])) / lam  # stable for lambda near 0
        if abs(lam - 2) > _EPS:
            out[neg] = -np.expm1((2 - lam) * np.log1p(-x[neg])) / (2 - lam)
        else:
            out[neg] = -np.log1p(-x[neg])
        return out

    def _yj_inv(self, y: np.ndarray) -> np.ndarray:
        lam = self.lmbda
        out = np.full_like(y, np.nan)
        pos = y >= 0
        neg = y < 0
        if abs(lam) < _EPS:
            out[pos] = np.expm1(y[pos])
        else:
            out[pos] = np.expm1(np.log1p(y[pos] * lam) / lam)
        if abs(lam - 2) > _EPS:
            out[neg] = -np.expm1(np.log1p(-(2 - lam) * y[neg]) / (2 - lam))
        else:
            out[neg] = -np.expm1(-y[neg])
        return out

    def image(self) -> tuple[float, float]:
        """The open interval this step maps onto (after standardising); its inverse is NaN outside it."""
        lam, inf = self.lmbda, float("inf")
        if self.method == "yeo-johnson":
            lo = 1.0 / (2.0 - lam) if lam > 2 + _EPS else -inf
            hi = -1.0 / lam if lam < -_EPS else inf
        elif abs(lam) < _EPS:
            lo, hi = -inf, inf
        else:
            lo, hi = (-1.0 / lam, inf) if lam > 0 else (-inf, -1.0 / lam)
        if not self.standardize:
            return lo, hi
        a, b = (lo - self.mean) / self.scale, (hi - self.mean) / self.scale
        return (a, b) if a <= b else (b, a)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.method == "yeo-johnson":
            z = self._yj(x)
        else:
            if np.any(x[np.isfinite(x)] <= 0):
                raise ValueError("box-cox requires strictly positive values.")
            # expm1 / log1p keep full precision for lambda near (but not at) 0, as scipy's boxcox does.
            z = np.log(x) if abs(self.lmbda) < _EPS else np.expm1(self.lmbda * np.log(x)) / self.lmbda
        return (z - self.mean) / self.scale if self.standardize else z

    def backward(self, y: np.ndarray) -> np.ndarray:
        z = y * self.scale + self.mean if self.standardize else y
        if self.method == "yeo-johnson":
            return self._yj_inv(z)
        return np.exp(z) if abs(self.lmbda) < _EPS else np.exp(np.log1p(z * self.lmbda) / self.lmbda)

    def params(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "lmbda": self.lmbda,
            "standardize": self.standardize,
            "mean": self.mean,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class Log1pStep:
    """``log1p(max(x, clip_lower))``; ``clip_lower = None`` applies no clip."""

    kind: ClassVar[str] = "log1p"
    clip_lower: float | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(x if self.clip_lower is None else np.maximum(x, self.clip_lower))

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.expm1(y)

    def params(self) -> dict[str, Any]:
        return {"clip_lower": self.clip_lower}


@dataclass(frozen=True)
class ArcsinhStep:
    kind: ClassVar[str] = "arcsinh"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.arcsinh(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.sinh(y)

    def params(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class ClipStep:
    """Bound the transformed value. The inverse clamps to the same bounds before passing values on.

    A model can output values beyond the bounds it was trained on; clamping them makes the inverse
    saturate at the bound's pre-image instead of extrapolating past it.
    """

    kind: ClassVar[str] = "clip"
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError("clip needs at least one of lower / upper.")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError(f"clip.lower must be < clip.upper, got {self.lower} >= {self.upper}.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.clip(y, self.lower, self.upper)

    def params(self) -> dict[str, Any]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class IdentityStep:
    kind: ClassVar[str] = "identity"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, y: np.ndarray) -> np.ndarray:
        return y

    def params(self) -> dict[str, Any]:
        return {}


Step = StandardScalerStep | PowerTransformerStep | Log1pStep | ArcsinhStep | ClipStep | IdentityStep
_STEP_TYPES: dict[str, type] = {
    cls.kind: cls for cls in (StandardScalerStep, PowerTransformerStep, Log1pStep, ArcsinhStep, ClipStep, IdentityStep)
}


# --- transform -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetTransform:
    """An ordered list of steps between physical values and the training scale.

    NaN passes through every step unchanged, matching the masked-target convention.
    """

    steps: tuple[Step, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        for step in self.steps:
            if type(step) not in _STEP_TYPES.values():
                raise UnsupportedTransformError(f"not a target-transform step: {type(step).__name__}.")

    def transform(self, values: Any) -> np.ndarray:
        """Physical values → training scale."""
        out = _as_1d(values).copy()
        for step in self.steps:
            out = step.forward(out)
        return out

    def inverse(self, values: Any) -> np.ndarray:
        """Training scale → physical values (values at or beyond a ``clip`` bound come back saturated)."""
        out = _as_1d(values).copy()
        for step in reversed(self.steps):
            out = step.backward(out)
        return out

    def inverse_domain(self) -> tuple[float, float]:
        """The interval of training-scale values the inverse is defined on (``±inf`` = unbounded).

        Built forward: start from the whole physical line and map the interval through each step, keeping
        the bounds every earlier step imposes.
        """
        lo, hi = -np.inf, np.inf
        for step in self.steps:
            lo, hi = _map_interval(step, lo, hi)
        return float(lo), float(hi)

    @property
    def saturates(self) -> bool:
        """True when a ``clip`` step makes the inverse lose values beyond its bounds."""
        return any(isinstance(step, ClipStep) for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [{"kind": step.kind, **step.params()} for step in self.steps]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetTransform:
        steps_raw = data.get("steps")
        if not isinstance(steps_raw, list):
            raise ValueError("a target transform needs a 'steps' list.")
        steps: list[Step] = []
        for i, raw in enumerate(steps_raw):
            raw = dict(raw)
            kind = raw.pop("kind", None)
            if kind not in _STEP_TYPES:
                raise UnsupportedTransformError(f"step {i}: unknown kind {kind!r}; supported: {sorted(_STEP_TYPES)}.")
            try:
                steps.append(_STEP_TYPES[kind](**raw))
            except TypeError as exc:
                raise ValueError(f"step {i} ({kind}): {exc}") from None
        return cls(tuple(steps))


@dataclass(frozen=True)
class ClassLabels:
    """Class names of a classification head, in label-index order (sklearn ``LabelEncoder.classes_``)."""

    classes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "classes", tuple(str(c) for c in self.classes))
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(f"class names must be unique, got {list(self.classes)}.")

    def names(self, indices: Any) -> list[str]:
        """Class names for integer label indices in ``[0, len(classes))``; anything else raises."""
        arr = np.asarray(indices).ravel()
        if arr.size and (not np.issubdtype(arr.dtype, np.number) or np.any(arr % 1 != 0)):
            raise ValueError(f"class indices must be integers, got {arr[:5].tolist()}.")
        ints = arr.astype(int)
        bad = ints[(ints < 0) | (ints >= len(self.classes))]
        if bad.size:
            raise ValueError(f"class indices out of range 0..{len(self.classes) - 1}: {bad[:5].tolist()}.")
        return [self.classes[i] for i in ints]

    def indices(self, names: Sequence[str]) -> list[int]:
        lookup = {c: i for i, c in enumerate(self.classes)}
        missing = [n for n in names if n not in lookup]
        if missing:
            raise KeyError(f"unknown class name(s) {missing}; classes are {list(self.classes)}.")
        return [lookup[n] for n in names]

    def to_dict(self) -> dict[str, Any]:
        return {"classes": list(self.classes)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassLabels:
        return cls(tuple(data["classes"]))


# --- conversion and fitting -----------------------------------------------------------------


def _scalar(arr: Any, name: str, owner: str) -> float:
    values = np.asarray(arr, dtype=float).ravel()
    if values.size != 1:
        raise UnsupportedTransformError(
            f"{owner}.{name} has {values.size} entries; a target transform is single-column."
        )
    return float(values[0])


def _steps_from_estimator(est: Any) -> list[Step]:
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import (  # type: ignore[import-untyped]
        FunctionTransformer,
        PowerTransformer,
        StandardScaler,
    )

    name = type(est).__name__
    if isinstance(est, Pipeline):
        return [step for _, sub in est.steps for step in _steps_from_estimator(sub)]
    n_features = getattr(est, "n_features_in_", 1)
    if n_features != 1:
        raise UnsupportedTransformError(
            f"{name} was fitted on {n_features} columns; a target transform is single-column."
        )
    if isinstance(est, StandardScaler):
        mean = _scalar(est.mean_, "mean_", name) if est.with_mean else 0.0
        scale = _scalar(est.scale_, "scale_", name) if est.with_std else 1.0
        return [StandardScalerStep(mean=mean, scale=scale)]
    if isinstance(est, PowerTransformer):
        inner = getattr(est, "_scaler", None)
        mean, scale = 0.0, 1.0
        if est.standardize:
            if inner is None:
                raise UnsupportedTransformError("PowerTransformer(standardize=True) without a fitted inner scaler.")
            mean, scale = _scalar(inner.mean_, "mean_", name), _scalar(inner.scale_, "scale_", name)
        return [
            PowerTransformerStep(
                method=est.method,
                lmbda=_scalar(est.lambdas_, "lambdas_", name),
                standardize=bool(est.standardize),
                mean=mean,
                scale=scale,
            )
        ]
    if isinstance(est, FunctionTransformer):
        if est.func is np.arcsinh and est.inverse_func is np.sinh:
            return [ArcsinhStep()]
        if est.func is np.log1p and est.inverse_func is np.expm1:
            return [Log1pStep()]
        if est.func is None and est.inverse_func is None:
            return [IdentityStep()]
        raise UnsupportedTransformError(
            f"FunctionTransformer({getattr(est.func, '__name__', est.func)!s}) has no plain-data equivalent."
        )
    raise UnsupportedTransformError(f"{name} has no plain-data equivalent; supported: {sorted(_STEP_TYPES)}.")


def from_sklearn(obj: Any) -> TargetTransform | ClassLabels:
    """Convert a fitted sklearn scaler / pipeline (→ :class:`TargetTransform`) or ``LabelEncoder``.

    Raises :class:`UnsupportedTransformError` naming the first estimator that cannot be expressed.
    """
    from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]

    if isinstance(obj, LabelEncoder):
        return ClassLabels(tuple(obj.classes_.tolist()))
    return TargetTransform(tuple(_steps_from_estimator(obj)))


def fit_default(train_values: Any) -> TargetTransform:
    """Fit ``StandardScaler → PowerTransformer(yeo-johnson)`` on finite training values."""
    from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import PowerTransformer, StandardScaler  # type: ignore[import-untyped]

    x = _as_1d(train_values)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError(f"fitting a target transform needs at least 2 finite training values, got {x.size}.")
    pipe = make_pipeline(StandardScaler(), PowerTransformer(method="yeo-johnson", standardize=True))
    pipe.fit(x.reshape(-1, 1))
    result = from_sklearn(pipe)
    assert isinstance(result, TargetTransform)
    return result


def _map_interval(step: Step, lo: float, hi: float) -> tuple[float, float]:
    """Where the inverse up to and including ``step`` is defined, given that of the steps before it.

    Every step is monotone, so an interval maps to the interval between its mapped ends (re-sorted for a
    negative scale). Infinite ends map to the step's limits.
    """
    ends = np.array([lo, hi], dtype=float)
    with np.errstate(all="ignore"):
        if isinstance(step, StandardScalerStep):
            mapped = step.forward(ends)
        elif isinstance(step, Log1pStep):
            # The inverse is expm1, defined everywhere and > -1; the forward clip does not bound it.
            mapped = np.log1p(np.maximum(ends, -1.0))
        elif isinstance(step, ArcsinhStep):
            mapped = np.arcsinh(ends)
        elif isinstance(step, PowerTransformerStep):
            if step.method == "yeo-johnson":
                mapped = step.forward(ends)  # a bijection of the real line; ±inf map to the image limits
            else:
                x = np.maximum(ends, 0.0)  # Box-Cox's inverse returns positive values only
                lam = step.lmbda
                z = np.log(x) if abs(lam) < _EPS else (np.power(x, lam) - 1.0) / lam
                mapped = (z - step.mean) / step.scale if step.standardize else z
        elif isinstance(step, ClipStep):
            # The inverse clamps first, so a side bounded inside the incoming interval becomes total.
            new_lo = -np.inf if step.lower is not None and step.lower > lo else lo
            new_hi = np.inf if step.upper is not None and step.upper < hi else hi
            return new_lo, new_hi
        else:  # identity
            return lo, hi
    a, b = (float(v) for v in mapped)
    return (a, b) if a <= b else (b, a)


def guard_inverse_domain(transform: TargetTransform, train_scaled: Any) -> TargetTransform:
    """``transform`` plus a ``clip`` at the training range on each side where its inverse is undefined.

    ``train_scaled`` are training targets on the training scale (``transform.transform(physical)``).
    A side whose inverse is defined everywhere stays open, so predictions there still extrapolate.
    A transform whose inverse is defined everywhere is returned unchanged.
    """
    lo_d, hi_d = transform.inverse_domain()
    if not (np.isfinite(lo_d) or np.isfinite(hi_d)):
        return transform
    x = _as_1d(train_scaled)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("guarding a transform's inverse needs finite training values.")
    lower = _inside(float(x.min()), transform, +1) if np.isfinite(lo_d) else None
    upper = _inside(float(x.max()), transform, -1) if np.isfinite(hi_d) else None
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(f"degenerate training range [{lower}, {upper}] for a guarded transform.")
    return TargetTransform((*transform.steps, ClipStep(lower=lower, upper=upper)))


def strip_inverse_guard(transform: TargetTransform) -> TargetTransform:
    """``transform`` without the clip :func:`guard_inverse_domain` added, if it ends with one.

    The guard only ever bounds a side whose inverse is undefined past it; a trailing clip that bounds an
    open side (the phonix-db ``kp`` / ``klat`` clip at +5, part of how those targets were normalised) is
    kept. Use the stripped transform to normalise labels, so held-out values past the training range are
    not clipped; keep the guarded one for inverting predictions.
    """
    if not transform.steps or not isinstance(transform.steps[-1], ClipStep):
        return transform
    clip, base = transform.steps[-1], TargetTransform(transform.steps[:-1])
    lo, hi = base.inverse_domain()
    if (clip.lower is None or np.isfinite(lo)) and (clip.upper is None or np.isfinite(hi)):
        return base
    return transform


def _inside(bound: float, transform: TargetTransform, direction: int) -> float:
    """``bound`` moved ``direction`` (towards the interior) one ulp at a time until its inverse is finite.

    A training value that sits on the domain edge in floating point inverts to ``inf``; the clip has to
    sit just inside it.
    """
    value = bound
    for _ in range(64):
        with np.errstate(all="ignore"):
            if np.isfinite(transform.inverse(np.array([value]))).all():
                return value
        value = float(np.nextafter(value, direction * np.inf))
    raise ValueError(f"no finite inverse within 64 ulps of the training bound {bound}.")


def normalize_max_abs(curve: Any) -> np.ndarray:
    """Divide one curve by its own max abs value — the per-row normalisation of shape-only heads.

    Matches ``InterpSmooth(normalize=True)``: an all-zero curve is returned unchanged.
    """
    y = _as_1d(curve)
    finite = y[np.isfinite(y)]
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    return y / peak if peak > 0 else y.copy()
