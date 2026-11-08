# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pytest
from sklearn.pipeline import Pipeline, make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import (  # type: ignore[import-untyped]
    FunctionTransformer,
    LabelEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)

from yammi.data.targets import (
    ArcsinhStep,
    ClassLabels,
    ClipStep,
    Log1pStep,
    PowerTransformerStep,
    StandardScalerStep,
    TargetTransform,
    UnsupportedTransformError,
    fit_default,
    from_sklearn,
    guard_inverse_domain,
    normalize_max_abs,
    strip_inverse_guard,
)

REPO = Path(__file__).resolve().parents[3]
TOL = 1e-12


def _rng_values(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mixed sign, skewed, with a heavy tail: exercises both Yeo-Johnson branches.
    return np.concatenate([rng.lognormal(1.0, 1.2, n), -rng.lognormal(0.0, 0.8, n // 4), [0.0]])


def _assert_matches(pipe: object, x: np.ndarray) -> None:
    transform = from_sklearn(pipe)
    assert isinstance(transform, TargetTransform)
    y_ref = pipe.transform(x.reshape(-1, 1)).ravel()  # type: ignore[attr-defined]
    np.testing.assert_allclose(transform.transform(x), y_ref, rtol=0, atol=TOL)
    x_ref = pipe.inverse_transform(y_ref.reshape(-1, 1)).ravel()  # type: ignore[attr-defined]
    np.testing.assert_allclose(transform.inverse(y_ref), x_ref, rtol=1e-12, atol=TOL)
    # JSON round trip keeps every number.
    again = TargetTransform.from_dict(json.loads(json.dumps(transform.to_dict())))
    np.testing.assert_array_equal(again.transform(x), transform.transform(x))


@pytest.mark.parametrize("standardize", [True, False])
def test_standard_then_yeo_johnson_matches_sklearn(standardize: bool) -> None:
    x = _rng_values()
    pipe = make_pipeline(StandardScaler(), PowerTransformer(method="yeo-johnson", standardize=standardize)).fit(
        x.reshape(-1, 1)
    )
    _assert_matches(pipe, x)


@pytest.mark.parametrize("lmbda", [0.0, 2.0, 0.37, -0.8, 2.9])
def test_yeo_johnson_special_lambdas_match_sklearn(lmbda: float) -> None:
    x = np.linspace(-3, 6, 101)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(x.reshape(-1, 1))
    pt.lambdas_ = np.array([lmbda])  # force the lambda == 0 / == 2 branches
    _assert_matches(pt, x)


def test_box_cox_matches_sklearn() -> None:
    x = np.abs(_rng_values()) + 0.1
    _assert_matches(PowerTransformer(method="box-cox").fit(x.reshape(-1, 1)), x)


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("lmbda", [1e-10, -1e-9, 2 + 1e-10])
def test_power_transformer_is_precise_near_singular_lambdas(method: str, lmbda: float) -> None:
    from scipy import stats  # type: ignore[import-untyped]
    from scipy.special import boxcox, inv_boxcox  # type: ignore[import-untyped]

    x = np.array([0.5, 2.0, 7.0]) if method == "box-cox" else np.array([-3.0, -0.5, 0.5, 2.0, 7.0])
    step = PowerTransformerStep(method=method, lmbda=lmbda, standardize=False)  # type: ignore[arg-type]
    ref = boxcox(x, lmbda) if method == "box-cox" else stats.yeojohnson(x, lmbda)
    np.testing.assert_allclose(step.forward(x), ref, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(step.backward(step.forward(x)), x, rtol=1e-12)
    if method == "box-cox":
        np.testing.assert_allclose(step.backward(ref), inv_boxcox(ref, lmbda), rtol=1e-12)


def test_agis_prescale_arcsinh_pipeline_matches_sklearn() -> None:
    # data/scripts/process_agis_data.py: StandardScaler(with_mean=False) -> arcsinh -> StandardScaler.
    x = _rng_values() * 50
    pipe = Pipeline(
        [
            ("prescale", StandardScaler(with_mean=False)),
            ("asinh", FunctionTransformer(np.arcsinh, inverse_func=np.sinh, validate=True)),
            ("standardscaler", StandardScaler()),
        ]
    ).fit(x.reshape(-1, 1))
    transform = from_sklearn(pipe)
    assert isinstance(transform, TargetTransform)
    assert [type(s) for s in transform.steps] == [StandardScalerStep, ArcsinhStep, StandardScalerStep]
    prescale = transform.steps[0]
    assert isinstance(prescale, StandardScalerStep) and prescale.mean == 0.0
    _assert_matches(pipe, x)


def test_label_encoder_becomes_class_labels() -> None:
    labels = from_sklearn(LabelEncoder().fit(["IQC", "DAC", "others", "IAC", "DQC"]))
    assert isinstance(labels, ClassLabels)
    assert labels.classes == ("DAC", "DQC", "IAC", "IQC", "others")
    assert labels.names([0, 4]) == ["DAC", "others"]
    assert labels.indices(["IQC"]) == [3]
    for bad in ([-1], [5], [0.9]):
        with pytest.raises(ValueError, match="class indices"):
            labels.names(bad)
    with pytest.raises(KeyError, match="unknown class"):
        labels.indices(["QC"])
    assert ClassLabels.from_dict(json.loads(json.dumps(labels.to_dict()))) == labels


def test_unsupported_estimator_is_rejected_by_name() -> None:
    x = _rng_values()
    with pytest.raises(UnsupportedTransformError, match="QuantileTransformer"):
        from_sklearn(make_pipeline(StandardScaler(), QuantileTransformer(n_quantiles=50)).fit(x.reshape(-1, 1)))
    with pytest.raises(UnsupportedTransformError, match="square"):
        from_sklearn(FunctionTransformer(np.square))


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_multi_column_scaler_is_rejected(with_mean: bool, with_std: bool) -> None:
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std).fit(np.arange(8.0).reshape(4, 2))
    with pytest.raises(UnsupportedTransformError, match="single-column"):
        from_sklearn(scaler)


def test_unknown_step_kind_in_json_is_rejected() -> None:
    with pytest.raises(UnsupportedTransformError, match="unknown kind 'quantile'"):
        TargetTransform.from_dict({"steps": [{"kind": "quantile"}]})
    with pytest.raises(ValueError, match="needs a 'steps' list"):
        TargetTransform.from_dict({})
    with pytest.raises(ValueError, match="step 0 \\(standard_scaler\\)"):
        TargetTransform.from_dict({"steps": [{"kind": "standard_scaler", "centre": 1.0}]})


def test_log1p_zscore_clip_recipe_and_saturation() -> None:
    # The recovered NEMAD / phonix-db recipe: log1p(max(x, 0)) -> train z-score -> clip at +5.
    t = TargetTransform((Log1pStep(clip_lower=0.0), StandardScalerStep(mean=1.5, scale=0.5), ClipStep(upper=5.0)))
    x = np.array([-1.0, 0.0, 3.0, 1e6, np.nan])
    y = t.transform(x)
    assert y[0] == y[1] == pytest.approx(-3.0)
    assert y[3] == 5.0 and np.isnan(y[4])
    back = t.inverse(y)
    assert back[2] == pytest.approx(3.0)
    assert back[3] == pytest.approx(np.expm1(5.0 * 0.5 + 1.5))  # saturated at the clip bound
    # A model output beyond the bound saturates too, instead of extrapolating past it.
    assert t.inverse([7.0])[0] == pytest.approx(np.expm1(5.0 * 0.5 + 1.5))
    both = TargetTransform((ClipStep(lower=-1.0, upper=1.0),))
    np.testing.assert_array_equal(both.inverse([-3.0, 0.5, 3.0]), [-1.0, 0.5, 1.0])
    assert t.saturates and not TargetTransform((Log1pStep(),)).saturates


def test_nan_passes_through_every_step() -> None:
    x = np.array([np.nan, 1.0])
    for step in (
        StandardScalerStep(mean=1.0, scale=2.0),
        PowerTransformerStep(method="yeo-johnson", lmbda=0.5),
        Log1pStep(clip_lower=0.0),
        ArcsinhStep(),
        ClipStep(lower=-1.0, upper=1.0),
    ):
        t = TargetTransform((step,))
        assert np.isnan(t.transform(x)[0]) and np.isnan(t.inverse(x)[0])


def test_step_validation() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        StandardScalerStep(scale=0.0)
    with pytest.raises(ValueError, match="at least one"):
        ClipStep()
    with pytest.raises(ValueError, match="lower must be < clip.upper"):
        ClipStep(lower=1.0, upper=1.0)
    with pytest.raises(ValueError, match="method"):
        PowerTransformerStep(method="quantile", lmbda=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly positive"):
        TargetTransform((PowerTransformerStep(method="box-cox", lmbda=0.5),)).transform([0.0, 1.0])
    with pytest.raises(ValueError, match="one column"):
        TargetTransform((ArcsinhStep(),)).transform(np.ones((3, 2)))


def test_fit_default_equals_the_sklearn_recipe_and_ignores_nan() -> None:
    x = _rng_values()
    with_nan = np.concatenate([x, [np.nan, np.inf]])
    fitted = fit_default(with_nan)
    pipe = make_pipeline(StandardScaler(), PowerTransformer(method="yeo-johnson")).fit(x.reshape(-1, 1))
    np.testing.assert_allclose(fitted.transform(x), pipe.transform(x.reshape(-1, 1)).ravel(), rtol=0, atol=TOL)
    with pytest.raises(ValueError, match="at least 2"):
        fit_default([1.0, np.nan])


def test_normalize_max_abs_matches_interp_smooth_convention() -> None:
    np.testing.assert_allclose(normalize_max_abs([1.0, -4.0, 2.0]), [0.25, -1.0, 0.5])
    np.testing.assert_array_equal(normalize_max_abs([0.0, 0.0]), [0.0, 0.0])
    out = normalize_max_abs([np.nan, 2.0])
    assert np.isnan(out[0]) and out[1] == 1.0


_PREPROCESSING = sorted((REPO / "data").glob("*preprocessing*.pkl.z"))


@pytest.mark.skipif(not _PREPROCESSING, reason="project preprocessing files are not present (data/ is untracked)")
@pytest.mark.parametrize("path", _PREPROCESSING, ids=lambda p: p.name)
def test_every_project_scaler_round_trips(path: Path) -> None:
    objects = joblib.load(path)
    checked = 0
    for key, obj in objects.items() if isinstance(objects, dict) else [("<root>", objects)]:
        if not (hasattr(obj, "transform") or isinstance(obj, LabelEncoder)):
            continue  # dropped_idx lists, provenance strings
        if getattr(obj, "n_features_in_", 1) != 1:
            # e.g. composition_desc_scaler (290 descriptor columns): not a target transform.
            with pytest.raises(UnsupportedTransformError, match="single-column"):
                from_sklearn(obj)
            continue
        converted = from_sklearn(obj)
        checked += 1
        if isinstance(converted, ClassLabels):
            assert list(converted.classes) == [str(c) for c in obj.classes_], key
            continue
        x = np.linspace(-3, 3, 61)
        ref = obj.inverse_transform(x.reshape(-1, 1)).ravel()
        np.testing.assert_allclose(converted.inverse(x), ref, rtol=1e-12, atol=TOL, err_msg=key)
        finite = np.isfinite(ref)
        np.testing.assert_allclose(
            converted.transform(ref[finite]),
            obj.transform(ref[finite].reshape(-1, 1)).ravel(),
            rtol=0,
            atol=1e-9,
            err_msg=key,
        )
    assert checked


# --- inverse domain ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "lmbda", "bounded"),
    [("yeo-johnson", -0.5, "upper"), ("yeo-johnson", 3.0, "lower"), ("yeo-johnson", 1.0, None),
     ("box-cox", 0.5, "lower"), ("box-cox", -0.5, "upper"), ("box-cox", 0.0, None)],
)  # fmt: skip
def test_power_transformer_image_is_where_the_inverse_is_defined(
    method: str, lmbda: float, bounded: str | None
) -> None:
    step = PowerTransformerStep(method=method, lmbda=lmbda, standardize=True, mean=0.3, scale=1.7)  # type: ignore[arg-type]
    lo, hi = TargetTransform((step,)).inverse_domain()
    assert (np.isfinite(lo), np.isfinite(hi)) == (bounded == "lower", bounded == "upper")
    with np.errstate(all="ignore"):
        if bounded == "upper":
            assert np.isfinite(step.backward(np.array([hi - 1e-6]))).all()
            assert not np.isfinite(step.backward(np.array([hi + 1e-6]))).any()
        if bounded == "lower":
            assert np.isfinite(step.backward(np.array([lo + 1e-6]))).all()
            assert not np.isfinite(step.backward(np.array([lo - 1e-6]))).any()


def test_guard_saturates_at_the_training_range_on_the_bounded_side_only() -> None:
    rng = np.random.default_rng(1)
    heavy = rng.lognormal(0.0, 2.0, 2000)  # heavy right tail: the fitted Yeo-Johnson lambda is negative
    t = fit_default(heavy)
    assert np.isfinite(t.inverse_domain()[1]) and not np.isfinite(t.inverse_domain()[0])
    scaled = t.transform(heavy)
    with np.errstate(all="ignore"):
        assert np.isnan(t.inverse(np.array([scaled.max() + 50.0]))).all()  # the defect being guarded

    g = guard_inverse_domain(t, scaled)
    assert isinstance(g.steps[-1], ClipStep) and g.steps[-1].lower is None
    assert g.steps[-1].upper == pytest.approx(scaled.max())
    far = g.inverse(np.array([scaled.max() + 50.0, scaled.min() - 50.0]))
    # Saturates at the largest training value; next to the singular edge the inverse is ill-conditioned.
    assert far[0] == pytest.approx(heavy.max(), rel=1e-4)
    assert np.isfinite(far[1]) and far[1] < heavy.min()  # the open side still extrapolates
    np.testing.assert_allclose(g.transform(heavy), scaled)  # training values are untouched
    assert TargetTransform.from_dict(json.loads(json.dumps(g.to_dict()))) == g


def test_guard_leaves_an_everywhere_invertible_transform_alone() -> None:
    plain = TargetTransform((StandardScalerStep(mean=1.0, scale=2.0),))
    assert guard_inverse_domain(plain, np.array([0.0, 1.0])) is plain
    # An existing clip (phonix-db kp / klat) already makes the inverse defined everywhere.
    clipped = TargetTransform((Log1pStep(clip_lower=0.0), StandardScalerStep(mean=1.0, scale=2.0), ClipStep(upper=5.0)))
    assert clipped.inverse_domain() == (-np.inf, np.inf)
    assert guard_inverse_domain(clipped, np.array([0.0, 5.0])) is clipped


def test_guard_steps_inside_a_training_value_on_the_domain_edge() -> None:
    step = PowerTransformerStep(method="yeo-johnson", lmbda=-2.0, standardize=False)
    t = TargetTransform((step,))
    edge = t.inverse_domain()[1]  # 0.5: inverts to +inf
    g = guard_inverse_domain(t, np.array([0.0, edge]))
    clip = g.steps[-1]
    assert isinstance(clip, ClipStep) and clip.upper is not None
    assert clip.upper < edge and np.isfinite(g.inverse(np.array([10.0]))).all()


def test_domain_keeps_bounds_from_earlier_steps() -> None:
    bounded = PowerTransformerStep(method="yeo-johnson", lmbda=-1.0, standardize=False)  # image (-inf, 1)
    identity_power = PowerTransformerStep(method="yeo-johnson", lmbda=1.0, standardize=False)
    stacked = TargetTransform((bounded, identity_power))
    assert stacked.inverse_domain() == (-np.inf, pytest.approx(1.0))
    then_log = TargetTransform((bounded, Log1pStep()))
    assert then_log.inverse_domain() == (-np.inf, pytest.approx(np.log(2.0)))
    for t in (stacked, then_log):
        g = guard_inverse_domain(t, t.transform(np.array([-3.0, 0.0, 5.0])))
        assert isinstance(g.steps[-1], ClipStep)
        assert np.isfinite(g.inverse(np.array([2.0]))).all()


def test_strip_removes_only_a_guard() -> None:
    t = fit_default(np.random.default_rng(2).lognormal(0.0, 2.0, 500))
    g = guard_inverse_domain(t, t.transform(np.random.default_rng(2).lognormal(0.0, 2.0, 500)))
    assert g is not t and strip_inverse_guard(g) == t
    assert strip_inverse_guard(t) is t
    phonix = TargetTransform((Log1pStep(clip_lower=0.0), StandardScalerStep(mean=1.0, scale=2.0), ClipStep(upper=5.0)))
    assert strip_inverse_guard(phonix) is phonix  # part of the normalisation, not a guard


def test_intrinsic_clip_on_a_bounded_side_is_not_a_guard() -> None:
    # A normalisation that itself ends with a clip on the side a power step bounds (schema v1 allows it).
    power = PowerTransformerStep(method="yeo-johnson", lmbda=-1.0, standardize=False)
    intrinsic = TargetTransform((power, ClipStep(upper=0.5)))
    assert strip_inverse_guard(intrinsic) is intrinsic
    guarded = guard_inverse_domain(TargetTransform((power,)), np.array([0.0, 0.4]))
    assert guarded.steps[-1] == ClipStep(upper=0.4, guard=True)


def test_guard_flag_serialises_only_when_set() -> None:
    assert ClipStep(upper=1.0).params() == {"lower": None, "upper": 1.0}  # identical to schema v1
    marked = TargetTransform((ClipStep(upper=1.0, guard=True),))
    assert marked.to_dict()["steps"][0]["guard"] is True
    assert TargetTransform.from_dict(marked.to_dict()) == marked
    v1 = TargetTransform.from_dict({"steps": [{"kind": "clip", "lower": None, "upper": 1.0}]})
    assert v1.steps[0] == ClipStep(upper=1.0) and not v1.steps[0].guard  # type: ignore[union-attr]
