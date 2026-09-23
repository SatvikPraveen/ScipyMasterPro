"""
Property-based tests driven by Hypothesis.

Instead of checking single hand-picked examples these tests assert invariants
that must hold for *any* valid input: ECDFs are monotone and bounded, effect
sizes are antisymmetric, confidence intervals contain the point estimate, and
so on. Hypothesis searches for counter-examples automatically.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from utils.inference_utils import compute_sem, confidence_interval, margin_of_error
from utils.interpolation_utils import cubic_interpolate, linear_interpolate
from utils.linear_algebra_utils import compute_svd
from utils.pdf_ecdf_utils import compute_manual_ecdf
from utils.sim_utils import bootstrap_sample, compute_ecdf, resample_with_replacement
from utils.stats_tests_utils import cohens_d_independent, p_adjust_bh, rank_biserial_effect_size

pytestmark = pytest.mark.property

finite_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def samples(min_size: int = 2, max_size: int = 200):
    return hnp.arrays(np.float64, st.integers(min_size, max_size), elements=finite_floats)


def _non_degenerate(x: np.ndarray) -> bool:
    return np.std(x) > 1e-9


# ---------------------------------------------------------------------------
# ECDF invariants
# ---------------------------------------------------------------------------
@given(samples())
def test_manual_ecdf_is_monotone_and_bounded(data):
    x, y = compute_manual_ecdf(data)
    assert np.all(np.diff(x) >= 0)
    assert np.all(np.diff(y) >= 0)
    assert 0 < y.min() <= 1 and np.isclose(y.max(), 1.0)


@given(samples())
def test_sim_ecdf_matches_manual_ecdf(data):
    x1, y1 = compute_manual_ecdf(data)
    x2, y2 = compute_ecdf(data)
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y1, y2)


# ---------------------------------------------------------------------------
# Resampling invariants
# ---------------------------------------------------------------------------
@given(samples(min_size=3, max_size=100), st.integers(1, 200), st.integers(0, 2**31 - 1))
@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
def test_bootstrap_means_stay_within_sample_range(data, n_iter, seed):
    boots = bootstrap_sample(data, n_iterations=n_iter, seed=seed)
    assert boots.shape == (n_iter,)
    assert boots.min() >= data.min() - 1e-9
    assert boots.max() <= data.max() + 1e-9


@given(samples(min_size=1, max_size=100), st.integers(1, 500), st.integers(0, 2**31 - 1))
def test_resample_with_replacement_only_uses_original_values(data, n_samples, seed):
    res = resample_with_replacement(data, n_samples=n_samples, seed=seed)
    assert res.shape == (n_samples,)
    assert set(np.unique(res)).issubset(set(np.unique(data)))


# ---------------------------------------------------------------------------
# Effect size invariants
# ---------------------------------------------------------------------------
@given(samples(min_size=3, max_size=100), samples(min_size=3, max_size=100))
def test_cohens_d_is_antisymmetric(a, b):
    if not (_non_degenerate(a) or _non_degenerate(b)):
        return
    d_ab = cohens_d_independent(a, b)
    d_ba = cohens_d_independent(b, a)
    assert np.isclose(d_ab, -d_ba, atol=1e-9)


@given(samples(min_size=3, max_size=60), samples(min_size=3, max_size=60))
def test_rank_biserial_bounded_and_antisymmetric(a, b):
    r_ab = rank_biserial_effect_size(a, b)
    r_ba = rank_biserial_effect_size(b, a)
    assert -1.0 - 1e-9 <= r_ab <= 1.0 + 1e-9
    assert np.isclose(r_ab, -r_ba, atol=1e-9)


@given(st.lists(st.floats(0.0, 1.0), min_size=1, max_size=50))
def test_bh_adjustment_never_decreases_pvalues_and_stays_in_unit_interval(pvals):
    adjusted = np.asarray(p_adjust_bh(np.asarray(pvals)))
    assert adjusted.shape == (len(pvals),)
    assert np.all(adjusted >= np.asarray(pvals) - 1e-12)
    assert np.all(adjusted <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# Inference invariants (summary-statistics API: mean, std_dev, n)
# ---------------------------------------------------------------------------
summary_stats = st.tuples(
    st.floats(-1e4, 1e4, allow_nan=False),  # mean
    st.floats(1e-3, 1e3, allow_nan=False),  # std_dev
    st.integers(2, 10_000),  # n
)


@given(summary_stats, st.floats(0.5, 0.999))
def test_confidence_interval_contains_mean_and_widens_with_confidence(stats_, conf):
    mean, sd, n = stats_
    lo, hi = confidence_interval(mean, sd, n, confidence=conf)
    assert lo <= mean <= hi
    lo2, hi2 = confidence_interval(mean, sd, n, confidence=min(conf + 0.0005, 0.9999))
    assert (hi2 - lo2) >= (hi - lo) - 1e-9


@given(st.floats(0.0, 1e3, allow_nan=False), st.integers(1, 10_000))
def test_sem_is_nonnegative_and_shrinks_with_n(sd, n):
    sem = compute_sem(sd, n)
    assert sem >= 0
    assert compute_sem(sd, 4 * n) <= sem + 1e-12
    assert np.isclose(compute_sem(sd, 4 * n), sem / 2, rtol=1e-9, atol=1e-12)


@given(summary_stats, st.floats(0.5, 0.999))
def test_margin_of_error_matches_half_interval_width(stats_, conf):
    mean, sd, n = stats_
    lo, hi = confidence_interval(mean, sd, n, confidence=conf)
    moe = margin_of_error(sd, n, confidence=conf)
    assert np.isclose(moe, (hi - lo) / 2, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Interpolation invariants (interpolators return callables)
# ---------------------------------------------------------------------------
@given(
    st.integers(4, 30).flatmap(
        lambda n: hnp.arrays(np.float64, n, elements=st.floats(-100, 100, allow_nan=False))
    )
)
def test_interpolators_reproduce_knots(y):
    x = np.linspace(0.0, 1.0, y.size)
    for make in (linear_interpolate, cubic_interpolate):
        f = make(x, y)
        np.testing.assert_allclose(f(x), y, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Linear algebra invariants
# ---------------------------------------------------------------------------
@given(
    hnp.arrays(
        np.float64,
        st.tuples(st.integers(2, 8), st.integers(2, 8)),
        elements=st.floats(-10, 10, allow_nan=False),
    )
)
def test_svd_reconstructs_matrix(matrix):
    parts = compute_svd(matrix)
    U, S, Vt = parts["U"], parts["S"], parts["Vt"]
    k = S.size
    reconstructed = (U[:, :k] * S) @ Vt[:k, :]
    np.testing.assert_allclose(reconstructed, matrix, atol=1e-8)
    assert np.all(np.diff(S) <= 1e-12)
