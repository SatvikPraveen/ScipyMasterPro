"""
Tests for pdf_ecdf_utils module.

Covers PDF computation, manual and statsmodels ECDF, and goodness-of-fit
utilities.
"""

import numpy as np
import pytest
from scipy.stats import norm, expon

from utils.pdf_ecdf_utils import (
    get_pdf,
    compute_manual_ecdf,
    compute_statsmodels_ecdf,
)


@pytest.fixture
def normal_sample(random_seed):
    np.random.seed(random_seed)
    return np.random.normal(loc=3.0, scale=1.5, size=300)


@pytest.fixture
def exponential_sample(random_seed):
    np.random.seed(random_seed)
    return np.random.exponential(scale=2.0, size=300)


class TestGetPDF:
    """Test PDF computation over data range."""

    def test_returns_two_arrays(self, normal_sample):
        """get_pdf should return (x, pdf) tuple."""
        x, pdf = get_pdf(normal_sample, norm)
        assert len(x) == len(pdf)

    def test_pdf_non_negative(self, normal_sample):
        """All PDF values must be >= 0."""
        _, pdf = get_pdf(normal_sample, norm)
        assert np.all(pdf >= 0)

    def test_x_range_covers_data(self, normal_sample):
        """x should span from min to max of data."""
        x, _ = get_pdf(normal_sample, norm)
        assert np.isclose(x[0], np.min(normal_sample), atol=1e-6)
        assert np.isclose(x[-1], np.max(normal_sample), atol=1e-6)

    def test_num_points_parameter(self, normal_sample):
        """num_points should control the length of returned arrays."""
        x, pdf = get_pdf(normal_sample, norm, num_points=50)
        assert len(x) == 50

    def test_with_pre_fitted_params(self, normal_sample):
        """Passing pre-fitted params should produce same result as fitting internally."""
        params = norm.fit(normal_sample)
        x1, pdf1 = get_pdf(normal_sample, norm, params=params)
        x2, pdf2 = get_pdf(normal_sample, norm)
        np.testing.assert_allclose(pdf1, pdf2, atol=1e-6)

    def test_pdf_peaks_near_data_mean(self, normal_sample):
        """For a normal fit, peak of PDF should be near the sample mean."""
        x, pdf = get_pdf(normal_sample, norm)
        peak_x = x[np.argmax(pdf)]
        assert abs(peak_x - np.mean(normal_sample)) < 0.5


class TestManualECDF:
    """Test manual empirical CDF computation."""

    def test_returns_sorted_x(self, normal_sample):
        """x values must be sorted ascending."""
        x, y = compute_manual_ecdf(normal_sample)
        assert np.all(np.diff(x) >= 0)

    def test_y_starts_near_zero_ends_at_one(self, normal_sample):
        """ECDF y must start just above 0 and end at 1."""
        x, y = compute_manual_ecdf(normal_sample)
        assert y[0] > 0
        assert np.isclose(y[-1], 1.0)

    def test_y_monotone(self, normal_sample):
        """ECDF y must be monotonically non-decreasing."""
        _, y = compute_manual_ecdf(normal_sample)
        assert np.all(np.diff(y) >= 0)

    def test_length_matches_data(self, normal_sample):
        """Output arrays should have same length as input."""
        x, y = compute_manual_ecdf(normal_sample)
        assert len(x) == len(normal_sample)
        assert len(y) == len(normal_sample)

    def test_uniform_steps(self):
        """For n points, each step should be 1/n."""
        data = np.array([3.0, 1.0, 4.0, 1.5, 2.0])
        x, y = compute_manual_ecdf(data)
        expected_step = 1.0 / len(data)
        diffs = np.diff(y)
        np.testing.assert_allclose(diffs, expected_step, atol=1e-10)


class TestStatsmodelsECDF:
    """Test statsmodels ECDF computation."""

    def test_returns_two_arrays(self, normal_sample):
        x, y = compute_statsmodels_ecdf(normal_sample)
        assert len(x) > 0
        assert len(y) > 0

    def test_y_in_zero_one(self, normal_sample):
        _, y = compute_statsmodels_ecdf(normal_sample)
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    def test_y_monotone(self, normal_sample):
        x, y = compute_statsmodels_ecdf(normal_sample)
        assert np.all(np.diff(y) >= 0)

    def test_agrees_with_manual_ecdf(self, normal_sample):
        """Both ECDF methods should give the same results."""
        x_manual, y_manual = compute_manual_ecdf(normal_sample)
        x_sm, y_sm = compute_statsmodels_ecdf(normal_sample)
        np.testing.assert_allclose(x_manual, x_sm, atol=1e-10)
        np.testing.assert_allclose(y_manual, y_sm, atol=1e-10)


class TestECDFProperties:
    """Cross-distribution ECDF property tests."""

    @pytest.mark.parametrize("n", [50, 200, 1000])
    def test_ecdf_length_matches_data(self, n, random_seed):
        """ECDF output length should always equal data length."""
        np.random.seed(random_seed)
        data = np.random.normal(size=n)
        x, y = compute_manual_ecdf(data)
        assert len(x) == n

    def test_ecdf_exponential_data(self, exponential_sample):
        """ECDF on exponential data should still be valid."""
        x, y = compute_manual_ecdf(exponential_sample)
        assert np.all(x >= 0)  # Exponential is non-negative
        assert np.isclose(y[-1], 1.0)
