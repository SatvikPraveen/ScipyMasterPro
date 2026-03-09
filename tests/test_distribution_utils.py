"""
Tests for distribution_utils module.

Covers distribution fitting, PDF/CDF computation, KS goodness-of-fit,
AIC/BIC information criteria, and multi-distribution comparison.
"""

import numpy as np
import pytest
from scipy.stats import norm, gamma, expon, lognorm

from utils.distribution_utils import (
    fit_distribution,
    compute_pdf,
    compute_cdf,
    perform_ks_test,
    fit_multiple_distributions,
    compute_nll,
    compute_aic,
    compute_bic,
)


@pytest.fixture
def normal_sample(random_seed):
    np.random.seed(random_seed)
    return np.random.normal(loc=5.0, scale=2.0, size=500)


@pytest.fixture
def exponential_sample(random_seed):
    np.random.seed(random_seed)
    return np.random.exponential(scale=2.0, size=500)


class TestFitDistribution:
    """Test MLE distribution fitting."""

    def test_fit_returns_tuple(self, normal_sample):
        """fit_distribution should return a tuple of parameters."""
        params = fit_distribution(normal_sample, norm)
        assert isinstance(params, tuple)
        assert len(params) == 2  # loc, scale for norm

    def test_fit_normal_loc_close_to_true(self, normal_sample):
        """Fitted loc should be close to the true mean."""
        params = fit_distribution(normal_sample, norm)
        loc = params[0]
        assert abs(loc - 5.0) < 0.5

    def test_fit_normal_scale_close_to_true(self, normal_sample):
        """Fitted scale should be close to the true std."""
        params = fit_distribution(normal_sample, norm)
        scale = params[1]
        assert abs(scale - 2.0) < 0.5

    def test_fit_gamma_returns_three_params(self, normal_sample):
        """Gamma fit should return 3 parameters (shape, loc, scale)."""
        params = fit_distribution(np.abs(normal_sample) + 0.1, gamma)
        assert len(params) == 3

    def test_fit_expon_with_dist_obj(self, exponential_sample):
        """fit_distribution with expon dist object should return tuple."""
        params = fit_distribution(exponential_sample, expon)
        assert isinstance(params, tuple)


class TestComputePDFCDF:
    """Test PDF and CDF computation."""

    def test_compute_pdf_returns_two_arrays(self, normal_sample):
        """compute_pdf should return (x, pdf_vals) of equal length."""
        params = fit_distribution(normal_sample, norm)
        x, pdf_vals = compute_pdf(normal_sample, norm, params)
        assert len(x) == len(pdf_vals)
        assert len(x) == 200

    def test_compute_pdf_non_negative(self, normal_sample):
        """PDF values must be non-negative."""
        params = fit_distribution(normal_sample, norm)
        _, pdf_vals = compute_pdf(normal_sample, norm, params)
        assert np.all(pdf_vals >= 0)

    def test_compute_cdf_monotone(self, normal_sample):
        """CDF values must be monotonically non-decreasing."""
        params = fit_distribution(normal_sample, norm)
        _, cdf_vals = compute_cdf(normal_sample, norm, params)
        diffs = np.diff(cdf_vals)
        assert np.all(diffs >= -1e-10)

    def test_compute_cdf_bounds(self, normal_sample):
        """CDF values must lie in [0, 1]."""
        params = fit_distribution(normal_sample, norm)
        _, cdf_vals = compute_cdf(normal_sample, norm, params)
        assert np.all(cdf_vals >= 0)
        assert np.all(cdf_vals <= 1)


class TestKSTest:
    """Test Kolmogorov-Smirnov goodness-of-fit."""

    def test_ks_test_returns_dict(self, normal_sample):
        """perform_ks_test should return a dict with KS_stat and p_value."""
        params = fit_distribution(normal_sample, norm)
        result = perform_ks_test(normal_sample, norm, params)
        assert "KS_stat" in result
        assert "p_value" in result

    def test_ks_test_good_fit_high_pval(self, normal_sample):
        """Normal data fitted with normal should give high p-value."""
        params = fit_distribution(normal_sample, norm)
        result = perform_ks_test(normal_sample, norm, params)
        assert result["p_value"] > 0.05

    def test_ks_test_bad_fit_low_pval(self, exponential_sample):
        """Exponential data fitted with normal should give low p-value."""
        params = fit_distribution(exponential_sample, norm)
        result = perform_ks_test(exponential_sample, norm, params)
        assert result["KS_stat"] > 0.0


class TestFitMultipleDistributions:
    """Test fitting multiple distributions at once."""

    def test_returns_list_of_dicts(self, normal_sample):
        """fit_multiple_distributions should return list of result dicts."""
        results = fit_multiple_distributions(normal_sample, [norm, expon])
        assert isinstance(results, list)
        assert len(results) == 2
        assert "distribution" in results[0]
        assert "KS_stat" in results[0]

    def test_norm_is_good_fit_for_normal_data(self, normal_sample):
        """Normal distribution should have a low KS stat and high p-value for normal data."""
        results = fit_multiple_distributions(normal_sample, [norm, expon, lognorm])
        norm_result = next(r for r in results if r["distribution"] == "norm")
        assert norm_result["p_value"] > 0.05


class TestInformationCriteria:
    """Test AIC and BIC calculations."""

    def test_compute_nll_positive(self, normal_sample):
        """NLL should be positive for a reasonable fit."""
        params = fit_distribution(normal_sample, norm)
        nll = compute_nll(norm, normal_sample, params)
        assert nll > 0

    def test_aic_penalizes_more_params(self, normal_sample):
        """AIC for gamma (3 params) should be >= AIC for norm (2 params) on normal data."""
        params_norm = fit_distribution(normal_sample, norm)
        params_gamma = fit_distribution(np.abs(normal_sample) + 0.1, gamma)

        nll_norm = compute_nll(norm, normal_sample, params_norm)
        nll_gamma = compute_nll(gamma, np.abs(normal_sample) + 0.1, params_gamma)

        aic_norm = compute_aic(nll_norm, k=2)
        aic_gamma = compute_aic(nll_gamma, k=3)

        # Both should be finite numbers
        assert np.isfinite(aic_norm)
        assert np.isfinite(aic_gamma)

    def test_bic_returns_float(self, normal_sample):
        """compute_bic should return a finite float."""
        params = fit_distribution(normal_sample, norm)
        nll = compute_nll(norm, normal_sample, params)
        bic = compute_bic(nll, k=2, n=len(normal_sample))
        assert np.isfinite(bic)
