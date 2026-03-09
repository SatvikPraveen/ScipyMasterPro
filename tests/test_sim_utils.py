"""
Tests for sim_utils module.

Covers bootstrap sampling, uniform/weighted/stratified sampling, multinomial
and Dirichlet sampling, ECDF computation, and bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
import pytest

from utils.sim_utils import (
    bootstrap_sample,
    sample_uniform,
    weighted_sample,
    stratified_sample,
    draw_multinomial_sample,
    draw_dirichlet_sample,
    sample_custom_discrete,
    resample_with_replacement,
    compute_ecdf,
    bootstrap_statistic,
    compute_bootstrap_ci,
    summarize_bootstrap,
)


@pytest.fixture
def population(random_seed):
    np.random.seed(random_seed)
    return np.random.normal(loc=10.0, scale=2.0, size=1000)


@pytest.fixture
def groups_df():
    """DataFrame with two groups for stratified sampling."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "value": rng.normal(0, 1, 200),
        "group": ["A"] * 100 + ["B"] * 100,
    })


class TestBootstrapSample:
    """Test basic bootstrap mean distribution."""

    def test_returns_array(self, population):
        result = bootstrap_sample(population, n_iterations=200)
        assert isinstance(result, np.ndarray)
        assert len(result) == 200

    def test_mean_close_to_population_mean(self, population):
        """Bootstrap mean-of-means should be close to the population mean."""
        result = bootstrap_sample(population, n_iterations=500)
        assert abs(np.mean(result) - np.mean(population)) < 0.5

    def test_reproducible_with_seed(self, population):
        r1 = bootstrap_sample(population, n_iterations=100, seed=0)
        r2 = bootstrap_sample(population, n_iterations=100, seed=0)
        np.testing.assert_array_equal(r1, r2)


class TestSampleUniform:
    """Test uniform random sampling."""

    def test_correct_sample_size(self, population):
        sample = sample_uniform(population, n=50)
        assert len(sample) == 50

    def test_with_replacement_allows_duplicates(self, population):
        """With replacement, duplicates are possible (very likely for large n)."""
        sample = sample_uniform(population, n=len(population), replace=True)
        assert len(sample) == len(population)

    def test_without_replacement_no_duplicates(self, population):
        sample = sample_uniform(population, n=50, replace=False)
        assert len(sample) == len(np.unique(sample))

    def test_reproducible_with_seed(self, population):
        s1 = sample_uniform(population, n=20, seed=7)
        s2 = sample_uniform(population, n=20, seed=7)
        np.testing.assert_array_equal(s1, s2)


class TestWeightedSample:
    """Test probability-weighted sampling."""

    def test_correct_size(self, population):
        n = 100
        weights = np.ones(len(population)) / len(population)
        sample = weighted_sample(population, weights=weights, n=n)
        assert len(sample) == n

    def test_biased_weights_affect_distribution(self):
        """Biased weights should over-sample certain values."""
        data = np.array([1, 2, 3, 4, 5])
        weights = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        sample = weighted_sample(data, weights=weights, n=100)
        assert np.all(sample == 5)


class TestStratifiedSample:
    """Test proportional stratified sampling."""

    def test_returns_dataframe(self, groups_df):
        result = stratified_sample(groups_df, stratify_col="group", frac=0.1)
        assert isinstance(result, pd.DataFrame)

    def test_proportions_preserved(self, groups_df):
        """Both groups should appear in the stratified sample."""
        result = stratified_sample(groups_df, stratify_col="group", frac=0.2)
        assert set(result["group"]) == {"A", "B"}

    def test_sample_size_approximately_correct(self, groups_df):
        """10% of 200 rows = ~20 rows."""
        result = stratified_sample(groups_df, stratify_col="group", frac=0.1)
        assert 15 <= len(result) <= 25


class TestMultinomialDirichlet:
    """Test multinomial and Dirichlet sampling."""

    def test_multinomial_shape(self):
        result = draw_multinomial_sample(n=100, probs=[0.2, 0.3, 0.5], size=5)
        assert result.shape == (5, 3)

    def test_multinomial_row_sums(self):
        """Each row of multinomial draw should sum to n."""
        result = draw_multinomial_sample(n=50, probs=[0.25, 0.25, 0.25, 0.25], size=10)
        np.testing.assert_array_equal(result.sum(axis=1), np.full(10, 50))

    def test_dirichlet_shape(self):
        result = draw_dirichlet_sample(alpha=[1.0, 2.0, 3.0], size=10)
        assert result.shape == (10, 3)

    def test_dirichlet_rows_sum_to_one(self):
        result = draw_dirichlet_sample(alpha=[2.0, 2.0, 2.0], size=20)
        np.testing.assert_allclose(result.sum(axis=1), np.ones(20), atol=1e-10)


class TestCustomDiscrete:
    """Test custom discrete distribution sampling."""

    def test_returns_correct_size(self):
        result = sample_custom_discrete([0, 1, 2], [0.2, 0.5, 0.3], size=200)
        assert len(result) == 200

    def test_only_support_values_present(self):
        result = sample_custom_discrete([0, 1, 2], [0.2, 0.5, 0.3], size=500)
        assert set(result).issubset({0, 1, 2})


class TestResampleWithReplacement:
    """Test basic bootstrap resampling."""

    def test_returns_correct_size(self, population):
        result = resample_with_replacement(population, n_samples=300)
        assert len(result) == 300

    def test_values_from_population(self, population):
        result = resample_with_replacement(population, n_samples=100)
        assert all(v in population for v in result)


class TestComputeECDF:
    """Test empirical CDF computation."""

    def test_returns_sorted_x(self, population):
        x, y = compute_ecdf(population)
        assert np.all(np.diff(x) >= 0)

    def test_y_in_zero_one(self, population):
        x, y = compute_ecdf(population)
        assert y[0] > 0
        assert np.isclose(y[-1], 1.0)

    def test_monotone_y(self, population):
        x, y = compute_ecdf(population)
        assert np.all(np.diff(y) >= 0)


class TestBootstrapStatistic:
    """Test generalized bootstrap statistic computation."""

    def test_bootstrap_mean(self, population):
        boot = bootstrap_statistic(population, stat_func=np.mean, n_resamples=200)
        assert len(boot) == 200
        assert abs(np.mean(boot) - np.mean(population)) < 0.5

    def test_bootstrap_median(self, population):
        boot = bootstrap_statistic(population, stat_func=np.median, n_resamples=100)
        assert len(boot) == 100

    def test_bootstrap_std(self, population):
        boot = bootstrap_statistic(population, stat_func=np.std, n_resamples=100)
        assert np.all(boot > 0)


class TestBootstrapCI:
    """Test percentile-method bootstrap confidence intervals."""

    def test_ci_lower_less_than_upper(self, population):
        boot = bootstrap_statistic(population, n_resamples=500)
        lower, upper = compute_bootstrap_ci(boot, ci=95)
        assert lower < upper

    def test_true_mean_within_ci(self, population):
        """The true population mean should fall inside the 95% CI."""
        boot = bootstrap_statistic(population, n_resamples=1000)
        lower, upper = compute_bootstrap_ci(boot, ci=95)
        assert lower < np.mean(population) < upper

    def test_wider_ci_for_higher_confidence(self, population):
        boot = bootstrap_statistic(population, n_resamples=500)
        lo90, hi90 = compute_bootstrap_ci(boot, ci=90)
        lo99, hi99 = compute_bootstrap_ci(boot, ci=99)
        assert (hi99 - lo99) > (hi90 - lo90)


class TestSummarizeBootstrap:
    """Test bootstrap summary report."""

    def test_summary_keys(self, population):
        boot = bootstrap_statistic(population, n_resamples=200)
        summary = summarize_bootstrap(boot)
        assert "mean" in summary
        assert "std" in summary
        assert "ci_lower" in summary
        assert "ci_upper" in summary

    def test_original_stat_included_when_provided(self, population):
        boot = bootstrap_statistic(population, n_resamples=200)
        original = np.mean(population)
        summary = summarize_bootstrap(boot, original_stat=original)
        assert "original" in summary
        assert np.isclose(summary["original"], original)
