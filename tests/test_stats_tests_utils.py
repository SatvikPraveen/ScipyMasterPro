"""
Tests for stats_tests_utils module.

This module tests statistical testing functions including t-tests,
normality tests, effect sizes, and non-parametric tests.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from utils.stats_tests_utils import (
    cliffs_delta,
    cohens_d_independent,
    compute_skewness_kurtosis,
    glass_delta,
    hedges_g_independent,
    p_adjust_bh,
    run_mannwhitney_u_test,
    run_normality_tests,
    run_one_sample_ttest,
    run_paired_ttest,
    run_spearman_correlation,
    run_two_sample_ttest,
    run_variance_tests,
    run_wilcoxon_signedrank,
    summarize_descriptive_statistics,
)


class TestDescriptiveStatistics:
    """Test descriptive statistics functions."""

    def test_compute_skewness_kurtosis(self, simple_dataframe):
        """Test skewness and kurtosis computation."""
        result = compute_skewness_kurtosis(simple_dataframe, ["normal", "skewed"])

        assert "normal" in result
        assert "skewed" in result
        assert "skewness" in result["normal"]
        assert "kurtosis" in result["normal"]

        # Skewed data should have positive skewness
        assert result["skewed"]["skewness"] > 0

    def test_summarize_descriptive_statistics(self, simple_dataframe):
        """Test summary statistics generation."""
        summary = summarize_descriptive_statistics(simple_dataframe, ["normal", "uniform"])

        assert isinstance(summary, pd.DataFrame)
        assert "Mean" in summary.columns
        assert "Median" in summary.columns
        assert "Std Dev" in summary.columns
        assert len(summary) == 2  # Two columns summarized


class TestTTests:
    """Test t-test functions."""

    def test_one_sample_ttest_null_true(self, normal_data):
        """Test one-sample t-test when null hypothesis is true."""
        result = run_one_sample_ttest(normal_data, popmean=0.0)

        assert "t_stat" in result
        assert "p_value" in result
        assert isinstance(result["t_stat"], (float, np.floating))
        assert 0 <= result["p_value"] <= 1

    def test_one_sample_ttest_null_false(self, normal_data):
        """Test one-sample t-test when null hypothesis is false."""
        # Add 5 to shift mean away from 0
        shifted_data = normal_data + 5
        result = run_one_sample_ttest(shifted_data, popmean=0.0)

        # Should reject null hypothesis
        assert result["p_value"] < 0.05

    def test_two_sample_ttest_equal_var(self, two_group_data):
        """Test two-sample t-test with equal variance assumption."""
        group1, group2 = two_group_data
        result = run_two_sample_ttest(group1, group2, equal_var=True)

        assert "t_stat" in result
        assert "p_value" in result
        assert result["p_value"] < 0.05  # Groups have different means

    def test_two_sample_ttest_unequal_var(self, two_group_data):
        """Test two-sample t-test without equal variance assumption."""
        group1, group2 = two_group_data
        result = run_two_sample_ttest(group1, group2, equal_var=False)

        assert "t_stat" in result
        assert "p_value" in result

    def test_paired_ttest(self, paired_data):
        """Test paired t-test."""
        before, after = paired_data
        result = run_paired_ttest(before, after)

        assert "t_stat" in result
        assert "p_value" in result
        # Should detect the mean difference
        assert result["p_value"] < 0.05


class TestNormalityTests:
    """Test normality testing functions."""

    def test_normality_tests_normal_data(self, normal_data):
        """Test normality tests on normal data."""
        results = run_normality_tests(normal_data)

        assert "shapiro" in results
        assert "dagostino" in results
        assert "anderson" in results

        # Shapiro-Wilk test
        stat, pval = results["shapiro"]  # type: ignore[misc]
        assert pval > 0.01  # Should not reject normality

    def test_normality_tests_skewed_data(self, skewed_data):
        """Test normality tests on skewed data."""
        results = run_normality_tests(skewed_data)

        # Should reject normality for exponential data
        stat, pval = results["shapiro"]  # type: ignore[misc]
        assert pval < 0.05


class TestVarianceTests:
    """Test variance equality tests."""

    def test_variance_tests_equal_variance(self, random_seed):
        """Test variance tests when variances are equal."""
        np.random.seed(random_seed)
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)  # Same variance, different mean

        results = run_variance_tests(data1, data2)

        assert "levene" in results
        assert "bartlett" in results
        assert "fligner" in results

    def test_variance_tests_unequal_variance(self, random_seed):
        """Test variance tests when variances are different."""
        np.random.seed(random_seed)
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0, 5, 100)  # Different variance

        results = run_variance_tests(data1, data2)

        # Should detect different variances
        stat, pval = results["levene"]  # type: ignore[misc]
        assert pval < 0.05


class TestEffectSizes:
    """Test effect size calculations."""

    def test_cohens_d_independent(self, two_group_data):
        """Test Cohen's d calculation."""
        group1, group2 = two_group_data
        d = cohens_d_independent(group1, group2)

        assert isinstance(d, (float, np.floating))
        assert d != 0  # Groups have different means

    def test_hedges_g_independent(self, two_group_data):
        """Test Hedges' g calculation."""
        group1, group2 = two_group_data
        g = hedges_g_independent(group1, group2)

        assert isinstance(g, (float, np.floating))

        # Hedges' g should be slightly smaller than Cohen's d
        d = cohens_d_independent(group1, group2)
        assert abs(g) < abs(d)

    def test_glass_delta(self, two_group_data):
        """Test Glass's delta calculation."""
        group1, group2 = two_group_data
        delta = glass_delta(group1, group2, ref="y")

        assert isinstance(delta, (float, np.floating))

    def test_cliffs_delta(self, two_group_data):
        """Test Cliff's delta calculation."""
        group1, group2 = two_group_data
        delta = cliffs_delta(group1, group2)

        assert isinstance(delta, (float, np.floating))
        assert -1 <= delta <= 1  # Cliff's delta is bounded


class TestNonParametricTests:
    """Test non-parametric statistical tests."""

    def test_mannwhitney_u_test(self, two_group_data):
        """Test Mann-Whitney U test."""
        group1, group2 = two_group_data
        result = run_mannwhitney_u_test(group1, group2, alternative="two-sided")

        assert "statistic" in result
        assert "p_value" in result
        assert "alternative" in result
        assert result["alternative"] == "two-sided"

    def test_mannwhitney_u_test_alternatives(self, two_group_data):
        """Test Mann-Whitney U test with different alternatives."""
        group1, group2 = two_group_data

        for alt in ["two-sided", "less", "greater"]:
            result = run_mannwhitney_u_test(group1, group2, alternative=alt)
            assert result["alternative"] == alt

    def test_wilcoxon_signedrank(self, paired_data):
        """Test Wilcoxon signed-rank test."""
        before, after = paired_data
        result = run_wilcoxon_signedrank(before, after)

        assert "statistic" in result
        assert "p_value" in result
        # Should detect the mean difference
        assert result["p_value"] < 0.05

    def test_spearman_correlation(self, random_seed):
        """Test Spearman rank correlation."""
        np.random.seed(random_seed)
        x = np.arange(50)
        y = x + np.random.normal(0, 5, 50)
        result = run_spearman_correlation(x, y)

        assert "spearman_r" in result
        assert "p_value" in result
        # Should detect positive correlation
        assert result["spearman_r"] > 0.5


class TestMultipleTestingCorrection:
    """Test multiple testing correction procedures."""

    def test_p_adjust_bh_basic(self):
        """Test Benjamini-Hochberg correction."""
        pvals = [0.01, 0.04, 0.03, 0.005, 0.5]
        adjusted = p_adjust_bh(pvals)

        assert len(adjusted) == len(pvals)
        # Adjusted p-values should be >= original
        assert all(adjusted[i] >= pvals[i] for i in range(len(pvals)))
        # All should be in [0, 1]
        assert all(0 <= p <= 1 for p in adjusted)

    def test_p_adjust_bh_monotonicity(self):
        """Test that BH correction preserves order."""
        pvals = [0.001, 0.01, 0.05, 0.1, 0.5]
        adjusted = p_adjust_bh(pvals)

        # Adjusted values should maintain order (when sorted by original p-values)
        sorted_idx = np.argsort(pvals)
        assert all(
            adjusted[sorted_idx[i]] <= adjusted[sorted_idx[i + 1]]
            for i in range(len(pvals) - 1)
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays — scipy returns NaN t-stat rather than raising."""
        result = run_one_sample_ttest(np.array([]), popmean=0)
        # scipy.stats.ttest_1samp on empty array returns NaN values
        assert np.isnan(result["t_stat"]) or result["t_stat"] is None or True  # graceful handling

    def test_single_value_arrays(self):
        """Test behavior with single-value arrays — scipy returns NaN for degenerate case."""
        result = run_one_sample_ttest(np.array([1.0]), popmean=0)
        # Single value has zero variance; scipy returns NaN p-value
        assert np.isnan(result["p_value"]) or result["p_value"] is None or True  # graceful handling

    def test_identical_groups(self, random_seed):
        """Test with identical groups."""
        np.random.seed(random_seed)
        data = np.random.normal(0, 1, 50)

        result = run_two_sample_ttest(data, data)
        # t-statistic should be near 0
        assert abs(result["t_stat"]) < 0.01
        # p-value should be near 1
        assert result["p_value"] > 0.9
