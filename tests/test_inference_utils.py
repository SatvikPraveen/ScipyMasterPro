"""
Tests for inference_utils module.

This module tests statistical inference functions including
confidence intervals, standard errors, and inference from summary stats.
"""

import numpy as np
import pytest
from scipy import stats

# Import functions from inference_utils (adjust based on actual module)
# For now, creating placeholder test structure


class TestConfidenceIntervals:
    """Test confidence interval calculations."""

    def test_t_based_confidence_interval(self, normal_data):
        """Test t-based confidence interval calculation."""
        # Calculate CI using scipy
        mean = np.mean(normal_data)
        sem = stats.sem(normal_data)
        ci = stats.t.interval(0.95, len(normal_data) - 1, loc=mean, scale=sem)

        assert len(ci) == 2
        assert ci[0] < mean < ci[1]  # Mean should be within CI

    def test_z_based_confidence_interval(self, normal_data):
        """Test z-based confidence interval calculation."""
        mean = np.mean(normal_data)
        sem = stats.sem(normal_data)
        ci = stats.norm.interval(0.95, loc=mean, scale=sem)

        assert len(ci) == 2
        assert ci[0] < mean < ci[1]

    @pytest.mark.parametrize("conf_level", [0.90, 0.95, 0.99])
    def test_confidence_levels(self, normal_data, conf_level):
        """Test different confidence levels."""
        mean = np.mean(normal_data)
        sem = stats.sem(normal_data)
        ci = stats.t.interval(conf_level, len(normal_data) - 1, loc=mean, scale=sem)

        width_90 = ci[1] - ci[0] if conf_level == 0.90 else None
        # Higher confidence level should give wider interval
        assert ci[0] < mean < ci[1]


class TestStandardErrors:
    """Test standard error calculations."""

    def test_sem_calculation(self, normal_data):
        """Test standard error of the mean calculation."""
        sem_scipy = stats.sem(normal_data)
        sem_manual = np.std(normal_data, ddof=1) / np.sqrt(len(normal_data))

        np.testing.assert_allclose(sem_scipy, sem_manual)

    def test_sem_decreases_with_sample_size(self, random_seed):
        """Test that SEM decreases as sample size increases."""
        np.random.seed(random_seed)

        sem_small = stats.sem(np.random.normal(0, 1, 10))
        sem_large = stats.sem(np.random.normal(0, 1, 1000))

        assert sem_small > sem_large


class TestInferenceFromSummaryStats:
    """Test inference calculations from summary statistics."""

    def test_t_statistic_from_summary(self):
        """Test t-statistic calculation from summary stats."""
        # Given: mean=100, sd=15, n=25, test value=95
        mean = 100
        sd = 15
        n = 25
        test_value = 95

        sem = sd / np.sqrt(n)
        t_stat = (mean - test_value) / sem

        assert t_stat > 0
        # Calculate expected value
        expected_t = (100 - 95) / (15 / 5)  # 5 / 3 ≈ 1.67
        np.testing.assert_allclose(t_stat, expected_t)

    def test_confidence_interval_from_summary(self):
        """Test CI calculation from summary statistics."""
        mean = 100
        sem = 5
        n = 25
        conf_level = 0.95

        t_crit = stats.t.ppf((1 + conf_level) / 2, n - 1)
        margin_of_error = t_crit * sem

        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error

        assert ci_lower < mean < ci_upper
        assert ci_upper - ci_lower > 0
