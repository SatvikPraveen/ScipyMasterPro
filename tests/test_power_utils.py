"""
Tests for power_utils module.

Covers Z-test and T-test manual power computation, statsmodels power API,
and Cohen's d effect size calculation.
"""

import numpy as np
import pytest

from utils.power_utils import (
    compute_cohens_d,
    compute_power_t,
    compute_power_z,
    statsmodels_power,
)


class TestComputePowerZ:
    """Test manual Z-test power calculation."""

    def test_power_in_zero_one(self):
        """Power must be a probability in [0, 1]."""
        power = compute_power_z(effect_size=0.5, alpha=0.05, n=100)
        assert 0.0 <= power <= 1.0

    def test_larger_n_gives_more_power(self):
        """Larger sample size should yield higher power."""
        p_small = compute_power_z(effect_size=0.5, alpha=0.05, n=20)
        p_large = compute_power_z(effect_size=0.5, alpha=0.05, n=200)
        assert p_large > p_small

    def test_larger_effect_gives_more_power(self):
        """Larger effect size should yield higher power."""
        p_small = compute_power_z(effect_size=0.2, alpha=0.05, n=50)
        p_large = compute_power_z(effect_size=0.8, alpha=0.05, n=50)
        assert p_large > p_small

    def test_smaller_alpha_gives_less_power(self):
        """Stricter significance level should give less power."""
        p_lenient = compute_power_z(effect_size=0.5, alpha=0.10, n=80)
        p_strict = compute_power_z(effect_size=0.5, alpha=0.01, n=80)
        assert p_lenient > p_strict

    def test_one_vs_two_tailed(self):
        """One-tailed test should have more power than two-tailed."""
        p_two = compute_power_z(effect_size=0.5, alpha=0.05, n=50, two_tailed=True)
        p_one = compute_power_z(effect_size=0.5, alpha=0.05, n=50, two_tailed=False)
        assert p_one > p_two

    def test_high_power_at_large_n(self):
        """Very large n with moderate effect should give power > 0.80."""
        power = compute_power_z(effect_size=0.5, alpha=0.05, n=500)
        assert power > 0.80


class TestComputePowerT:
    """Test manual T-test power calculation."""

    def test_power_in_zero_one(self):
        power = compute_power_t(effect_size=0.5, alpha=0.05, n=50)
        assert 0.0 <= power <= 1.0

    def test_larger_n_gives_more_power(self):
        p_small = compute_power_t(effect_size=0.5, alpha=0.05, n=20)
        p_large = compute_power_t(effect_size=0.5, alpha=0.05, n=200)
        assert p_large > p_small

    def test_larger_effect_gives_more_power(self):
        p_small = compute_power_t(effect_size=0.2, alpha=0.05, n=50)
        p_large = compute_power_t(effect_size=0.8, alpha=0.05, n=50)
        assert p_large > p_small

    def test_t_power_vs_z_power(self):
        """T-test power should be slightly lower than Z-test power (wider t tails)."""
        p_z = compute_power_z(effect_size=0.5, alpha=0.05, n=30)
        p_t = compute_power_t(effect_size=0.5, alpha=0.05, n=30)
        assert p_z >= p_t

    def test_one_vs_two_tailed(self):
        p_two = compute_power_t(effect_size=0.5, alpha=0.05, n=50, two_tailed=True)
        p_one = compute_power_t(effect_size=0.5, alpha=0.05, n=50, two_tailed=False)
        assert p_one > p_two


class TestStatsmodelsPower:
    """Test statsmodels TTestPower integration."""

    def test_power_in_zero_one(self):
        power = statsmodels_power(effect_size=0.5, alpha=0.05, n=50)
        assert 0.0 <= power <= 1.0

    def test_larger_n_gives_more_power(self):
        p_small = statsmodels_power(effect_size=0.5, alpha=0.05, n=20)
        p_large = statsmodels_power(effect_size=0.5, alpha=0.05, n=200)
        assert p_large > p_small

    def test_agrees_with_manual_t_power(self):
        """statsmodels and manual T power should be close for same inputs."""
        n = 80
        d = 0.5
        p_manual = compute_power_t(effect_size=d, alpha=0.05, n=n)
        p_sm = statsmodels_power(effect_size=d, alpha=0.05, n=n)
        assert abs(p_manual - p_sm) < 0.05

    @pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
    def test_alternative_hypothesis_variants(self, alternative):
        power = statsmodels_power(effect_size=0.5, alpha=0.05, n=50, alternative=alternative)
        assert 0.0 <= power <= 1.0


class TestComputeCohensd:
    """Test Cohen's d effect size computation."""

    def test_zero_difference_gives_zero_d(self):
        d = compute_cohens_d(mean1=5.0, mean2=5.0, std_dev=1.0)
        assert d == 0.0

    def test_positive_difference(self):
        d = compute_cohens_d(mean1=6.0, mean2=4.0, std_dev=2.0)
        assert np.isclose(d, 1.0)

    def test_negative_difference(self):
        d = compute_cohens_d(mean1=3.0, mean2=5.0, std_dev=2.0)
        assert np.isclose(d, -1.0)

    def test_small_medium_large_conventions(self):
        """Sanity: small (0.2), medium (0.5), large (0.8) Cohen benchmarks."""
        d_small = compute_cohens_d(1.2, 1.0, 1.0)
        d_medium = compute_cohens_d(1.5, 1.0, 1.0)
        d_large = compute_cohens_d(1.8, 1.0, 1.0)
        assert d_small < d_medium < d_large

    def test_scales_with_std(self):
        """Larger std_dev should give smaller d for the same mean difference."""
        d_narrow = compute_cohens_d(6.0, 4.0, std_dev=1.0)
        d_wide = compute_cohens_d(6.0, 4.0, std_dev=4.0)
        assert d_narrow > d_wide
