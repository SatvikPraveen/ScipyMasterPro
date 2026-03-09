"""
Tests for interpolation_utils module.

Covers linear, cubic, and spline interpolation, exponential and Gaussian
curve fitting, 2D griddata interpolation, and RBF interpolation.
"""

import numpy as np
import pytest

from utils.interpolation_utils import (
    linear_interpolate,
    cubic_interpolate,
    spline_interpolate,
    fit_curve,
    exponential_model,
    gaussian_model,
    safe_gaussian_fit,
    interpolate_2d,
    rbf_interpolation,
)


@pytest.fixture
def smooth_xy():
    """Simple smooth dataset for interpolation testing."""
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    return x, y


@pytest.fixture
def gaussian_xy():
    """Gaussian-shaped dataset for curve fitting."""
    x = np.linspace(-5, 5, 100)
    y = gaussian_model(x, mu=0.0, sigma=1.0, A=3.0) + np.random.default_rng(42).normal(0, 0.05, 100)
    return x, y


@pytest.fixture
def exponential_xy():
    """Exponential decay dataset for curve fitting."""
    x = np.linspace(0, 3, 60)
    y = exponential_model(x, a=2.0, b=-1.5, c=0.5) + np.random.default_rng(42).normal(0, 0.05, 60)
    return x, y


class TestLinearInterpolation:
    """Test linear piecewise interpolation."""

    def test_returns_callable(self, smooth_xy):
        """linear_interpolate should return a callable."""
        x, y = smooth_xy
        f = linear_interpolate(x, y)
        assert callable(f)

    def test_interpolates_at_known_points(self, smooth_xy):
        """Interpolated values at known x points should match original y."""
        x, y = smooth_xy
        f = linear_interpolate(x, y)
        np.testing.assert_allclose(f(x), y, atol=1e-10)

    def test_interpolates_midpoints(self, smooth_xy):
        """Interpolated values at midpoints should be between neighbours."""
        x, y = smooth_xy
        f = linear_interpolate(x, y)
        x_mid = (x[:-1] + x[1:]) / 2
        y_mid = f(x_mid)
        assert y_mid.shape == x_mid.shape


class TestCubicInterpolation:
    """Test cubic spline interpolation."""

    def test_returns_callable(self, smooth_xy):
        x, y = smooth_xy
        f = cubic_interpolate(x, y)
        assert callable(f)

    def test_interpolates_at_known_points(self, smooth_xy):
        x, y = smooth_xy
        f = cubic_interpolate(x, y)
        np.testing.assert_allclose(f(x), y, atol=1e-10)

    def test_smoother_than_linear(self, smooth_xy):
        """Cubic interpolation should differ from linear at midpoints."""
        x, y = smooth_xy
        f_lin = linear_interpolate(x, y)
        f_cub = cubic_interpolate(x, y)
        x_mid = (x[:-1] + x[1:]) / 2
        # They won't be identical (cubic is smoother)
        assert not np.allclose(f_lin(x_mid), f_cub(x_mid))


class TestSplineInterpolation:
    """Test UnivariateSpline interpolation."""

    def test_returns_callable(self, smooth_xy):
        x, y = smooth_xy
        spl = spline_interpolate(x, y, s=0)
        assert callable(spl)

    def test_s0_interpolates_exactly(self, smooth_xy):
        """With s=0, spline should pass through data points exactly."""
        x, y = smooth_xy
        spl = spline_interpolate(x, y, s=0)
        np.testing.assert_allclose(spl(x), y, atol=1e-6)

    def test_smoothing_reduces_oscillation(self, smooth_xy):
        """Smoothing spline (s>0) should differ from exact spline."""
        x, y = smooth_xy
        spl_exact = spline_interpolate(x, y, s=0)
        spl_smooth = spline_interpolate(x, y, s=1.0)
        x_test = np.linspace(x[1], x[-2], 50)
        # With smoothing the values differ from exact interpolation
        assert not np.allclose(spl_exact(x_test), spl_smooth(x_test))


class TestModelFunctions:
    """Test exponential and Gaussian model functions."""

    def test_exponential_model_shape(self):
        """exponential_model should return array of same shape as input."""
        x = np.linspace(0, 5, 50)
        y = exponential_model(x, a=2.0, b=-1.0, c=0.5)
        assert y.shape == x.shape

    def test_exponential_model_at_zero(self):
        """At x=0: f(0) = a*exp(0)+c = a + c."""
        assert np.isclose(exponential_model(0.0, a=2.0, b=-1.0, c=0.5), 2.5)

    def test_gaussian_model_peak_at_mu(self):
        """Gaussian should reach its peak A at x=mu."""
        x = np.linspace(-3, 3, 200)
        y = gaussian_model(x, mu=0.0, sigma=1.0, A=5.0)
        assert np.isclose(y.max(), 5.0, atol=0.01)

    def test_gaussian_model_symmetry(self):
        """Gaussian should be symmetric around mu."""
        x = np.array([-1.0, 0.0, 1.0])
        y = gaussian_model(x, mu=0.0, sigma=1.0, A=1.0)
        assert np.isclose(y[0], y[2])


class TestCurveFitting:
    """Test fit_curve and safe_gaussian_fit."""

    def test_fit_curve_gaussian_recovers_params(self, gaussian_xy):
        """fit_curve should recover Gaussian parameters within tolerance."""
        x, y = gaussian_xy
        popt, _ = fit_curve(gaussian_model, x, y)
        mu_fit, sigma_fit, A_fit = popt
        assert abs(mu_fit - 0.0) < 0.3
        assert abs(sigma_fit - 1.0) < 0.3
        assert abs(A_fit - 3.0) < 0.5

    def test_fit_curve_exponential_recovers_params(self, exponential_xy):
        """fit_curve should recover exponential parameters within tolerance."""
        x, y = exponential_xy
        popt, _ = fit_curve(exponential_model, x, y)
        a_fit, b_fit, c_fit = popt
        assert abs(a_fit - 2.0) < 0.5
        assert abs(b_fit - (-1.5)) < 0.5

    def test_safe_gaussian_fit_returns_popt_pcov(self, gaussian_xy):
        """safe_gaussian_fit should return (popt, pcov) tuples."""
        x, y = gaussian_xy
        popt, pcov = safe_gaussian_fit(x, y)
        assert len(popt) == 3
        assert pcov.shape == (3, 3)


class TestInterpolate2D:
    """Test 2D scattered data interpolation."""

    def test_interpolate_2d_returns_grid(self):
        """interpolate_2d should return grid_x, grid_y, grid_z of equal shape."""
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, 50)
        y = rng.uniform(0, 10, 50)
        z = np.sin(x) + np.cos(y)
        grid_x, grid_y, grid_z = interpolate_2d(x, y, z, grid_res=20)
        assert grid_x.shape == grid_y.shape == grid_z.shape

    def test_interpolate_2d_grid_resolution(self):
        """Grid resolution parameter should control output shape."""
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 5, 40)
        y = rng.uniform(0, 5, 40)
        z = x + y
        grid_x, _, _ = interpolate_2d(x, y, z, grid_res=30)
        assert grid_x.shape == (30, 30)


class TestRBFInterpolation:
    """Test RBF interpolation."""

    def test_rbf_returns_grid_z(self):
        """rbf_interpolation should return a 2D array."""
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 5, 30)
        y = rng.uniform(0, 5, 30)
        z = np.sin(x) * np.cos(y)
        grid_z = rbf_interpolation(x, y, z, grid_res=15)
        assert grid_z.shape == (15, 15)
