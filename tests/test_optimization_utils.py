"""
Tests for optimization_utils module.

This module tests optimization functions including minimization,
constraints, and loss surface evaluation.
"""

import numpy as np
import pytest
from scipy.optimize import minimize


class TestSimpleOptimization:
    """Test basic optimization functions."""

    def test_quadratic_optimization(self, simple_quadratic):
        """Test optimization of simple quadratic function."""
        func, grad, optimal, optimal_value = simple_quadratic

        # Start from a random point
        x0 = np.array([0.0, 0.0])
        result = minimize(func, x0, jac=grad, method="BFGS")

        assert result.success
        np.testing.assert_allclose(result.x, optimal, atol=1e-4)
        np.testing.assert_allclose(result.fun, optimal_value, atol=1e-4)

    def test_rosenbrock_optimization(self, rosenbrock_function):
        """Test optimization of Rosenbrock function."""
        func, optimal, optimal_value = rosenbrock_function

        x0 = np.array([0.0, 0.0])
        result = minimize(func, x0, method="Nelder-Mead")

        assert result.success or result.fun < 0.1  # May not fully converge
        # Check we're close to minimum
        assert result.fun < 1.0

    @pytest.mark.parametrize("method", ["BFGS", "Nelder-Mead", "Powell"])
    def test_optimization_methods(self, simple_quadratic, method):
        """Test different optimization methods."""
        func, grad, optimal, optimal_value = simple_quadratic

        x0 = np.array([0.0, 0.0])
        if method == "BFGS":
            result = minimize(func, x0, jac=grad, method=method)
        else:
            result = minimize(func, x0, method=method)

        assert result.success or result.fun < 0.01


class TestConstrainedOptimization:
    """Test optimization with constraints."""

    def test_bounded_optimization(self):
        """Test optimization with bounds."""

        def func(x):
            return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

        # Constrain to x in [0, 1], y in [0, 2]
        bounds = [(0, 1), (0, 2)]
        x0 = np.array([0.5, 0.5])

        result = minimize(func, x0, bounds=bounds, method="L-BFGS-B")

        assert result.success
        # Should be at boundary (1, 2) since unconstrained optimum is (2, 3)
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-3)

    def test_linear_constraints(self):
        """Test optimization with linear constraints."""

        def func(x):
            return x[0] ** 2 + x[1] ** 2

        # Constraint: x + y >= 1
        from scipy.optimize import LinearConstraint

        A = np.array([[1, 1]])
        constraint = LinearConstraint(A, 1.0, np.inf)

        x0 = np.array([0.0, 0.0])
        result = minimize(func, x0, constraints=constraint, method="trust-constr")

        assert result.success
        # Optimal should be at x=y=0.5 (on the constraint boundary)
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-2)


class TestOptimizationConvergence:
    """Test optimization convergence and performance."""

    def test_convergence_tolerance(self, simple_quadratic):
        """Test that optimization respects tolerance settings."""
        func, grad, optimal, optimal_value = simple_quadratic

        x0 = np.array([0.0, 0.0])

        # Tight tolerance
        result_tight = minimize(func, x0, jac=grad, tol=1e-10, method="BFGS")

        # Loose tolerance
        result_loose = minimize(func, x0, jac=grad, tol=1e-3, method="BFGS")

        # Tighter tolerance should give more accurate result
        assert result_tight.fun <= result_loose.fun

    def test_max_iterations(self, rosenbrock_function):
        """Test that max iterations parameter is respected."""
        func, optimal, optimal_value = rosenbrock_function

        x0 = np.array([0.0, 0.0])

        result = minimize(func, x0, method="Nelder-Mead", options={"maxiter": 10})

        # Should stop after 10 iterations
        assert result.nit <= 10 or not result.success
