import numpy as np
from scipy.optimize import minimize, minimize_scalar, Bounds, LinearConstraint
import pandas as pd


# 🔹 1. Example cost functions
def cost_quadratic(x):
    """Simple quadratic function: f(x) = (x - 3)^2"""
    return (x - 3) ** 2


def cost_nonconvex(x):
    """Non-convex function for testing global vs local optimization"""
    return np.sin(2 * x) + 0.1 * x ** 2


def multi_var_cost(x):
    """2-variable cost function: f(x, y) = (x - 1)^2 + (y - 2.5)^2"""
    return (x[0] - 1)**2 + (x[1] - 2.5)**2


# 🔹 2. Constraint setup examples
def get_linear_constraint():
    """
    Example: x + y ≤ 3 → [1, 1] @ x ≤ 3
    """
    A = np.array([[1, 1]])
    ub = np.array([3])
    lb = np.array([-np.inf])
    return LinearConstraint(A, lb, ub)


def get_bounds_2d():
    """
    Bounds: 0 ≤ x ≤ 5, 0 ≤ y ≤ 5
    """
    return Bounds([0, 0], [5, 5])


# 🔹 3. Wrapper to run minimization
def run_minimization(func, x0, bounds=None, constraints=None, method='trust-constr'):
    result = minimize(
        func,
        x0,
        bounds=bounds,
        constraints=constraints,
        method=method
    )
    return result


# 🔹 4. Scalar minimization wrapper
def run_scalar_minimization(func, bracket=(0, 5), method='Brent'):
    result = minimize_scalar(func, bracket=bracket, method=method)
    return result


# 🔹 5. Create grid and evaluate loss for surface visualization
def evaluate_loss_surface(func, x_range=(0, 5), y_range=(0, 5), steps=50):
    x = np.linspace(*x_range, steps)
    y = np.linspace(*y_range, steps)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(steps):
        for j in range(steps):
            Z[i, j] = func([X[i, j], Y[i, j]])

    return X, Y, Z

