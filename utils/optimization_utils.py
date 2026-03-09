from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize, minimize_scalar


# 🔹 1. Example cost functions
def cost_quadratic(x: float) -> float:
    """Simple quadratic function: f(x) = (x - 3)^2"""
    return (x - 3) ** 2


def cost_nonconvex(x: float) -> float:
    """Non-convex function for testing global vs local optimization."""
    return np.sin(2 * x) + 0.1 * x**2


def multi_var_cost(x: "np.ndarray") -> float:
    """
    2-variable cost function: f(x, y) = (x - 1)^2 + (y - 2.5)^2

    Parameters
    ----------
    x : array-like of length 2
        Input vector [x0, x1].

    Returns
    -------
    float
        Scalar cost value.
    """
    return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2


# 🔹 2. Constraint setup examples
def get_linear_constraint() -> LinearConstraint:
    """
    Build a linear inequality constraint: x + y <= 3.

    Returns
    -------
    scipy.optimize.LinearConstraint
        Constraint object compatible with scipy.optimize.minimize.
    """
    A = np.array([[1, 1]])
    ub = np.array([3])
    lb = np.array([-np.inf])
    return LinearConstraint(A, lb, ub)


def get_bounds_2d() -> Bounds:
    """
    Create box bounds for a 2D optimization problem: 0 <= x,y <= 5.

    Returns
    -------
    scipy.optimize.Bounds
        Bounds object compatible with scipy.optimize.minimize.
    """
    return Bounds([0, 0], [5, 5])


# 🔹 3. Wrapper to run minimization
def run_minimization(
    func: callable,
    x0: "np.ndarray",
    bounds: Bounds | None = None,
    constraints: LinearConstraint | dict | list | None = None,
    method: str = "trust-constr",
) -> object:
    """
    Run multivariate minimization using scipy.optimize.minimize.

    Parameters
    ----------
    func : callable
        Objective function to minimize, taking a 1D array and returning a scalar.
    x0 : array-like
        Initial guess for the solution.
    bounds : scipy.optimize.Bounds, optional
        Variable bounds.
    constraints : constraint or list of constraints, optional
        Linear or nonlinear constraints.
    method : str, default='trust-constr'
        Optimization algorithm. Common choices: 'SLSQP', 'L-BFGS-B', 'trust-constr'.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result object with attributes: x (solution), fun (min value), success, message.
    """
    result = minimize(func, x0, bounds=bounds, constraints=constraints, method=method)
    return result


# 🔹 4. Scalar minimization wrapper
def run_scalar_minimization(
    func: callable,
    bracket: tuple[float, float] = (0, 5),
    method: str = "Brent",
) -> object:
    """
    Run scalar (1D) minimization using scipy.optimize.minimize_scalar.

    Parameters
    ----------
    func : callable
        Single-variable objective function.
    bracket : tuple of (float, float), default=(0, 5)
        Bracket interval for the search.
    method : str, default='Brent'
        Scalar minimization method: 'Brent', 'Bounded', or 'Golden'.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result with attributes: x (solution), fun (min value).
    """
    result = minimize_scalar(func, bracket=bracket, method=method)
    return result


# 🔹 5. Create grid and evaluate loss for surface visualization
def evaluate_loss_surface(
    func: callable,
    x_range: tuple[float, float] = (0, 5),
    y_range: tuple[float, float] = (0, 5),
    steps: int = 50,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Evaluate a 2D loss function over a grid for surface visualization.

    Parameters
    ----------
    func : callable
        Two-variable objective function taking a list/array [x, y].
    x_range : tuple of (float, float), default=(0, 5)
        Range for the x-axis grid.
    y_range : tuple of (float, float), default=(0, 5)
        Range for the y-axis grid.
    steps : int, default=50
        Number of grid points along each axis.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        (X, Y, Z) meshgrid arrays where Z[i,j] = func([X[i,j], Y[i,j]]).

    Examples
    --------
    >>> X, Y, Z = evaluate_loss_surface(multi_var_cost, x_range=(0,3), y_range=(0,5))
    >>> Z.shape
    (50, 50)
    """
    x = np.linspace(*x_range, steps)
    y = np.linspace(*y_range, steps)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(steps):
        for j in range(steps):
            Z[i, j] = func([X[i, j], Y[i, j]])

    return X, Y, Z
