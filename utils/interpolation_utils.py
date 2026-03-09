# utils/interpolation_utils.py

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline, griddata, Rbf
from scipy.optimize import curve_fit

# --------------------------
# 1. Interpolation Functions
# --------------------------

def linear_interpolate(
    x: "np.ndarray", y: "np.ndarray"
) -> object:
    """
    Create a linear piecewise interpolation function.

    Parameters
    ----------
    x : array-like
        1D array of x-coordinates (must be monotonically increasing).
    y : array-like
        1D array of y-values at each x point.

    Returns
    -------
    scipy.interpolate.interp1d
        Callable interpolator. Raises ValueError outside the x range.
    """
    return interp1d(x, y, kind='linear')

def cubic_interpolate(
    x: "np.ndarray", y: "np.ndarray"
) -> object:
    """
    Create a cubic spline interpolation function.

    Parameters
    ----------
    x : array-like
        1D array of x-coordinates (must be monotonically increasing).
    y : array-like
        1D array of y-values at each x point.

    Returns
    -------
    scipy.interpolate.interp1d
        Callable cubic interpolator. Smoother than linear but may oscillate.
    """
    return interp1d(x, y, kind='cubic')

def spline_interpolate(
    x: "np.ndarray", y: "np.ndarray", s: float = 0
) -> object:
    """
    Fit a univariate smoothing spline to data.

    Parameters
    ----------
    x : array-like
        1D array of x-coordinates.
    y : array-like
        1D array of y-values.
    s : float, default=0
        Smoothing factor. s=0 means the spline interpolates exactly.
        Larger s allows more deviation from data points for smoother fit.

    Returns
    -------
    scipy.interpolate.UnivariateSpline
        Callable spline object.
    """
    return UnivariateSpline(x, y, s=s)

def multivariate_interpolate(points, values, xi, method='linear'):
    """
    `points`: shape (n, 2) → input x, y
    `values`: shape (n,) → values at those points
    `xi`: query points for interpolation
    """
    return griddata(points, values, xi, method=method)

# --------------------------
# 2. Curve Fitting Utilities
# --------------------------

def fit_curve(func, xdata, ydata, p0=None, maxfev=5000, bounds=(-np.inf, np.inf)):
    """
    Fits a parametric curve to data using scipy's curve_fit.
    Dynamically chooses p0 if not provided. Handles convergence errors.
    """
    if p0 is None:
        if func.__name__ == "exponential_model":
            p0 = (ydata.max(), -0.5, ydata.min())
        elif func.__name__ == "gaussian_model":
            p0 = (np.mean(xdata), np.std(xdata), np.max(ydata))
        else:
            p0 = np.ones(3)  # generic fallback

    try:
        popt, pcov = curve_fit(func, xdata, ydata, p0=p0, maxfev=maxfev, bounds=bounds)
    except RuntimeError:
        print(f"⚠️ Warning: {func.__name__} fit did not converge. Returning initial guess.")
        popt, pcov = p0, np.zeros((len(p0), len(p0)))
    return popt, pcov



def exponential_model(x: "np.ndarray", a: float, b: float, c: float) -> "np.ndarray":
    """
    Exponential model: f(x) = a * exp(b * x) + c.

    Parameters
    ----------
    x : array-like
        Input values.
    a : float
        Amplitude scaling factor.
    b : float
        Exponential growth/decay rate.
    c : float
        Vertical offset.

    Returns
    -------
    np.ndarray
        Computed values.
    """
    return a * np.exp(b * x) + c


def gaussian_model(x: "np.ndarray", mu: float, sigma: float, A: float) -> "np.ndarray":
    """
    Gaussian (normal bell curve) model: f(x) = A * exp(-(x - mu)^2 / (2 * sigma^2)).

    Parameters
    ----------
    x : array-like
        Input values.
    mu : float
        Center (mean) of the Gaussian.
    sigma : float
        Width (standard deviation).
    A : float
        Peak amplitude.

    Returns
    -------
    np.ndarray
        Computed values.
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def safe_gaussian_fit(xdata, ydata, maxfev=10000):
    """
    Performs a robust Gaussian curve fit with safe initial guesses and bounds.
    Returns popt, pcov.
    """
    p0 = (np.mean(xdata), np.std(xdata), np.max(ydata))
    bounds = ([xdata.min(), 0, 0], [xdata.max(), np.inf, np.inf])
    try:
        popt, pcov = curve_fit(
            gaussian_model,
            xdata,
            ydata,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev
        )
    except RuntimeError:
        print("⚠️ Warning: Gaussian fit did not converge. Using initial guess.")
        popt, pcov = p0, np.zeros((len(p0), len(p0)))
    return popt, pcov


def interpolate_2d(x, y, z, method="linear", grid_res=100):
    """
    Performs 2D interpolation over scattered data points.

    Parameters
    ----------
    x, y : array-like
        1D coordinates of known data points.
    z : array-like
        Values at data points (same length as x and y).
    method : str, default='linear'
        Interpolation method ('linear', 'nearest', 'cubic').
    grid_res : int, default=100
        Resolution of the output grid.

    Returns
    -------
    grid_x, grid_y, grid_z : ndarray
        Interpolated grid coordinates and interpolated values.
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(x), max(x), grid_res),
        np.linspace(min(y), max(y), grid_res)
    )

    grid_z = griddata(
        points=(x, y),
        values=z,
        xi=(grid_x, grid_y),
        method=method
    )

    return grid_x, grid_y, grid_z


def rbf_interpolation(x, y, z, function='multiquadric', grid_res=100):
    """
    Performs RBF (Radial Basis Function) interpolation for 2D data.

    Parameters
    ----------
    x, y : array-like
        1D coordinates of known data points.
    z : array-like
        Values at data points (same length as x and y).
    function : str, default='multiquadric'
        Type of RBF kernel ('multiquadric', 'inverse', 'gaussian', etc.)
    grid_res : int, default=100
        Resolution of the output grid.

    Returns
    -------
    grid_z : ndarray
        Interpolated Z values on the grid.
    """
    rbf = Rbf(x, y, z, function=function)
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(x), max(x), grid_res),
        np.linspace(min(y), max(y), grid_res)
    )
    grid_z = rbf(grid_x, grid_y)
    return grid_z
