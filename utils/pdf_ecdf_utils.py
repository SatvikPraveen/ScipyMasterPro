from __future__ import annotations

import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

import os


# ✅ 1. Compute PDF from a scipy.stats distribution
def get_pdf(
    data: "np.ndarray",
    dist: object,
    params: tuple | None = None,
    num_points: int = 100,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Compute PDF values for a distribution over the data range.

    Parameters
    ----------
    data : array-like
        Observed data (used to determine x-axis range).
    dist : scipy.stats continuous_rv_generic
        Distribution object.
    params : tuple, optional
        Pre-fitted parameters. If None, the distribution is fitted to data.
    num_points : int, default=100
        Number of evenly spaced x points for the PDF curve.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (x, pdf_values) over [min(data), max(data)].
    """
    x = np.linspace(np.min(data), np.max(data), num_points)
    if params:
        pdf = dist.pdf(x, *params)
    else:
        params = dist.fit(data)
        pdf = dist.pdf(x, *params)
    return x, pdf

# ✅ 2. Compute ECDF manually (raw empirical CDF)
def compute_manual_ecdf(
    data: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Compute the empirical CDF from data.

    Parameters
    ----------
    data : array-like
        Observed data.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (x, y) where x is sorted data and y is cumulative probability in [0, 1].
    """
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

# ✅ 3. Compute ECDF using statsmodels
def compute_statsmodels_ecdf(
    data: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Compute the empirical CDF using the statsmodels ECDF implementation.

    Parameters
    ----------
    data : array-like
        Observed data.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (x, y) where x is sorted data and y is the ECDF evaluated at each point.

    Notes
    -----
    This uses statsmodels.distributions.ECDF which handles ties consistently.
    For most use cases, compute_manual_ecdf gives identical results.
    """
    ecdf_obj = sm.distributions.ECDF(data)
    x = np.sort(data)
    y = ecdf_obj(x)
    return x, y

import matplotlib.pyplot as plt
from scipy.stats import kstest

def plot_pdf_ecdf_overlay(
    data: "np.ndarray",
    dist: object,
    params: tuple | None = None,
    title: str = "",
    annotate_ks: bool = True,
    save_path: str | None = None,
) -> "plt.Figure":
    """
    Plot ECDF with theoretical CDF overlay and optional KS test annotation.

    Parameters
    ----------
    data : array-like
        Observed data.
    dist : scipy.stats continuous_rv_generic
        Distribution object.
    params : tuple, optional
        Pre-fitted parameters. If None, the distribution is fitted to data.
    title : str, default=''
        Plot title.
    annotate_ks : bool, default=True
        If True, annotates the plot with the KS test p-value.
    save_path : str, optional
        File path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x_pdf, y_pdf = get_pdf(data, dist, params)
    x_ecdf, y_ecdf = compute_manual_ecdf(data)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot ECDF
    ax.step(x_ecdf, y_ecdf, where='post', label="Empirical CDF", color='green')

    # Plot CDF from fitted distribution
    if params:
        y_theoretical = dist.cdf(x_pdf, *params)
    else:
        params = dist.fit(data)
        y_theoretical = dist.cdf(x_pdf, *params)

    ax.plot(x_pdf, y_theoretical, label=f"{dist.name.capitalize()} CDF", color='blue')

    # Optional shaded area between ECDF and CDF
    ax.fill_between(x_pdf, y_theoretical, y_ecdf[:len(y_theoretical)],
                    color='orange', alpha=0.2, label='ECDF–CDF gap')

    # KS test annotation
    if annotate_ks:
        ks_stat, p_val = kstest(data, dist.name, args=params)
        ax.text(0.05, 0.1, f"KS p-value: {p_val:.4f}", transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    #plt.show()

    return fig


import matplotlib.pyplot as plt
from scipy.stats import kstest, anderson, shapiro

def plot_enhanced_ecdf_comparison(
    data: "np.ndarray",
    dist_list: list,
    dist_labels: list[str] | None = None,
    annotate_tests: bool = True,
    title: str = "ECDF vs Multiple Distributions",
    save_path: str | None = None,
) -> "plt.Figure":
    """
    Plot ECDF alongside theoretical CDFs for multiple distributions with goodness-of-fit annotations.

    Parameters
    ----------
    data : array-like
        Observed data.
    dist_list : list of scipy.stats continuous_rv_generic
        List of distribution objects to compare against.
    dist_labels : list of str, optional
        Custom labels for each distribution. Defaults to dist.name.
    annotate_tests : bool, default=True
        If True, adds KS, Anderson-Darling, and Shapiro-Wilk test results as text annotation.
    title : str, default='ECDF vs Multiple Distributions'
        Plot title.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x_ecdf, y_ecdf = compute_manual_ecdf(data)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ECDF
    ax.step(x_ecdf, y_ecdf, where='post', label="Empirical CDF", color="black", linewidth=2)

    annotations = []

    for idx, dist in enumerate(dist_list):
        color = plt.cm.tab10(idx)
        label = dist.name if not dist_labels else dist_labels[idx]
        params = dist.fit(data)
        x_pdf, _ = get_pdf(data, dist, params)
        y_theoretical = dist.cdf(x_pdf, *params)

        ax.plot(x_pdf, y_theoretical, label=f"{label} CDF", color=color)

        if annotate_tests:
            ks_stat, ks_p = kstest(data, dist.name, args=params)
            ad_stat = anderson(data, dist=dist.name if dist.name in ['norm', 'expon', 'logistic'] else 'norm').statistic
            shapiro_stat, shapiro_p = shapiro(data[:5000])  # limit Shapiro to 5000

            annotations.append(
                f"{label}:\n"
                f"KS p={ks_p:.4f}, AD={ad_stat:.4f}, Shapiro p={shapiro_p:.4f}"
            )

    # Annotate test results
    if annotate_tests:
        full_note = "\n\n".join(annotations)
        ax.text(1.02, 0.5, full_note, transform=ax.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.tight_layout()
    #plt.show()
    return fig


from scipy.stats import kstest, anderson, shapiro

def run_goodness_of_fit_tests(
    data: "np.ndarray", dist: object
) -> dict[str, float]:
    """
    Run a battery of goodness-of-fit tests against a specified distribution.

    Parameters
    ----------
    data : array-like
        Observed data.
    dist : scipy.stats continuous_rv_generic
        Distribution object to test against (must have .fit and .cdf methods).

    Returns
    -------
    dict of str -> float
        {'KS_stat': float, 'KS_p': float, 'AD_stat': float, 'Shapiro_p': float}

    Notes
    -----
    - KS (Kolmogorov-Smirnov): tests max deviation between ECDF and fitted CDF.
    - Anderson-Darling: weights differences more heavily in the tails.
    - Shapiro-Wilk: specifically tests for normality (not the fitted dist).
    """
    ks_stat, ks_p = kstest(data, dist.cdf, args=dist.fit(data))
    ad_result = anderson(data)
    shapiro_stat, shapiro_p = shapiro(data)
    return {
        "KS_stat": ks_stat, "KS_p": ks_p,
        "AD_stat": ad_result.statistic,
        "Shapiro_p": shapiro_p
    }
