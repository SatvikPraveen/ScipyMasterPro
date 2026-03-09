from __future__ import annotations

import numpy as np
from scipy.stats import (rv_discrete,
                         multinomial, 
                         dirichlet                        
    )
import matplotlib.pyplot as plt


def bootstrap_sample(
    data: "np.ndarray", n_iterations: int = 1000, seed: int = 42
) -> "np.ndarray":
    """
    Generate bootstrap distribution of sample means.

    Parameters
    ----------
    data : array-like
        Original sample data.
    n_iterations : int, default=1000
        Number of bootstrap resamples.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of n_iterations bootstrap sample means.
    """
    np.random.seed(seed)
    result = []
    n = len(data)
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=n, replace=True)
        result.append(np.mean(sample))
    return np.array(result)


# Uniform Sampling with/without Replacement
def sample_uniform(
    data: "np.ndarray", n: int, replace: bool = True, seed: int = 42
) -> "np.ndarray":
    """
    Draw a uniform random sample from data.

    Parameters
    ----------
    data : array-like
        Population to sample from.
    n : int
        Number of samples to draw.
    replace : bool, default=True
        Whether to sample with replacement.
    seed : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Array of n sampled values.
    """
    np.random.seed(seed)
    return np.random.choice(data, size=n, replace=replace)


# Stratified Sampling (Basic Category Proportions)
def stratified_sample(
    df: "pd.DataFrame", stratify_col: str, frac: float = 0.1, seed: int = 42
) -> "pd.DataFrame":
    """
    Draw a proportionally stratified sample from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    stratify_col : str
        Column to stratify by (categorical grouping variable).
    frac : float, default=0.1
        Fraction of each group to sample.
    seed : int, default=42
        Random seed.

    Returns
    -------
    pd.DataFrame
        Stratified sample preserving group proportions.
    """
    return (
        df.groupby(stratify_col, group_keys=False)
        .apply(lambda x: x.sample(frac=frac, random_state=seed))
    )


# Weighted Sampling
def weighted_sample(
    data: "np.ndarray", weights: "np.ndarray", n: int, replace: bool = True, seed: int = 42
) -> "np.ndarray":
    """
    Draw a weighted random sample from data.

    Parameters
    ----------
    data : array-like
        Values to sample from.
    weights : array-like
        Probability weights for each element (must sum to 1).
    n : int
        Number of samples to draw.
    replace : bool, default=True
        Whether to sample with replacement.
    seed : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Array of n sampled values.
    """
    np.random.seed(seed)
    return np.random.choice(data, size=n, replace=replace, p=weights)


# Multinomial & Dirichlet Sampling
def draw_multinomial_sample(
    n: int, probs: list[float], size: int = 1, seed: int = 42
) -> "np.ndarray":
    """
    Draw samples from a multinomial distribution.

    Parameters
    ----------
    n : int
        Number of trials per draw.
    probs : array-like
        Probability of each category (must sum to 1).
    size : int, default=1
        Number of independent draws.
    seed : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Array of shape (size, len(probs)) with trial counts per category.
    """
    np.random.seed(seed)
    return multinomial.rvs(n=n, p=probs, size=size)

def draw_dirichlet_sample(
    alpha: list[float], size: int = 1, seed: int = 42
) -> "np.ndarray":
    """
    Draw samples from a Dirichlet distribution.

    Parameters
    ----------
    alpha : array-like
        Concentration parameters (positive values). Length determines number of categories.
    size : int, default=1
        Number of random samples to draw.
    seed : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Array of shape (size, len(alpha)) where each row sums to 1.
    """
    np.random.seed(seed)
    return dirichlet.rvs(alpha=alpha, size=size)


# Custom Discrete Distribution Sampling (rv_discrete)
def sample_custom_discrete(
    support_vals: list[int], probs: list[float], size: int = 1000, seed: int = 42
) -> "np.ndarray":
    """
    Sample from a custom discrete probability distribution.

    Parameters
    ----------
    support_vals : array-like of int
        Discrete support values (e.g., [0, 1, 2, 3]).
    probs : array-like of float
        Probability for each value (must sum to 1).
    size : int, default=1000
        Number of samples to draw.
    seed : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Array of sampled values from the custom distribution.
    """
    np.random.seed(seed)
    custom_rv = rv_discrete(name='custom', values=(support_vals, probs))
    return custom_rv.rvs(size=size)


# Manual Resampling with Replacement (Bootstrap Base)
def resample_with_replacement(
    data: "np.ndarray", n_samples: int = 1000, seed: int = 42
) -> "np.ndarray":
    """
    Resample from data with replacement (bootstrap base operation).

    Parameters
    ----------
    data : array-like
        Original data to resample from.
    n_samples : int, default=1000
        Number of samples in the resample.
    seed : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Resampled array of length n_samples.
    """
    np.random.seed(seed)
    return np.random.choice(data, size=n_samples, replace=True)


# Construct ECDF (for visual comparison)
def compute_ecdf(
    data: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Compute the empirical cumulative distribution function.

    Parameters
    ----------
    data : array-like
        Observed data.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (x, y) where x is sorted data and y is cumulative probability [1/n ... 1].
    """
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y


# -------------------------------------------
# 🔁 BOOTSTRAP SIMULATION UTILITIES
# -------------------------------------------

# Basic Bootstrap Resample Generator
def bootstrap_statistic(
    data: "np.ndarray",
    stat_func: callable = np.mean,
    n_resamples: int = 1000,
    seed: int = 42,
) -> "np.ndarray":
    """
    Compute a bootstrap distribution for any statistic.

    Parameters
    ----------
    data : array-like
        Original sample data.
    stat_func : callable, default=np.mean
        Statistic function to apply to each resample (e.g., np.median, np.std).
    n_resamples : int, default=1000
        Number of bootstrap resamples.
    seed : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Bootstrap distribution of the statistic (length = n_resamples).
    """
    np.random.seed(seed)
    boot_stats = []
    n = len(data)
    for _ in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))
    return np.array(boot_stats)


# Confidence Interval Calculation (Percentile Method)
def compute_bootstrap_ci(
    boot_stats: "np.ndarray", ci: float = 95
) -> tuple[float, float]:
    """
    Compute percentile-method confidence interval from bootstrap distribution.

    Parameters
    ----------
    boot_stats : array-like
        Bootstrap distribution of a statistic.
    ci : float, default=95
        Confidence level as a percentage (e.g., 95 for 95% CI).

    Returns
    -------
    tuple of (float, float)
        (lower, upper) confidence interval bounds.
    """
    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return lower, upper


# Bootstrap Summary Report
def summarize_bootstrap(
    estimates: "np.ndarray",
    original_stat: float | None = None,
    ci: float = 95,
) -> dict[str, float]:
    """
    Generate a summary report of a bootstrap distribution.

    Parameters
    ----------
    estimates : array-like
        Bootstrap distribution of a statistic.
    original_stat : float, optional
        The original (observed) statistic from the full sample.
    ci : float, default=95
        Confidence level as a percentage.

    Returns
    -------
    dict of str -> float
        Keys: 'mean', 'std', 'ci_lower', 'ci_upper', and optionally 'original'.
    """
    lower, upper = compute_bootstrap_ci(estimates, ci)
    summary = {
        "mean": np.mean(estimates),
        "std": np.std(estimates),
        "ci_lower": lower,
        "ci_upper": upper,
    }
    if original_stat is not None:
        summary["original"] = original_stat
    return summary


# Bootstrap Distribution with CI
def plot_bootstrap_distribution(
    estimates: "np.ndarray",
    ci_bounds: tuple[float, float] | None = None,
    title: str = "Bootstrap Distribution",
    bins: int = 30,
) -> "plt.Figure":
    """
    Plot a histogram of bootstrap estimates with optional CI bounds.

    Parameters
    ----------
    estimates : array-like
        Bootstrap distribution values.
    ci_bounds : tuple of (float, float), optional
        (lower, upper) bounds to annotate as vertical lines.
    title : str, default='Bootstrap Distribution'
        Plot title.
    bins : int, default=30
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(estimates, bins=bins, edgecolor="k", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Estimate")
    ax.set_ylabel("Frequency")
    
    if ci_bounds:
        ax.axvline(ci_bounds[0], color="red", linestyle="--", label="CI Lower")
        ax.axvline(ci_bounds[1], color="red", linestyle="--", label="CI Upper")
        ax.legend()

    return fig


# Mahalanobis Distance Calculator
from scipy.spatial.distance import mahalanobis

def compute_mahalanobis_distances(data):
    """
    Computes Mahalanobis distance of each row from the multivariate mean.

    Args:
        data (pd.DataFrame): Input data (n x d)

    Returns:
        np.ndarray: Mahalanobis distances (n,)
    """
    cov = np.cov(data.T)
    cov_inv = np.linalg.inv(cov)
    mean_vec = data.mean(axis=0).values

    distances = data.apply(lambda row: mahalanobis(row, mean_vec, cov_inv), axis=1)
    return distances


# Chi-Square Test for Mahalanobis Distances
from scipy.stats import chi2

def evaluate_mahalanobis_outliers(distances, df_dim, alpha=0.01):
    """
    Compares Mahalanobis distances to Chi-Square threshold for outlier detection.

    Args:
        distances (np.ndarray): Mahalanobis distances
        df_dim (int): Degrees of freedom (number of features)
        alpha (float): Significance level

    Returns:
        pd.Series: Boolean mask of outliers
    """
    threshold = chi2.ppf(1 - alpha, df_dim)
    return distances > threshold

