import numpy as np
from scipy.stats import (rv_discrete,
                         multinomial, 
                         dirichlet                        
    )
import matplotlib.pyplot as plt


def bootstrap_sample(data, n_iterations=1000, seed=42):
    np.random.seed(seed)
    stats = []
    n = len(data)
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=n, replace=True)
        stats.append(np.mean(sample))
    return np.array(stats)


# Uniform Sampling with/without Replacement
def sample_uniform(data, n, replace=True, seed=42):
    np.random.seed(seed)
    return np.random.choice(data, size=n, replace=replace)


# Stratified Sampling (Basic Category Proportions)
def stratified_sample(df, stratify_col, frac=0.1, seed=42):
    return (
        df.groupby(stratify_col, group_keys=False)
        .apply(lambda x: x.sample(frac=frac, random_state=seed))
    )


# Weighted Sampling
def weighted_sample(data, weights, n, replace=True, seed=42):
    np.random.seed(seed)
    return np.random.choice(data, size=n, replace=replace, p=weights)


# Multinomial & Dirichlet Sampling
def draw_multinomial_sample(n, probs, size=1, seed=42):
    np.random.seed(seed)
    return multinomial.rvs(n=n, p=probs, size=size)

def draw_dirichlet_sample(alpha, size=1, seed=42):
    np.random.seed(seed)
    return dirichlet.rvs(alpha=alpha, size=size)


# Custom Discrete Distribution Sampling (rv_discrete)
def sample_custom_discrete(support_vals, probs, size=1000, seed=42):
    np.random.seed(seed)
    custom_rv = rv_discrete(name='custom', values=(support_vals, probs))
    return custom_rv.rvs(size=size)


# Manual Resampling with Replacement (Bootstrap Base)
def resample_with_replacement(data, n_samples=1000, seed=42):
    np.random.seed(seed)
    return np.random.choice(data, size=n_samples, replace=True)


# Construct ECDF (for visual comparison)
def compute_ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y


# -------------------------------------------
# 🔁 BOOTSTRAP SIMULATION UTILITIES
# -------------------------------------------

# Basic Bootstrap Resample Generator
def bootstrap_statistic(data, stat_func=np.mean, n_resamples=1000, seed=42):
    np.random.seed(seed)
    boot_stats = []
    n = len(data)
    for _ in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))
    return np.array(boot_stats)


# Confidence Interval Calculation (Percentile Method)
def compute_bootstrap_ci(boot_stats, ci=95):
    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return lower, upper


# Bootstrap Summary Report
def summarize_bootstrap(estimates, original_stat=None, ci=95):
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
def plot_bootstrap_distribution(estimates, ci_bounds=None, title="Bootstrap Distribution", bins=30):
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

