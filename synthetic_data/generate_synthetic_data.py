"""
📦 Synthetic Data Generator for SciPyMasterPro

This script generates reusable synthetic datasets for each module in the SciPyMasterPro project.
All data is created to focus on statistical clarity, simulation control, and functional use with SciPy.

Author: Satvik Praveen
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, expon, gamma, multivariate_normal, norm, poisson, skewnorm

# Export directory lives next to this file, so the script works from any working directory.
EXPORT_DIR = Path(__file__).resolve().parent / "exports"


# 🔹 1. Basic descriptive statistics (normal + skewed)
def generate_normal_skewed(seed=42, n=1000):
    np.random.seed(seed)
    normal_data = np.random.normal(loc=50, scale=10, size=n)
    skewed_data = skewnorm.rvs(a=10, loc=60, scale=15, size=n)
    df = pd.DataFrame({"normal": normal_data, "skewed": skewed_data})
    return df


# 🔹 2. Distribution fitting (beta, gamma, exponential, lognormal)
def generate_mixed_distributions(seed=42, n=1000):
    np.random.seed(seed)
    data = pd.DataFrame(
        {
            "beta": beta.rvs(a=2, b=5, size=n),
            "gamma": gamma.rvs(a=2, scale=2, size=n),
            "exponential": expon.rvs(scale=1.5, size=n),
            "lognorm": np.random.lognormal(mean=0.8, sigma=0.4, size=n),
            "normal": norm.rvs(loc=70, scale=12, size=n),
            "poisson": poisson.rvs(mu=5, size=n),
        }
    )
    return data


# 🔹 3. Multivariate Gaussian for Mahalanobis and PCA
def generate_multivariate_gaussian(n=500, seed=42):
    np.random.seed(seed)
    mean = [0, 1, 2]
    cov = [[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]]
    data = multivariate_normal.rvs(mean=mean, cov=cov, size=n)
    return pd.DataFrame(data, columns=["X1", "X2", "X3"])


# 🔹 4. Data for optimization (e.g., cost function minimization)
def generate_sample_for_optimization(seed=42, n=100):
    np.random.seed(seed)
    x = np.linspace(0, 10, n)
    y = 3 * np.sin(x) + 0.5 * x + np.random.normal(0, 0.5, size=n)
    return pd.DataFrame({"x": x, "y": y})


# 🔹 5. Curve fitting with noise
def generate_noisy_curve_fitting_data(seed=42, n=150):
    np.random.seed(seed)
    x = np.linspace(0, 5, n)
    y = 2 * np.exp(-0.5 * x) + np.random.normal(scale=0.05, size=n)
    return pd.DataFrame({"x": x, "y": y})


# 🔹 6. Categorical distribution (for sampling & chi-square)
def generate_categorical_counts(seed=42):
    np.random.seed(seed)
    categories = ["A", "B", "C", "D"]
    probs = [0.3, 0.4, 0.2, 0.1]
    sampled = np.random.choice(categories, p=probs, size=1000)
    counts = pd.Series(sampled).value_counts().sort_index()
    return counts


# 🔹 7. Poisson for discrete sampling
def generate_poisson_data(seed=42, n=500, lam=4):
    np.random.seed(seed)
    return pd.DataFrame({"counts": poisson.rvs(mu=lam, size=n)})


# 🔹 8. Grouped Continuous for Violin/Box Comparisons
def generate_grouped_continuous(seed=42, n_per_group=300):
    np.random.seed(seed)
    groups = ["A", "B", "C"]
    data: list[tuple[str, float]] = []

    for group in groups:
        if group == "A":
            vals = np.random.normal(loc=50, scale=5, size=n_per_group)
        elif group == "B":
            vals = np.random.normal(loc=60, scale=7, size=n_per_group)
        else:
            vals = np.random.normal(loc=55, scale=4, size=n_per_group)
        data.extend(zip([group] * n_per_group, vals))

    return pd.DataFrame(data, columns=["group", "value"])


# 🔹 9. Bootstrap-specific dataset (small sample from normal distribution)
def generate_bootstrap_sample_data(seed=42, n=200):
    """
    Generates a small synthetic dataset for bootstrap demonstrations.
    Ideal for computing mean, median, and confidence intervals.
    """
    np.random.seed(seed)
    data = np.random.normal(loc=100, scale=15, size=n)
    return pd.DataFrame({"normal_sample": data})


# Save all to destined location in .csv format
def export_all_datasets(export_dir: Path | str = EXPORT_DIR) -> Path:
    """Generate every synthetic dataset and write it as CSV into ``export_dir``."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving all datasets to {export_dir}/")

    generate_normal_skewed().to_csv(export_dir / "normal_skewed.csv", index=False)
    generate_mixed_distributions().to_csv(export_dir / "mixed_distributions.csv", index=False)
    generate_multivariate_gaussian().to_csv(export_dir / "multivariate_gaussian.csv", index=False)
    generate_sample_for_optimization().to_csv(
        export_dir / "sample_for_optimization.csv", index=False
    )
    generate_noisy_curve_fitting_data().to_csv(export_dir / "curve_fitting_data.csv", index=False)
    generate_poisson_data().to_csv(export_dir / "poisson_data.csv", index=False)
    generate_grouped_continuous().to_csv(export_dir / "grouped_continuous.csv", index=False)
    generate_bootstrap_sample_data().to_csv(export_dir / "bootstrap_sample_data.csv", index=False)

    cat_counts = generate_categorical_counts()
    cat_counts.to_csv(export_dir / "categorical_counts.csv", header=True)

    print("All synthetic datasets exported successfully!")
    return export_dir


def main() -> None:
    """Console-script entry point (``scipy-generate-data``)."""
    export_all_datasets()


if __name__ == "__main__":
    main()
