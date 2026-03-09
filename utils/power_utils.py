from __future__ import annotations

from scipy.stats import norm, t
from statsmodels.stats.power import TTestPower
import numpy as np

# ✅ 1. Manual Power Calculation for One-Sample Z-test
def compute_power_z(
    effect_size: float,
    alpha: float = 0.05,
    n: int | None = None,
    two_tailed: bool = True,
) -> float:
    """
    Compute statistical power for a one-sample Z-test.

    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d).
    alpha : float, default=0.05
        Significance level (Type I error rate).
    n : int
        Sample size.
    two_tailed : bool, default=True
        If True, uses two-tailed critical value; otherwise one-tailed.

    Returns
    -------
    float
        Statistical power in [0, 1]. Power > 0.8 is typically desirable.

    Notes
    -----
    Uses the normal distribution (appropriate when population variance is known
    or n is large enough for CLT to apply).
    """
    z_alpha = norm.ppf(1 - alpha/2 if two_tailed else 1 - alpha)
    z_power = z_alpha - (effect_size * np.sqrt(n))
    power = 1 - norm.cdf(z_power)
    return power

# ✅ 2. Manual Power Calculation for One-Sample T-test
def compute_power_t(
    effect_size: float,
    alpha: float = 0.05,
    n: int | None = None,
    two_tailed: bool = True,
) -> float:
    """
    Compute statistical power for a one-sample t-test.

    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d).
    alpha : float, default=0.05
        Significance level.
    n : int
        Sample size (must be >= 2).
    two_tailed : bool, default=True
        If True, uses two-tailed critical value.

    Returns
    -------
    float
        Statistical power in [0, 1].

    Notes
    -----
    Uses the t-distribution with df = n - 1. Preferred over Z-test power
    when population variance is unknown.
    """
    df = n - 1
    t_alpha = t.ppf(1 - alpha/2 if two_tailed else 1 - alpha, df)
    t_power = t_alpha - (effect_size * np.sqrt(n))
    power = 1 - t.cdf(t_power, df)
    return power

# ✅ 3. Use Statsmodels API to compute power
def statsmodels_power(
    effect_size: float,
    alpha: float = 0.05,
    n: int | None = None,
    alternative: str = "two-sided",
) -> float:
    """
    Compute t-test power using the statsmodels TTestPower API.

    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d).
    alpha : float, default=0.05
        Significance level.
    n : int
        Number of observations.
    alternative : str, default='two-sided'
        Direction: 'two-sided', 'larger', or 'smaller'.

    Returns
    -------
    float
        Statistical power in [0, 1].
    """
    power_obj = TTestPower()
    power = power_obj.power(effect_size=effect_size, nobs=n, alpha=alpha, alternative=alternative)
    return power

# ✅ 4. Compute Cohen's d for mean difference
def compute_cohens_d(
    mean1: float, mean2: float, std_dev: float
) -> float:
    """
    Compute Cohen's d effect size from two group means.

    Parameters
    ----------
    mean1 : float
        Mean of the first group.
    mean2 : float
        Mean of the second group.
    std_dev : float
        Pooled or common standard deviation.

    Returns
    -------
    float
        Cohen's d = (mean1 - mean2) / std_dev.
        Interpretation: 0.2 small, 0.5 medium, 0.8 large.
    """
    return (mean1 - mean2) / std_dev
