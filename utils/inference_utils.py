import numpy as np
import scipy.stats as stats
from statsmodels.stats.power import TTestPower


# ✅ 1. Compute Standard Error of the Mean
def compute_sem(std_dev: float, n: int) -> float:
    """
    Compute the Standard Error of the Mean (SEM).

    Parameters
    ----------
    std_dev : float
        Sample standard deviation.
    n : int
        Sample size.

    Returns
    -------
    float
        SEM = std_dev / sqrt(n).
    """
    return std_dev / np.sqrt(n)


# ✅ 2. Compute confidence interval from summary stats (t-based)
def confidence_interval(
    mean: float, std_dev: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute a t-based confidence interval from summary statistics.

    Parameters
    ----------
    mean : float
        Sample mean.
    std_dev : float
        Sample standard deviation.
    n : int
        Sample size.
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for a 95% CI).

    Returns
    -------
    tuple of (float, float)
        (lower, upper) confidence interval bounds.
    """
    sem = compute_sem(std_dev, n)
    df = n - 1
    t_crit = stats.t.ppf((1 + confidence) / 2.0, df)
    margin = t_crit * sem
    return (mean - margin, mean + margin)


# ✅ 3. Z-score confidence interval (when population std is known)
def z_confidence_interval(
    mean: float, pop_std: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute a Z-based confidence interval using the known population standard deviation.

    Parameters
    ----------
    mean : float
        Sample mean.
    pop_std : float
        Known population standard deviation.
    n : int
        Sample size.
    confidence : float, default=0.95
        Confidence level.

    Returns
    -------
    tuple of (float, float)
        (lower, upper) confidence interval bounds.

    Notes
    -----
    Use this when population variance is known (e.g., large samples or known process).
    For unknown variance, use confidence_interval() which uses the t-distribution.
    """
    z_crit = stats.norm.ppf((1 + confidence) / 2.0)
    margin = z_crit * (pop_std / np.sqrt(n))
    return (mean - margin, mean + margin)


# ✅ 4. Compute t-statistic from summary stats
def compute_t_stat(sample_mean: float, pop_mean: float, sample_std: float, n: int) -> float:
    """
    Compute the one-sample t-statistic from summary statistics.

    Parameters
    ----------
    sample_mean : float
        Observed sample mean.
    pop_mean : float
        Hypothesized population mean (null hypothesis).
    sample_std : float
        Sample standard deviation.
    n : int
        Sample size.

    Returns
    -------
    float
        t-statistic = (sample_mean - pop_mean) / (sample_std / sqrt(n)).
    """
    return (sample_mean - pop_mean) / (sample_std / np.sqrt(n))


# ✅ 5. Perform one-sample t-test manually (returns t, p)
def manual_t_test(
    sample_mean: float, pop_mean: float, sample_std: float, n: int
) -> tuple[float, float]:
    """
    Perform a one-sample t-test manually from summary statistics.

    Parameters
    ----------
    sample_mean : float
        Observed sample mean.
    pop_mean : float
        Hypothesized population mean (null hypothesis).
    sample_std : float
        Sample standard deviation.
    n : int
        Sample size.

    Returns
    -------
    tuple of (float, float)
        (t_stat, p_value) for a two-tailed test.
    """
    t_stat = compute_t_stat(sample_mean, pop_mean, sample_std, n)
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return t_stat, p_value


# ✅ 6. Margin of Error (can be reused for survey/interval calc)
def margin_of_error(std_dev: float, n: int, confidence: float = 0.95) -> float:
    """
    Compute the margin of error for a t-based confidence interval.

    Parameters
    ----------
    std_dev : float
        Sample standard deviation.
    n : int
        Sample size.
    confidence : float, default=0.95
        Confidence level.

    Returns
    -------
    float
        Margin of error (half-width of the confidence interval).
    """
    sem = compute_sem(std_dev, n)
    t_crit = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return t_crit * sem


def compute_sample_size(effect_size, alpha=0.05, power=0.8):
    """
    Compute required sample size for a given effect size, alpha, and power.
    Uses one-sample t-test power analysis.
    """
    analysis = TTestPower()
    return analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)
