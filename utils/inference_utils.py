import numpy as np
import scipy.stats as stats
from statsmodels.stats.power import TTestPower

# ✅ 1. Compute Standard Error of the Mean
def compute_sem(std_dev: float, n: int) -> float:
    return std_dev / np.sqrt(n)


# ✅ 2. Compute confidence interval from summary stats (t-based)
def confidence_interval(mean: float, std_dev: float, n: int, confidence: float = 0.95) -> tuple:
    sem = compute_sem(std_dev, n)
    df = n - 1
    t_crit = stats.t.ppf((1 + confidence) / 2., df)
    margin = t_crit * sem
    return (mean - margin, mean + margin)


# ✅ 3. Z-score confidence interval (when population std is known)
def z_confidence_interval(mean: float, pop_std: float, n: int, confidence: float = 0.95) -> tuple:
    z_crit = stats.norm.ppf((1 + confidence) / 2.)
    margin = z_crit * (pop_std / np.sqrt(n))
    return (mean - margin, mean + margin)


# ✅ 4. Compute t-statistic from summary stats
def compute_t_stat(sample_mean: float, pop_mean: float, sample_std: float, n: int) -> float:
    return (sample_mean - pop_mean) / (sample_std / np.sqrt(n))


# ✅ 5. Perform one-sample t-test manually (returns t, p)
def manual_t_test(sample_mean: float, pop_mean: float, sample_std: float, n: int) -> tuple:
    t_stat = compute_t_stat(sample_mean, pop_mean, sample_std, n)
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return t_stat, p_value


# ✅ 6. Margin of Error (can be reused for survey/interval calc)
def margin_of_error(std_dev: float, n: int, confidence: float = 0.95) -> float:
    sem = compute_sem(std_dev, n)
    t_crit = stats.t.ppf((1 + confidence) / 2., n - 1)
    return t_crit * sem


def compute_sample_size(effect_size, alpha=0.05, power=0.8):
    """
    Compute required sample size for a given effect size, alpha, and power.
    Uses one-sample t-test power analysis.
    """
    analysis = TTestPower()
    return analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)

