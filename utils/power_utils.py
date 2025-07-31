from scipy.stats import norm, t
from statsmodels.stats.power import TTestPower
import numpy as np

# ✅ 1. Manual Power Calculation for One-Sample Z-test
def compute_power_z(effect_size, alpha=0.05, n=None, two_tailed=True):
    z_alpha = norm.ppf(1 - alpha/2 if two_tailed else 1 - alpha)
    z_power = z_alpha - (effect_size * np.sqrt(n))
    power = 1 - norm.cdf(z_power)
    return power

# ✅ 2. Manual Power Calculation for One-Sample T-test
def compute_power_t(effect_size, alpha=0.05, n=None, two_tailed=True):
    df = n - 1
    t_alpha = t.ppf(1 - alpha/2 if two_tailed else 1 - alpha, df)
    t_power = t_alpha - (effect_size * np.sqrt(n))
    power = 1 - t.cdf(t_power, df)
    return power

# ✅ 3. Use Statsmodels API to compute power
def statsmodels_power(effect_size, alpha=0.05, n=None, alternative='two-sided'):
    power_obj = TTestPower()
    power = power_obj.power(effect_size=effect_size, nobs=n, alpha=alpha, alternative=alternative)
    return power

# ✅ 4. Compute Cohen's d for mean difference
def compute_cohens_d(mean1, mean2, std_dev):
    return (mean1 - mean2) / std_dev
