import numpy as np
from scipy.stats import (
        skew, 
        kurtosis,
        levene, # Variance Equality Tests
        bartlett, # Variance Equality Tests
        fligner, # Variance Equality Tests
        ttest_1samp, # One-sample t-test
        ttest_ind, # Two-sample t-test (equal/unequal variance)
        ttest_rel, # Paired t-test
        shapiro, # Normality Tests
        normaltest, # Normality Tests
        anderson, # Normality Tests
        mannwhitneyu, # Mann-Whitney U Test
        wilcoxon,
        spearmanr,
        kendalltau,
        rankdata,
        levene,
        shapiro
    )

import numpy as np
import pandas as pd
from scipy import stats


def compute_skewness_kurtosis(df, columns):
    stats = {}
    for col in columns:
        stats[col] = {
            "skewness": skew(df[col]),
            "kurtosis": kurtosis(df[col], fisher=True)
        }
    return stats


# Summary Stats Function
def summarize_descriptive_statistics(df, columns):
    summary = df[columns].agg(['mean', 'median', 'std', 'min', 'max', 'var', 'sem']).T
    summary = summary.rename(columns={
        'mean': 'Mean',
        'median': 'Median',
        'std': 'Std Dev',
        'min': 'Min',
        'max': 'Max',
        'var': 'Variance',
        'sem': 'Std Error'
    })
    return summary


# One-sample t-test
def run_one_sample_ttest(data, popmean):
    stat, pval = ttest_1samp(data, popmean)
    return {"t_stat": stat, "p_value": pval}


# Two-sample t-test (equal/unequal variance)
def run_two_sample_ttest(data1, data2, equal_var=True):
    stat, pval = ttest_ind(data1, data2, equal_var=equal_var)
    return {"t_stat": stat, "p_value": pval}


# Paired t-test
def run_paired_ttest(before, after):
    stat, pval = ttest_rel(before, after)
    return {"t_stat": stat, "p_value": pval}


# Normality Tests
def run_normality_tests(data):
    results = {
        "shapiro": shapiro(data),
        "dagostino": normaltest(data),
        "anderson": anderson(data)
    }
    return results


# Variance Equality Tests
def run_variance_tests(data1, data2):
    return {
        "levene": levene(data1, data2),
        "bartlett": bartlett(data1, data2),
        "fligner": fligner(data1, data2)
    }


# Hypothesis Test Result Formatter
def format_test_result(result_dict, test_name):
    print(f"📌 {test_name} Results")
    for k, v in result_dict.items():
        print(f"  {k}: {v:.4f}")



# -------------------------------------------
# 📏 EFFECT SIZE UTILITIES (parametric & nonparametric)
# -------------------------------------------
def cohens_d_independent(x, y, equal_var=True):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    if equal_var:
        s_pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    else:
        # use average SD when not assuming equal variances
        s_pooled = np.sqrt((vx + vy) / 2)
    d = (np.mean(x) - np.mean(y)) / s_pooled
    return d

def hedges_g_independent(x, y, equal_var=True):
    d = cohens_d_independent(x, y, equal_var=equal_var)
    n = len(x) + len(y)
    J = 1 - (3 / (4*n - 9))
    return J * d

def glass_delta(x, y, ref="y"):
    x = np.asarray(x); y = np.asarray(y)
    sd_ref = np.std(y, ddof=1) if ref == "y" else np.std(x, ddof=1)
    return (np.mean(x) - np.mean(y)) / sd_ref

def cliffs_delta(x, y):
    # Nonparametric effect size (|δ| thresholds: 0.147 small, 0.33 medium, 0.474 large)
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    greater = sum((xi > y).sum() for xi in x)
    less = sum((xi < y).sum() for xi in x)
    delta = (greater - less) / (nx * ny)
    return delta

# -------------------------------------------
# 🧪 MULTIPLE TESTING CORRECTION (Benjamini–Hochberg)
# -------------------------------------------
def p_adjust_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    cummin = 1.0
    for i in range(n-1, -1, -1):
        frac = ranked[i] * n / (i+1)
        cummin = min(cummin, frac)
        adj[i] = cummin
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out


def run_mannwhitney_u_test(group1, group2, alternative='two-sided'):
    """
    Perform Mann–Whitney U test (non-parametric test for two independent samples).

    Parameters:
    - group1 (array-like): First independent sample
    - group2 (array-like): Second independent sample
    - alternative (str): 'two-sided', 'less', or 'greater'

    Returns:
    - dict: { 'statistic': float, 'p_value': float, 'alternative': str }
    """
    stat, p = mannwhitneyu(group1, group2, alternative=alternative)
    return {
        'statistic': stat,
        'p_value': p,
        'alternative': alternative
    }


def format_mannwhitney_result(result_dict, test_name="Mann–Whitney U Test"):
    """
    Format the result dictionary from run_mannwhitney_u_test for readable output.
    """
    stat = result_dict['statistic']
    p = result_dict['p_value']
    alt = result_dict['alternative']
    print(f"{test_name} ({alt}): stat = {stat:.3f}, p = {p:.4f}")


def format_effect_sizes(cohens_d: float, hedges_g: float, cliffs_delta: float):
    """Nicely formatted output for effect size metrics."""
    print("📐 Effect Size Metrics:")
    print(f"  • Cohen’s d     : {cohens_d:.3f}")
    print(f"  • Hedges’ g     : {hedges_g:.3f}")
    print(f"  • Cliff’s Delta : {cliffs_delta:.3f}")


# Wilcoxon Signed-Rank Test (Paired, non-parametric)
def run_wilcoxon_signedrank(x, y):
    stat, p = wilcoxon(x, y)
    return {"statistic": stat, "p_value": p}

# Spearman Rank Correlation
def run_spearman_correlation(x, y):
    corr, p = spearmanr(x, y)
    return {"spearman_r": corr, "p_value": p}

# Kendall’s Tau
def run_kendall_tau(x, y):
    tau, p = kendalltau(x, y)
    return {"kendall_tau": tau, "p_value": p}

# Rank-Biserial Effect Size (for Mann–Whitney or Wilcoxon)
def rank_biserial_effect_size(x, y):
    nx, ny = len(x), len(y)
    ranks = rankdata(np.concatenate([x, y]))
    rx = np.sum(ranks[:nx])
    U = rx - nx * (nx + 1) / 2
    R = U / (nx * ny)
    return 2 * R - 1


def compute_trimmed_stats(series, trim=0.1):
    """
    Computes trimmed mean and trimmed standard deviation for a given series.
    :param series: pd.Series or array-like
    :param trim: proportion to cut from each tail (0 to 0.5)
    :return: dict with trimmed mean and std
    """
    series = pd.Series(series).dropna()
    trimmed_mean = stats.trim_mean(series, proportiontocut=trim)
    lower, upper = np.percentile(series, [100*trim, 100*(1-trim)])
    trimmed_std = series[(series >= lower) & (series <= upper)].std()

    return {
        "trim_percentage": trim,
        "trimmed_mean": round(trimmed_mean, 4),
        "trimmed_std": round(trimmed_std, 4),
        "n_after_trim": len(series[(series >= lower) & (series <= upper)])
    }

def compute_robust_summaries(series):
    """
    Computes robust statistics (median, MAD, IQR) that are less sensitive to outliers.
    :param series: pd.Series or array-like
    :return: dict with robust metrics
    """
    series = pd.Series(series).dropna()
    median = series.median()
    mad = stats.median_abs_deviation(series, scale='normal')
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1

    return {
        "median": round(median, 4),
        "mad": round(mad, 4),
        "iqr": round(iqr, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4)
    }


def perform_shapiro_test(data):
    """
    Perform Shapiro-Wilk test for normality.

    Parameters:
        data (array-like): Sample data.

    Returns:
        dict: Test statistic and p-value.
    """
    stat, p = shapiro(data.dropna() if hasattr(data, "dropna") else data)
    return {
        "test": "Shapiro-Wilk",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "interpretation": "Data looks normal (p > 0.05)" if p > 0.05 else "Data is likely non-normal (p ≤ 0.05)"
    }


def levene_variance_test(group1, group2):
    """
    Perform Levene's test to assess equality of variances.

    Parameters:
        group1, group2 (array-like): Two samples to compare.

    Returns:
        dict: Test statistic and p-value.
    """
    stat, p = levene(
        group1.dropna() if hasattr(group1, "dropna") else group1,
        group2.dropna() if hasattr(group2, "dropna") else group2
    )
    return {
        "test": "Levene's Test",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "interpretation": "Variances are equal (p > 0.05)" if p > 0.05 else "Variances are unequal (p ≤ 0.05)"
    }

