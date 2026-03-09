from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import anderson  # Normality Tests
from scipy.stats import bartlett  # Variance Equality Tests
from scipy.stats import fligner  # Variance Equality Tests
from scipy.stats import levene  # Variance Equality Tests
from scipy.stats import mannwhitneyu  # Mann-Whitney U Test
from scipy.stats import normaltest  # Normality Tests
from scipy.stats import shapiro  # Normality Tests
from scipy.stats import ttest_1samp  # One-sample t-test
from scipy.stats import ttest_ind  # Two-sample t-test (equal/unequal variance)
from scipy.stats import ttest_rel  # Paired t-test
from scipy.stats import (
    kendalltau,
    kurtosis,
    rankdata,
    skew,
    spearmanr,
    wilcoxon,
)


def compute_skewness_kurtosis(
    df: "pd.DataFrame", columns: list[str]
) -> dict[str, dict[str, float]]:
    """
    Compute skewness and excess kurtosis for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numeric columns.
    columns : list of str
        Column names to compute skewness and kurtosis for.

    Returns
    -------
    dict of str -> dict
        Nested dict mapping column name to {'skewness': float, 'kurtosis': float}.
        Kurtosis is Fisher's (excess) definition, where normal distribution = 0.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': np.random.normal(0, 1, 100)})
    >>> result = compute_skewness_kurtosis(df, ['a'])
    >>> 'skewness' in result['a']
    True
    """
    result: dict[str, dict[str, float]] = {}
    for col in columns:
        result[col] = {"skewness": skew(df[col]), "kurtosis": kurtosis(df[col], fisher=True)}
    return result


# Summary Stats Function
def summarize_descriptive_statistics(df: "pd.DataFrame", columns: list[str]) -> "pd.DataFrame":
    """
    Compute a descriptive statistics summary table for selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Columns to include in the summary.

    Returns
    -------
    pd.DataFrame
        Transposed summary DataFrame with columns: Mean, Median, Std Dev,
        Min, Max, Variance, Std Error.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'x': np.random.normal(0, 1, 200)})
    >>> summary = summarize_descriptive_statistics(df, ['x'])
    >>> list(summary.columns)
    ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Variance', 'Std Error']
    """
    summary = df[columns].agg(["mean", "median", "std", "min", "max", "var", "sem"]).T
    summary = summary.rename(
        columns={
            "mean": "Mean",
            "median": "Median",
            "std": "Std Dev",
            "min": "Min",
            "max": "Max",
            "var": "Variance",
            "sem": "Std Error",
        }
    )
    return summary


# One-sample t-test
def run_one_sample_ttest(data: "np.ndarray", popmean: float) -> dict[str, float]:
    """
    Perform a one-sample t-test to determine if the sample mean differs from a population mean.

    Parameters
    ----------
    data : array-like
        Sample data.
    popmean : float
        Hypothesized population mean (null hypothesis H0: mean == popmean).

    Returns
    -------
    dict of str -> float
        {'t_stat': float, 'p_value': float}

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.normal(5, 1, 100)
    >>> result = run_one_sample_ttest(data, 5.0)
    >>> 'p_value' in result
    True
    """
    stat, pval = ttest_1samp(data, popmean)
    return {"t_stat": stat, "p_value": pval}


# Two-sample t-test (equal/unequal variance)
def run_two_sample_ttest(
    data1: "np.ndarray", data2: "np.ndarray", equal_var: bool = True
) -> dict[str, float]:
    """
    Perform an independent two-sample t-test.

    Parameters
    ----------
    data1 : array-like
        First sample.
    data2 : array-like
        Second sample.
    equal_var : bool, default=True
        If True, assumes equal population variances (Student's t-test).
        If False, uses Welch's t-test (does not assume equal variance).

    Returns
    -------
    dict of str -> float
        {'t_stat': float, 'p_value': float}
    """
    stat, pval = ttest_ind(data1, data2, equal_var=equal_var)
    return {"t_stat": stat, "p_value": pval}


# Paired t-test
def run_paired_ttest(before: "np.ndarray", after: "np.ndarray") -> dict[str, float]:
    """
    Perform a paired (related samples) t-test.

    Parameters
    ----------
    before : array-like
        Measurements from the first condition (e.g., pre-treatment).
    after : array-like
        Measurements from the second condition (e.g., post-treatment).
        Must be the same length as `before`.

    Returns
    -------
    dict of str -> float
        {'t_stat': float, 'p_value': float}
    """
    stat, pval = ttest_rel(before, after)
    return {"t_stat": stat, "p_value": pval}


# Normality Tests
def run_normality_tests(data: "np.ndarray") -> dict[str, object]:
    """
    Run a battery of normality tests on the data.

    Parameters
    ----------
    data : array-like
        Sample data to test for normality.

    Returns
    -------
    dict
        Results from three normality tests:
        - 'shapiro': ShapiroResult(statistic, pvalue)
        - 'dagostino': NormaltestResult(statistic, pvalue) (D'Agostino & Pearson)
        - 'anderson': AndersonResult(statistic, critical_values, significance_level)

    Notes
    -----
    Shapiro-Wilk is most reliable for small samples (n < 50).
    D'Agostino's test is preferred for larger samples.
    Anderson-Darling provides critical values rather than a p-value.
    """
    results = {"shapiro": shapiro(data), "dagostino": normaltest(data), "anderson": anderson(data)}
    return results


# Variance Equality Tests
def run_variance_tests(data1: "np.ndarray", data2: "np.ndarray") -> dict[str, object]:
    """
    Test equality of variances between two samples using three methods.

    Parameters
    ----------
    data1 : array-like
        First sample.
    data2 : array-like
        Second sample.

    Returns
    -------
    dict
        Results from three variance equality tests:
        - 'levene': LeveneResult(statistic, pvalue) — robust to non-normality
        - 'bartlett': BartlettResult(statistic, pvalue) — for normally distributed data
        - 'fligner': FlignerResult(statistic, pvalue) — non-parametric alternative
    """
    return {
        "levene": levene(data1, data2),
        "bartlett": bartlett(data1, data2),
        "fligner": fligner(data1, data2),
    }


# Hypothesis Test Result Formatter
def format_test_result(result_dict: dict[str, float], test_name: str) -> None:
    """
    Print a formatted summary of a hypothesis test result dictionary.

    Parameters
    ----------
    result_dict : dict of str -> float
        Dictionary of test results with string keys and numeric values.
    test_name : str
        Human-readable name for the test (used as the header in output).
    """
    print(f"📌 {test_name} Results")
    for k, v in result_dict.items():
        print(f"  {k}: {v:.4f}")


# -------------------------------------------
# 📏 EFFECT SIZE UTILITIES (parametric & nonparametric)
# -------------------------------------------
def cohens_d_independent(x: "np.ndarray", y: "np.ndarray", equal_var: bool = True) -> float:
    """
    Compute Cohen's d effect size for two independent samples.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    equal_var : bool, default=True
        If True, uses pooled standard deviation.
        If False, uses average of the two group SDs.

    Returns
    -------
    float
        Cohen's d value. Interpretation: |d| < 0.2 small, 0.2–0.5 medium, > 0.8 large.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    if equal_var:
        s_pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    else:
        # use average SD when not assuming equal variances
        s_pooled = np.sqrt((vx + vy) / 2)
    d = (np.mean(x) - np.mean(y)) / s_pooled
    return d


def hedges_g_independent(x: "np.ndarray", y: "np.ndarray", equal_var: bool = True) -> float:
    """
    Compute Hedges' g effect size — a bias-corrected version of Cohen's d.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    equal_var : bool, default=True
        Passed to cohens_d_independent for the underlying calculation.

    Returns
    -------
    float
        Hedges' g value. Slightly smaller than Cohen's d due to small-sample
        bias correction factor J = 1 - 3/(4N - 9).
    """
    d = cohens_d_independent(x, y, equal_var=equal_var)
    n = len(x) + len(y)
    J = 1 - (3 / (4 * n - 9))
    return J * d


def glass_delta(x: "np.ndarray", y: "np.ndarray", ref: str = "y") -> float:
    """
    Compute Glass's delta effect size using only the control group SD.

    Parameters
    ----------
    x : array-like
        Experimental or treatment group.
    y : array-like
        Control group (reference group by default).
    ref : {'y', 'x'}, default='y'
        Which group's standard deviation to use as the reference.

    Returns
    -------
    float
        Glass's delta value. Preferred when variances are unequal and
        one group is clearly the control.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sd_ref = np.std(y, ddof=1) if ref == "y" else np.std(x, ddof=1)
    return (np.mean(x) - np.mean(y)) / sd_ref


def cliffs_delta(x: "np.ndarray", y: "np.ndarray") -> float:
    """
    Compute Cliff's delta — a non-parametric effect size measure.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.

    Returns
    -------
    float
        Cliff's delta in [-1, 1]. Interpretation thresholds:
        |δ| < 0.147 negligible, 0.147–0.33 small, 0.33–0.474 medium, > 0.474 large.

    Notes
    -----
    Cliff's delta measures how often values in one distribution are
    larger than values in another. It is robust to non-normality.
    """
    # Nonparametric effect size (|δ| thresholds: 0.147 small, 0.33 medium, 0.474 large)
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    greater = sum((xi > y).sum() for xi in x)
    less = sum((xi < y).sum() for xi in x)
    delta = (greater - less) / (nx * ny)
    return delta


# -------------------------------------------
# 🧪 MULTIPLE TESTING CORRECTION (Benjamini–Hochberg)
# -------------------------------------------
def p_adjust_bh(pvals: list[float] | "np.ndarray") -> "np.ndarray":
    """
    Apply Benjamini-Hochberg (BH) False Discovery Rate correction to p-values.

    Parameters
    ----------
    pvals : array-like of float
        Raw p-values from multiple hypothesis tests.

    Returns
    -------
    np.ndarray
        BH-adjusted p-values, clipped to [0, 1].

    Notes
    -----
    The BH procedure controls the expected proportion of false positives
    among rejected hypotheses (FDR), making it less conservative than
    Bonferroni correction.

    Examples
    --------
    >>> pvals = [0.01, 0.04, 0.03, 0.2, 0.5]
    >>> adjusted = p_adjust_bh(pvals)
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        frac = ranked[i] * n / (i + 1)
        cummin = min(cummin, frac)
        adj[i] = cummin
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out


def run_mannwhitney_u_test(group1, group2, alternative="two-sided"):
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
    return {"statistic": stat, "p_value": p, "alternative": alternative}


def format_mannwhitney_result(result_dict, test_name="Mann–Whitney U Test"):
    """
    Format the result dictionary from run_mannwhitney_u_test for readable output.
    """
    stat = result_dict["statistic"]
    p = result_dict["p_value"]
    alt = result_dict["alternative"]
    print(f"{test_name} ({alt}): stat = {stat:.3f}, p = {p:.4f}")


def format_effect_sizes(cohens_d: float, hedges_g: float, cliffs_delta: float):
    """Nicely formatted output for effect size metrics."""
    print("📐 Effect Size Metrics:")
    print(f"  • Cohen’s d     : {cohens_d:.3f}")
    print(f"  • Hedges’ g     : {hedges_g:.3f}")
    print(f"  • Cliff’s Delta : {cliffs_delta:.3f}")


# Wilcoxon Signed-Rank Test (Paired, non-parametric)
def run_wilcoxon_signedrank(x: "np.ndarray", y: "np.ndarray") -> dict[str, float]:
    """
    Perform Wilcoxon signed-rank test for paired samples (non-parametric).

    Parameters
    ----------
    x : array-like
        First set of measurements (e.g., pre-treatment).
    y : array-like
        Second set of measurements (e.g., post-treatment).
        Must be the same length as x.

    Returns
    -------
    dict of str -> float
        {'statistic': float, 'p_value': float}

    Notes
    -----
    Non-parametric alternative to the paired t-test. Use when normality
    of differences cannot be assumed.
    """
    stat, p = wilcoxon(x, y)
    return {"statistic": stat, "p_value": p}


# Spearman Rank Correlation
def run_spearman_correlation(x: "np.ndarray", y: "np.ndarray") -> dict[str, float]:
    """
    Compute Spearman rank correlation coefficient between two variables.

    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.

    Returns
    -------
    dict of str -> float
        {'spearman_r': float, 'p_value': float}
        spearman_r is in [-1, 1] where ±1 means perfect monotonic relationship.
    """
    corr, p = spearmanr(x, y)
    return {"spearman_r": corr, "p_value": p}


# Kendall’s Tau
def run_kendall_tau(x: "np.ndarray", y: "np.ndarray") -> dict[str, float]:
    """
    Compute Kendall's tau rank correlation coefficient.

    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.

    Returns
    -------
    dict of str -> float
        {'kendall_tau': float, 'p_value': float}
        tau is in [-1, 1]; more robust to ties than Spearman's r.
    """
    tau, p = kendalltau(x, y)
    return {"kendall_tau": tau, "p_value": p}


# Rank-Biserial Effect Size (for Mann–Whitney or Wilcoxon)
def rank_biserial_effect_size(x: "np.ndarray", y: "np.ndarray") -> float:
    """
    Compute rank-biserial correlation effect size for Mann-Whitney U or Wilcoxon test.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.

    Returns
    -------
    float
        Rank-biserial correlation in [-1, 1].
        |r| < 0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, > 0.5 large.
    """
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
    lower, upper = np.percentile(series, [100 * trim, 100 * (1 - trim)])
    trimmed_std = series[(series >= lower) & (series <= upper)].std()

    return {
        "trim_percentage": trim,
        "trimmed_mean": round(trimmed_mean, 4),
        "trimmed_std": round(trimmed_std, 4),
        "n_after_trim": len(series[(series >= lower) & (series <= upper)]),
    }


def compute_robust_summaries(series):
    """
    Computes robust statistics (median, MAD, IQR) that are less sensitive to outliers.
    :param series: pd.Series or array-like
    :return: dict with robust metrics
    """
    series = pd.Series(series).dropna()
    median = series.median()
    mad = stats.median_abs_deviation(series, scale="normal")
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1

    return {
        "median": round(median, 4),
        "mad": round(mad, 4),
        "iqr": round(iqr, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
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
        "interpretation": (
            "Data looks normal (p > 0.05)" if p > 0.05 else "Data is likely non-normal (p ≤ 0.05)"
        ),
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
        group2.dropna() if hasattr(group2, "dropna") else group2,
    )
    return {
        "test": "Levene's Test",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "interpretation": (
            "Variances are equal (p > 0.05)" if p > 0.05 else "Variances are unequal (p ≤ 0.05)"
        ),
    }
