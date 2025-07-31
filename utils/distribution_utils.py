import numpy as np
import pandas as pd
from scipy.stats import (norm, 
                         beta, 
                         gamma, 
                         expon, 
                         lognorm, 
                         kstest,
                         anderson
    )


def fit_distribution(data, dist_name):
    dist_map = {
        "normal": norm,
        "beta": beta,
        "gamma": gamma,
        "exponential": expon,
        "lognorm": lognorm,
    }
    dist = dist_map.get(dist_name.lower())
    if dist:
        return dist.fit(data)
    else:
        raise ValueError("Distribution not supported")


# Fit Distribution to Data
def fit_distribution(data, dist_obj):
    """
    Fit a distribution object (e.g., scipy.stats.norm) to the data.
    Returns the fitted parameters.
    """
    return dist_obj.fit(data)


# Compute PDF & CDF from Fitted Parameters
def compute_pdf(data, dist_obj, params):
    x = np.linspace(min(data), max(data), 200)
    pdf_vals = dist_obj.pdf(x, *params)
    return x, pdf_vals


def compute_cdf(data, dist_obj, params):
    x = np.linspace(min(data), max(data), 200)
    cdf_vals = dist_obj.cdf(x, *params)
    return x, cdf_vals


# Perform Goodness-of-Fit Test (e.g., KS)
def perform_ks_test(data, dist_obj, params):
    D, p = kstest(data, dist_obj.name, args=params)
    return {"KS_stat": D, "p_value": p}


# Wrapper to Fit Multiple Distributions
def fit_multiple_distributions(data, dist_list):
    results = []
    for dist in dist_list:
        try:
            params = dist.fit(data)
            D, p = kstest(data, dist.name, args=params)
            results.append({
                "distribution": dist.name,
                "params": params,
                "KS_stat": round(D, 4),
                "p_value": round(p, 4)
            })
        except Exception as e:
            results.append({
                "distribution": dist.name,
                "error": str(e)
            })
    return results


def compute_nll(dist, data, params):
    """Compute Negative Log-Likelihood for a fitted distribution."""
    pdf_vals = dist.pdf(data, *params)
    pdf_vals = np.where(pdf_vals == 0, 1e-12, pdf_vals)  # avoid log(0)
    return -np.sum(np.log(pdf_vals))


def compute_aic(nll, k):
    """Akaike Information Criterion."""
    return 2 * k + 2 * nll


def compute_bic(nll, k, n):
    """Bayesian Information Criterion."""
    return k * np.log(n) + 2 * nll


def perform_anderson_darling(data, dist='norm'):
    """
    Perform Anderson-Darling test (default: normality test).
    Returns statistic and critical values.
    """
    result = anderson(data, dist=dist)
    return {
        "statistic": result.statistic,
        "critical_values": result.critical_values.tolist(),
        "significance_levels": result.significance_level.tolist()
    }


def fit_multiple_distributions_extended(data, distribution_list):
    """
    Fit multiple candidate distributions, compute AIC/BIC, KS test, AD test.
    Returns a list of results with metrics for comparison.
    """
    results = []
    n = len(data)

    for dist in distribution_list:
        params = dist.fit(data)
        nll = compute_nll(dist, data, params)
        k = len(params)

        # KS Test
        ks_stat, ks_p = kstest(data, dist.name, args=params)

        # Anderson-Darling (only for normal)
        ad_result = perform_anderson_darling(data, dist='norm') if dist.name == 'norm' else None

        results.append({
            "distribution": dist.name,
            "params": params,
            "nll": nll,
            "aic": compute_aic(nll, k),
            "bic": compute_bic(nll, k, n),
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "ad_stat": ad_result["statistic"] if ad_result else None
        })
    return results

def fit_distributions_all_columns(df, distribution_list):
    """
    Apply fit_multiple_distributions_extended to all numeric columns in a DataFrame.
    """
    all_results = []
    for col in df.select_dtypes(include=np.number).columns:
        data = df[col].dropna()
        fit_res = fit_multiple_distributions_extended(data, distribution_list)
        for res in fit_res:
            res["column"] = col
        all_results.extend(fit_res)
    return pd.DataFrame(all_results)
