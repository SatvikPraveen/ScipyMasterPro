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


def fit_distribution(data: "np.ndarray", dist_name: str) -> tuple:
    """
    Fit a named distribution to data using MLE.

    Parameters
    ----------
    data : array-like
        Observed data to fit.
    dist_name : str
        Distribution name: 'normal', 'beta', 'gamma', 'exponential', or 'lognorm'.

    Returns
    -------
    tuple
        Fitted distribution parameters (loc, scale, and shape params if applicable).

    Raises
    ------
    ValueError
        If dist_name is not one of the supported distributions.
    """
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
def fit_distribution(data: "np.ndarray", dist_obj: object) -> tuple:  # type: ignore[misc]
    """
    Fit a scipy.stats distribution object to the data using MLE.

    Parameters
    ----------
    data : array-like
        Observed data to fit.
    dist_obj : scipy.stats continuous_rv_generic
        A scipy distribution object (e.g., scipy.stats.norm, scipy.stats.gamma).

    Returns
    -------
    tuple
        Fitted parameters as returned by the distribution's ``.fit()`` method.
        For most distributions: (shape_params..., loc, scale).

    Examples
    --------
    >>> from scipy.stats import norm
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, 1000)
    >>> params = fit_distribution(data, norm)
    >>> len(params)  # loc, scale
    2
    """
    return dist_obj.fit(data)


# Compute PDF & CDF from Fitted Parameters
def compute_pdf(
    data: "np.ndarray", dist_obj: object, params: tuple
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Compute PDF values over the data range using fitted distribution parameters.

    Parameters
    ----------
    data : array-like
        Observed data (used only for determining the x-axis range).
    dist_obj : scipy.stats continuous_rv_generic
        Fitted distribution object.
    params : tuple
        Parameters as returned by dist_obj.fit(data).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (x, pdf_vals) where x has 200 evenly spaced points over [min(data), max(data)].
    """
    x = np.linspace(min(data), max(data), 200)
    pdf_vals = dist_obj.pdf(x, *params)
    return x, pdf_vals


def compute_cdf(
    data: "np.ndarray", dist_obj: object, params: tuple
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Compute CDF values over the data range using fitted distribution parameters.

    Parameters
    ----------
    data : array-like
        Observed data (used only for determining the x-axis range).
    dist_obj : scipy.stats continuous_rv_generic
        Fitted distribution object.
    params : tuple
        Parameters as returned by dist_obj.fit(data).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (x, cdf_vals) where x has 200 evenly spaced points over [min(data), max(data)].
    """
    x = np.linspace(min(data), max(data), 200)
    cdf_vals = dist_obj.cdf(x, *params)
    return x, cdf_vals


# Perform Goodness-of-Fit Test (e.g., KS)
def perform_ks_test(
    data: "np.ndarray", dist_obj: object, params: tuple
) -> dict[str, float]:
    """
    Perform a Kolmogorov-Smirnov goodness-of-fit test.

    Parameters
    ----------
    data : array-like
        Observed data.
    dist_obj : scipy.stats continuous_rv_generic
        Distribution to test against.
    params : tuple
        Fitted distribution parameters.

    Returns
    -------
    dict of str -> float
        {'KS_stat': float, 'p_value': float}
        Large KS stat or small p-value indicates poor fit.
    """
    D, p = kstest(data, dist_obj.name, args=params)
    return {"KS_stat": D, "p_value": p}


# Wrapper to Fit Multiple Distributions
def fit_multiple_distributions(
    data: "np.ndarray", dist_list: list
) -> list[dict]:
    """
    Fit multiple distributions to the same dataset and compare KS test results.

    Parameters
    ----------
    data : array-like
        Observed data.
    dist_list : list of scipy.stats continuous_rv_generic
        List of distribution objects to try (e.g., [norm, gamma, expon]).

    Returns
    -------
    list of dict
        Each dict contains:
        - 'distribution': str — distribution name
        - 'params': tuple — fitted parameters
        - 'KS_stat': float — KS test statistic
        - 'p_value': float — KS test p-value
        - 'error': str — only present if fitting failed
    """
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


def compute_nll(dist: object, data: "np.ndarray", params: tuple) -> float:
    """
    Compute Negative Log-Likelihood for a fitted distribution.

    Parameters
    ----------
    dist : scipy.stats continuous_rv_generic
        Distribution object with a .pdf method.
    data : array-like
        Observed data.
    params : tuple
        Fitted distribution parameters.

    Returns
    -------
    float
        Negative log-likelihood. Lower values indicate a better fit.
    """
    pdf_vals = dist.pdf(data, *params)
    pdf_vals = np.where(pdf_vals == 0, 1e-12, pdf_vals)  # avoid log(0)
    return -np.sum(np.log(pdf_vals))


def compute_aic(nll: float, k: int) -> float:
    """
    Compute Akaike Information Criterion (AIC).

    Parameters
    ----------
    nll : float
        Negative log-likelihood of the fitted model.
    k : int
        Number of free parameters in the model.

    Returns
    -------
    float
        AIC value. Lower is better. Penalizes model complexity.
    """
    return 2 * k + 2 * nll


def compute_bic(nll: float, k: int, n: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC).

    Parameters
    ----------
    nll : float
        Negative log-likelihood of the fitted model.
    k : int
        Number of free parameters in the model.
    n : int
        Number of observations.

    Returns
    -------
    float
        BIC value. Lower is better. Penalizes complexity more strongly than AIC
        for larger sample sizes.
    """
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


def fit_multiple_distributions_extended(
    data: "np.ndarray", distribution_list: list
) -> list[dict]:
    """
    Fit multiple distributions and compute AIC, BIC, KS test, and Anderson-Darling test.

    Parameters
    ----------
    data : array-like
        Observed data.
    distribution_list : list of scipy.stats continuous_rv_generic
        List of distribution objects to fit.

    Returns
    -------
    list of dict
        Each dict contains:
        - 'distribution': str
        - 'params': tuple of fitted params
        - 'nll': float — negative log-likelihood
        - 'aic': float
        - 'bic': float
        - 'ks_stat': float
        - 'ks_pvalue': float
        - 'ad_stat': float or None (only for normal distribution)
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

def fit_distributions_all_columns(
    df: "pd.DataFrame", distribution_list: list
) -> "pd.DataFrame":
    """
    Fit multiple distributions to every numeric column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Only numeric columns are processed.
    distribution_list : list of scipy.stats continuous_rv_generic
        List of distributions to fit against each column.

    Returns
    -------
    pd.DataFrame
        Combined results from fit_multiple_distributions_extended for all columns,
        with an additional 'column' field identifying the source column.
    """
    all_results = []
    for col in df.select_dtypes(include=np.number).columns:
        data = df[col].dropna()
        fit_res = fit_multiple_distributions_extended(data, distribution_list)
        for res in fit_res:
            res["column"] = col
        all_results.extend(fit_res)
    return pd.DataFrame(all_results)
