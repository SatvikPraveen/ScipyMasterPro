import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

import os


# ✅ 1. Compute PDF from a scipy.stats distribution
def get_pdf(data, dist, params=None, num_points=100):
    x = np.linspace(np.min(data), np.max(data), num_points)
    if params:
        pdf = dist.pdf(x, *params)
    else:
        params = dist.fit(data)
        pdf = dist.pdf(x, *params)
    return x, pdf

# ✅ 2. Compute ECDF manually (raw empirical CDF)
def compute_manual_ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

# ✅ 3. Compute ECDF using statsmodels
def compute_statsmodels_ecdf(data):
    ecdf_obj = sm.distributions.ECDF(data)
    x = np.sort(data)
    y = ecdf_obj(x)
    return x, y

import matplotlib.pyplot as plt
from scipy.stats import kstest

def plot_pdf_ecdf_overlay(data, dist, params=None, title="", annotate_ks=True, save_path=None):
    x_pdf, y_pdf = get_pdf(data, dist, params)
    x_ecdf, y_ecdf = compute_manual_ecdf(data)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot ECDF
    ax.step(x_ecdf, y_ecdf, where='post', label="Empirical CDF", color='green')

    # Plot CDF from fitted distribution
    if params:
        y_theoretical = dist.cdf(x_pdf, *params)
    else:
        params = dist.fit(data)
        y_theoretical = dist.cdf(x_pdf, *params)

    ax.plot(x_pdf, y_theoretical, label=f"{dist.name.capitalize()} CDF", color='blue')

    # Optional shaded area between ECDF and CDF
    ax.fill_between(x_pdf, y_theoretical, y_ecdf[:len(y_theoretical)],
                    color='orange', alpha=0.2, label='ECDF–CDF gap')

    # KS test annotation
    if annotate_ks:
        ks_stat, p_val = kstest(data, dist.name, args=params)
        ax.text(0.05, 0.1, f"KS p-value: {p_val:.4f}", transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    #plt.show()

    return fig


import matplotlib.pyplot as plt
from scipy.stats import kstest, anderson, shapiro

def plot_enhanced_ecdf_comparison(
    data,
    dist_list,
    dist_labels=None,
    annotate_tests=True,
    title="ECDF vs Multiple Distributions",
    save_path=None
):
    x_ecdf, y_ecdf = compute_manual_ecdf(data)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ECDF
    ax.step(x_ecdf, y_ecdf, where='post', label="Empirical CDF", color="black", linewidth=2)

    annotations = []

    for idx, dist in enumerate(dist_list):
        color = plt.cm.tab10(idx)
        label = dist.name if not dist_labels else dist_labels[idx]
        params = dist.fit(data)
        x_pdf, _ = get_pdf(data, dist, params)
        y_theoretical = dist.cdf(x_pdf, *params)

        ax.plot(x_pdf, y_theoretical, label=f"{label} CDF", color=color)

        if annotate_tests:
            ks_stat, ks_p = kstest(data, dist.name, args=params)
            ad_stat = anderson(data, dist=dist.name if dist.name in ['norm', 'expon', 'logistic'] else 'norm').statistic
            shapiro_stat, shapiro_p = shapiro(data[:5000])  # limit Shapiro to 5000

            annotations.append(
                f"{label}:\n"
                f"KS p={ks_p:.4f}, AD={ad_stat:.4f}, Shapiro p={shapiro_p:.4f}"
            )

    # Annotate test results
    if annotate_tests:
        full_note = "\n\n".join(annotations)
        ax.text(1.02, 0.5, full_note, transform=ax.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.tight_layout()
    #plt.show()
    return fig


from scipy.stats import kstest, anderson, shapiro

def run_goodness_of_fit_tests(data, dist):
    ks_stat, ks_p = kstest(data, dist.cdf, args=dist.fit(data))
    ad_result = anderson(data)
    shapiro_stat, shapiro_p = shapiro(data)
    return {
        "KS_stat": ks_stat, "KS_p": ks_p,
        "AD_stat": ad_result.statistic,
        "Shapiro_p": shapiro_p
    }
