# streamlit_app/pages/09_interpolation_curvefitting.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

from streamlit_utils import sidebar_section
from utils.interpolation_utils import (
    linear_interpolate,
    cubic_interpolate,
    spline_interpolate,
    fit_curve,
    exponential_model,
    gaussian_model,
    safe_gaussian_fit,
    interpolate_2d,
    rbf_interpolation
)
from utils.viz_utils import (
    plot_interpolation_comparison,
    plot_curve_fits_with_bands,
    plot_gaussian_fit_with_band,
    plot_polynomial_fit,
    plot_weighted_vs_unweighted_fit,
    plot_multivariate_griddata,
    plot_rbf_interpolation,
    plot_residuals_comparison,
    plot_all_fits_comparison
)

# -------------------------------
# 📌 Page Config
# -------------------------------
st.title("🧩 Curve Fitting & Interpolation Explorer")
sidebar_section("Interpolation & Curve Fitting Settings")

demo_type = st.sidebar.selectbox("Select Demo", ["1D Curve Fitting & Interpolation", "2D Surface Interpolation"])
num_points = st.sidebar.slider("Number of Data Points", 10, 200, 50)

# -------------------------------
# 🔹 1D Curve Fitting & Interpolation
# -------------------------------
if demo_type == "1D Curve Fitting & Interpolation":
    st.subheader("📈 1D Curve Fitting, Interpolation & Residual Analysis")

    # Synthetic data
    np.random.seed(42)
    x = np.linspace(0, 6 * np.pi, num_points)
    y = 3 * np.exp(-0.1 * x) * np.sin(x) + np.random.normal(scale=0.3, size=len(x))

    x_new = np.linspace(x.min(), x.max(), 300)

    # --- Interpolation ---
    lin_fn = linear_interpolate(x, y)
    cub_fn = cubic_interpolate(x, y)
    spl_fn = spline_interpolate(x, y)

    y_lin = lin_fn(x_new)
    y_cub = cub_fn(x_new)
    y_spl = spl_fn(x_new)

    fig_interp = plot_interpolation_comparison(x, y, x_new, y_lin, y_cub, y_spl)
    st.pyplot(fig_interp, use_container_width=True)

    # RMSE for methods
    rmse_lin = root_mean_squared_error(y, lin_fn(x))
    rmse_cub = root_mean_squared_error(y, cub_fn(x))
    rmse_spl = root_mean_squared_error(y, spl_fn(x))

    interp_metrics = pd.DataFrame({
        "Method": ["Linear", "Cubic", "Spline"],
        "RMSE": [rmse_lin, rmse_cub, rmse_spl]
    })
    st.markdown("### 📊 Interpolation Metrics")
    st.dataframe(interp_metrics)

    # --- Curve Fitting ---
    popt_exp, _ = fit_curve(exponential_model, x, y, p0=(y.max(), -0.5, 0.1))
    y_exp = exponential_model(x_new, *popt_exp)

    popt_gauss, pcov_gauss = safe_gaussian_fit(x, y)
    y_gauss = gaussian_model(x_new, *popt_gauss)

    fig_curve = plot_curve_fits_with_bands(x, y, x_new, y_exp, y_exp-0.5, y_exp+0.5, y_gauss)
    st.pyplot(fig_curve, use_container_width=True)

    # Weighted fit
    weights = np.linspace(1, 3, len(x))
    from scipy.optimize import curve_fit
    popt_weighted, _ = curve_fit(exponential_model, x, y, sigma=1/weights, absolute_sigma=True, maxfev=10000)
    y_weighted = exponential_model(x_new, *popt_weighted)

    fig_weighted = plot_weighted_vs_unweighted_fit(x, y, x_new, y_exp, y_weighted)
    st.pyplot(fig_weighted, use_container_width=True)

    # Polynomial fit
    coeffs = np.polyfit(x, y, deg=2)
    y_poly = np.poly1d(coeffs)(x_new)
    fig_poly = plot_polynomial_fit(x, y, x_new, y_poly, degree=2)
    st.pyplot(fig_poly, use_container_width=True)

    # Residuals comparison
    res_exp = y - exponential_model(x, *popt_exp)
    res_gauss = y - gaussian_model(x, *popt_gauss)
    res_poly = y - np.poly1d(coeffs)(x)
    res_weighted = y - exponential_model(x, *popt_weighted)

    fig_res = plot_residuals_comparison(x, res_exp, res_gauss, res_poly, res_weighted)
    st.pyplot(fig_res, use_container_width=True)

    # Overlay of all fits
    fig_all = plot_all_fits_comparison(x, y, x_new, y_lin, y_cub, y_spl, y_exp, y_gauss, y_poly)
    st.pyplot(fig_all, use_container_width=True)

    # Metrics
    summary = pd.DataFrame({
        "Technique": ["Linear", "Cubic", "Spline", "Exponential", "Weighted Exp", "Gaussian", "Polynomial"],
        "RMSE": [rmse_lin, rmse_cub, rmse_spl,
                 root_mean_squared_error(y, exponential_model(x, *popt_exp)),
                 root_mean_squared_error(y, exponential_model(x, *popt_weighted)),
                 root_mean_squared_error(y, gaussian_model(x, *popt_gauss)),
                 root_mean_squared_error(y, np.poly1d(coeffs)(x))],
        "R2_Score": [np.nan, np.nan, np.nan,
                     r2_score(y, exponential_model(x, *popt_exp)),
                     r2_score(y, exponential_model(x, *popt_weighted)),
                     r2_score(y, gaussian_model(x, *popt_gauss)),
                     r2_score(y, np.poly1d(coeffs)(x))]
    })
    st.markdown("### 📑 Summary of Fitting Techniques")
    st.dataframe(summary)

# -------------------------------
# 🔹 2D Surface Interpolation
# -------------------------------
else:
    st.subheader("🌐 2D Interpolation of Random Points")

    np.random.seed(0)
    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    z = np.sin(x * 4) * np.cos(y * 4)

    grid_x, grid_y, grid_z = interpolate_2d(x, y, z, method="linear")
    fig2d = plot_multivariate_griddata(grid_x, grid_y, grid_z)
    st.pyplot(fig2d, use_container_width=True)

    z_rbf = rbf_interpolation(x, y, z)
    fig_rbf = plot_rbf_interpolation(grid_x, grid_y, z_rbf)
    st.pyplot(fig_rbf, use_container_width=True)

    error_grid = np.abs(np.sin(grid_x * 4) * np.cos(grid_y * 4) - grid_z)
    st.markdown("### 🔎 Approximation Error (Linear vs True Surface)")
    st.dataframe(pd.DataFrame(error_grid).round(4).head())

# -------------------------------
# ✅ Summary
# -------------------------------
st.markdown("## ✅ Summary")
if demo_type == "1D Curve Fitting & Interpolation":
    st.markdown("""
    - Compared **linear, cubic, and spline interpolation**.
    - Fitted **exponential, Gaussian, weighted, and polynomial models**.
    - Visualized residuals and overlay of all fits for quality check.
    - Displayed **RMSE and R² metrics** for all fitting methods.
    """)
else:
    st.markdown("""
    - Interpolated random 2D data using **linear griddata** and **RBF kernels**.
    - Visualized surfaces interactively.
    - Displayed approximation errors against the true function.
    """)
