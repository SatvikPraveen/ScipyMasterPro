# streamlit_app/pages/shared_pdf_ecdf.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import anderson, gamma, kstest, lognorm, norm, shapiro
from statsmodels.distributions.empirical_distribution import ECDF

from streamlit_app.ui_components import (
    add_sidebar_notes,
    load_synthetic_distribution_data,
    select_distribution_column,
)
from utils.distribution_utils import fit_distribution
from utils.pdf_ecdf_utils import (
    compute_manual_ecdf,
    compute_statsmodels_ecdf,
    plot_enhanced_ecdf_comparison,
    plot_pdf_ecdf_overlay,
    run_goodness_of_fit_tests,
)

# -------------------------------
# 🎛️ Page Config
# -------------------------------
st.title("📊 PDF & ECDF Comparison — SciPy vs Statsmodels")
add_sidebar_notes()

# -------------------------------
# 📂 Load Dataset
# -------------------------------
df = load_synthetic_distribution_data()
col = select_distribution_column(df)
data = df[col].dropna()

# -------------------------------
# 🔹 Auto map column to SciPy distribution
# -------------------------------
dist_map = {"normal": norm, "gamma": gamma, "lognorm": lognorm}
dist = dist_map.get(col.lower(), norm)

# -------------------------------
# 🔹 Fit distribution parameters
# -------------------------------
params = fit_distribution(data, dist)
st.subheader(f"📌 Fitted Parameters for {col.title()} ({dist.name})")

param_labels = ["loc", "scale"] if len(params) == 2 else ["shape"] + ["loc", "scale"]
df_params = pd.DataFrame([params], columns=param_labels)
st.dataframe(df_params.style.format(precision=4))

# -------------------------------
# 🔹 PDF & ECDF Overlay Plot
# -------------------------------
st.subheader(f"🔹 PDF–ECDF Overlay for {col.title()}")
fig_pdf_ecdf = plot_pdf_ecdf_overlay(data, dist, title=f"{col.title()} PDF–ECDF Overlay")
st.pyplot(fig_pdf_ecdf)

# -------------------------------
# 🔹 Manual vs Statsmodels ECDF Comparison
# -------------------------------
st.subheader("🔹 Manual vs Statsmodels ECDF")
x_manual, y_manual = compute_manual_ecdf(data)
x_sm, y_sm = compute_statsmodels_ecdf(data)

fig_ecdf = plot_enhanced_ecdf_comparison(
    data, dist_list=[dist], dist_labels=[col.title()], title="Manual vs Statsmodels ECDF"
)
st.pyplot(fig_ecdf)

# -------------------------------
# 🔹 ECDF against multiple candidate distributions
# -------------------------------
st.subheader("🔹 ECDF vs Multiple Fits (Normal, Gamma, LogNorm)")
fig_multi = plot_enhanced_ecdf_comparison(
    data,
    dist_list=[norm, gamma, lognorm],
    dist_labels=["Normal", "Gamma", "LogNorm"],
    title="ECDF vs Multiple Distribution Fits",
)
st.pyplot(fig_multi)

# -------------------------------
# 🔹 Goodness-of-Fit Statistical Tests
# -------------------------------
st.subheader("📋 Goodness-of-Fit Test Results")
results = []
for dist_candidate, label in zip([norm, gamma, lognorm], ["Normal", "Gamma", "LogNorm"]):
    test_res = run_goodness_of_fit_tests(data, dist_candidate)
    test_res["Distribution"] = label
    results.append(test_res)

results_df = pd.DataFrame(results)
st.dataframe(results_df)

# -------------------------------
# ✅ Summary
# -------------------------------
st.markdown("## ✅ Summary")
st.markdown(
    """
- **Empirical CDF (ECDF)** compared against fitted PDFs from SciPy  
- Manual vs statsmodels ECDF provides sanity check for computation methods  
- Goodness-of-fit tests (KS, Shapiro, Anderson) indicate statistical fit quality  
- Visual overlays combined with test results help validate distribution assumptions
"""
)
