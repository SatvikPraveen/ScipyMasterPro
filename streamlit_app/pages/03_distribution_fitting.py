import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, gamma, lognorm, beta, expon, kstest
from statsmodels.distributions.empirical_distribution import ECDF
import plotly.graph_objects as go

from utils.distribution_utils import (
    fit_distribution,
    fit_multiple_distributions,
    fit_distributions_all_columns,
    compute_pdf,
    compute_cdf,
    perform_ks_test
)
from streamlit_app.ui_components import (
    load_synthetic_distribution_data,
    select_distribution_column,
    add_sidebar_notes
)

# -------------------------------------
# PAGE CONFIGURATION
# -------------------------------------
st.title("🔍 Distribution Fitting & Comparison")
add_sidebar_notes()

# Load dataset
df = load_synthetic_distribution_data()
col = select_distribution_column(df)
data = df[col]

# -------------------------------------
# DISTRIBUTION SELECTION
# -------------------------------------
st.sidebar.subheader("📐 Choose Distribution to Fit")
dist_map = {
    "Normal": norm,
    "Gamma": gamma,
    "Lognormal": lognorm,
    "Beta": beta,
    "Exponential": expon
}
dist_name = st.sidebar.selectbox("Choose Distribution", list(dist_map.keys()))
dist = dist_map[dist_name]

# -------------------------------------
# FIT SELECTED DISTRIBUTION
# -------------------------------------
import warnings

# Fit selected distribution safely
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    params = fit_distribution(data, dist)

params = fit_distribution(data, dist)
# Parameter labels mapping
param_labels = {
    "norm": ["loc", "scale"],
    "gamma": ["shape (a)", "loc", "scale"],
    "lognorm": ["shape (s)", "loc", "scale"],
    "beta": ["alpha (a)", "beta (b)", "loc", "scale"],
    "expon": ["loc", "scale"]
}

labels = param_labels.get(dist.name, [f"param_{i}" for i in range(len(params))])
df_params = pd.DataFrame([params], columns=labels)
st.markdown(f"### 📌 Fitted Parameters for {dist_name}")
st.dataframe(df_params.style.format(precision=4))


# Compute PDF
x = np.linspace(min(data), max(data), 500)
pdf_vals = dist.pdf(x, *params)

# Plot PDF overlay
fig_pdf = go.Figure()
fig_pdf.add_trace(go.Histogram(x=data, histnorm='probability density',
                               name='Data', opacity=0.6))
fig_pdf.add_trace(go.Scatter(x=x, y=pdf_vals, mode='lines',
                             name=f'{dist_name} PDF', line=dict(color='red')))
fig_pdf.update_layout(title=f"{dist_name} PDF Overlay",
                      xaxis_title="Value", yaxis_title="Density")
st.plotly_chart(fig_pdf, use_container_width=True)

# -------------------------------------
# CDF vs ECDF Plot
# -------------------------------------
ecdf = ECDF(data)
cdf_vals = dist.cdf(x, *params)

fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(x=x, y=ecdf(x), mode='lines',
                             name="ECDF", line=dict(color='blue')))
fig_cdf.add_trace(go.Scatter(x=x, y=cdf_vals, mode='lines',
                             name=f'{dist_name} CDF', line=dict(color='green')))
fig_cdf.update_layout(title=f"{dist_name} CDF vs ECDF",
                      xaxis_title="Value", yaxis_title="Probability")
st.plotly_chart(fig_cdf, use_container_width=True)

# KS Test for goodness-of-fit
ks = perform_ks_test(data, dist, params)
st.subheader("📊 Goodness-of-Fit (KS Test)")
st.write(f"**Statistic**: `{ks['KS_stat']:.4f}`")
st.write(f"**p-value**: `{ks['p_value']:.4f}`")

# -------------------------------------
# MULTIPLE DISTRIBUTION COMPARISON
# -------------------------------------
st.markdown("### 🔄 Compare Multiple Candidate Distributions")

distribution_list = [norm, gamma, lognorm, beta, expon]
fit_results = fit_multiple_distributions(data, distribution_list)
results_df = pd.DataFrame(fit_results)

st.dataframe(results_df)

# Download results
st.download_button(
    "⬇️ Download Fit Results (CSV)",
    data=results_df.to_csv(index=False),
    file_name=f"{col}_fit_results.csv",
    mime="text/csv"
)

# Multi-distribution overlay
fig_multi = go.Figure()
fig_multi.add_trace(go.Histogram(x=data, histnorm='probability density',
                                 name='Data', opacity=0.5))
for d in distribution_list:
    params_d = fit_distribution(data, d)
    pdf_d = d.pdf(x, *params_d)
    fig_multi.add_trace(go.Scatter(x=x, y=pdf_d, mode='lines',
                                   name=f"{d.name.capitalize()} PDF"))
fig_multi.update_layout(title=f"{col}: Multiple Distribution Fits",
                        xaxis_title="Value", yaxis_title="Density")
st.plotly_chart(fig_multi, use_container_width=True)

# -------------------------------------
# PDF & CDF VALUE EXPORT (for selected distribution)
# -------------------------------------
st.markdown("### 📥 Export PDF & CDF values for analytical use")
x_pdf, pdf_vals = compute_pdf(data, dist, params)
x_cdf, cdf_vals = compute_cdf(data, dist, params)

export_df = pd.DataFrame({"x": x_pdf, "pdf": pdf_vals, "cdf": cdf_vals})
st.download_button(
    "⬇️ Download PDF & CDF Values",
    data=export_df.to_csv(index=False),
    file_name=f"{col}_{dist_name}_pdf_cdf.csv",
    mime="text/csv"
)

# -------------------------------------
# SUMMARY
# -------------------------------------
st.markdown("## ✅ Summary")
st.markdown(f"""
- **Fit parametric distributions** (Normal, Gamma, Lognorm, Beta, Exponential) to your data  
- Visualize **PDF overlays** and **ECDF vs CDF fits**  
- Run **KS test** for statistical goodness-of-fit  
- Compare multiple distributions to identify the **best candidate model**  
- Export fitted values for further analysis  
""")
