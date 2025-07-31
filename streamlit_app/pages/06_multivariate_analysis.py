# streamlit_app/pages/06_multivariate_analysis.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import chi2

from utils.sim_utils import (
    compute_mahalanobis_distances,
    evaluate_mahalanobis_outliers
)
from utils.viz_utils import (
    plot_covariance_heatmap,
    plot_correlation_heatmap, 
    plot_mahalanobis_distance_distribution,
    plot_mahalanobis_outliers,
    plot_mahalanobis_outliers_3d
)
from streamlit_app.streamlit_utils import (
    load_dataset,
    sidebar_section
)

# -------------------------------
# 📌 Page Configuration
# -------------------------------
st.title("📊 Multivariate Outlier Detection (Mahalanobis Distance)")
sidebar_section("Multivariate Settings")

# -------------------------------
# 🎛️ User Inputs
# -------------------------------
dataset_name = st.sidebar.selectbox("Select Dataset", ["multivariate_gaussian.csv", "mixed_distributions.csv"])
df = load_dataset(dataset_name)

if df.empty:
    st.stop()

# Feature selection
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
selected_features = st.sidebar.multiselect(
    "Select 2 or 3 Features for Analysis",
    options=numeric_cols,
    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
)

alpha = st.sidebar.select_slider("Significance Level (α)", options=[0.10, 0.05, 0.01], value=0.01)

if len(selected_features) < 2:
    st.warning("Please select at least 2 numeric features.")
    st.stop()

data = df[selected_features].dropna()

# -------------------------------
# 🔹 Covariance / Correlation Plot
# -------------------------------

st.subheader("📌 Covariance / Correlation Structure")

matrix_type = st.radio("Choose Matrix Type:", ["Covariance", "Correlation"], horizontal=True)

if matrix_type == "Covariance":
    fig_cov = plot_covariance_heatmap(df[selected_features], title="Covariance Matrix of Selected Features")
    st.pyplot(fig_cov, use_container_width=True)
else:
    fig_corr = plot_correlation_heatmap(df[selected_features], annot=True, cmap="coolwarm")
    st.pyplot(fig_corr, use_container_width=True)

# -------------------------------
# 🔁 Mahalanobis Distances
# -------------------------------
st.subheader("📏 Computing Mahalanobis Distances")

distances = compute_mahalanobis_distances(data)
threshold = chi2.ppf(1 - alpha, df=data.shape[1])
mask_outliers = evaluate_mahalanobis_outliers(distances, df_dim=data.shape[1], alpha=alpha)

result_df = data.copy()
result_df["Mahalanobis"] = distances
result_df["Outlier"] = mask_outliers

st.dataframe(result_df.head())

# -------------------------------
# 📊 Summary Table
# -------------------------------
outlier_percentage = mask_outliers.mean() * 100
summary_df = pd.DataFrame([{
    "Total Points": len(result_df),
    "Outliers Detected": mask_outliers.sum(),
    "Alpha": alpha,
    "Chi-Square Threshold": round(threshold, 3),
    "Percent Outliers": round(outlier_percentage, 2)
}])

st.subheader("📋 Outlier Detection Summary")
st.dataframe(summary_df)

st.download_button(
    "⬇️ Download Annotated Results (CSV)",
    data=result_df.to_csv(index=False),
    file_name=f"mahalanobis_outlier_results.csv",
    mime="text/csv"
)

# -------------------------------
# 📉 Visualizations
# -------------------------------
st.subheader("📈 Mahalanobis Distance Distribution")
fig_dist = plot_mahalanobis_distance_distribution(result_df, distance_col="Mahalanobis", threshold=threshold)
st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("🔹 2D Scatterplot of Outliers")
x_col, y_col = selected_features[:2]
fig_scatter = plot_mahalanobis_outliers(result_df, x_col=x_col, y_col=y_col, outlier_col="Outlier")
st.plotly_chart(fig_scatter, use_container_width=True)

if len(selected_features) == 3:
    st.subheader("🔹 3D Visualization of Outliers")
    fig_3d = plot_mahalanobis_outliers_3d(result_df, *selected_features, outlier_col="Outlier")
    st.plotly_chart(fig_3d, use_container_width=True)

# -------------------------------
# ✅ Summary
# -------------------------------
st.markdown("## ✅ Interpretation")
st.markdown(f"""
- Mahalanobis distance identifies points far from the multivariate mean while accounting for covariance.
- Threshold is based on **Chi-Square distribution (α = {alpha})**, degrees of freedom = `{data.shape[1]}`.
- **Detected Outliers:** `{mask_outliers.sum()}` out of `{len(result_df)}` points ({outlier_percentage:.2f}%).
- Use this module to validate multivariate assumptions and flag anomalies interactively.
""")
