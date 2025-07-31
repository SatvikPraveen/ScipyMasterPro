# streamlit_app/pages/05_bootstrap_simulation.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import streamlit as st
import numpy as np
import pandas as pd
from utils.sim_utils import bootstrap_statistic, compute_bootstrap_ci, summarize_bootstrap
from utils.viz_utils import plot_bootstrap_distribution
from streamlit_app.streamlit_utils import (
    load_dataset,
    sidebar_section
)

# -------------------------------
# 📌 Page Config
# -------------------------------
MODULE_ID = "05_bootstrap_simulation"
st.title("📊 Bootstrap Simulation & Confidence Interval Explorer")
sidebar_section("Bootstrap Settings")

# -------------------------------
# 🎛️ User Inputs
# -------------------------------
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["bootstrap_sample_data.csv", "mixed_distributions.csv"]
)

# Load dataset early to dynamically populate features
df = load_dataset(dataset_name)

# Filter available numeric columns dynamically
available_features = (
    df.select_dtypes(include=np.number).columns.tolist()
    if not df.empty else []
)

# Provide safe fallback if no numeric columns found
if not available_features:
    st.error(f"No numeric features found in `{dataset_name}`.")
    st.stop()

# Dynamic feature selection
feature = st.sidebar.selectbox("Select Variable", available_features)

# Remaining inputs
statistic_choice = st.sidebar.radio("Statistic to Bootstrap", ["Mean", "Median", "Std Dev"])
n_iterations = st.sidebar.slider("Bootstrap Iterations", 500, 5000, 2000, step=500)
ci_level = st.sidebar.select_slider("Confidence Level", options=[90, 95, 99], value=95)


# -------------------------------
# 📊 Load Data
# -------------------------------
df = load_dataset(dataset_name)


if df.empty:
    st.error(f"Dataset `{dataset_name}` could not be loaded or is empty.")
    st.stop()


data = df[feature].dropna()

# -------------------------------
# 🔁 Perform Bootstrapping
# -------------------------------
st.markdown(f"### 🔄 Bootstrapping **{statistic_choice}** for `{feature}` ({n_iterations} iterations)")

if statistic_choice == "Mean":
    true_val = data.mean()
    boot_samples = bootstrap_statistic(data, np.mean, n_resamples=n_iterations)
elif statistic_choice == "Median":
    true_val = data.median()
    boot_samples = bootstrap_statistic(data, np.median, n_resamples=n_iterations)
else:
    true_val = data.std()
    boot_samples = bootstrap_statistic(data, np.std, n_resamples=n_iterations)

ci_bounds = compute_bootstrap_ci(boot_samples, ci=ci_level)

# -------------------------------
# 📊 Display Results
# -------------------------------
summary = summarize_bootstrap(boot_samples)
summary["True Value"] = true_val
summary["CI Lower"] = ci_bounds[0]
summary["CI Upper"] = ci_bounds[1]
summary_df = pd.DataFrame([summary])

st.subheader("📄 Bootstrap Summary")
st.dataframe(summary_df)

# Download option
st.download_button(
    "⬇️ Download Bootstrap Summary (CSV)",
    data=summary_df.to_csv(index=False),
    file_name=f"{feature}_{statistic_choice.lower()}_bootstrap_summary.csv",
    mime="text/csv"
)

# -------------------------------
# 📉 Plot Bootstrap Distribution
# -------------------------------
fig = plot_bootstrap_distribution(
    boot_samples,
    ci_bounds,
    title=f"Bootstrap {statistic_choice} Distribution (CI {ci_level}%)"
)
st.pyplot(fig, use_container_width=True)

# -------------------------------
# ✅ Summary Section
# -------------------------------
st.markdown("## ✅ Interpretation")
st.markdown(f"""
- Bootstrapping resamples data **with replacement** to estimate the distribution of a statistic.  
- Here we calculated **{statistic_choice}** across `{n_iterations}` resamples.  
- **True Value:** `{true_val:.4f}`  
- **{ci_level}% Confidence Interval:** `({ci_bounds[0]:.4f}, {ci_bounds[1]:.4f})`  
- Non-parametric approach → no distribution assumptions.  
- You can **download summary results** above for reporting.
""")
