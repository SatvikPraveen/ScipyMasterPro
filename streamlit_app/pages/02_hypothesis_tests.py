import sys
from pathlib import Path
import os
from streamlit_app.config import DATA_PATH

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # goes up from pages/ to streamlit_app
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT).rsplit('/', 1)[0])  # adds main project root



import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from streamlit_app.streamlit_utils import sidebar_section, load_dataset
from utils.stats_tests_utils import (
    perform_shapiro_test, levene_variance_test,
    run_one_sample_ttest, run_two_sample_ttest, run_paired_ttest,
    run_mannwhitney_u_test,
    cohens_d_independent, hedges_g_independent, cliffs_delta,
    run_spearman_correlation, run_kendall_tau, run_wilcoxon_signedrank,
    rank_biserial_effect_size
)


# ------------------------------
# Config
# ------------------------------
MODULE_ID = "02_hypothesis_tests"
sidebar_section("🔬 Hypothesis Testing Explorer", 
                "Perform parametric and non-parametric tests with visualizations and effect size metrics.")

st.title("🔬 Hypothesis Testing Explorer")

# ----------------------
# Load Filtered Dataset 
# ----------------------
st.sidebar.markdown("### 📂 Dataset Selection")

# Only include relevant datasets
allowed_datasets = {
    "normal_skewed.csv",
    "mixed_distributions.csv",
    "multivariate_gaussian.csv",
    "sample_for_optimization.csv",
    "curve_fitting_data.csv"
}

# Filter files
available_files = [
    f.name for f in DATA_PATH.glob("*.csv")
    if f.name in allowed_datasets
]

# Default selection
default_index = (
    available_files.index("normal_skewed.csv")
    if "normal_skewed.csv" in available_files else 0
)

dataset_choice = st.sidebar.selectbox("Choose built-in dataset", available_files, index=default_index)

# File upload option
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV", type=["csv"])

# Load dataset
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")
        st.stop()
else:
    df = load_dataset(dataset_choice)

# ✅ Validate dataset
if df is None or df.empty:
    st.warning("⚠️ Dataset not found or empty. Please select or upload a valid file.")
    st.stop()

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) < 2:
    st.error("Dataset must have at least two numeric columns for comparison tests.")
    st.stop()


# ------------------------------
# Column Selection
# ------------------------------
st.subheader("📌 Select Columns for Comparison")
col1 = st.selectbox("Group 1 Column", num_cols)
col2 = st.selectbox("Group 2 Column", num_cols, index=1 if len(num_cols) > 1 else 0)

# Hypothesis Test Type
test_type = st.radio("Choose Test", 
                     ["One-sample t-test", "t-test (independent)", "t-test (paired)", "Mann–Whitney U", 
                      "Rank-based Nonparametric"])

# Hypothesis Parameters
pop_mean = None
if test_type == "One-sample t-test":
    pop_mean = st.number_input("Population mean (μ₀)", value=50.0, step=1.0)

shapiro = st.checkbox("✅ Check Normality (Shapiro–Wilk)", value=True)
equal_var = st.checkbox("✅ Assume Equal Variance (Levene's Test)", value=False)

# ------------------------------
# Visualization
# ------------------------------
st.subheader("📊 Distribution Comparison")
stacked_df = pd.DataFrame({col1: df[col1], col2: df[col2]}).melt(var_name="Group", value_name="Value")
fig_box = px.box(stacked_df, x="Group", y="Value", points="all", title="Group Comparison with Data Points")
st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------
# Normality & Variance Checks
# ------------------------------
if shapiro:
    st.subheader("📏 Normality Tests (Shapiro–Wilk)")
    st.write(f"**{col1}** →", perform_shapiro_test(df[col1]))
    st.write(f"**{col2}** →", perform_shapiro_test(df[col2]))

if equal_var:
    st.subheader("📐 Levene's Test for Equal Variance")
    st.write(levene_variance_test(df[col1], df[col2]))

# ------------------------------
# Run Hypothesis Test
# ------------------------------
st.subheader("📈 Hypothesis Test Results")

if test_type == "One-sample t-test":
    result = run_one_sample_ttest(df[col1], popmean=pop_mean)
elif test_type == "t-test (independent)":
    result = run_two_sample_ttest(df[col1], df[col2], equal_var=equal_var)
elif test_type == "t-test (paired)":
    result = run_paired_ttest(df[col1], df[col2])
elif test_type == "Mann–Whitney U":
    result = run_mannwhitney_u_test(df[col1], df[col2])
else:
    # Rank-based tests
    spearman_res = run_spearman_correlation(df[col1], df[col2])
    kendall_res = run_kendall_tau(df[col1], df[col2])
    wilcoxon_res = run_wilcoxon_signedrank(df[col1], df[col2])
    rb_effect = rank_biserial_effect_size(df[col1], df[col2])
    
    result = {
        "Spearman_rho": round(spearman_res['spearman_r'], 3),
        "Spearman_p": round(spearman_res['p_value'], 4),
        "Kendall_tau": round(kendall_res['kendall_tau'], 3),
        "Kendall_p": round(kendall_res['p_value'], 4),
        "Wilcoxon_stat": round(wilcoxon_res['statistic'], 3),
        "Wilcoxon_p": round(wilcoxon_res['p_value'], 4),
        "Rank-Biserial": round(rb_effect, 3)
    }

st.json(result)

# ------------------------------
# Effect Size Calculations (for independent samples)
# ------------------------------
if test_type in ["t-test (independent)", "t-test (paired)"]:
    st.subheader("📏 Effect Sizes")
    effect_d = cohens_d_independent(df[col1], df[col2])
    effect_g = hedges_g_independent(df[col1], df[col2])
    effect_delta = cliffs_delta(df[col1], df[col2])
    
    effect_df = pd.DataFrame([{
        "Cohen’s d": round(effect_d, 3),
        "Hedges’ g": round(effect_g, 3),
        "Cliff’s Delta": round(effect_delta, 3)
    }])
    st.dataframe(effect_df)

# ------------------------------
# Export Option
# ------------------------------
res_df = pd.DataFrame([result])
st.download_button(
    "⬇️ Download Result as CSV",
    data=res_df.to_csv(index=False),
    file_name=f"{col1}_vs_{col2}_test_result.csv",
    mime="text/csv"
)

# ------------------------------
# Summary
# ------------------------------
st.markdown("## ✅ Summary")
st.markdown(f"""
- **Parametric tests (t-tests)** require normality and sometimes equal variances  
- **Non-parametric tests** (Mann–Whitney, Wilcoxon) are robust alternatives  
- **Effect sizes** complement p-values by quantifying practical significance  
- **Rank-based tests** handle ordinal or non-normal data scenarios  
- Use this module to **test hypotheses interactively** and compare results visually
""")
