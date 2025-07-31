# streamlit_app/pages/04_sampling_resampling.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import streamlit as st
import numpy as np
import pandas as pd

from utils.sim_utils import (
    sample_uniform,
    weighted_sample,
    stratified_sample,
    draw_multinomial_sample,
    draw_dirichlet_sample,
    sample_custom_discrete,
    resample_with_replacement,
    compute_ecdf
)
from streamlit_app.streamlit_utils import (
    load_dataset,
    sidebar_section
)

import plotly.graph_objects as go

# -------------------------------------
# PAGE CONFIGURATION
# -------------------------------------
MODULE_ID = "04_sampling_resampling"
st.title("🎯 Sampling & Resampling Simulation")
sidebar_section("Sampling & Resampling Settings")

# -------------------------------------
# LOAD DATA
# -------------------------------------
cat_counts = load_dataset("categorical_counts.csv")

# ✅ Handle multiple formats and ensure numeric counts
if isinstance(cat_counts, pd.DataFrame):
    # Convert all columns to numeric (coerce non-numeric to NaN)
    cat_counts = cat_counts.apply(pd.to_numeric, errors='coerce')
    cat_counts = cat_counts.fillna(0)  # Replace NaN with 0
    if cat_counts.shape[1] == 1:
        cat_counts = cat_counts.iloc[:, 0]
    else:
        cat_counts = cat_counts.sum(axis=1)  # Now safely sums numeric values
elif isinstance(cat_counts, pd.Series):
    cat_counts = pd.to_numeric(cat_counts, errors='coerce').fillna(0)

cat_counts = pd.Series(cat_counts, name="counts").astype(float)
weights = cat_counts.values / cat_counts.sum()


poisson_df = load_dataset("poisson_data.csv")



if cat_counts.empty or poisson_df.empty:
    st.error("Required datasets not found. Please ensure both categorical_counts.csv and poisson_data.csv are available.")
    st.stop()

# -------------------------------------
# USER CONTROLS
# -------------------------------------
sample_size = st.sidebar.slider("Sample Size", 50, 1000, 200, step=50)
sampling_type = st.sidebar.selectbox(
    "Select Sampling Method",
    ["Uniform", "Weighted", "Stratified", "Multinomial", "Dirichlet", "Custom Discrete", "Bootstrap (Poisson)"]
)

# -------------------------------------
# PERFORM SAMPLING
# -------------------------------------
result = None

if sampling_type == "Uniform":
    result = sample_uniform(cat_counts.index, n=sample_size)
elif sampling_type == "Weighted":
    weights = cat_counts.values.flatten() / cat_counts.sum()
    result = weighted_sample(cat_counts.index, weights, n=sample_size)
elif sampling_type == "Stratified":
    df_strat = pd.DataFrame({
        "category": np.random.choice(cat_counts.index, p=cat_counts.values.flatten()/cat_counts.sum(), size=1000),
        "value": np.random.randn(1000)
    })
    result = stratified_sample(df_strat, stratify_col="category", frac=min(1.0, sample_size / 1000))
elif sampling_type == "Multinomial":
    weights = cat_counts.values.flatten() / cat_counts.sum()
    result = draw_multinomial_sample(n=sample_size, probs=weights, size=1)[0]
elif sampling_type == "Dirichlet":
    result = draw_dirichlet_sample(alpha=[1]*len(cat_counts.index), size=1)[0]
elif sampling_type == "Custom Discrete":
    result = sample_custom_discrete([0, 1, 2, 3], [0.1, 0.3, 0.4, 0.2], size=sample_size)
elif sampling_type == "Bootstrap (Poisson)":
    result = resample_with_replacement(poisson_df['counts'], n_samples=sample_size)
else:
    st.warning("Unsupported sampling method selected.")

# -------------------------------------
# DISPLAY RESULTS
# -------------------------------------
if result is not None:
    # Case 1: DataFrame output
    if isinstance(result, pd.DataFrame):
        st.subheader("📄 Sampled Data Preview")
        st.dataframe(result.head(20))
        csv_data = result.to_csv(index=False)
    
    # Case 2: 2D numpy array
    elif isinstance(result, np.ndarray) and result.ndim == 2:
        # If two columns, treat as DataFrame
        if result.shape[1] > 1:
            result_df = pd.DataFrame(result, columns=[f"col{i+1}" for i in range(result.shape[1])])
            st.subheader("📄 Sampled Data Preview")
            st.dataframe(result_df.head(20))
            csv_data = result_df.to_csv(index=False)
        else:
            # Single column, flatten
            result_series = pd.Series(result.flatten(), name="Sample")
            st.subheader("📄 Sampled Data Preview")
            st.write(result_series.head(20))
            csv_data = result_series.to_csv(index=False)
    
    # Case 3: List or 1D array
    else:
        result_series = pd.Series(np.array(result).flatten(), name="Sample")
        st.subheader("📄 Sampled Data Preview")
        st.write(result_series.head(20))
        csv_data = result_series.to_csv(index=False)
    
    # Common download button
    st.download_button(
        label="⬇️ Download Sample Data (CSV)",
        data=csv_data,
        file_name=f"{sampling_type.lower()}_sample.csv",
        mime="text/csv"
    )


# -------------------------------------
# FREQUENCY PLOT FOR NON-BOOTSTRAP METHODS
# -------------------------------------
if sampling_type != "Bootstrap (Poisson)" and result is not None:
    # Convert result safely
    if isinstance(result, pd.DataFrame):
        # Choose category column or first column
        if "category" in result.columns:
            data_for_plot = result["category"]
        else:
            data_for_plot = result.iloc[:, 0]
    elif isinstance(result, np.ndarray):
        # Handle multi-column arrays
        data_for_plot = result.flatten() if result.ndim == 1 else result[:, 0]
    else:
        data_for_plot = pd.Series(result, name="Sample")

    result_series = pd.Series(data_for_plot, name="Sample")
    freq_table = result_series.value_counts(normalize=True)

    st.subheader("📊 Sampling Distribution Plot")
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Bar(
        x=freq_table.index.astype(str),
        y=freq_table.values,
        marker_color="steelblue"
    ))
    fig_freq.update_layout(
        title=f"{sampling_type} Sampling Distribution",
        xaxis_title="Category",
        yaxis_title="Relative Frequency"
    )
    st.plotly_chart(fig_freq, use_container_width=True)


# -------------------------------------
# ECDF COMPARISON (for Poisson bootstrap)
# -------------------------------------
if sampling_type == "Bootstrap (Poisson)":
    st.subheader("📈 ECDF Comparison: Original vs Resampled")
    x_emp, y_emp = compute_ecdf(poisson_df['counts'])
    x_res, y_res = compute_ecdf(result)

    fig_ecdf = go.Figure()
    fig_ecdf.add_trace(go.Scatter(x=x_emp, y=y_emp, mode='lines', name='Original Poisson'))
    fig_ecdf.add_trace(go.Scatter(x=x_res, y=y_res, mode='lines', name='Bootstrap Sample'))
    fig_ecdf.update_layout(title="ECDF Comparison", xaxis_title="Value", yaxis_title="Cumulative Probability")
    st.plotly_chart(fig_ecdf, use_container_width=True)

# -------------------------------------
# SUMMARY
# -------------------------------------
st.markdown("## ✅ Summary")
st.markdown(f"""
- Sampling Method: **{sampling_type}**
- Sample Size: **{sample_size}**
- This tool demonstrates:
  - **Uniform & Weighted Sampling** for categorical data
  - **Stratified sampling** for balanced group selection
  - **Custom discrete** distributions for full control
  - **Multinomial & Dirichlet** for probabilistic modeling
  - **Bootstrap** resampling for inference and variability estimation
- Use ECDF plots to check if your resample preserves original data characteristics.
""")
