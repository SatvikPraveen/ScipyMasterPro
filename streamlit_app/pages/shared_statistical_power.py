# streamlit_app/pages/shared_statistical_power.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from streamlit_utils import sidebar_section
from utils.power_utils import (
    compute_cohens_d,
    compute_power_z,
    compute_power_t,
    statsmodels_power
)

# -------------------------------
# 🎛️ Sidebar Configuration
# -------------------------------
sidebar_section("Statistical Power Analysis")

st.title("📊 Statistical Power — Manual vs Statsmodels")
st.markdown("""
Explore **power analysis** interactively:
- Compare **manual Z-test & T-test calculations** with `statsmodels` estimates
- Understand how **effect size, alpha, and sample size** impact test sensitivity
""")

# -------------------------------
# 🔹 User Inputs
# -------------------------------
st.subheader("🧮 Input Parameters")
mean_observed = st.number_input("Observed Mean", value=74.2)
mean_expected = st.number_input("Expected Mean (Null Hypothesis μ₀)", value=70.0)
std_dev = st.number_input("Sample Standard Deviation (σ)", value=10.0)
n = st.slider("Sample Size (n)", 5, 300, 40, step=5)
alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)
two_tailed = st.checkbox("Two-Tailed Test?", value=True)

# -------------------------------
# 🔹 Compute Effect Size
# -------------------------------
d = compute_cohens_d(mean_observed, mean_expected, std_dev)
st.metric("🎯 Effect Size (Cohen's d)", f"{d:.3f}")

# -------------------------------
# 🔹 Manual Power Calculations
# -------------------------------
st.subheader("🧮 Manual Power Estimates")
power_z = compute_power_z(d, alpha=alpha, n=n, two_tailed=two_tailed)
power_t = compute_power_t(d, alpha=alpha, n=n, two_tailed=two_tailed)

st.write(f"🔹 **Z-test Power:** {power_z:.4f}")
st.write(f"🔹 **T-test Power:** {power_t:.4f}")

# -------------------------------
# 🔹 Statsmodels Power Estimate
# -------------------------------
st.subheader("⚡ Statsmodels Power Estimate")
sm_power = statsmodels_power(effect_size=d, alpha=alpha, n=n)
st.write(f"✅ **Statsmodels Power:** {sm_power:.4f}")

# -------------------------------
# 🔹 Power Curve Visualization
# -------------------------------
st.subheader("📈 Power vs Sample Size")
sample_sizes = np.arange(5, 300, 5)
power_curve = [statsmodels_power(effect_size=d, alpha=alpha, n=n_val) for n_val in sample_sizes]

fig_curve = px.line(
    x=sample_sizes, y=power_curve,
    labels={'x': 'Sample Size (n)', 'y': 'Power'},
    title='Statistical Power Curve'
)
fig_curve.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Target Power 0.8")
st.plotly_chart(fig_curve, use_container_width=True)

# -------------------------------
# 🔹 Comparison Table
# -------------------------------
st.subheader("📊 Power Comparison Table")
df_power = pd.DataFrame({
    "Method": [
        f"Manual Z-test ({'Two' if two_tailed else 'One'}-tailed)",
        f"Manual T-test ({'Two' if two_tailed else 'One'}-tailed)",
        "Statsmodels"
    ],
    "Power": [power_z, power_t, sm_power]
})
st.dataframe(df_power)

# Effect size interpretation
effect_table = pd.DataFrame({
    "Cohen's d": [0.2, 0.5, 0.8],
    "Interpretation": ["Small effect", "Medium effect", "Large effect"]
})
st.markdown("### 🎯 Effect Size Interpretation")
st.dataframe(effect_table)

# -------------------------------
# ✅ Summary
# -------------------------------
st.markdown("## ✅ Key Takeaways")
st.markdown("""
- **Effect size (Cohen’s d)** standardizes the difference between observed and expected means  
- **Manual power (Z & T-tests)** is based on normal and t-distributions  
- `statsmodels` provides validated power estimates given test parameters  
- **Sample size directly impacts power**, visualize this with the curve above  
- Aim for **≥ 0.8 power** to reduce Type II errors in your study
""")
