# streamlit_app/pages/10_inference_from_raw.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import streamlit as st
import numpy as np
import pandas as pd
from streamlit_utils import sidebar_section, EXPORT_TABLES
from utils.inference_utils import (
    compute_sem,
    confidence_interval,
    z_confidence_interval,
    compute_t_stat,
    manual_t_test,
    margin_of_error,
    compute_sample_size
)
from utils.viz_utils import (
    plot_confidence_interval,
    plot_residuals_vs_population,
    plot_multiple_confidence_intervals
)

# -------------------------------
# 📌 Constants
# -------------------------------
MODULE_ID = "10_inference_from_raw"
TABLE_PATH = EXPORT_TABLES / MODULE_ID
TABLE_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 🎛️ Sidebar
# -------------------------------
sidebar_section("Inference from Summary Statistics")

inference_type = st.sidebar.selectbox(
    "Inference Type",
    [
        "Compute SEM",
        "Confidence Interval (t)",
        "Confidence Interval (z)",
        "Multiple CI Levels",
        "Margin of Error",
        "Manual t-Test",
        "Power Analysis"
    ]
)

st.title("🧩 Inference from Summary Statistics")
st.markdown("Perform estimation, hypothesis testing, and sample size planning using only **mean, SD, and n** — no raw data required.")

# -------------------------------
# 🔹 Input Section
# -------------------------------
st.subheader("🧮 Input Parameters")
mean = st.number_input("Sample Mean", value=72.5)
std = st.number_input("Sample Standard Deviation (SD)", value=10.0)
n = st.number_input("Sample Size (n)", value=30, min_value=1)
confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)

# Optional inputs
pop_mean = None
if inference_type in ["Manual t-Test", "Power Analysis"]:
    pop_mean = st.number_input("Population / Hypothesized Mean (μ₀)", value=70.0)

pop_std = None
if inference_type == "Confidence Interval (z)":
    pop_std = st.number_input("Population Standard Deviation (σ)", value=9.5)

# -------------------------------
# 🔍 Inference Computation
# -------------------------------
st.subheader("📊 Inference Output")

result = {
    "mean": mean,
    "std_dev": std,
    "n": n,
    "confidence": confidence,
    "type": inference_type
}

if inference_type == "Compute SEM":
    sem = compute_sem(std, n)
    st.info(f"Standard Error of Mean (SEM): **{sem:.4f}**")
    result.update({"SEM": sem})

elif inference_type == "Confidence Interval (t)":
    lower, upper = confidence_interval(mean, std, n, confidence)
    st.success(f"t-Confidence Interval: **[{lower:.2f}, {upper:.2f}]**")
    fig = plot_confidence_interval(mean, (lower, upper), pop_mean=None)
    st.pyplot(fig, use_container_width=True)
    result.update({"lower_bound": lower, "upper_bound": upper})

elif inference_type == "Confidence Interval (z)":
    if not pop_std:
        st.warning("Population standard deviation is required for z-based CI.")
    else:
        lower, upper = z_confidence_interval(mean, pop_std, n, confidence)
        st.success(f"z-Confidence Interval: **[{lower:.2f}, {upper:.2f}]**")
        fig = plot_confidence_interval(mean, (lower, upper), pop_mean=None)
        st.pyplot(fig, use_container_width=True)
        result.update({"lower_bound": lower, "upper_bound": upper})


elif inference_type == "Multiple CI Levels":
    levels = [0.90, 0.95, 0.99]
    # ✅ Keep keys as floats for plotting
    ci_results = {lvl: confidence_interval(mean, std, n, lvl) for lvl in levels}

    # Convert to DataFrame (convert float to string only for the table)
    df_ci = pd.DataFrame(
        [(f"{lvl:.0%}", bounds[0], bounds[1]) for lvl, bounds in ci_results.items()],
        columns=["Confidence_Level", "CI_Low", "CI_High"]
    )
    st.dataframe(df_ci)

    # Plot using float keys
    fig = plot_multiple_confidence_intervals(mean, ci_results)
    st.pyplot(fig, use_container_width=True)

    # Save results
    df_ci.to_csv(TABLE_PATH / "multiple_confidence_intervals.csv", index=False)
    result.update({"multi_CI": {f"{lvl:.0%}": bounds for lvl, bounds in ci_results.items()}})


elif inference_type == "Margin of Error":
    moe = margin_of_error(std, n, confidence)
    st.success(f"Margin of Error: **±{moe:.3f}**")
    result.update({"Margin_of_Error": moe})


elif inference_type == "Manual t-Test":
    t_stat = compute_t_stat(mean, pop_mean, std, n)
    
    # ✅ Manual t-test returns a tuple (t_statistic, p_value)
    t_stat_result, p_val = manual_t_test(t_stat, n - 1, std, n)
    
    st.success(f"t-statistic = **{t_stat_result:.3f}**, p-value = **{p_val:.4f}**")
    
    fig_res = plot_residuals_vs_population(mean, pop_mean, mean - pop_mean)
    st.pyplot(fig_res, use_container_width=True)
    result.update({"t_stat": t_stat_result, "p_value": p_val})


elif inference_type == "Power Analysis":
    effect_size = abs(mean - pop_mean) / std
    req_n = compute_sample_size(effect_size)
    st.success(f"Effect Size (Cohen's d): **{effect_size:.3f}**")
    st.info(f"Required Sample Size for 80% Power: **{req_n:.2f}**")
    result.update({"Effect_Size": effect_size, "Required_Sample_Size": req_n})

# -------------------------------
# 💾 Display Table
# -------------------------------
df = pd.DataFrame([result])
st.dataframe(df)

# -------------------------------
# ✅ Summary
# -------------------------------
st.markdown("## ✅ Summary")
st.markdown("""
- Supports multiple inference tasks: **SEM, t/z confidence intervals, margin of error, manual t-tests, power analysis**  
- Confidence intervals and test results visualized when applicable  
- Suitable for cases where only summary statistics are available (no raw dataset)
""")
