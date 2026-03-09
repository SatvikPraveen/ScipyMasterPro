import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from streamlit_app.streamlit_utils import load_dataset, sidebar_section
from utils.stats_tests_utils import (
    compute_robust_summaries,
    compute_skewness_kurtosis,
    compute_trimmed_stats,
    summarize_descriptive_statistics,
)

# ------------------------------
# Config
# ------------------------------
MODULE_ID = "01_descriptive_stats"
sidebar_section(
    "📊 Descriptive Statistics Explorer",
    "Explore summary statistics, distribution shapes, and relationships interactively.",
)

st.title("📊 Descriptive Statistics Explorer")

# ------------------------------
# Load Dataset
# ------------------------------
filename = st.sidebar.text_input(
    "Enter dataset filename (default: normal_skewed.csv)", "normal_skewed.csv"
)
df = load_dataset(filename)

if df.empty:
    st.warning("⚠️ Dataset not found or empty. Please check the filename.")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in the dataset.")
    st.stop()

# ------------------------------
# Dataset Overview
# ------------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# ------------------------------
# Column Selection
# ------------------------------
col = st.selectbox("Choose a Numerical Column for Detailed Stats", numeric_cols)

# ------------------------------
# Summary Statistics
# ------------------------------
st.subheader("📑 Summary Statistics (All Columns)")
summary_stats = summarize_descriptive_statistics(df, numeric_cols)
st.dataframe(summary_stats)

# ------------------------------
# Skewness & Kurtosis
# ------------------------------
st.subheader("🔹 Shape Statistics (Skewness & Kurtosis)")
shape_stats = compute_skewness_kurtosis(df, numeric_cols)
st.dataframe(pd.DataFrame(shape_stats).T)

# ------------------------------
# Trimmed & Robust Statistics
# ------------------------------
trim = st.slider("Trim Percentage", 0.0, 0.5, 0.1, step=0.05)
trimmed_stats = compute_trimmed_stats(df[col], trim)
robust_stats = compute_robust_summaries(df[col])

st.subheader("✂️ Trimmed Statistics")
st.json(trimmed_stats)

st.subheader("🧠 Robust Summary Statistics")
st.json(robust_stats)

# ------------------------------
# Histograms
# ------------------------------
st.subheader("📊 Histogram")
fig_hist = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------
# Boxplots
# ------------------------------
st.subheader("📦 Boxplot")
fig_box = px.box(df, y=col, title=f"Boxplot of {col}")
st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------
# ECDF Plot
# ------------------------------
st.subheader("📈 Empirical Cumulative Distribution Function (ECDF)")

x = np.sort(df[col])
y = np.arange(1, len(x) + 1) / len(x)

fig_ecdf = px.line(
    x=x, y=y, title=f"ECDF of {col}", labels={"x": col, "y": "Cumulative Probability"}
)
st.plotly_chart(fig_ecdf, use_container_width=True)

# ------------------------------
# Pairplot & Correlation Heatmap
# ------------------------------
st.subheader("🔗 Pairplot & Correlation Heatmap (All Numeric Columns)")

# Pairplot
st.write("📌 Pairplot of numeric columns:")
pairplot_fig = sns.pairplot(df[numeric_cols])
st.pyplot(pairplot_fig)

# Correlation Heatmap
st.write("📌 Correlation Heatmap:")
corr = df[numeric_cols].corr()
fig_corr, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig_corr)

# ------------------------------
# Summary Section
# ------------------------------
st.markdown("## ✅ Summary")
st.markdown(
    f"""
- **Summary stats** show central tendency and dispersion for all numeric variables  
- **Skewness & kurtosis** quantify asymmetry and tailedness of distributions  
- **Trimmed & robust measures** provide stability against outliers  
- **ECDF** adds deeper insight into distribution shape  
- **Pairplot and heatmap** help visualize potential relationships between variables  
- Use this interactive module to complement offline analysis done in the notebook
"""
)
