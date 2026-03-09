# streamlit_app/pages/08_linear_algebra_stats.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.streamlit_utils import sidebar_section
from utils.linear_algebra_utils import (
    compute_determinant,
    compute_eigendecomposition,
    compute_inverse,
    compute_svd_adv,
    generate_matrix,
    least_squares_solution,
    matrix_summary_df,
)
from utils.viz_utils import plot_eigenvectors_safe, plot_residuals, plot_singular_values_safe

# -------------------------------
# 📌 Page Configuration
# -------------------------------
st.title("🔢 Linear Algebra for Statistical Modeling")
sidebar_section("Matrix Operations & Diagnostics")

# -------------------------------
# 🎛️ Sidebar Options
# -------------------------------
matrix_type = st.sidebar.selectbox("Matrix Type", ["Random Symmetric", "Tall Matrix"])
dim = st.sidebar.slider("Matrix Dimension", min_value=2, max_value=8, value=4)

# -------------------------------
# 🔹 Matrix Generation
# -------------------------------
A = generate_matrix(matrix_type, dim)
st.subheader("🧮 Generated Matrix A")
st.dataframe(A)

# -------------------------------
# 🔹 Eigen Decomposition
# -------------------------------
if matrix_type == "Random Symmetric":
    st.subheader("🧮 Eigen Decomposition")
    eigvals, eigvecs = compute_eigendecomposition(A)
    st.write("Eigenvalues:", np.round(eigvals, 4))
    st.dataframe(pd.DataFrame(eigvecs, columns=[f"v{i+1}" for i in range(len(eigvals))]))

    fig_eigen = plot_eigenvectors_safe(A, eigvals, eigvecs)
    st.pyplot(fig_eigen, use_container_width=True)

# -------------------------------
# 🔹 Singular Value Decomposition
# -------------------------------
st.subheader("🔬 Singular Value Decomposition")
U, s, VT = compute_svd_adv(A)
st.write("Singular Values:", np.round(s, 4))
fig_svd = plot_singular_values_safe(s, title="Singular Values")
st.pyplot(fig_svd, use_container_width=True)

# -------------------------------
# 🔹 Least Squares Solution
# -------------------------------
if matrix_type == "Tall Matrix":
    st.subheader("📈 Least Squares Regression (Ax = b)")
    b = np.random.randn(A.shape[0])
    x, residuals = least_squares_solution(A, b)
    st.write("Solution x:", np.round(x, 4))
    st.write("Residual Norm:", np.linalg.norm(residuals))

    fig_res = plot_residuals(A @ x, b)
    st.plotly_chart(fig_res, use_container_width=True)

    df_res = pd.DataFrame({"Predicted": A @ x, "Actual": b})
    st.dataframe(df_res)

# -------------------------------
# 🔹 Matrix Diagnostics
# -------------------------------
st.subheader("🔎 Matrix Diagnostics")
inverse = compute_inverse(A) if A.shape[0] == A.shape[1] else "Not square"
determinant = compute_determinant(A) if A.shape[0] == A.shape[1] else "Not square"
rank = np.linalg.matrix_rank(A)
trace = np.trace(A) if A.shape[0] == A.shape[1] else "Not square"
condition_number = np.linalg.cond(A)

diag_data = {
    "Determinant": [determinant],
    "Rank": [rank],
    "Trace": [trace],
    "Condition Number": [round(condition_number, 4)],
}
st.dataframe(pd.DataFrame(diag_data))

if isinstance(inverse, np.ndarray):
    st.subheader("🔄 Matrix Inverse")
    st.dataframe(pd.DataFrame(inverse))

# -------------------------------
# 🔹 Matrix Summary Table
# -------------------------------
st.subheader("📋 Matrix Summary")
summary_df = matrix_summary_df(A)
st.dataframe(summary_df)

# -------------------------------
# ✅ Summary
# -------------------------------
st.markdown("## ✅ Summary")
st.markdown(
    f"""
- Matrix Type: **{matrix_type}** | Dimension: **{dim}**
- Performed:
  - Eigen Decomposition (if symmetric)
  - Singular Value Decomposition (SVD)
  - Least Squares solution (for tall matrix)
  - Matrix diagnostics: determinant, rank, trace, condition number
- Interactive plots provided for eigenvectors, singular values, and residuals.
- No files saved locally (interactive only).
"""
)
