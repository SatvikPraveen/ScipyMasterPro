# streamlit_app/pages/07_optimization_minimization.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


import streamlit as st
import numpy as np
import pandas as pd

from utils.optimization_utils import (
    cost_quadratic,
    cost_nonconvex,
    multi_var_cost,
    get_bounds_2d,
    get_linear_constraint,
    run_minimization,
    run_scalar_minimization,
    evaluate_loss_surface
)
from utils.viz_utils import (
    plot_contour_loss_surface,
    plot_3d_loss_surface,
    plot_scalar_function
)
from streamlit_app.streamlit_utils import sidebar_section

# -------------------------------
# 📌 Page Configuration
# -------------------------------
st.title("⚙️ Optimization & Minimization Explorer")
sidebar_section("Optimization Settings")

# -------------------------------
# 🎛️ Sidebar Options
# -------------------------------
opt_type = st.sidebar.radio(
    "Select Optimization Scenario",
    ["Scalar (Convex)", "Scalar (Non-convex)", "Multivariate (Constrained)"]
)

x0_scalar = st.sidebar.slider("Initial Guess (for scalar)", -10.0, 10.0, 3.0)
x0_multi_x = st.sidebar.slider("Initial x (multivariate)", 0.0, 5.0, 1.0)
x0_multi_y = st.sidebar.slider("Initial y (multivariate)", 0.0, 5.0, 1.0)


# -------------------------------
# 🔹 Scalar Minimization (Convex/Nonconvex)
# -------------------------------
if opt_type in ["Scalar (Convex)", "Scalar (Non-convex)"]:
    fn = cost_quadratic if opt_type == "Scalar (Convex)" else cost_nonconvex
    result_scalar = run_scalar_minimization(fn)

    st.subheader("📊 Scalar Minimization Result")
    st.json({
        "Solver": "Brent",
        "Success": bool(result_scalar.success),
        "Optimal x": round(float(result_scalar.x), 4),
        "Function Value": round(result_scalar.fun, 4),
        "Iterations": result_scalar.nfev
    })

    # ✅ Add function plot visualization
    fig_scalar = plot_scalar_function(
        fn,
        x_range=(-10, 10),
        optimum=(result_scalar.x, result_scalar.fun),
        title=f"{opt_type} Cost Function"
    )
    st.pyplot(fig_scalar)


# -------------------------------
# 🔹 Multivariate Minimization
# -------------------------------
if opt_type == "Multivariate (Constrained)":
    x0 = [x0_multi_x, x0_multi_y]
    result_multi = run_minimization(
        func=multi_var_cost,
        x0=x0,
        bounds=get_bounds_2d(),
        constraints=get_linear_constraint()
    )

    st.subheader("📊 Multivariate Minimization Result")
    st.json({
        "Solver": getattr(result_multi, "method", "trust-constr"),
        "Success": bool(result_multi.success),
        "Optimal x": round(result_multi.x[0], 4),
        "Optimal y": round(result_multi.x[1], 4),
        "Function Value": round(result_multi.fun, 4),
        "Iterations": result_multi.nit
    })

    # Generate loss surface for visualization
    X, Y, Z = evaluate_loss_surface(multi_var_cost, x_range=(0, 5), y_range=(0, 5))

    st.subheader("📈 Contour Plot of Loss Surface")
    fig_contour = plot_contour_loss_surface(X, Y, Z, optimum=(result_multi.x[0], result_multi.x[1]))
    st.pyplot(fig_contour, use_container_width=True)

    st.subheader("📈 3D Loss Surface Visualization")
    fig_3d = plot_3d_loss_surface(X, Y, Z, optimum=(result_multi.x[0], result_multi.x[1], result_multi.fun))
    st.pyplot(fig_3d, use_container_width=True)

# -------------------------------
# ✅ Summary
# -------------------------------
st.markdown("## ✅ Summary")
st.markdown(f"""
- **Scenario:** `{opt_type}`
- Explored scalar (convex & non-convex) and constrained multivariate optimization.
- Used SciPy's `minimize` API for flexible optimization routines.
- Visualized loss surfaces to understand solution landscapes.
- Results shown interactively (no files saved locally).
""")
