# 🧠 SciPyMasterPro Cheatsheet — Master-Level Statistical Utilities

A categorized, **ready-to-copy reference** for all utility functions available in **SciPyMasterPro**, making it quick to run advanced statistical workflows inside your Streamlit or Jupyter environment.

---

## 📦 **1️⃣ Linear Algebra Utilities (`linear_algebra_utils.py`)**

### 🏗️ Matrix Generation

```python
generate_matrix("Random Symmetric", dim=4)
generate_matrix("Tall Matrix", dim=4)
```

- Creates symmetric or tall matrices for linear algebra demos.

### 🔹 Eigen Decomposition

```python
eigvals, eigvecs = compute_eigendecomposition(A)
```

- Computes eigenvalues & eigenvectors of a square matrix.

### 🔹 Singular Value Decomposition (SVD)

```python
U, s, VT = compute_svd_adv(A)
```

- Advanced, type-safe SVD returning U, singular values, and VT.

### 🔹 Least Squares Solution

```python
x, residuals = least_squares_solution(A, b)
```

- Solves **Ax = b** for over-determined systems with residual output.

### 🧮 Matrix Diagnostics

```python
compute_inverse(A)
compute_determinant(A)
matrix_summary_df(A)
```

- Quickly inspect determinant, rank, trace, condition number, inverse.

### 💻 Syntax Examples

```python
# Generate 4x4 symmetric matrix
A = generate_matrix("Random Symmetric", dim=4)

# Eigen decomposition
eigvals, eigvecs = compute_eigendecomposition(A)
print(eigvals)      # → array([...])
print(eigvecs)      # → 2D eigenvector matrix

# Singular Value Decomposition
U, s, VT = compute_svd_adv(A)
print(s)            # → singular values

# Solve Least Squares Ax = b
b = np.random.randn(A.shape[0])
x, residuals = least_squares_solution(A, b)

# Diagnostics table
matrix_summary_df(A)
```

---

## ⚙️ **2️⃣ Optimization Utilities (`optimization_utils.py`)**

### 🔹 Scalar Cost Functions

```python
cost_quadratic(x)      # Convex quadratic cost
cost_nonconvex(x)      # Non-convex function
```

### 🔹 Minimization Routines

```python
run_scalar_minimization(fn)
run_minimization(func, x0, bounds, constraints)
```

- Uses SciPy's Brent or trust-constr solvers to find minima.

### 🎨 Visualization

```python
plot_scalar_function(fn, x_range=(-10,10), optimum=(x_opt, f_opt))
plot_contour_loss_surface(X, Y, Z, optimum=(x,y))
plot_3d_loss_surface(X, Y, Z, optimum=(x,y,z))
```

---

## 📊 **3️⃣ Hypothesis Testing (`stats_tests_utils.py`)**

### ✅ Assumption Checks

```python
perform_shapiro_test(series)      # Normality test
levene_variance_test(x, y)        # Equal variance test
```

### 🔹 Parametric Tests

```python
run_one_sample_ttest(x, popmean=50)
run_two_sample_ttest(x, y, equal_var=False)
run_paired_ttest(before, after)
```

### 🔹 Non-Parametric Tests

```python
run_mannwhitney_u_test(x, y)
run_wilcoxon_signedrank(x, y)
run_spearman_correlation(x, y)
run_kendall_tau(x, y)
rank_biserial_effect_size(x, y)
```

### 📏 Effect Sizes

```python
cohens_d_independent(x, y)
hedges_g_independent(x, y)
cliffs_delta(x, y)
```

### 💻 Hypothesis Testing Related Syntax Examples

```python
# Normality test
perform_shapiro_test(df["col1"])

# Independent t-test
run_two_sample_ttest(df["col1"], df["col2"], equal_var=False)

# Paired t-test
run_paired_ttest(before, after)

# Mann–Whitney
run_mannwhitney_u_test(group1, group2)

# Effect sizes
cohens_d_independent(group1, group2)
hedges_g_independent(group1, group2)
cliffs_delta(group1, group2)
```

---

## 🧮 **4️⃣ Inference Utilities (`inference_utils.py`)**

### 🔹 Standard Errors & Margins

```python
compute_sem(std, n)
margin_of_error(std, n, confidence=0.95)
```

### 🔹 Confidence Intervals

```python
confidence_interval(mean, std, n, conf=0.95)   # t-based
z_confidence_interval(mean, pop_std, n, conf=0.95)
plot_confidence_interval(mean, (low, high))
plot_multiple_confidence_intervals(mean, ci_dict)
```

### 🔹 Hypothesis Tests

```python
compute_t_stat(sample_mean, pop_mean, std, n)
manual_t_test(t_stat, df, sample_std, n)
```

### 🔹 Power Analysis

```python
compute_sample_size(effect_size, power=0.8)
```

### 💻 Inference Utilities Related Syntax Examples

```python
# t-based CI
confidence_interval(mean=72, std=10, n=30, conf=0.95)
# → (lower, upper)

# z-based CI
z_confidence_interval(mean=72, pop_std=9.5, n=30)

# Standard error
compute_sem(std=12.4, n=50)

# Margin of error
margin_of_error(std=12, n=60, confidence=0.99)

# Manual t-test
t_stat = compute_t_stat(72, 70, 10, 30)
p_val = manual_t_test(t_stat, df=29, sample_std=10, n=30)

# Power analysis
compute_sample_size(effect_size=0.6)
```

---

## 📈 **5️⃣ Interpolation & Curve Fitting (`interpolation_utils.py`)**

### 🔹 1D Interpolation

```python
linear_interpolate(x, y)
cubic_interpolate(x, y)
spline_interpolate(x, y)
```

### 🔹 Curve Fitting

```python
fit_curve(exponential_model, x, y)
safe_gaussian_fit(x, y)
```

### 🔹 2D Interpolation

```python
interpolate_2d(x, y, z, method="linear")
rbf_interpolation(x, y, z)
```

### 🎨 Related Visualization

```python
plot_interpolation_comparison(x, y, x_new, y_lin, y_cub, y_spl)
plot_curve_fits_with_bands(...)
plot_all_fits_comparison(...)
plot_rbf_interpolation(...)
```

### 💻 Interpolation & Curve-fitting related Syntax Examples

```python
# Linear interpolation
fn_lin = linear_interpolate(x, y)
y_new = fn_lin(x_new)

# Gaussian fit
params, _ = safe_gaussian_fit(x, y)

# 2D interpolation
grid_x, grid_y, grid_z = interpolate_2d(x, y, z, method="linear")

# RBF interpolation
z_rbf = rbf_interpolation(x, y, z)
```

---

## 📊 **6️⃣ PDF & ECDF Utilities (`pdf_ecdf_utils.py`)**

### 🔹 Empirical Distributions

```python
compute_manual_ecdf(data)
compute_statsmodels_ecdf(data)
```

### 🔹 PDF and CDF

```python
get_pdf(data, norm)
plot_pdf_ecdf_overlay(data, norm)
plot_enhanced_ecdf_comparison(data, [norm, gamma, lognorm])
```

### 🔹 Goodness of Fit

```python
run_goodness_of_fit_tests(data, norm)
```

### 💻 PDF & ECDF Related Syntax Examples

```python
# Compute ECDF manually
x_ecdf, y_ecdf = compute_manual_ecdf(data)

# ECDF using statsmodels
x_sm, y_sm = compute_statsmodels_ecdf(data)

# Plot PDF & ECDF
plot_pdf_ecdf_overlay(data, norm)

# Compare multiple distributions
plot_enhanced_ecdf_comparison(data, [norm, gamma, lognorm])
```

---

## 🎨 **7️⃣ Visualization Utilities (`viz_utils.py`)**

### 📦 Linear Algebra Plots

```python
plot_singular_values_safe(s)
plot_residuals(pred, actual)
plot_eigenvectors_safe(A, eigvals, eigvecs)
```

### 📦 Interpolation Plots

```python
plot_polynomial_fit(x, y, x_new, y_poly)
plot_weighted_vs_unweighted_fit(...)
plot_residuals_comparison(...)
```

### 📦 Optimization Plots

```python
plot_scalar_function(fn)
plot_contour_loss_surface(X,Y,Z)
plot_3d_loss_surface(X,Y,Z)
```

### 💻 Visualization Syntax Examples

```python
# Singular values plot
plot_singular_values_safe(s)

# Residual plot for Ax = b
plot_residuals(A @ x, b)

# Eigenvector plot
plot_eigenvectors_safe(A, eigvals, eigvecs)

# Polynomial fit visualization
plot_polynomial_fit(x, y, x_new, y_poly)
```

---

## 💾 **8️⃣ Export Helpers**

```python
df.to_csv(EXPORT_TABLES / "filename.csv")
st.download_button("⬇️ Download Result", data=df.to_csv())
```

- Exports results and tables from modules interactively.

---

## 🧠 **Tips for Mastery**

- Use **modular utils** across multiple pages: reusable in Jupyter and Streamlit.
- Chain **assumption checks → tests → effect sizes** for robust inference.
- Combine **interpolation + hypothesis testing + optimization** for end-to-end experiments.
- Use `least_squares_solution()` and residual plots for regression diagnostics.
- Keep exports versioned for reproducibility.

---
