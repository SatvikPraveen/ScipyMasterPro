# Streamlit App

The Streamlit app exposes the same analyses as the notebooks through an interactive UI: pick a
dataset, tune parameters, and watch the statistics and plots update live.

```bash
scipymasterpro app           # after `pip install -e .`
# or
streamlit run streamlit_app/app.py
# or
docker run -p 8501:8501 ghcr.io/satvikpraveen/scipymasterpro:latest
```

Open <http://localhost:8501>.

## Pages

| Page | What you can do |
| --- | --- |
| Descriptive Stats | summary tables, skew/kurtosis, histograms, box plots, correlation heatmap |
| Hypothesis Tests | one/two-sample and paired t-tests, Mann-Whitney U, Wilcoxon, effect sizes |
| Distribution Fitting | fit normal / gamma / lognormal / beta / exponential, KS test, PDF and CDF overlays |
| Sampling and Resampling | stratified, weighted, multinomial and Dirichlet samples |
| Bootstrap Simulation | bootstrap the mean or median, percentile confidence intervals |
| Multivariate Analysis | covariance, Mahalanobis distances, 2-D and 3-D outlier views |
| Optimization | minimise quadratic and non-convex costs with bounds and constraints |
| Linear Algebra | eigendecomposition, SVD, least squares, matrix diagnostics |
| Interpolation and Curve Fitting | compare interpolators, fit exponential and Gaussian models with confidence bands |
| Inference from Raw | confidence intervals and t-tests from summary statistics |
| Shared: PDF and ECDF | SciPy vs statsmodels ECDF comparison and goodness-of-fit |
| Shared: Statistical Power | power curves for z- and t-tests |

## How the app is structured

```
streamlit_app/
├── app.py               # landing page + sidebar navigation
├── config.py            # data and export paths, theme constants
├── streamlit_utils.py   # dataset loader, save/show helpers, sidebar section
├── ui_components.py     # reusable widgets
└── pages/               # one script per module (auto-discovered by Streamlit)
```

Every page is exercised headlessly in the test suite with `streamlit.testing.v1.AppTest`
(`pytest -m app`), so a page that raises on load fails CI.
