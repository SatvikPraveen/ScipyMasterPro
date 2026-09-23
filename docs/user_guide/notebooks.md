# Notebooks

Ten concept notebooks in `notebooks/` build up SciPy fluency one topic at a time, and two
notebooks in `shared_notebooks/` compare SciPy against statsmodels. Every notebook is executed
end-to-end in CI (`pytest --nbmake`), so they are guaranteed to run on the supported
dependency versions.

| # | Notebook | Focus | Key SciPy APIs |
| --- | --- | --- | --- |
| 01 | `01_descriptive_stats` | moments, trimmed stats, robust summaries, ECDF | `stats.describe`, `stats.skew`, `stats.kurtosis`, `stats.trim_mean` |
| 02 | `02_hypothesis_tests` | parametric and non-parametric tests, assumption checks, effect sizes | `ttest_*`, `mannwhitneyu`, `wilcoxon`, `shapiro`, `levene` |
| 03 | `03_distribution_fitting` | MLE fitting, PDF/CDF, goodness-of-fit | `rv_continuous.fit`, `kstest`, `anderson` |
| 04 | `04_sampling_resampling` | stratified and weighted sampling, custom discrete RVs | `rv_discrete`, `dirichlet`, `multinomial` |
| 05 | `05_bootstrap_simulation` | bootstrap distributions and confidence intervals | `stats.bootstrap`, percentiles |
| 06 | `06_multivariate_analysis` | covariance, Mahalanobis distance, chi-square outlier thresholds | `spatial.distance.mahalanobis`, `stats.chi2` |
| 07 | `07_optimization_minimalization` | unconstrained, bounded and constrained minimisation, loss surfaces | `optimize.minimize`, `minimize_scalar`, `LinearConstraint` |
| 08 | `08_linear_algebra_stats` | eigen, SVD, least squares, condition numbers | `linalg.eig`, `linalg.svd`, `linalg.lstsq` |
| 09 | `09_interpolation_curvefitting` | 1-D/2-D interpolation, splines, RBF, `curve_fit` | `interpolate.*`, `optimize.curve_fit` |
| 10 | `10_inference_from_raw` | inference from summary statistics | `stats.sem`, `stats.t`, `stats.norm` |
| S1 | `shared_pdf_ecdf` | ECDF and PDF overlays: SciPy vs statsmodels | `kstest`, `ECDF` |
| S2 | `shared_statistical_power` | power analysis by hand vs statsmodels | `stats.norm`, `stats.t`, `TTestIndPower` |

## Running them

```bash
make jupyter                 # JupyterLab from the project root
make run-notebooks           # execute every notebook with nbmake (same as CI)
```

Each notebook writes its figures to `exports/plots/<module>/` and its tables to
`exports/tables/<module>/`. Those exports are committed, so you can browse results on GitHub
without running anything.

!!! tip "Working directory"
    Notebooks resolve data with paths relative to the repository root. Start JupyterLab from the
    root (the `make jupyter` target does this) or the CSV loads will fail.
