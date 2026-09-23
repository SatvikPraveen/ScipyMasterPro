# Utilities

The `utils` package is the reusable core of the project. The notebooks and the Streamlit app
are thin layers over these functions, so learning the package is the fastest way to get value
from the repository.

```python
from utils.sim_utils import bootstrap_sample, compute_bootstrap_ci
from utils.stats_tests_utils import run_normality_tests

boots = bootstrap_sample(data, n_iterations=2000, seed=7)
low, high = compute_bootstrap_ci(boots, ci=0.95)
print(run_normality_tests(data))
```

## Design rules

- **Pure SciPy.** Statistical logic is implemented with `scipy.stats`, `scipy.optimize`,
  `scipy.linalg` and `scipy.interpolate`; statsmodels appears only in the comparison notebooks.
- **Plain return types.** Functions return dicts, tuples, NumPy arrays or DataFrames so they can
  be dropped into any workflow.
- **Deterministic.** Every function that draws random numbers takes a `seed` argument.
- **Documented.** Every public function has a NumPy-style docstring; the
  [API reference](../api_reference/index.md) is generated from them.
- **Tested.** Unit tests, Hypothesis property tests and plot smoke tests cover the package
  (`make test`).

## Module map

| Module | Use it for |
| --- | --- |
| `stats_tests_utils` | hypothesis tests, effect sizes, multiple-comparison correction |
| `distribution_utils` | fitting distributions and checking fit quality |
| `pdf_ecdf_utils` | ECDFs and PDF/ECDF overlays |
| `sim_utils` | bootstrap and other resampling schemes, Mahalanobis outliers |
| `inference_utils` | confidence intervals and tests from summary statistics |
| `power_utils` | power and sample-size reasoning |
| `optimization_utils` | cost functions and `minimize` wrappers |
| `linear_algebra_utils` | decompositions and least squares |
| `interpolation_utils` | interpolation and curve fitting |
| `viz_utils` | all plotting helpers |
