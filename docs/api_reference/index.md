# API Reference

The reusable code lives in the `utils` package. Every public function has a NumPy-style
docstring, and the pages in this section are generated directly from those docstrings
with [mkdocstrings](https://mkdocstrings.github.io/), so they never drift from the code.

| Module | What it covers |
| --- | --- |
| [`stats_tests_utils`](stats_tests.md) | t-tests, Mann-Whitney, Wilcoxon, normality and variance tests, effect sizes, BH correction |
| [`distribution_utils`](distribution.md) | MLE fitting, PDF/CDF evaluation, KS and Anderson goodness-of-fit |
| [`pdf_ecdf_utils`](pdf_ecdf.md) | ECDF computation and PDF/ECDF overlays |
| [`sim_utils`](simulation.md) | bootstrap, stratified/weighted sampling, Mahalanobis outliers |
| [`inference_utils`](inference.md) | SEM, confidence intervals, margin of error, sample size |
| [`power_utils`](power.md) | power of z- and t-tests, Cohen's d |
| [`optimization_utils`](optimization.md) | cost functions, constraints, `minimize` wrappers, loss surfaces |
| [`linear_algebra_utils`](linear_algebra.md) | eigen, SVD, least squares, matrix diagnostics |
| [`interpolation_utils`](interpolation.md) | 1-D/2-D interpolation, RBF, `curve_fit` models |
| [`viz_utils`](visualization.md) | every plotting helper |
| [`scipymasterpro.cli`](cli.md) | the command line interface |

```python
import numpy as np
from scipy import stats

from utils.distribution_utils import fit_distribution, perform_ks_test
from utils.stats_tests_utils import cohens_d_independent, run_two_sample_ttest

a = np.random.default_rng(0).normal(5, 2, 200)
b = np.random.default_rng(1).normal(6, 2, 200)

print(run_two_sample_ttest(a, b))            # {'t_stat': ..., 'p_value': ...}
print(cohens_d_independent(a, b))            # ~ -0.5
params = fit_distribution(a, stats.norm)      # (loc, scale)
print(perform_ks_test(a, stats.norm, params)) # {'KS_stat': ..., 'p_value': ...}
```
