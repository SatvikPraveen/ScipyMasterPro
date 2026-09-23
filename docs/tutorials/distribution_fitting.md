# Tutorial: Distribution Fitting

Mirrors `03_distribution_fitting.ipynb` and the *Distribution Fitting* app page.

## Fit one distribution

```python
import numpy as np
from scipy import stats
from utils.distribution_utils import fit_distribution, perform_ks_test, compute_pdf, compute_cdf

data = np.random.default_rng(0).gamma(shape=2.0, scale=1.5, size=500)

params = fit_distribution(data, stats.gamma)   # (a, loc, scale) via MLE
print(params)
print(perform_ks_test(data, stats.gamma, params))
```

## Compare several candidates

```python
from utils.distribution_utils import fit_multiple_distributions

results = fit_multiple_distributions(data, [stats.norm, stats.gamma, stats.lognorm, stats.expon])
for r in sorted(results, key=lambda r: r.get("KS_stat", 1)):
    print(r["distribution"], round(r.get("KS_stat", float("nan")), 4), round(r.get("p_value", 0), 4))
```

The smallest KS statistic (largest p-value) is the best-fitting candidate.

## Visualise the fit

```python
from utils.viz_utils import plot_pdf_overlay, plot_cdf_overlay

plot_pdf_overlay(data, stats.gamma, params, title="Gamma PDF vs histogram")
plot_cdf_overlay(data, stats.gamma, params, title="Gamma CDF vs ECDF")
```

!!! note "Why the frozen distribution matters"
    Goodness-of-fit is evaluated against the *frozen* distribution `dist(*params).cdf`.
    Passing the distribution name plus `args=` to `kstest` breaks on recent SciPy releases,
    which is why the utilities always freeze first.
