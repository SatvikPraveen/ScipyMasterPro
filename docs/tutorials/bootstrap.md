# Tutorial: Bootstrap Methods

Mirrors `05_bootstrap_simulation.ipynb` and the *Bootstrap Simulation* app page.

## Bootstrap the mean

```python
import numpy as np
from utils.sim_utils import bootstrap_sample, compute_bootstrap_ci, summarize_bootstrap

data = np.random.default_rng(1).exponential(scale=3.0, size=150)
boots = bootstrap_sample(data, n_iterations=5000, seed=1)   # array of resampled means
low, high = compute_bootstrap_ci(boots, ci=95)                 # ci is a percentage
print(summarize_bootstrap(boots, original_stat=data.mean(), ci=95))
```

## Bootstrap any statistic

```python
from utils.sim_utils import bootstrap_statistic

medians = bootstrap_statistic(data, np.median, n_resamples=5000, seed=1)
print(np.percentile(medians, [2.5, 97.5]))
```

## Plot the distribution

```python
from utils.viz_utils import plot_bootstrap_distribution

plot_bootstrap_distribution(boots, ci_bounds=(low, high), true_stat=data.mean())
```

## Property that always holds

Every bootstrap mean lies inside `[data.min(), data.max()]`. This invariant is checked
automatically by the Hypothesis property tests in `tests/test_properties.py`.
