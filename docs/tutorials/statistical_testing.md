# Tutorial: Statistical Testing

This walkthrough mirrors `02_hypothesis_tests.ipynb` and the *Hypothesis Tests* app page.

## Check assumptions first

```python
import numpy as np
from utils.stats_tests_utils import run_normality_tests, run_variance_tests

rng = np.random.default_rng(42)
a = rng.normal(10, 2, 80)
b = rng.normal(11, 2.5, 90)

print(run_normality_tests(a))      # Shapiro, D'Agostino, Anderson
print(run_variance_tests(a, b))    # Levene, Bartlett, Fligner
```

## Choose the test

| Situation | Function |
| --- | --- |
| one sample vs a known mean | `run_one_sample_ttest(data, popmean)` |
| two independent samples, equal variance | `run_two_sample_ttest(a, b, equal_var=True)` |
| two independent samples, unequal variance (Welch) | `run_two_sample_ttest(a, b, equal_var=False)` |
| paired measurements | `run_paired_ttest(before, after)` |
| non-normal independent samples | `run_mannwhitney_u_test(a, b)` |
| non-normal paired samples | `run_wilcoxon_signedrank(before, after)` |

```python
from utils.stats_tests_utils import run_two_sample_ttest, run_mannwhitney_u_test

print(run_two_sample_ttest(a, b, equal_var=False))
print(run_mannwhitney_u_test(a, b))
```

## Report an effect size, not just a p-value

```python
from utils.stats_tests_utils import cohens_d_independent, hedges_g_independent, cliffs_delta

print(cohens_d_independent(a, b))
print(hedges_g_independent(a, b))   # small-sample corrected
print(cliffs_delta(a, b))           # non-parametric
```

## Correct for multiple comparisons

```python
from utils.stats_tests_utils import p_adjust_bh

p_values = [0.001, 0.02, 0.04, 0.2]
print(p_adjust_bh(p_values))        # Benjamini-Hochberg adjusted
```
