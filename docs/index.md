# SciPyMasterPro

**A hands-on toolkit for mastering SciPy**: ten concept notebooks, an interactive Streamlit
app, a tested utility library and reproducible synthetic datasets, all wired into a modern
CI/CD pipeline.

[![CI](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/ci.yml/badge.svg)](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/ci.yml)
[![Docker](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/docker.yml/badge.svg)](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/docker.yml)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Start here

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Install with pip, uv or Docker and run your first analysis in five minutes.

    [:octicons-arrow-right-24: Getting started](tutorials/getting_started.md)

-   :material-notebook: **Notebooks**

    ---

    Ten notebooks from descriptive statistics to curve fitting, each executed in CI.

    [:octicons-arrow-right-24: Notebook guide](user_guide/notebooks.md)

-   :material-application: **Streamlit App**

    ---

    Twelve interactive pages covering every module.

    [:octicons-arrow-right-24: App guide](user_guide/streamlit_app.md)

-   :material-api: **API Reference**

    ---

    Every utility function, generated from its docstring.

    [:octicons-arrow-right-24: API reference](api_reference/index.md)

</div>

## What is inside

| Layer | Contents |
| --- | --- |
| `notebooks/`, `shared_notebooks/` | 10 concept notebooks + 2 SciPy-vs-statsmodels comparisons |
| `streamlit_app/` | landing page + 12 interactive pages |
| `utils/` | 10 modules, ~4,300 lines, NumPy-style docstrings, type hints |
| `synthetic_data/` | 9 reproducible datasets and their generator |
| `tests/` | unit, Hypothesis property, plot smoke, app, CLI and docs-example tests |
| `scipymasterpro/` | the `scipymasterpro` command line |

## Quick taste

```python
import numpy as np
from scipy import stats

from utils.distribution_utils import fit_distribution, perform_ks_test
from utils.sim_utils import bootstrap_sample, compute_bootstrap_ci

data = np.random.default_rng(0).gamma(2.0, 1.5, size=400)

params = fit_distribution(data, stats.gamma)
print(perform_ks_test(data, stats.gamma, params))

boots = bootstrap_sample(data, n_iterations=2000, seed=0)
print(compute_bootstrap_ci(boots, ci=95))
```

## Why synthetic data?

- concepts stay in focus, not domain noise
- every experiment is repeatable
- assumption violations can be constructed on purpose
- edge cases needed for tests are one function call away

## Project links

- [Repository](https://github.com/SatvikPraveen/ScipyMasterPro)
- [Issues](https://github.com/SatvikPraveen/ScipyMasterPro/issues)
- [Container image](https://github.com/SatvikPraveen/ScipyMasterPro/pkgs/container/scipymasterpro)
- [Changelog](about/changelog.md)
