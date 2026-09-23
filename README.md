# 🧠 SciPyMasterPro

[![CI](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/ci.yml/badge.svg)](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/ci.yml)
[![Docker](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/docker.yml/badge.svg)](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/docker.yml)
[![Docs](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/docs.yml/badge.svg)](https://satvikpraveen.github.io/ScipyMasterPro/)
[![CodeQL](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/codeql.yml/badge.svg)](https://github.com/SatvikPraveen/ScipyMasterPro/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/SatvikPraveen/ScipyMasterPro/branch/main/graph/badge.svg)](https://codecov.io/gh/SatvikPraveen/ScipyMasterPro)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**A hands-on, production-grade toolkit for mastering [SciPy](https://docs.scipy.org/doc/scipy/).**
Ten concept notebooks, a twelve-page Streamlit app, a tested utility library, reproducible synthetic
datasets, a CLI, a Docker image and a documentation site, all validated by CI on every push.

📖 **Documentation:** <https://satvikpraveen.github.io/ScipyMasterPro/>

---

## 🚀 Quick start

```bash
git clone https://github.com/SatvikPraveen/ScipyMasterPro.git
cd ScipyMasterPro
python -m venv .venv && source .venv/bin/activate     # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"                                # or: uv pip install -e ".[dev]"

scipymasterpro info            # versions of Python, NumPy, SciPy, pandas, ...
scipymasterpro app             # Streamlit app on http://localhost:8501
scipymasterpro generate-data   # regenerate the synthetic CSVs
make jupyter                   # JupyterLab for the notebooks
```

Or skip the install entirely:

```bash
docker run -p 8501:8501 -p 8888:8888 ghcr.io/satvikpraveen/scipymasterpro:latest both
```

---

## ✨ What you get

| Layer | Contents |
| --- | --- |
| **Notebooks** | 10 concept notebooks + 2 SciPy-vs-statsmodels comparisons, every one executed in CI |
| **Streamlit app** | landing page + 12 interactive pages, each smoke-tested headlessly |
| **`utils/` library** | 10 modules, NumPy-style docstrings, type hints, pure SciPy statistics |
| **Synthetic data** | 9 seeded datasets and a path-independent generator |
| **Tests** | 240+ tests: unit, [Hypothesis](https://hypothesis.readthedocs.io/) property tests, plot smoke tests, app tests, CLI tests, and tests that execute the docs' code samples |
| **CLI** | `scipymasterpro {info,app,generate-data}` |
| **Docker** | multi-stage, non-root image with health checks; multi-arch (amd64/arm64) on GHCR |
| **Docs** | MkDocs Material site with API reference generated from docstrings |
| **CI/CD** | lint, tests on Linux/macOS/Windows and Python 3.11-3.13 (+3.14 experimental), notebook execution, package build, docs build, CodeQL, Trivy, tag-driven releases |

---

## 📘 Modules

| Notebook | App page | Concepts | Key SciPy APIs |
| --- | --- | --- | --- |
| `01_descriptive_stats` | Descriptive Stats | moments, trimmed stats, robust summaries, ECDF | `stats.describe`, `skew`, `kurtosis`, `trim_mean` |
| `02_hypothesis_tests` | Hypothesis Tests | parametric and non-parametric tests, assumption checks, effect sizes, BH correction | `ttest_*`, `mannwhitneyu`, `wilcoxon`, `shapiro`, `levene` |
| `03_distribution_fitting` | Distribution Fitting | MLE fitting, PDF/CDF, goodness-of-fit | `rv_continuous.fit`, `kstest`, `anderson` |
| `04_sampling_resampling` | Sampling and Resampling | stratified, weighted, multinomial, Dirichlet, custom discrete RVs | `rv_discrete`, `dirichlet`, `multinomial` |
| `05_bootstrap_simulation` | Bootstrap Simulation | bootstrap distributions and percentile CIs | `stats.bootstrap` |
| `06_multivariate_analysis` | Multivariate Analysis | covariance, Mahalanobis distance, chi-square outlier thresholds | `spatial.distance.mahalanobis`, `stats.chi2` |
| `07_optimization_minimalization` | Optimization | unconstrained, bounded and constrained minimisation, loss surfaces | `optimize.minimize`, `minimize_scalar`, `LinearConstraint` |
| `08_linear_algebra_stats` | Linear Algebra | eigen, SVD, least squares, condition numbers | `linalg.eig`, `linalg.svd`, `linalg.lstsq` |
| `09_interpolation_curvefitting` | Interpolation and Curve Fitting | 1-D/2-D interpolation, splines, RBF, `curve_fit` | `interpolate.*`, `optimize.curve_fit` |
| `10_inference_from_raw` | Inference from Raw | CIs and t-tests from summary statistics | `stats.sem`, `stats.t`, `stats.norm` |
| `shared_pdf_ecdf` | Shared: PDF and ECDF | ECDF/PDF overlays, SciPy vs statsmodels | `kstest`, `ECDF` |
| `shared_statistical_power` | Shared: Statistical Power | power by hand vs statsmodels | `stats.norm`, `stats.t`, `TTestIndPower` |

---

## 🛠 Using the library

```python
import numpy as np
from scipy import stats

from utils.distribution_utils import fit_distribution, perform_ks_test
from utils.sim_utils import bootstrap_sample, compute_bootstrap_ci
from utils.stats_tests_utils import cohens_d_independent, run_two_sample_ttest

rng = np.random.default_rng(0)
a, b = rng.normal(5, 2, 200), rng.normal(6, 2, 200)

print(run_two_sample_ttest(a, b, equal_var=False))   # Welch t-test -> {'t_stat': ..., 'p_value': ...}
print(cohens_d_independent(a, b))                    # effect size

params = fit_distribution(a, stats.norm)             # MLE (loc, scale)
print(perform_ks_test(a, stats.norm, params))        # {'KS_stat': ..., 'p_value': ...}

boots = bootstrap_sample(a, n_iterations=2000, seed=0)
print(compute_bootstrap_ci(boots, ci=95))            # percentile CI for the mean
```

| Module | Purpose |
| --- | --- |
| `utils/stats_tests_utils.py` | t-tests, Mann-Whitney, Wilcoxon, normality and variance tests, effect sizes, Benjamini-Hochberg |
| `utils/distribution_utils.py` | MLE fitting, PDF/CDF evaluation, KS and Anderson goodness-of-fit, AIC/BIC |
| `utils/pdf_ecdf_utils.py` | manual and statsmodels ECDFs, PDF/ECDF overlays |
| `utils/sim_utils.py` | bootstrap, stratified and weighted sampling, Mahalanobis outliers |
| `utils/inference_utils.py` | SEM, confidence intervals, margin of error, sample size |
| `utils/power_utils.py` | power of z- and t-tests, Cohen's d |
| `utils/optimization_utils.py` | cost functions, constraints, `minimize` wrappers, loss surfaces |
| `utils/linear_algebra_utils.py` | eigen, SVD, least squares, matrix diagnostics |
| `utils/interpolation_utils.py` | interpolation, RBF, `curve_fit` models |
| `utils/viz_utils.py` | every plotting helper (Matplotlib, Seaborn, Plotly) |

Full API reference: <https://satvikpraveen.github.io/ScipyMasterPro/api_reference/>

---

## 🧱 Project structure

```
ScipyMasterPro/
├── notebooks/               # 10 concept notebooks
├── shared_notebooks/        # SciPy vs statsmodels comparisons
├── streamlit_app/           # app.py + pages/ (12 pages)
├── utils/                   # reusable statistics, optimisation, viz
├── scipymasterpro/          # package metadata + `scipymasterpro` CLI
├── synthetic_data/          # generator + committed CSV exports
├── tests/                   # unit, property, app, CLI, docs-example tests
├── docs/ + mkdocs.yml       # documentation site
├── exports/                 # plots and tables produced by the notebooks
├── cheatsheets/             # SciPy cheatsheet
├── docker/ + Dockerfile     # multi-stage image, entrypoint (app | jupyter | both)
├── docker-compose.yml       # Streamlit + JupyterLab services
├── .github/workflows/       # ci, docker, docs, codeql, release
├── pyproject.toml           # packaging, extras, tool config
├── requirements.txt         # loose runtime pins
└── requirements_dev.txt     # resolved lock used by the Docker image
```

---

## 🧪 Development

```bash
pip install -e ".[dev,docs]"
pre-commit install

make lint             # black, isort, ruff, bandit (what CI blocks on)
make test             # unit + property + plot tests
make test-app         # Streamlit pages
make test-notebooks   # execute every notebook
make docs-serve       # live docs at http://127.0.0.1:8000
make docker-build && make docker-up
```

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md), and the
[developer guide](https://satvikpraveen.github.io/ScipyMasterPro/developer/setup/) for the full
setup, testing and release process.

---

## 🐳 Docker

```bash
docker build -t scipymasterpro .
docker run -p 8501:8501 scipymasterpro                 # Streamlit
docker run -p 8888:8888 scipymasterpro jupyter         # JupyterLab
docker run -p 8501:8501 -p 8888:8888 scipymasterpro both
docker compose up -d                                   # both as separate services
```

Pre-built multi-arch images are published to
[`ghcr.io/satvikpraveen/scipymasterpro`](https://github.com/SatvikPraveen/ScipyMasterPro/pkgs/container/scipymasterpro)
on every push to `main` and every release tag.

---

## 🌱 Why synthetic data?

Synthetic datasets keep the focus on **concepts** rather than domain noise, make every
simulation and inference **repeatable**, let assumption violations be **constructed on purpose**,
and provide the exact edge cases the test suite needs.

---

## 📜 License

GNU General Public License v3.0. See [LICENSE](LICENSE).

## 🔗 Related projects

- [PandasPlayground](https://github.com/SatvikPraveen/PandasPlayground) — data manipulation with pandas
- [NumPyMasterPro](https://github.com/SatvikPraveen/NumPyMasterPro) — vectorisation and broadcasting
- [StatsmodelsMasterPro](https://github.com/SatvikPraveen/StatsmodelsMasterPro) — modelling and inference with statsmodels
- [SeabornMasterPro](https://github.com/SatvikPraveen/SeabornMasterPro) — statistical plotting with Seaborn
- [PlotlyVizPro](https://github.com/SatvikPraveen/PlotlyVizPro) — interactive dashboards with Plotly
