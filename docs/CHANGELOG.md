# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-09-23

### Added
- `scipymasterpro` command line (`info`, `app`, `generate-data`) and a `scipymasterpro` package
  holding the version; the previous `scipy-app` / `scipy-generate-data` entry points now work.
- Streamlit smoke tests: every page is executed headlessly with `streamlit.testing.v1.AppTest`.
- Plot smoke tests covering every helper in `utils/viz_utils.py`.
- Hypothesis property tests (ECDF monotonicity, bootstrap bounds, effect-size antisymmetry,
  confidence-interval containment, interpolation at knots, SVD reconstruction).
- Tests that execute every Python code block in the tutorials and API overview.
- Notebook execution job in CI (`nbmake`) for all 12 notebooks.
- Documentation site content: user guide, tutorials, developer guide, per-module API reference
  generated from docstrings; deployed to GitHub Pages by `docs.yml`.
- CodeQL workflow, tag-driven release workflow (GitHub Release + optional PyPI trusted publishing).
- Multi-stage, non-root Docker image with health check and an entrypoint that runs
  `app`, `jupyter` or `both`; multi-arch (amd64/arm64) publishing to GHCR.
- `requirements_dev.txt` is now a fully resolved lock (`make freeze-deps`).
- ruff linting; Makefile targets `test-app`, `test-notebooks`, `test-all`, `lint-advisory`,
  `docker-test`.

### Changed
- Minimum Python is 3.11; CI tests 3.11-3.13 with 3.14 as an experimental job.
- Dependency floors raised (numpy>=1.26, pandas>=2.0, scipy>=1.11, streamlit>=1.50) and the
  Dependabot updates for pandas, matplotlib, plotly, jupyter-core, jupyterlab-server, pyarrow,
  cffi, executing, narwhals, ipykernel and wcwidth applied via the new lock.
- GitHub Actions updated to current majors (checkout v7, setup-python v7, upload-artifact v7,
  codecov v7, build-push-action v7, metadata-action v6, CodeQL v4); Docker workflow uses
  `docker compose` v2.
- CI lint job is now blocking (black, isort, ruff, bandit); mypy and pylint report as advisory.
- Streamlit pages use a single, cross-platform `PROJECT_ROOT` bootstrap and package-qualified
  imports; `use_container_width` replaced by `width="stretch"`.
- Synthetic data generator writes next to its own file instead of relative to the working
  directory, and exposes `main()`.
- Coverage is no longer forced through `pytest` `addopts`; use `make test-cov`.
- pre-commit configuration modernised (ruff, current hook versions, no `types-all`).

### Fixed
- `perform_ks_test`, `fit_multiple_distributions*` and the ECDF goodness-of-fit helpers failed
  on SciPy >= 1.18 (`ndtr() takes from 1 to 2 positional arguments`); they now test against the
  frozen distribution.
- `stratified_sample` dropped the stratification column on pandas >= 3; it now uses
  `GroupBy.sample`.
- Three Streamlit pages imported `streamlit_utils` unqualified and failed outside `streamlit run`.
- Duplicate `fit_distribution` definition (the name-based variant is now `fit_named_distribution`).
- Mid-module imports, unused imports and an unused walrus assignment in `utils`.
- Docker workflow failed on current runners (`docker-compose: command not found`).

## [1.0.0] - 2026-03-09

### Added
- Initial release of ScipyMasterPro
- 10 comprehensive Jupyter notebooks covering all SciPy domains:
  - 01: Descriptive Statistics (moments, ECDF, skewness, kurtosis)
  - 02: Hypothesis Tests (t-tests, Mann-Whitney U, normality tests)
  - 03: Distribution Fitting (MLE, KS tests, multiple distributions)
  - 04: Sampling & Resampling (stratified, weighted, multinomial)
  - 05: Bootstrap & Simulation (confidence intervals, resampling)
  - 06: Multivariate Analysis (PCA, Mahalanobis distance, chi²)
  - 07: Optimization & Minimization (constraints, cost functions)
  - 08: Linear Algebra & Statistics (SVD, eigendecomposition, least squares)
  - 09: Interpolation & Curve Fitting (splines, curve_fit)
  - 10: Inference from Raw Data (SEM, confidence intervals)
- 2 shared notebooks for comparative analysis:
  - PDF/ECDF comparison with statsmodels
  - Statistical power analysis
- Interactive Streamlit application with 12 pages
  - Clean UI with custom CSS styling
  - Dataset selector and filtering
  - Interactive visualizations with Plotly
  - Export functionality for plots and tables
- 10 modular utility modules:
  - `stats_tests_utils.py` - Statistical hypothesis testing
  - `distribution_utils.py` - Distribution fitting and evaluation
  - `viz_utils.py` - Statistical visualization functions
  - `optimization_utils.py` - Optimization algorithms and constraints
  - `linear_algebra_utils.py` - Matrix operations and decomposition
  - `interpolation_utils.py` - Interpolation and curve fitting
  - `pdf_ecdf_utils.py` - PDF and ECDF visualization tools
  - `inference_utils.py` - Statistical inference calculations
  - `sim_utils.py` - Simulation and bootstrap utilities
  - `power_utils.py` - Statistical power analysis
- Synthetic data generator with 6+ reproducible datasets:
  - Normal and skewed distributions
  - Mixed distributions (beta, gamma, exponential, lognormal, Poisson)
  - Multivariate Gaussian with covariance
  - Optimization test data (sinusoidal + noise)
  - Curve fitting data (exponential decay + noise)
  - Categorical counts for chi-square tests
  - Poisson discrete data
- Docker support for containerized deployment
  - Python 3.11-slim base image
  - Both JupyterLab and Streamlit in one container
  - Optimized system and Python dependencies
- Comprehensive documentation:
  - Detailed README with setup instructions
  - CONTRIBUTING.md with contributor guidelines
  - CODE_OF_CONDUCT.md for community standards
  - SciPy cheatsheet for quick reference
- Proper .gitignore covering Python, Jupyter, and OS files
- GPL-3.0 license for open-source distribution
- Consistent export structure for all plots and tables

### Project Structure
```
ScipyMasterPro/
├── notebooks/               # 10 core concept notebooks
├── shared_notebooks/        # 2 comparison notebooks
├── streamlit_app/           # Interactive web application
│   └── pages/              # 12 Streamlit pages
├── synthetic_data/          # Data generation scripts and outputs
├── utils/                   # 10 reusable utility modules
├── cheatsheets/            # SciPy syntax reference
├── exports/                # Generated plots and tables
│   ├── plots/
│   └── tables/
├── requirements.txt        # Production dependencies
├── requirements_dev.txt    # Development dependencies
├── Dockerfile             # Container configuration
├── README.md              # Project documentation
├── CONTRIBUTING.md        # Contribution guidelines
├── CODE_OF_CONDUCT.md     # Community standards
└── LICENSE               # GPL-3.0 license

```

### Features
- ✅ Pure SciPy implementation (minimal statsmodels dependency)
- ✅ Reproducible synthetic datasets (seed=42)
- ✅ Modular, reusable utility functions
- ✅ Consistent notebook structure across all modules
- ✅ Professional Streamlit UI with custom styling
- ✅ Automated plot and table exports
- ✅ Docker-ready for easy deployment
- ✅ Portfolio and interview prep ready

### Dependencies
- Python ≥ 3.10
- NumPy ≥ 1.23
- Pandas ≥ 1.5
- SciPy ≥ 1.10
- Matplotlib ≥ 3.6
- Seaborn ≥ 0.12
- Plotly ≥ 5.18
- Statsmodels ≥ 0.14
- Streamlit ≥ 1.33
- JupyterLab ≥ 3.6
- scikit-learn ≥ 1.3

---

## Release Notes

### Version 1.0.0 - Initial Release

This is the first official release of **ScipyMasterPro**, a comprehensive toolkit for mastering the SciPy library through hands-on practice with synthetic datasets, interactive notebooks, and a professional web application.

**Target Audience:**
- Data scientists learning SciPy
- Students studying statistical methods
- Professionals preparing for technical interviews
- Researchers needing quick SciPy reference materials

**Key Highlights:**
- Complete coverage of `scipy.stats`, `scipy.optimize`, `scipy.interpolate`, and `scipy.linalg`
- Clean, synthetic datasets for focused learning without domain noise
- Interactive exploration via both Jupyter and Streamlit
- Production-ready utilities for real-world statistical analysis
- Fully containerized with Docker for reproducible environments

**What's Next:**
- See CONTRIBUTING.md to get involved
- Check out the notebooks in sequential order for structured learning
- Experiment with the Streamlit app for interactive exploration
- Use utilities in your own projects by importing from the utils/ folder

---

**Full Changelog**: https://github.com/SatvikPraveen/ScipyMasterPro/commits/v1.0.0
