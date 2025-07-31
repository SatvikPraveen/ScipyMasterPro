# 🧠 SciPyMasterPro

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-darkgreen.svg)](https://www.python.org/)
[![Notebooks](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![SciPy Focused](https://img.shields.io/badge/SciPy-100%25-brightgreen.svg)](https://docs.scipy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-ff4b4b.svg)](https://streamlit.io/)
[![Synthetic Data](https://img.shields.io/badge/Data-Synthetic-lightblue.svg)](./synthetic_data/)
[![Portfolio Ready](https://img.shields.io/badge/Project-Portfolio--Ready-blueviolet.svg)](./README.md)

---

## 🎯 Project Goal

**SciPyMasterPro** is a hands-on, deep-dive project built to master the complete range of functionality offered by [`SciPy`](https://docs.scipy.org/doc/scipy/). It emphasizes **numerical computing**, **distribution fitting**, **hypothesis testing**, **optimization**, and **simulation** through **clean, synthetic data**.

This project helps you build deep fluency with `scipy.stats`, `scipy.optimize`, `scipy.interpolate`, and `scipy.linalg` — and supporting libraries.

---

## 🚀 Key Features

✅ 10 concept-driven Jupyter notebooks
✅ Interactive **Streamlit web application** for live statistical exploration
✅ All statistical logic done with pure **SciPy** (no heavy reliance on statsmodels)
✅ Modular utility functions for resampling, optimization, diagnostics
✅ Synthetic data generator for reproducible, controlled experiments
✅ Shared notebooks comparing **SciPy vs Statsmodels**
✅ Markdown cheatsheet and mastery checklist for fast recall
✅ **Docker-ready** for seamless environment setup (Jupyter + Streamlit in one container)
✅ Perfect for **interview prep**, **portfolio building**, and **teaching use cases**

---

## 🌱 Why Synthetic?

This project uses synthetic datasets to:

- ✨ Focus on **concepts**, not domain-specific noise
- 🔁 Enable **repeatable simulation and inference**
- 🧪 Make assumption validation crystal clear
- 📐 Generate precise shapes and edge cases needed for testing

---

## 🧱 Project Structure

```bash
SciPyMasterPro/
├── notebooks/               # Core concept notebooks (distribution fitting, optimization, etc.)
├── shared_notebooks/        # Comparison notebooks with statsmodels (PDF/ECDF, power analysis)
├── streamlit_app/           # Interactive Streamlit web app (hypothesis tests, inference tools)
├── synthetic_data/          # Scripts + outputs for synthetic datasets
├── utils/                   # Reusable code: bootstrapping, fitting, plotting, diagnostics
├── cheatsheets/             # Markdown cheatsheet + mastery checklist
├── exports/                 # All plots and tabular results from notebooks and app
│   ├── plots/
│   └── tables/
├── requirements.txt         # Main dependencies
├── requirements_dev.txt     # Full development environment
├── Dockerfile               # Docker environment for Jupyter + Streamlit
├── README.md                # This file
```

---

## 📘 Notebook Modules

| Notebook                        | Conceptual Focus                                           |
| ------------------------------- | ---------------------------------------------------------- |
| `01_descriptive_stats`          | Moments, trimmed stats, robust summaries                   |
| `02_hypothesis_tests`           | Parametric and nonparametric tests, assumption checks      |
| `03_distribution_fitting`       | `.fit()`, `.pdf()`, `.cdf()`, MLE                          |
| `04_sampling_resampling`        | Stratified sampling, `rv_discrete`, Dirichlet, multinomial |
| `05_bootstrap_simulation`       | Manual bootstrapping, CI, distribution shape checking      |
| `06_multivariate_analysis`      | Mahalanobis, covariance, chi², permutation tests           |
| `07_optimization_minimization`  | Minimize functions, constraints, real-world losses         |
| `08_linear_algebra_stats`       | SVD, eigen, least squares, matrix ops                      |
| `09_interpolation_curvefitting` | Splines, interpolators, `curve_fit()`                      |
| `10_inference_from_raw`         | Inference from summary stats, `sem()`, interval estimation |

---

## 🔁 Shared Notebooks with Statsmodels

| Notebook                         | Topics Compared                                    |
| -------------------------------- | -------------------------------------------------- |
| `shared_pdf_ecdf.ipynb`          | ECDF, fitted PDFs, visual fit quality              |
| `shared_statistical_power.ipynb` | Manual power analysis using SciPy vs `statsmodels` |

---

## 🧬 Synthetic Data Preview

| Dataset Source                        | Use Case                                       |
| ------------------------------------- | ---------------------------------------------- |
| `generate_normal_skewed()`            | Skew/kurtosis comparison and descriptive stats |
| `generate_mixed_distributions()`      | Distribution fitting & tail analysis           |
| `generate_multivariate_gaussian()`    | Mahalanobis distance, PCA                      |
| `generate_sample_for_optimization()`  | Optimization curve, cost function              |
| `generate_noisy_curve_fitting_data()` | Model calibration and smoothing                |
| `generate_poisson_data()`             | Discrete probability testing                   |

---

## 📊 Exports Example

```bash
exports/
├── plots/
│   ├── ecdf_vs_pdf.png
│   ├── bootstrap_distribution.png
│   └── optimization_convergence.png
├── tables/
│   ├── fitted_parameters_gamma.csv
│   ├── mahalanobis_distances.csv
│   └── power_curve_results.csv
```

---

## ✅ Cheatsheet & Mastery Checklist

📁 `cheatsheets/` includes:

- `scipy_cheatsheet.md` → syntax, use cases, formulas

---

## 🛠 Utilities in `utils/`

- **`stats_tests_utils.py`** → Wrapper for t-tests, chi², normality tests, rank-based methods
- **`distribution_utils.py`** → Fit, sample, evaluate PDFs/CDFs for multiple distributions
- **`sim_utils.py`** → Bootstrap, permutation tests, resampling utilities
- **`viz_utils.py`** → ECDF, diagnostic plots, confidence bands, linear algebra plots
- **`inference_utils.py`** → Compute SEM, confidence intervals, t-tests from summary stats
- **`linear_algebra_utils.py`** → Matrix generation, eigen decomposition, SVD, least squares solutions
- **`optimization_utils.py`** → Solve constrained and unconstrained optimization problems
- **`pdf_ecdf_utils.py`** → Manual ECDF computation, PDF–ECDF overlays, fit quality visualization
- **`power_utils.py`** → Statistical power analysis, effect size estimation, sample size planning
- **`interpolation_utils.py`** → Curve fitting, splines, polynomial interpolation

---

All results export to `exports/` automatically with timestamp/version control.

---

## 📦 Installation Instructions

```bash
# Clone repo
git clone https://github.com/SatvikPraveen/SciPyMasterPro.git
cd SciPyMasterPro

# Create virtualenv
python3 -m venv scipy_env
source scipy_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🐳 Docker Setup

Build Docker image:

```bash
docker build -t scipy-masterpro .
```

Run **Streamlit app**:

```bash
docker run -p 8501:8501 scipy-masterpro
```

Run **JupyterLab**:

```bash
docker run -p 8888:8888 scipy-masterpro
```

Run **both Streamlit + Jupyter in background**:

```bash
docker run -d -p 8501:8501 -p 8888:8888 scipy-masterpro
```

---

## 💼 Portfolio Impact

This project was designed to:

- ✅ Fill gaps from `statsmodels` and `NumPy`
- ✅ Build working fluency with `SciPy`'s major submodules
- ✅ Provide clean synthetic demonstrations of core stats ideas
- ✅ Enable faster recall via organized notebooks, exports, and cheatsheets
- ✅ Become your go-to resource for reviewing stats & optimization in interviews

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**.

> You are free to **use, study, share, and modify** this project under the terms of the GPLv3.
> Contributions are welcome and must also be licensed under GPLv3.

---

## 🙌 Acknowledgements

Thanks to the contributors of the SciPy ecosystem — especially the authors behind `scipy.stats`, `scipy.optimize`, and `scipy.linalg` — for making scientific computing accessible and extensible in Python.

---

## 🔗 Related Projects

- 📊 [PandasPlayground](https://github.com/SatvikPraveen/PandasPlayground) — Data manipulation workflows with pandas
- 🔢 [NumPyMasterPro](https://github.com/SatvikPraveen/NumPyMasterPro) — Deep dive into vectorization and broadcasting
- 📘 [StatsmodelsMasterPro](https://github.com/SatvikPraveen/StatsmodelsMasterPro) — Modeling & inference with `statsmodels`
- 🎨 [SeabornMasterPro](https://github.com/SatvikPraveen/SeabornMasterPro) — Statistical plotting with Seaborn
- 🌐 [PlotlyVizPro](https://github.com/SatvikPraveen/PlotlyVizPro) — Interactive dashboards with Plotly

---
