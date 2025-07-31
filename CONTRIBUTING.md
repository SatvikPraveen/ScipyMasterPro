# 🤝 Contributing to SciPyMasterPro

Thank you for your interest in contributing! 🚀
SciPyMasterPro is a modular learning project aimed at mastering **SciPy for statistics, optimization, linear algebra, and simulations** through reusable utilities, synthetic datasets, interactive apps, and notebooks.

We welcome contributions that improve **functionality, performance, documentation, or educational clarity** of this project.

---

## 📌 How You Can Contribute

* 🛠 **New Features** – Add utilities, helper functions, or workflow enhancements.
* 📚 **Documentation** – Improve the README, cheatsheets, or add learning notes.
* 🧪 **Notebooks** – Create or improve Jupyter notebooks with practical demonstrations.
* 🐞 **Bug Fixes** – Identify and fix issues in code, visualizations, or logic.
* 🎨 **UX Improvements** – Enhance Streamlit dashboards, plots, or exports.
* 🧹 **Refactoring** – Clean up code for readability and maintainability.

---

## 🛠 Project Setup

1. **Fork the Repository**
   Click the "Fork" button on the top-right of this page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/<your-username>/SciPyMasterPro.git
   cd SciPyMasterPro
   ```

3. **Create a Virtual Environment**

   ```bash
   python3 -m venv scipy_env
   source scipy_env/bin/activate
   pip install -r requirements_dev.txt
   ```

4. **Run Streamlit App**

   ```bash
   streamlit run streamlit_app/app.py
   ```

5. **Run Jupyter Lab**

   ```bash
   jupyter lab --allow-root --ip=0.0.0.0 --no-browser
   ```

---

## ✅ Contribution Guidelines

1. **Fork** the repository and create your branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Follow **PEP8 coding style** and keep code modular.
3. Add **docstrings** for all new functions or modules.
4. If your contribution changes outputs or adds new functionality:

   * Update the relevant **Jupyter notebooks** or **Streamlit app pages** to demonstrate usage.
   * Update **cheatsheets or utility references** if necessary.
5. Use **descriptive commit messages** and keep commits focused.

---

## 🔄 Pull Request (PR) Process

1. **Test your changes locally**:

   * Run the Streamlit app (`streamlit run streamlit_app/app.py`)
   * Open and validate relevant notebooks in JupyterLab.
2. Check that:

   * No existing functionality is broken.
   * Exports, plots, and computed outputs remain correct.
3. Push your branch and open a PR to the `main` branch of this repository.
4. Include in your PR:

   * A **clear description** of what you changed or added.
   * Any **dependencies** introduced (if applicable).
   * Screenshots or sample outputs (for UI or visualization changes).
5. Wait for code review and feedback. 🎉

---

## 📝 Code of Conduct

We follow a **friendly and inclusive collaboration style**:

* Respect all contributors and their time.
* Keep discussions constructive and focused on the project.
* Credit others where due.

---

💡 **Tip:** If you are unsure whether a feature or idea fits the project, feel free to open an **issue** first to discuss it.

---
