# 🤝 Contributing to SciPyMasterPro

Thanks for your interest! SciPyMasterPro is a modular learning project for mastering
**SciPy for statistics, optimisation, linear algebra and simulation** through reusable
utilities, synthetic datasets, an interactive app and notebooks. Contributions that improve
functionality, correctness, documentation or educational clarity are all welcome.

## 📌 Ways to contribute

- 🛠 **Utilities** – new functions in `utils/`, with docstrings and tests.
- 🧪 **Notebooks** – new or improved demonstrations (they must execute cleanly: `make test-notebooks`).
- 🎨 **App pages** – Streamlit pages in `streamlit_app/pages/` (they are smoke-tested: `make test-app`).
- 📚 **Docs** – tutorials and guides in `docs/`; code samples there are executed by the test suite.
- 🐞 **Bug reports and fixes** – open an issue with a minimal reproduction, or a PR with a regression test.

## 🛠 Development setup

```bash
git clone https://github.com/<you>/ScipyMasterPro.git
cd ScipyMasterPro
python -m venv .venv && source .venv/bin/activate     # or: uv venv && source .venv/bin/activate
pip install -e ".[dev,docs]"                           # or: uv pip install -e ".[dev,docs]"
pre-commit install
scipymasterpro info                                    # sanity check
```

Useful targets (`make help` lists them all):

| Command | Purpose |
| --- | --- |
| `make format` | ruff --fix, black, isort |
| `make lint` | the blocking CI checks |
| `make test` | unit + property + plot tests |
| `make test-app` | Streamlit page smoke tests |
| `make test-notebooks` | execute all notebooks |
| `make docs-serve` | live documentation preview |
| `make freeze-deps` | regenerate `requirements_dev.txt` after changing `pyproject.toml` |

## ✅ Guidelines

1. **Branch** from `main`: `git checkout -b feat/short-description`.
2. **Style**: black (line length 100), isort, ruff. `make format` before committing.
3. **Docstrings**: NumPy style on every public function; they feed the API reference.
4. **Tests**: add or update tests for behaviour you change. Prefer a Hypothesis property test
   when the behaviour is an invariant.
5. **Determinism**: functions that draw random numbers take a `seed` argument.
6. **Dependencies**: add them to `pyproject.toml`, then run `make freeze-deps`.
7. **Changelog**: add a line under `## [Unreleased]` in `docs/CHANGELOG.md`.
8. **Commits**: focused, descriptive messages (`fix:`, `feat:`, `docs:`, `ci:` prefixes are used).

## 🔄 Pull requests

1. Run `make lint test test-app` locally; CI runs the same plus notebooks, build and docs.
2. Open the PR against `main` and fill in the template (what, why, how it was tested).
3. Include screenshots for UI or plot changes.
4. The `All checks passed` job must be green before merge.

## 📦 Releases (maintainers)

Bump `version` in `pyproject.toml`, move the `[Unreleased]` notes into a new section in
`docs/CHANGELOG.md`, merge, then tag `vX.Y.Z` and push the tag. The release workflow builds the
package, creates the GitHub Release and (when enabled) publishes to PyPI.

## 📝 Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Be kind, constructive and
credit others where due.

💡 Unsure whether an idea fits? Open an issue first and we will figure it out together.
