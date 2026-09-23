# Code Style

| Tool | Role | Blocking in CI |
| --- | --- | --- |
| black (25.11.0) | formatting, line length 100 | yes |
| isort (6.1.0) | import ordering, black profile | yes |
| ruff | linting (`E`, `F`, `W`, `B`, `UP`) | yes |
| bandit | security lint | yes |
| mypy | type checking | advisory |
| pylint | additional lint, score reported | advisory |

```bash
make format     # black + isort + ruff --fix
make lint       # what CI runs
```

Conventions:

- NumPy-style docstrings on every public function (they feed the API reference).
- Type hints on utilities. Streamlit pages are scripts and are exempt.
- Random behaviour takes a `seed` argument.
- Streamlit pages begin with the standard `PROJECT_ROOT` bootstrap so they run from any
  working directory.

`pre-commit install` runs the same checks before each commit.
