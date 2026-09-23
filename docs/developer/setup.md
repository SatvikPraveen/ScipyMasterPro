# Development Setup

```bash
git clone https://github.com/SatvikPraveen/ScipyMasterPro.git
cd ScipyMasterPro
uv venv --python 3.12 && source .venv/bin/activate   # or python -m venv .venv
uv pip install -e ".[dev,docs]"                       # or pip install -e ".[dev,docs]"
pre-commit install
scipymasterpro info
```

`make help` lists every automation target.

## Dependency files

| File | Purpose |
| --- | --- |
| `pyproject.toml` | source of truth: runtime deps and the `dev`, `docs`, `notebook` extras |
| `requirements.txt` | loose runtime pins for quick installs and CI |
| `requirements_dev.txt` | fully resolved lock (`make freeze-deps`) used by the Docker image |

Regenerate the lock after changing `pyproject.toml`:

```bash
make freeze-deps
```

## Python versions

Python 3.11 through 3.13 are tested on every push; 3.14 runs as an experimental job.
