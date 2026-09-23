# Getting Started

## 1. Install

=== "pip"

    ```bash
    git clone https://github.com/SatvikPraveen/ScipyMasterPro.git
    cd ScipyMasterPro
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    ```

=== "uv"

    ```bash
    git clone https://github.com/SatvikPraveen/ScipyMasterPro.git
    cd ScipyMasterPro
    uv venv && source .venv/bin/activate
    uv pip install -e ".[dev]"
    ```

=== "Docker"

    ```bash
    docker run -p 8501:8501 -p 8888:8888 ghcr.io/satvikpraveen/scipymasterpro:latest both
    ```

## 2. Check the environment

```bash
scipymasterpro info
```

## 3. Pick an entry point

- **Explore interactively:** `scipymasterpro app` and open <http://localhost:8501>.
- **Read and run the notebooks:** `make jupyter`, then start with `01_descriptive_stats.ipynb`.
- **Use the utilities in your own code:**

```python
import numpy as np
from utils.inference_utils import confidence_interval

data = np.random.default_rng(0).normal(50, 8, size=120)
print(confidence_interval(data.mean(), data.std(ddof=1), data.size, confidence=0.95))
```

## 4. Run the test suite

```bash
make test        # unit + property + plot smoke tests
make test-app    # Streamlit pages
make test-notebooks
```
