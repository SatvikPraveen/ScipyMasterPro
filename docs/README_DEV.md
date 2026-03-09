# 🛠️ Developer Quick Start Guide

> **For Contributors and Developers working on ScipyMasterPro**

This guide helps you get up and running with the development environment quickly.

---

## 🚀 Quick Setup (5 Minutes)

### Option 1: Manual Setup (Recommended for Development)

```bash
# 1. Clone and navigate
git clone https://github.com/SatvikPraveen/ScipyMasterPro.git
cd ScipyMasterPro

# 2. Setup development environment (one command does everything!)
make dev

# This runs:
# - Creates virtual environment (if needed)
# - Installs all dev dependencies
# - Sets up pre-commit hooks
# - Generates synthetic data
```

### Option 2: Docker Setup (Fastest)

```bash
# Clone
git clone https://github.com/SatvikPraveen/ScipyMasterPro.git
cd ScipyMasterPro

# Start everything
make docker-up

# Access:
# - Jupyter: http://localhost:8888
# - Streamlit: http://localhost:8501
```

---

## 📋 Essential Commands

### Most Used Commands
```bash
make help               # See all available commands
make test               # Run tests with coverage
make lint               # Check code quality
make format             # Auto-format code
make clean              # Clean generated files
```

### Development Workflow
```bash
# Before starting work
source venv/bin/activate
make generate-data      # If you need fresh synthetic data

# During development
make format             # Format your code
make lint               # Check for issues
make test-fast          # Quick test run

# Before committing
make check              # Run lint + test
git add .
git commit -m "Your message"  # Pre-commit hooks run automatically
```

---

## 🧪 Testing

### Running Tests
```bash
# Run all tests
make test

# Fast tests (no coverage)
make test-fast

# With coverage report
make test-cov           # Opens HTML coverage in browser

# Specific test file
pytest tests/test_stats_tests_utils.py -v

# Specific test
pytest tests/test_stats_tests_utils.py::TestTTests::test_one_sample_ttest -v

# Run only unit tests
make test-unit

# Watch mode (re-run on file changes)
make test-watch
```

### Writing Tests
```python
# tests/test_my_module.py
import pytest
import numpy as np
from utils.my_module import my_function


class TestMyFunction:
    """Test my_function."""
    
    def test_basic_case(self, normal_data):
        """Test basic functionality."""
        result = my_function(normal_data)
        assert result > 0
    
    @pytest.mark.parametrize("value", [1, 2, 3])
    def test_multiple_values(self, value):
        """Test with different values."""
        result = my_function(np.array([value]))
        assert result is not None
```

---

## 🎨 Code Quality

### Auto-Formatting
```bash
# Format all code
make format

# This runs:
# - black (code formatter)
# - isort (import sorter)
```

### Linting
```bash
# Check everything
make lint

# Individual tools
make pylint         # Pylint
make black-check    # Check black formatting
make isort-check    # Check import sorting
make type-check     # MyPy type checking
```

### Pre-commit Hooks
```bash
# Install hooks (done automatically in make dev)
make pre-commit-install

# Run manually on all files
make pre-commit-run

# Update hooks
make pre-commit-update
```

---

## 🐳 Docker Development

### Working with Docker
```bash
# Build image
make docker-build

# Start services
make docker-up

# View logs
make docker-logs

# Jupyter logs only
make docker-logs-jupyter

# Streamlit logs only
make docker-logs-streamlit

# Open shell in container
make docker-shell

# Stop services
make docker-down

# Clean everything
make docker-clean
```

### Development with Docker
The Docker setup includes volume mounts, so changes you make locally are reflected in the container immediately.

---

## 📝 Code Style Guide

### Python Code Standards
- **Line length**: 100 characters
- **Formatter**: Black
- **Import sorting**: isort (black-compatible profile)
- **Type hints**: Required for all public functions
- **Docstrings**: Google or NumPy style

### Example: Well-Documented Function
```python
from typing import Dict, Union
import numpy as np
from scipy import stats


def calculate_effect_size(
    group1: np.ndarray,
    group2: np.ndarray,
    method: str = "cohen_d"
) -> Dict[str, Union[float, str]]:
    """
    Calculate effect size between two groups.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group data.
    group2 : np.ndarray
        Second group data.
    method : str, default="cohen_d"
        Effect size method: "cohen_d", "hedges_g", or "glass_delta".
    
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing:
        - 'effect_size': Calculated effect size
        - 'interpretation': Size interpretation (small/medium/large)
    
    Examples
    --------
    >>> group1 = np.array([1, 2, 3, 4, 5])
    >>> group2 = np.array([3, 4, 5, 6, 7])
    >>> result = calculate_effect_size(group1, group2)
    >>> print(result['effect_size'])
    1.414
    
    Notes
    -----
    Cohen's d interpretation:
    - Small: 0.2
    - Medium: 0.5
    - Large: 0.8
    
    See Also
    --------
    cohens_d_independent : Raw Cohen's d calculation
    """
    if method == "cohen_d":
        d = cohen_d_independent(group1, group2)
        interpretation = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
        return {"effect_size": d, "interpretation": interpretation}
    else:
        raise ValueError(f"Unknown method: {method}")
```

---

## 🐛 Debugging

### Common Issues

**Issue: Tests failing with import errors**
```bash
# Solution: Install in editable mode
pip install -e .
```

**Issue: Pre-commit hooks failing**
```bash
# Solution: Format code first
make format

# Or skip hooks temporarily (not recommended)
git commit --no-verify
```

**Issue: Docker container port conflicts**
```bash
# Solution: Stop existing containers
make docker-down

# Or use different ports in docker-compose.yml
```

**Issue: Outdated dependencies**
```bash
# Solution: Update deps
pip install --upgrade -r requirements.txt
```

---

## 📚 Project Structure

```
ScipyMasterPro/
├── .github/              # GitHub templates and workflows
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   └── dependabot.yml
├── docs/                 # Documentation source
│   └── index.md
├── notebooks/           # Jupyter notebooks (10 core)
├── shared_notebooks/    # Comparison notebooks (2)
├── streamlit_app/       # Streamlit application
│   ├── pages/          # App pages
│   └── app.py
├── synthetic_data/      # Data generation
│   ├── exports/        # Generated datasets
│   └── generate_synthetic_data.py
├── tests/              # Test suite
│   ├── conftest.py     # Shared fixtures
│   └── test_*.py       # Test modules
├── utils/              # Utility modules (10 files)
│   ├── stats_tests_utils.py
│   ├── distribution_utils.py
│   ├── optimization_utils.py
│   └── ...
├── pyproject.toml      # Package configuration
├── Makefile           # Development commands
├── docker-compose.yml  # Multi-service setup
├── mkdocs.yml         # Docs configuration
└── README.md          # Project overview
```

---

## 🔄 Git Workflow

### Creating a Feature
```bash
# 1. Create branch
git checkout -b feature/my-feature

# 2. Make changes and test
make test
make lint

# 3. Commit (pre-commit runs automatically)
git add .
git commit -m "feat: add new feature"

# 4. Push
git push origin feature/my-feature

# 5. Open PR on GitHub
```

### Commit Message Convention
```
feat: Add new statistical test
fix: Correct calculation in effect size
docs: Update API reference
test: Add tests for distribution fitting
refactor: Simplify optimization code
style: Format code with black
chore: Update dependencies
```

---

## 🆘 Getting Help

### Resources
- **Documentation**: Run `make mkdocs-serve` and visit `http://localhost:8000`
- **Tests**: Look at `tests/` for usage examples
- **Notebooks**: Check `notebooks/` for demonstrations
- **Issues**: Search [GitHub Issues](https://github.com/SatvikPraveen/ScipyMasterPro/issues)

### Making Changes to Documentation
```bash
# Edit files in docs/

# Serve locally to preview
make mkdocs-serve

# Build static site
make mkdocs-build

# Deploy to GitHub Pages
make mkdocs-deploy
```

---

## ⚡ Performance Tips

### Speeding Up Tests
```bash
# Run in parallel
pytest -n auto

# Run only failed tests
pytest --lf

# Skip slow tests
pytest -m "not slow"
```

### Speeding Up Docker Builds
```bash
# Use build cache
make docker-build

# Build without cache (clean build)
make docker-build-no-cache
```

---

## 🎯 Before Submitting PR

### Checklist
- [ ] **Code formatted**: `make format`
- [ ] **Tests pass**: `make test`
- [ ] **Linting passes**: `make lint`
- [ ] **Type hints added**: All new functions
- [ ] **Docstrings added**: All new functions
- [ ] **Tests added**: For new functionality
- [ ] **CHANGELOG updated**: Describe your changes
- [ ] **Documentation updated**: If changing APIs

### Quick Check
```bash
# Run everything at once
make all

# This runs:
# - install-dev
# - generate-data  
# - test
# - lint
```

---

## 📞 Contact

- **Maintainer**: Satvik Praveen
- **Issues**: [GitHub Issues](https://github.com/SatvikPraveen/ScipyMasterPro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SatvikPraveen/ScipyMasterPro/discussions)

---

**Happy coding! 🚀**
