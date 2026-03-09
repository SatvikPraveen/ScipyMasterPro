"""
Pytest configuration and shared fixtures for ScipyMasterPro tests.

This module provides reusable fixtures for testing utility functions across
the project. Fixtures include synthetic datasets, temporary directories,
and common test parameters.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Test Configuration
# ============================================================================

def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# ============================================================================
# Random Seed Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def random_seed() -> int:
    """Global random seed for reproducible tests."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed: int) -> None:
    """Automatically set random seeds before each test."""
    np.random.seed(random_seed)


# ============================================================================
# Sample Size Fixtures
# ============================================================================

@pytest.fixture(params=[10, 50, 100])
def small_sample_size(request: pytest.FixtureRequest) -> int:
    """Small sample sizes for testing."""
    return request.param


@pytest.fixture
def medium_sample_size() -> int:
    """Medium sample size for most tests."""
    return 100


@pytest.fixture
def large_sample_size() -> int:
    """Large sample size for performance tests."""
    return 1000


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================

@pytest.fixture
def normal_data(medium_sample_size: int, random_seed: int) -> np.ndarray:
    """Generate normally distributed data."""
    np.random.seed(random_seed)
    return np.random.normal(loc=0, scale=1, size=medium_sample_size)


@pytest.fixture
def skewed_data(medium_sample_size: int, random_seed: int) -> np.ndarray:
    """Generate skewed data using exponential distribution."""
    np.random.seed(random_seed)
    return np.random.exponential(scale=2.0, size=medium_sample_size)


@pytest.fixture
def uniform_data(medium_sample_size: int, random_seed: int) -> np.ndarray:
    """Generate uniformly distributed data."""
    np.random.seed(random_seed)
    return np.random.uniform(low=0, high=10, size=medium_sample_size)


@pytest.fixture
def bimodal_data(medium_sample_size: int, random_seed: int) -> np.ndarray:
    """Generate bimodal data."""
    np.random.seed(random_seed)
    n_half = medium_sample_size // 2
    part1 = np.random.normal(loc=-2, scale=0.5, size=n_half)
    part2 = np.random.normal(loc=2, scale=0.5, size=medium_sample_size - n_half)
    return np.concatenate([part1, part2])


@pytest.fixture
def paired_data(medium_sample_size: int, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate paired data with correlation."""
    np.random.seed(random_seed)
    before = np.random.normal(loc=10, scale=2, size=medium_sample_size)
    noise = np.random.normal(loc=0, scale=1, size=medium_sample_size)
    after = before + 1.5 + noise  # Mean difference of 1.5
    return before, after


@pytest.fixture
def two_group_data(
    medium_sample_size: int, random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent groups with different means."""
    np.random.seed(random_seed)
    group1 = np.random.normal(loc=5, scale=2, size=medium_sample_size)
    group2 = np.random.normal(loc=7, scale=2, size=medium_sample_size)
    return group1, group2


# ============================================================================
# DataFrame Fixtures
# ============================================================================

@pytest.fixture
def simple_dataframe(medium_sample_size: int, random_seed: int) -> pd.DataFrame:
    """Generate a simple DataFrame with numeric columns."""
    np.random.seed(random_seed)
    return pd.DataFrame(
        {
            "normal": np.random.normal(0, 1, medium_sample_size),
            "skewed": np.random.exponential(2, medium_sample_size),
            "uniform": np.random.uniform(0, 10, medium_sample_size),
        }
    )


@pytest.fixture
def multivariate_dataframe(medium_sample_size: int, random_seed: int) -> pd.DataFrame:
    """Generate a multivariate DataFrame."""
    np.random.seed(random_seed)
    mean = [0, 0, 0]
    cov = [[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]]
    data = np.random.multivariate_normal(mean, cov, size=medium_sample_size)
    return pd.DataFrame(data, columns=["var1", "var2", "var3"])


@pytest.fixture
def categorical_dataframe(medium_sample_size: int, random_seed: int) -> pd.DataFrame:
    """Generate a DataFrame with categorical and continuous variables."""
    np.random.seed(random_seed)
    return pd.DataFrame(
        {
            "category": np.random.choice(["A", "B", "C"], size=medium_sample_size),
            "value": np.random.normal(10, 2, medium_sample_size),
            "group": np.random.choice([0, 1], size=medium_sample_size),
        }
    )


# ============================================================================
# Distribution Fixtures
# ============================================================================

@pytest.fixture
def distribution_params() -> Dict[str, Dict[str, Any]]:
    """Common distribution parameters for testing."""
    return {
        "normal": {"loc": 0, "scale": 1},
        "exponential": {"scale": 2.0},
        "gamma": {"a": 2.0, "scale": 1.5},
        "beta": {"a": 2.0, "b": 5.0},
        "uniform": {"loc": 0, "scale": 10},
        "lognormal": {"s": 0.5, "scale": 1.0},
    }


@pytest.fixture
def common_distributions() -> list[Any]:
    """List of common scipy distributions for testing."""
    return [
        stats.norm(0, 1),
        stats.expon(scale=2),
        stats.gamma(a=2, scale=1.5),
        stats.beta(a=2, b=5),
        stats.uniform(0, 10),
    ]


# ============================================================================
# Optimization Fixtures
# ============================================================================

@pytest.fixture
def simple_quadratic() -> tuple:
    """Simple quadratic function for optimization testing."""

    def func(x: np.ndarray) -> float:
        return (x[0] - 3) ** 2 + (x[1] + 2) ** 2

    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([2 * (x[0] - 3), 2 * (x[1] + 2)])

    optimal = np.array([3.0, -2.0])
    optimal_value = 0.0

    return func, grad, optimal, optimal_value


@pytest.fixture
def rosenbrock_function() -> tuple:
    """Rosenbrock function for optimization testing."""

    def func(x: np.ndarray) -> float:
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    optimal = np.array([1.0, 1.0])
    optimal_value = 0.0

    return func, optimal, optimal_value


# ============================================================================
# Linear Algebra Fixtures
# ============================================================================

@pytest.fixture
def square_matrix(random_seed: int) -> np.ndarray:
    """Generate a random square matrix."""
    np.random.seed(random_seed)
    return np.random.randn(5, 5)


@pytest.fixture
def symmetric_matrix(random_seed: int) -> np.ndarray:
    """Generate a symmetric positive definite matrix."""
    np.random.seed(random_seed)
    A = np.random.randn(5, 5)
    return A @ A.T  # This ensures positive semi-definiteness


@pytest.fixture
def rectangular_matrix(random_seed: int) -> np.ndarray:
    """Generate a rectangular matrix."""
    np.random.seed(random_seed)
    return np.random.randn(10, 5)


# ============================================================================
# Interpolation Fixtures
# ============================================================================

@pytest.fixture
def interpolation_data(random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data for interpolation testing."""
    np.random.seed(random_seed)
    x = np.linspace(0, 10, 20)
    y = np.sin(x) + 0.1 * np.random.randn(20)
    return x, y


@pytest.fixture
def curve_fitting_data(random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data for curve fitting testing."""
    np.random.seed(random_seed)
    x = np.linspace(0, 5, 50)
    y_true = 2.5 * np.exp(-x / 2)
    y = y_true + 0.2 * np.random.randn(50)
    return x, y


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_export_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for export files."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "plots").mkdir()
    (export_dir / "tables").mkdir()
    return export_dir


# ============================================================================
# Tolerance Fixtures
# ============================================================================

@pytest.fixture
def numerical_tolerance() -> float:
    """Standard numerical tolerance for float comparisons."""
    return 1e-6


@pytest.fixture
def statistical_significance() -> float:
    """Standard significance level for statistical tests."""
    return 0.05


# ============================================================================
# Utility Functions
# ============================================================================

@pytest.fixture
def assert_arrays_close():
    """Fixture for comparing arrays with tolerance."""

    def _assert_close(arr1: np.ndarray, arr2: np.ndarray, rtol: float = 1e-5) -> None:
        np.testing.assert_allclose(arr1, arr2, rtol=rtol)

    return _assert_close


@pytest.fixture
def assert_dict_close():
    """Fixture for comparing dictionaries with numeric values."""

    def _assert_dict_close(
        dict1: Dict[str, float], dict2: Dict[str, float], rtol: float = 1e-5
    ) -> None:
        assert dict1.keys() == dict2.keys()
        for key in dict1.keys():
            np.testing.assert_allclose(dict1[key], dict2[key], rtol=rtol)

    return _assert_dict_close
