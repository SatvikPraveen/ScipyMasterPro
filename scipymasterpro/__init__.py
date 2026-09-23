"""
SciPyMasterPro: a hands-on toolkit for mastering SciPy.

The statistical utilities live in :mod:`utils`, the interactive app in
:mod:`streamlit_app`, and the reproducible datasets in :mod:`synthetic_data`.
This package hosts the version number and the ``scipymasterpro`` command line.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scipymasterpro")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "1.1.0"

__all__ = ["__version__"]
