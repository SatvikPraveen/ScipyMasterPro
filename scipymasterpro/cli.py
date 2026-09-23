"""
Command line interface for SciPyMasterPro.

Examples
--------
Launch the Streamlit app::

    scipymasterpro app

Regenerate the synthetic datasets::

    scipymasterpro generate-data

Print the environment summary (Python, NumPy, SciPy, ... versions)::

    scipymasterpro info
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from scipymasterpro import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "streamlit_app" / "app.py"


def _library_versions() -> dict[str, str]:
    """Return a mapping of core scientific libraries to their installed versions."""
    versions: dict[str, str] = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "plotly",
        "statsmodels",
        "streamlit",
    ):
        try:
            versions[name] = import_module(name).__version__
        except Exception:  # pragma: no cover - only hit when an optional dep is missing
            versions[name] = "not installed"
    return versions


def cmd_info(_: argparse.Namespace) -> int:
    """Print version information for the toolkit and its dependencies."""
    print(f"SciPyMasterPro {__version__}")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    for name, ver in _library_versions().items():
        print(f"  {name:<12} {ver}")
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    """Run the Streamlit application."""
    if not APP_PATH.exists():
        print(f"Streamlit app not found at {APP_PATH}", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        f"--server.port={args.port}",
        f"--server.address={args.address}",
    ]
    if args.headless:
        cmd.append("--server.headless=true")
    return subprocess.call(cmd)  # noqa: S603 - arguments are built from validated CLI inputs


def cmd_generate_data(args: argparse.Namespace) -> int:
    """Regenerate every synthetic dataset."""
    from synthetic_data.generate_synthetic_data import EXPORT_DIR, export_all_datasets

    export_all_datasets(args.output or EXPORT_DIR)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``scipymasterpro`` command."""
    parser = argparse.ArgumentParser(
        prog="scipymasterpro",
        description="SciPyMasterPro: notebooks, utilities and an interactive app for mastering SciPy.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show toolkit and dependency versions")
    p_info.set_defaults(func=cmd_info)

    p_app = sub.add_parser("app", help="launch the Streamlit application")
    p_app.add_argument("--port", type=int, default=8501, help="port to serve on (default: 8501)")
    p_app.add_argument(
        "--address", default="localhost", help="address to bind (default: localhost)"
    )
    p_app.add_argument("--headless", action="store_true", help="do not open a browser window")
    p_app.set_defaults(func=cmd_app)

    p_gen = sub.add_parser("generate-data", help="regenerate the synthetic datasets")
    p_gen.add_argument(
        "-o", "--output", type=Path, default=None, help="directory to write CSV files to"
    )
    p_gen.set_defaults(func=cmd_generate_data)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``scipymasterpro`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def run_app() -> int:
    """Entry point for the legacy ``scipy-app`` console script."""
    return main(["app"])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
