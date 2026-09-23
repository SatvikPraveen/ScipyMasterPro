"""
Smoke tests for the Streamlit application.

Every page is executed headlessly through :class:`streamlit.testing.v1.AppTest`.
A page passes when it renders without raising an exception. These tests are
marked ``app`` so they can be selected or deselected with ``-m app``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "streamlit_app"
PAGES = sorted((APP_DIR / "pages").glob("*.py"))
PAGES = [p for p in PAGES if p.name != "__init__.py"]

pytestmark = [pytest.mark.app, pytest.mark.integration]


def _run(script: Path) -> AppTest:
    at = AppTest.from_file(str(script), default_timeout=180)
    at.run()
    return at


def _format_exceptions(at: AppTest) -> str:
    return "\n".join(f"{e.value}\n{e.stack_trace}" for e in at.exception)


def test_home_page_renders() -> None:
    at = _run(APP_DIR / "app.py")
    assert not at.exception, _format_exceptions(at)
    assert at.title or at.markdown, "home page rendered nothing"


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_renders_without_exception(page: Path) -> None:
    at = _run(page)
    assert not at.exception, _format_exceptions(at)


def test_all_pages_are_covered() -> None:
    """Guard against silently adding a page that is not exercised here."""
    assert len(PAGES) >= 12
