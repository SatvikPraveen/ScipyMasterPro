"""Tests for the ``scipymasterpro`` command line interface."""

from __future__ import annotations

import pandas as pd
import pytest

from scipymasterpro import __version__, cli


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_info_lists_core_libraries(capsys):
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    for lib in ("numpy", "scipy", "pandas", "streamlit"):
        assert lib in out


def test_generate_data_writes_every_dataset(tmp_path):
    assert cli.main(["generate-data", "--output", str(tmp_path)]) == 0
    csvs = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert len(csvs) == 9
    df = pd.read_csv(tmp_path / "normal_skewed.csv")
    assert not df.empty


def test_app_reports_missing_script(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "APP_PATH", tmp_path / "missing.py")
    assert cli.main(["app"]) == 1


def test_app_invokes_streamlit(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert cli.main(["app", "--port", "9999", "--headless"]) == 0
    assert calls and "streamlit" in calls[0] and "--server.port=9999" in calls[0]
    assert "--server.headless=true" in calls[0]
