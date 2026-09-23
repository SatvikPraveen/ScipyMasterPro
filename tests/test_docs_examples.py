"""
Execute the Python code blocks in the documentation.

Every fenced ``python`` block in the tutorials and the API-reference index is run
in a fresh namespace. This keeps the documentation honest: a tutorial that calls a
renamed function fails here before it reaches the published site.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
DOC_FILES = sorted(DOCS.glob("tutorials/*.md")) + [DOCS / "api_reference" / "index.md"]
BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks_for(md: Path) -> list[str]:
    text = md.read_text(encoding="utf-8")
    # Blocks inside pymdownx tabs are indented by four spaces; dedent restores them.
    return [textwrap.dedent(block) for block in BLOCK_RE.findall(text)]


@pytest.mark.parametrize("md", DOC_FILES, ids=[f"{m.parent.name}/{m.stem}" for m in DOC_FILES])
def test_doc_code_blocks_execute_in_order(md: Path, monkeypatch, tmp_path):
    """Blocks within one page share a namespace, exactly as a reader typing along would."""
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__doc_example__"}
    blocks = _blocks_for(md)
    assert blocks, f"{md} has no python code blocks"
    for i, code in enumerate(blocks, start=1):
        try:
            exec(
                compile(code, f"<{md.stem}#{i}>", "exec"), namespace
            )  # noqa: S102 - trusted repo docs
        except Exception as exc:  # pragma: no cover - only on failure
            pytest.fail(
                f"{md.relative_to(DOCS.parent)} block #{i} raised {type(exc).__name__}: {exc}\n{code}"
            )
