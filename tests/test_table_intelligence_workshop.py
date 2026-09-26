# ruff: noqa: E501,I001
"""Static contract tests for the Table Intelligence workshop notebook."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_Table_Intelligence_Workshop.ipynb"
SPEC = REPO / "docs" / "table-intelligence-workshop-spec.md"


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells():
    return ["".join(cell["source"]) for cell in load()["cells"] if cell["cell_type"] == "code"]


def test_notebook_is_complete_valid_json():
    # A truncated upload once landed here as 1,000 lines of a 2,000-line notebook; json.loads catches that.
    notebook = load()
    assert notebook["nbformat"] == 4
    assert code_cells()[-1].rstrip().endswith('print(f"Outputs: {out_dir}/")')


def test_code_cells_compile():
    for cell in code_cells():
        compile(cell, "cell", "exec")


def test_spec_is_complete():
    text = SPEC.read_text(encoding="utf-8")
    assert "# 103. Workshop learning arc" in text or "# 103. " in text
    assert text.rstrip().endswith("must each be measured independently.*")


def test_metadata():
    meta = load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True


def test_clean_notebook():
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
