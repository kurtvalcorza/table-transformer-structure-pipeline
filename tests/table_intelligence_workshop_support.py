# ruff: noqa: E501
"""Shared helpers for the Table Intelligence workshop tests (revision 0.3.0: uv isolated environment).

The workshop's model code lives in ``tools/table_intelligence_workshop.py``, which the notebook carries
byte-for-byte and runs stage by stage in its isolated environment. Tests load a fresh copy of that module,
set its globals the way ``configure()`` and the restored stage state would, and call stage functions
directly with stand-ins for the models (CI installs torch, transformers, numpy and pillow only).
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_Table_Intelligence_Workshop.ipynb"
STAGE_FILE = ROOT / "tools" / "table_intelligence_workshop.py"
LOCK_FILE = ROOT / "tools" / "table-intelligence-workshop-requirements.lock"
REQUIREMENTS_IN = ROOT / "tools" / "table-intelligence-workshop-requirements.in"
_counter = itertools.count()


def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def cells() -> list[dict]:
    return notebook()["cells"]


def joined(cell: dict) -> str:
    return "".join(cell["source"])


def cell_by_id(cell_id: str) -> str:
    return joined(next(c for c in cells() if c["id"] == cell_id))


def markdown(marker: str) -> str:
    matches = [joined(c) for c in cells() if c["cell_type"] == "markdown" and marker in joined(c)]
    assert len(matches) == 1, f"{marker!r} matched {len(matches)} markdown cells"
    return matches[0]


def host_cell(marker: str) -> str:
    """The one notebook code cell (other than the carrier, which holds the stage file) containing marker."""
    matches = [joined(c) for c in cells() if c["cell_type"] == "code" and c["id"] != "uvcarrier" and marker in joined(c)]
    assert len(matches) == 1, f"{marker!r} matched {len(matches)} code cells"
    return matches[0]


def stage_text() -> str:
    return STAGE_FILE.read_text(encoding="utf-8")


def fn_src(name: str) -> str:
    """Source of one top-level function of the carried stage file."""
    text = stage_text()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{name} is not a top-level function of {STAGE_FILE.name}")


def carried_files() -> dict[str, str]:
    """The files the notebook's carrier cell writes, evaluated from the cell itself."""
    module = ast.parse(cell_by_id("uvcarrier"))
    for node in module.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "CARRIED_FILES":
            return ast.literal_eval(node.value)
    raise AssertionError("CARRIED_FILES not found in the carrier cell")


def load_stages(tmp_path: Path | None = None, **overrides) -> dict:
    """A fresh copy of the carried stage module; returns its globals (mutations reach its functions)."""
    spec = importlib.util.spec_from_file_location(f"table_intelligence_workshop_{next(_counter)}", STAGE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ns = vars(module)
    if tmp_path is not None:
        ns.update(OUTPUT_DIR=str(tmp_path / "outputs"), WORK_ROOT=tmp_path / "work")
    try:
        import torch
    except ImportError:  # pragma: no cover - CI installs torch
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    ns["torch"] = torch
    for loader in ("load_torch", "load_transformers", "load_pandas", "load_hub", "load_tapas"):
        ns[loader] = lambda: None
    ns.update(overrides)
    return ns
