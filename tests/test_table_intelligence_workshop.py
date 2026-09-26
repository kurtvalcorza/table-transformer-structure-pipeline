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


def test_generator_parity():
    import subprocess
    import sys
    subprocess.run([sys.executable, str(REPO / "tools" / "build_table_intelligence_workshop.py"), "--check"], cwd=REPO, check=True)


def test_structure_manifest_matches_the_repository_snapshot():
    import glob
    import re
    body = "\n".join(code_cells())
    embedded = json.loads(re.search(r'STRUCTURE_MANIFEST=json\.loads\(r"""(.*?)"""\)', body, re.S).group(1))
    committed = json.loads(Path(glob.glob(str(REPO / "weights" / "*" / "dimer-base-manifest.json"))[0]).read_text(encoding="utf-8"))
    assert embedded["modelId"] == committed["modelId"]
    assert embedded["revision"] == committed["revision"]
    assert {f["path"]: (f["bytes"], f["sha256"]) for f in embedded["files"]} == {f["path"]: (f["bytes"], f["sha256"]) for f in committed["files"]}


def test_modules_are_imported_before_first_use():
    import ast
    cells = code_cells()
    for module in ("json", "re"):
        first_use = next(i for i, cell in enumerate(cells) if f"{module}." in cell)
        imported = {
            alias.name
            for cell in cells[: first_use + 1]
            for node in ast.walk(ast.parse(cell))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert module in imported, f"{module} is used in code cell {first_use} before it is imported"


def test_geometry_is_the_carrier_geometry():
    import hashlib
    import re
    from table_transformer_structure_pipeline.sample_data import SAMPLE_RECORDS
    body = "\n".join(code_cells())
    text = re.search(r'CARRIER_GEOMETRY_JSON=r"""(.*?)"""', body, re.S).group(1)
    digest = re.search(r'CARRIER_GEOMETRY_SHA256="([0-9a-f]{64})"', body).group(1)
    assert hashlib.sha256(text.encode()).hexdigest() == digest
    geometry = json.loads(text)
    codes = {"table": "t", "table row": "r", "table column": "c", "table spanning cell": "s"}
    assert set(geometry) == {e["table_id"] for e in SAMPLE_RECORDS}
    for e in SAMPLE_RECORDS:
        g = geometry[e["table_id"]]
        assert (g["paper_id"], g["dx"], g["dy"], g["n_rows"], g["n_cols"]) == (
            e["paper_id"], e["alignment"]["dx"], e["alignment"]["dy"], e["n_rows"], e["n_cols"]
        )
        assert g["objects"] == [[codes[o["label"]], *o["box"]] for o in e["objects"]]


def test_pixel_digests_match_the_carrier_images():
    import hashlib
    import re
    from table_transformer_structure_pipeline import samples
    body = "\n".join(code_cells())
    geometry = json.loads(re.search(r'CARRIER_GEOMETRY_JSON=r"""(.*?)"""', body, re.S).group(1))
    for record in samples.load_corpus():
        rgb = record["image"].convert("RGB")
        digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
        assert geometry[record["id"]]["pixel_sha256"] == digest


def test_qa_tables_come_from_the_test_split_only():
    # Spec §18: never borrow train/validation tables for the canonical QA set.
    assert 'splits["validation"]+splits["train"]' not in "\n".join(code_cells())


def test_timm_is_pinned_for_the_detection_backbone():
    # microsoft/table-transformer-detection builds its ResNet-18 backbone through timm.
    assert any('"timm":"' in cell for cell in code_cells())


def test_splits_do_not_iterate_sets():
    # Set iteration order of strings changes between processes, which made the complex probe non-deterministic.
    split_cell = next(cell for cell in code_cells() if "rng=random.Random(42)" in cell)
    assert "set(papers" not in split_cell
