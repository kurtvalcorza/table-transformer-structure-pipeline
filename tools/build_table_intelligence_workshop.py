# ruff: noqa: E501,I001
"""Generate the DIMER Table Intelligence workshop notebook."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from table_intelligence_workshop_source import CELLS

NOTEBOOK_NAME = "DIMER_Table_Intelligence_Workshop.ipynb"
DIMER_METADATA = {
    "notebook_spec": "2.1",
    "notebook_profile": "TASK-INFERENCE",
    "profile": "TASK-INFERENCE",
    "notebook_mode": "WORKSHOP",
    "workflow_scope": "COMPOSED-PIPELINE",
    "standalone": True,
    "canonical_runtime": "NVIDIA Tesla T4",
    "worker_required": False,
    "credentials_required": False,
    "clean_runtime_evidence": "pending",
    "dataset": {
        "id": "bevaya/SciTSR-pd",
        "license": "CC0-1.0",
        "revision": "dae336efa7af07d69194a510ff5568980e8ef253"
    },
    "models": [
        {
            "id": "microsoft/table-transformer-detection",
            "revision": "2357cbe2b5a5d1c03e54f32764f06058933b65ab"
        },
        {
            "id": "microsoft/table-transformer-structure-recognition-v1.1-all",
            "revision": "7587a7ef111d9dcbf8ac695f1376ab7014340a0c"
        },
        {
            "id": "google/tapas-large-finetuned-wtq",
            "revision": "f58317ab2577d17647d9acafa790c744a0388b30"
        }
    ],
    "generated_from": {
        "repository": "kurtvalcorza/table-transformer-structure-pipeline",
        "source": "tools/table_intelligence_workshop_source.py",
        "generator": "tools/build_table_intelligence_workshop.py"
    }
}


def build_notebook():
    rendered = []
    for index, cell in enumerate(CELLS):
        # Infrastructure cells (GDL11) are collapsed where the notebook viewer supports it.
        metadata = {"jupyter": {"source_hidden": True}} if cell.get("infrastructure") else {}
        base = {"id": f"dimer-table-workshop-{index:02d}", "metadata": metadata, "source": cell["source"].splitlines(keepends=True)}
        if cell["kind"] == "markdown":
            rendered.append({"cell_type": "markdown", **base})
        else:
            rendered.append({"cell_type": "code", "execution_count": None, "outputs": [], **base})
    return {
        "cells": rendered,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "dimer": DIMER_METADATA,
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "workshop_revision": "0.1.0-candidate",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def serialized():
    return json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.out or repo / "tutorials" / NOTEBOOK_NAME
    content = serialized()
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != content:
            raise SystemExit(f"STALE: {out}; regenerate the workshop notebook")
        print(f"OK: {out}")
        return 0
    out.write_text(content, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
