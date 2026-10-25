"""Tier C review findings T100-1..T100-5 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "table_transformer_structure_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    md = "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    return " ".join(md.split())


def test_t100_1_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "FinTabNet" in md and "not verified" in md


def test_t100_2_and_3_numbers_name_their_run():
    md = _markdown()
    assert "Kaggle T4 release run" not in md and "build record" not in md
    assert "`dea5747f`" in md and "`5d70d67`" in md


def test_t100_4_capacity_choice_is_validation_based():
    md = _markdown()
    assert "no better than two layers" not in md and "not as the basis of the choice" in md


def test_t100_5_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] B. Smock" in md and "`DIMER_ISOLATED_ENV`" in md
