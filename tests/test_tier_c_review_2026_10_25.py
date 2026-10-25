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


def test_prose_holds_across_both_recorded_colab_t4_runs():
    # Tier C prose-mismatch fix (2026-10-25): T4-attributed numbers are T4 values;
    # CPU pre-flight values stay labelled.
    md = " ".join(_markdown().split())
    for gone in ("by a wide margin (0.295 vs 0.771)", "neither was re-run for this revision"):
        assert gone not in md, gone
    for quoted in ("(0.279 vs 0.771 in the three Colab T4 runs; 0.295 in the CPU pre-flight)",
                   "0.2762 → 0.3028 → 0.2842", "the first unfreeze epoch (`best_epoch` 2) was kept",
                   "blob `629e15bc`", "412.8 s for blob `629e15bc`"):
        assert quoted in md, quoted


def test_historical_wording_records_every_hosted_run():
    # Historical wording (2026-10-26): timings are records of every named hosted run, never a span to land in;
    # the third Colab T4 run (blob fcd83d4d) is named; no run-status sentence. Fails on PR #14 head 1196a33.
    md = " ".join(_markdown().split())
    for gone in ("about 7 to 7.5 minutes", "about 100 to 110 s", "about 110 to 120 s", "takes about 3 minutes",
                 "No experiment run is recorded on a hosted runtime", "both recorded Colab T4 runs"):
        assert gone not in md, gone
    for quoted in ("413.7 s for blob `fcd83d4d`", "101.0 s", "111.2 s", "blobs `629e15bc` and `fcd83d4d`",
                   "your run may take a different time"):
        assert quoted in md, quoted
