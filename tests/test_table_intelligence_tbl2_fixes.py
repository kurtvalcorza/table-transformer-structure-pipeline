# ruff: noqa: E501
"""Regression tests for the 2026-10-14 notebook review of the Table Intelligence notebook (prefix TBL2).

They call the carried stage file's own functions with fabricated state (no model runs) and check the
generated notebook's markdown, so they need only CI's dependencies.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import warnings

import pytest
from PIL import Image

from table_intelligence_workshop_support import (
    NOTEBOOK,
    cell_by_id,
    cells,
    fn_src,
    joined,
    load_stages,
    markdown,
    notebook,
)

BOOT_ID = "dimer-table-workshop-06"


# ---------------------------------------------------------------- TBL2-M1


def _qa(correct, status=None, path="gold_table"):
    status = status or ["ok"] * len(correct)
    return [{"question_id": f"q{i:02d}", "table_id": f"t{i // 5}", "path": path, "value_correct": c, "strict_correct": c,
             "aggregation_correct": True, "pipeline_status": s} for i, (c, s) in enumerate(zip(correct, status, strict=True))]


def _synthetic_state(ns, gold, structure, e2e, e2e_status):
    """SYNTHETIC state for the waterfall stage: the correctness of each question on each path is made up by the test.
    It is not the recorded run's question-level data (that run saved no per-question records)."""
    tables = [{"source_id": f"t{i}", "gold_recon": {"valid": True}, "det_recon": {"valid": i < 8}} for i in range(10)]
    ns.update(
        canonical_tables=tables, qa_records=[{}] * 50,
        gold_qa=_qa(gold), structure_qa=_qa(structure, path="structure_only"),
        end_to_end_qa=_qa(e2e, e2e_status, path="end_to_end"),
        det_rows=[{"hit50": i != 8, "hit75": i != 8} for i in range(10)],
        structure_rows=[{"path": p, "grid_shape_exact": not (p == "detected_crop" and i >= 8)}
                        for p in ("gold_crop", "detected_crop") for i in range(10)],
    )


def test_waterfall_attribution_on_a_synthetic_state_with_the_recorded_counts(tmp_path, capsys):
    # Synthetic: 40 / 40 / 30 correct, with the ten gold errors placed on t0..t7 and t8 (miss) / t9 (unusable) failing.
    gold = [i % 5 != 0 or i >= 40 for i in range(50)]
    gold[1] = gold[6] = False
    e2e_status = ["ok"] * 40 + ["detection_miss"] * 5 + ["header_or_limit_invalid"] * 5
    e2e = [c and s == "ok" for c, s in zip(gold, e2e_status, strict=True)]
    ns = load_stages(tmp_path)
    _synthetic_state(ns, gold, gold, e2e, e2e_status)
    ns["stage_waterfall"](None)
    attribution = ns["waterfall"]["loss_attribution"]["value"]
    assert ns["waterfall"]["loss_attribution"]["strict"] == attribution  # synthetic: both rules agree here
    assert attribution["wrong_on_gold_table"] == 10
    assert attribution["gold_table_to_structure_only"] == {"lost": 0, "gained": 0, "net_correct_change": 0, "lost_by_status": {}}
    assert attribution["structure_only_to_end_to_end"] == {
        "lost": 10, "gained": 0, "net_correct_change": -10,
        "lost_by_status": {"detection_miss": 5, "header_or_limit_invalid": 5}}
    assert attribution["net_correct_change_vs_gold"] == {"structure_only": 0, "end_to_end": -10}
    assert "lost_vs_gold_table" not in attribution
    out = capsys.readouterr().out
    assert "Per-question trace: 20 of 50 questions" in out
    assert out.count("value: wrong on the gold table") == 10 and out.count("strict: wrong on the gold table") == 10
    assert out.count("(detection miss)") == 10 and out.count("(header_or_limit_invalid on the detected crop)") == 10


def test_waterfall_reports_recoveries_and_cancelling_losses_and_gains(tmp_path, capsys):
    # Synthetic: equal accuracies on gold and structure-only that hide one loss and one gain (q00 lost, q01 recovered),
    # and an end-to-end path that recovers q02, which was wrong on both earlier paths.
    gold = [True] * 50
    gold[1] = gold[2] = False
    structure = list(gold)
    structure[0], structure[1] = False, True
    e2e = list(structure)
    e2e[2] = True
    ns = load_stages(tmp_path)
    _synthetic_state(ns, gold, structure, e2e, ["ok"] * 50)
    ns["stage_waterfall"](None)
    attribution = ns["waterfall"]["loss_attribution"]["value"]
    assert attribution["gold_table_to_structure_only"]["lost"] == 1
    assert attribution["gold_table_to_structure_only"]["gained"] == 1
    assert attribution["gold_table_to_structure_only"]["net_correct_change"] == 0
    assert attribution["net_correct_change_vs_gold"]["structure_only"] == 0
    assert attribution["structure_only_to_end_to_end"]["gained"] == 1
    assert attribution["net_correct_change_vs_gold"]["end_to_end"] == 1
    out = capsys.readouterr().out
    assert "lost gold -> structure-only (answered differently)" in out
    assert "recovered on structure-only" in out and "recovered end to end" in out


def test_final_denotation_uses_the_predicted_operator(tmp_path):
    ns = load_stages(tmp_path)
    lookup = {"gold_denotation": ["12"], "gold_coordinates": [[0, 1]]}
    right_cell_with_count = {"cells": ["12"], "aggregation": "COUNT", "numeric_answer": 1.0, "coordinates": [[0, 1]]}
    assert not ns["value_correct"](right_cell_with_count, lookup)  # final answer is 1, not 12
    assert not ns["strict_correct"](right_cell_with_count, lookup)
    assert ns["selected_cells_match"](right_cell_with_count, lookup) is True  # diagnostic: the cell was right
    assert ns["coordinates_exact"](right_cell_with_count, lookup) is True
    right_cell_with_sum = {"cells": ["12"], "aggregation": "SUM", "numeric_answer": 12.0, "coordinates": [[0, 1]]}
    assert ns["value_correct"](right_cell_with_sum, lookup)  # the final number equals the gold value
    assert not ns["strict_correct"](right_cell_with_sum, lookup)  # strict: a lookup needs NONE
    plain = {"cells": ["12"], "aggregation": "NONE", "numeric_answer": None, "coordinates": [[0, 1]]}
    assert ns["value_correct"](plain, lookup) and ns["strict_correct"](plain, lookup) and ns["final_denotation"](plain) == ["12"]
    total = {"gold_denotation": 30.0, "gold_coordinates": [[0, 1], [1, 1]]}
    summed = {"cells": ["10", "20"], "aggregation": "SUM", "numeric_answer": 30.0, "coordinates": [[0, 1], [1, 1]]}
    total = {**total, "gold_aggregation": "SUM"}
    assert ns["value_correct"](summed, total) and ns["strict_correct"](summed, total)
    # Right value from the wrong operator: AVERAGE of 30 and 30 is 30, accepted by the value rule only.
    wrong_op = {"cells": ["30", "30"], "aggregation": "AVERAGE", "numeric_answer": 30.0, "coordinates": [[0, 1], [1, 1]]}
    assert ns["value_correct"](wrong_op, total) and not ns["strict_correct"](wrong_op, total)
    as_lookup = {"cells": ["30"], "aggregation": "NONE", "numeric_answer": None, "coordinates": [[0, 1]]}
    assert not ns["value_correct"](as_lookup, total) and not ns["strict_correct"](as_lookup, total)
    assert ns["selected_cells_match"](plain, total) is None


def test_reconstruction_fidelity_and_text_visibility(tmp_path):
    ns = load_stages(tmp_path)
    item = {"grid": [["A", "B"], ["x", "1"]]}
    assert ns["table_equals_gold"]([["a", "b"], ["x", "1"]], item) and ns["headers_equal_gold"]([["A", "B"]], item)
    assert not ns["table_equals_gold"]([["A", "B"], ["x", "2"]], item)
    assert not ns["table_equals_gold"](None, item) and not ns["headers_equal_gold"](None, item)
    inside = {"crop_box": [10, 10, 20, 20]}
    outside = {"crop_box": [110, 10, 130, 20]}
    assert ns["center_in_crop"](inside, [0, 0, 100, 50]) and not ns["center_in_crop"](outside, [0, 0, 100, 50])
    assert ns["TEXT_VISIBILITY"] in ("center-in-crop", "all")


def test_waterfall_reading_rule_is_anchored_to_the_gold_table_path():
    text = markdown("### What to notice — waterfall")
    assert "largest drop" not in text
    assert "gold-table path" in text and "loss_attribution" in text
    sample = markdown("## 24. Checkpoint — trace the losses")
    assert "40 / 40 / 30" in sample and "revision 0.3.0-candidate" in sample and "not hosted evidence" in sample
    assert "on every path" not in sample and "ceiling" not in markdown("### What to notice — QA")
    assert "TEXT_VISIBILITY" in markdown("## 17. Assign SciTSR source cell text")
    conclusion = markdown("## Your conclusion")
    assert "largest drop" not in conclusion and "loss_attribution" in conclusion


# ---------------------------------------------------------------- TBL2-m1 (DAT20)


def _byod_folder(tmp_path, ocr_lines):
    folder = tmp_path / "byod_in"
    (folder / "pages").mkdir(parents=True)
    Image.new("RGB", (400, 300), "white").save(folder / "pages" / "page001.png")
    (folder / "ocr.csv").write_text("\n".join(["page_id,text,x0,y0,x1,y1,order", *ocr_lines]) + "\n", encoding="utf-8")
    return folder


def test_byod_refusal_lists_every_bad_row_and_leaves_no_stale_results(tmp_path):
    folder = _byod_folder(tmp_path, ["page001,a,10,10,20,20,1", "page001,b,abc,10,20,20,2", "page001,c,10,10,20,20,x"])
    ns = load_stages(tmp_path, USE_BYOD=True, BYOD_PATH=str(folder))
    stale = tmp_path / "outputs" / "byod" / "byod_results.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"byod_path": "earlier attempt"}', encoding="utf-8")
    with pytest.raises(ns["ByodRefusal"]) as refused:
        ns["stage_byod"](None)
    message = str(refused.value)
    assert "ocr.csv line 3: x0,y0,x1,y1 must be numbers" in message and "ocr.csv line 4: order must be a whole number" in message
    assert not stale.exists()


def test_byod_refusal_exits_with_one_message_and_no_traceback(tmp_path):
    folder = _byod_folder(tmp_path, ["page001,b,abc,10,20,20,1"])
    config = {"use_byod": True, "byod_path": str(folder), "detection_threshold": 0.9, "structure_threshold": 0.5,
              "crop_padding": 10, "grid_nms_iou": 0.5, "end_to_end_tables": 10, "questions_per_table": 5,
              "run_complex_structure_probe": True, "complex_structure_tables": 3,
              "output_dir": str(tmp_path / "outputs"), "work_dir": str(tmp_path / "work")}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    ns = load_stages(tmp_path)
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = ns["main"](["--config", str(path), "--stage", "byod"])
    assert code == ns["REFUSED_EXIT"] == 3
    out = buffer.getvalue()
    assert out.startswith("BYOD input refused") and "ocr.csv line 2" in out and "run the BYOD cell" in out
    assert "Traceback" not in out


def _run_stage(tmp_path, exit_code):
    """The notebook's own run_stage (boot cell) against a stand-in stage script that exits with exit_code."""
    fn = next(n for n in ast.parse(cell_by_id(BOOT_ID)).body if isinstance(n, ast.FunctionDef) and n.name == "run_stage")
    script = tmp_path / "stage.py"
    script.write_text(f"print('stand-in message')\nraise SystemExit({exit_code})\n", encoding="utf-8")
    import os
    g = {"USE_BYOD": True, "BYOD_PATH": "x", "DETECTION_THRESHOLD": 0.9, "STRUCTURE_THRESHOLD": 0.5, "CROP_PADDING": 10,
         "GRID_NMS_IOU": 0.5, "END_TO_END_TABLES": 10, "QUESTIONS_PER_TABLE": 5, "RUN_COMPLEX_STRUCTURE_PROBE": True,
         "COMPLEX_STRUCTURE_TABLES": 3, "OUTPUT_DIR": "out", "WORK_ROOT": tmp_path, "PYTHON": sys.executable,
         "STAGE_SCRIPT": script, "ENV": dict(os.environ), "subprocess": subprocess, "os": os, "json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "boot", "exec"), g)
    return g["run_stage"]


def test_run_stage_reports_a_refusal_without_raising_but_still_raises_on_a_crash(tmp_path, capsys):
    _run_stage(tmp_path, 3)("byod")
    assert "stand-in message" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="byod failed with exit 1"):
        _run_stage(tmp_path, 1)("byod")


# ---------------------------------------------------------------- TBL2-m2, m3, m7, S2


def test_tapas_futurewarning_is_filtered_in_the_stage_process():
    load_stages()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        load_stages()  # module import re-installs the filter after simplefilter
        warnings.warn("Series.__getitem__ treating keys as positions is deprecated. x", FutureWarning, stacklevel=1)
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]


def test_detection_metrics_report_a_mean_over_detected_pages(tmp_path):
    ns = load_stages(tmp_path)
    ious = [0.8, 0.9, 0.0]
    ns["canonical_tables"] = [
        {"source_id": f"t{i}", "page_gt_box": [0, 0, 10, 10],
         "detections": [] if iou == 0 else [{"score": 0.95, "box": [0, 0, 10, 10 * iou]}]} for i, iou in enumerate(ious)]
    ns["stage_detection_metrics"](None)
    metrics = ns["det_metrics"]
    assert metrics["detected_pages"] == 2
    assert metrics["mean_best_iou_detected"] == pytest.approx(0.85)
    assert metrics["mean_best_iou"] == pytest.approx(1.7 / 3)  # unchanged: a miss counts as 0
    text = markdown("## 13. Detection metrics")
    assert "no detection counts as IoU 0" in text and "mean_best_iou_detected" in text


def test_detection_prints_the_best_score_below_the_threshold_for_a_missed_page():
    source = fn_src("stage_detection")
    assert "best_table_score_below_threshold" in source and "threshold=0.0" in source
    # Display only: it runs after the timings and the peak memory are recorded.
    assert source.index("det_peak=") < source.index("threshold=0.0")
    assert "num_batches_tracked" in markdown("## 12. Stage 1")


def test_crop_trace_prints_the_clipped_box(tmp_path):
    ns = load_stages(tmp_path)
    assert ns["clip_to_crop"]([-1.0, 9.0, 518.0, 35.1], [100, 100, 600, 300]) == [0, 9.0, 500, 35.1]
    assert "detected_crop_frame_clipped" in fn_src("stage_crops")
    assert "before clipping" in markdown("## 14. Build gold and detected crops")


# ---------------------------------------------------------------- TBL2-m4, m5, m6, m8, m9, S1


def test_disclosures_lineage_references_and_notice():
    six = markdown("## 6. SciTSR-PD provenance")
    for term in ("PubTables-1M", "FinTabNet.c", "WikiTableQuestions", "unknown, because it was not checked"):
        assert term in six
    seven = markdown("## 7. Build the table-geometry")
    assert "mid-gaps" in seven and "14 of the 108" in seven and "The builder is not published" in seven
    activity = markdown("### Activity — Predict")
    assert "short or sparse" not in activity and "No recorded run shows what 0.30 or 0.98 does" in activity
    refs = markdown("## References")
    assert len(re.findall(r"https://", refs)) >= 6
    first_code = next(i for i, c in enumerate(cells()) if c["cell_type"] == "code")
    assert cells()[1]["id"] == "use-notice" and first_code > 1
    assert joined(cells()[1]).startswith("> **⚠️ Notice.**")
    for n in range(1, 7):  # every reference is cited (DOC16)
        assert sum(f"[{n}]" in joined(c) for c in cells() if c["id"] != "references") >= 1, n


def test_title_uses_notebook_and_file_name_keeps_the_workshop_kind_pattern():
    assert NOTEBOOK.name == "DIMER_Table_Intelligence_Workshop.ipynb"  # KND2: the kind word in the file name is an identifier
    title = joined(cells()[0]).splitlines()[0]
    assert title == "# DIMER Notebook: Table Intelligence"
    meta = notebook()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5" and meta["notebook_kind"] == "WORKSHOP"
    assert "renamed_from" not in meta["review_revisions"][-1]
    assert 'print("DIMER Notebook: Table Intelligence")' in fn_src("stage_summary")
