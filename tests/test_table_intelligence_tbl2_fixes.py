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
    return [{"question_id": f"q{i:02d}", "table_id": f"t{i // 5}", "path": path, "denotation_correct": c,
             "aggregation_correct": True, "pipeline_status": s} for i, (c, s) in enumerate(zip(correct, status, strict=True))]


def _recorded_run_state(ns):
    """The 2026-10-03 counts: gold 40, structure-only 40, end-to-end 30; tables t8 (miss) and t9 (extra column) fail."""
    gold = [i % 5 != 0 or i >= 40 for i in range(50)]  # 8 wrong on t0..t7 ... plus 2 more below
    gold[1] = gold[6] = False  # 10 wrong on the gold table, all on t0..t7
    e2e_status = ["ok"] * 40 + ["detection_miss"] * 5 + ["header_or_limit_invalid"] * 5
    e2e = [c and s == "ok" for c, s in zip(gold, e2e_status, strict=True)]
    tables = [{"source_id": f"t{i}", "gold_recon": {"valid": True}, "det_recon": {"valid": i < 8}} for i in range(10)]
    ns.update(
        canonical_tables=tables, qa_records=[{}] * 50,
        gold_qa=_qa(gold), structure_qa=_qa(gold, path="structure_only"),
        end_to_end_qa=_qa(e2e, e2e_status, path="end_to_end"),
        det_rows=[{"hit50": i != 8, "hit75": i != 8} for i in range(10)],
        structure_rows=[{"path": p, "grid_shape_exact": not (p == "detected_crop" and i >= 8)}
                        for p in ("gold_crop", "detected_crop") for i in range(10)],
    )
    return sum(gold), sum(e2e)


def test_waterfall_attributes_each_lost_question_to_a_stage(tmp_path, capsys):
    ns = load_stages(tmp_path)
    gold_correct, e2e_correct = _recorded_run_state(ns)
    assert (gold_correct, e2e_correct) == (40, 30)
    ns["stage_waterfall"](None)
    attribution = ns["waterfall"]["loss_attribution"]
    assert attribution["wrong_on_gold_table"] == 10
    assert attribution["gold_table_to_structure_only"] == {"lost": 0, "gained": 0, "lost_by_status": {}}
    assert attribution["structure_only_to_end_to_end"] == {
        "lost": 10, "gained": 0, "lost_by_status": {"detection_miss": 5, "header_or_limit_invalid": 5}}
    assert attribution["lost_vs_gold_table"] == {"structure_only": 0, "end_to_end": 10}
    # The counts that existed before are unchanged.
    assert ns["waterfall"]["qa"]["gold_table_correct"] == 40 and ns["waterfall"]["qa"]["end_to_end_correct"] == 30
    out = capsys.readouterr().out
    assert "Per-question trace: 20 of 50 questions" in out
    assert out.count("TAPAS on the gold table") == 10
    assert out.count("detection miss") == 5 and out.count("detected crop (header_or_limit_invalid)") == 5


def test_waterfall_reading_rule_is_anchored_to_the_gold_table_path():
    text = markdown("### What to notice — waterfall")
    assert "largest drop" not in text
    assert "gold-table path" in text and "loss_attribution" in text
    sample = markdown("## 24. Checkpoint — trace the losses")
    assert "40 of 50" in sample and "TAPAS's own errors" in sample and "revision 0.3.0-candidate" in sample
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
    for term in ("PubTables-1M", "FinTabNet.c", "WikiTableQuestions", "no check for overlap"):
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
