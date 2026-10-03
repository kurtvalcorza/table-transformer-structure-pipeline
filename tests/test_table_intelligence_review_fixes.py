# ruff: noqa: E501,I001
"""Regression tests for the 2026-10-02 notebook review of the Table Intelligence workshop (TBL-M1..M3, TBL-m1..m7).

Since revision 0.3.0 the workshop's code runs in an isolated environment, one stage per process, from the carried
stage file ``tools/table_intelligence_workshop.py``. These tests call that file's own functions and stages with
small synthetic inputs and inert stand-ins for the three models, so they stay inside CI's install budget (torch,
transformers, numpy, pillow; no pandas, datasets or weights).
"""
from __future__ import annotations

import csv
import json
import re
import types

import pytest
from PIL import Image

from table_intelligence_workshop_support import cells, fn_src, host_cell, joined, load_stages, markdown, notebook, stage_text


def grid_boxes(n_rows, n_cols, x0=0.0, y0=0.0, w=200.0, h=100.0):
    rows = [{"label": "table row", "score": 0.99, "box": [x0, y0 + h * i / n_rows, x0 + w, y0 + h * (i + 1) / n_rows]} for i in range(n_rows)]
    cols = [{"label": "table column", "score": 0.99, "box": [x0 + w * j / n_cols, y0, x0 + w * (j + 1) / n_cols, y0 + h]} for j in range(n_cols)]
    return rows + cols


# ---------------------------------------------------------------- TBL-M1


def test_detection_miss_is_a_failed_detected_crop_grid():
    ns = load_stages()
    raw = grid_boxes(3, 2)
    hit = {"source_id": "hit", "n_rows": 3, "n_cols": 2, "det_crop": object(),
           "gold_structure_raw": raw, "det_structure_raw": raw, "gold_crop_objects": raw, "det_crop_objects": raw}
    miss = {"source_id": "miss", "n_rows": 3, "n_cols": 2, "det_crop": None,
            "gold_structure_raw": raw, "det_structure_raw": [], "gold_crop_objects": raw, "det_crop_objects": []}
    for item in (hit, miss):
        item["gold_grid_pred"] = ns["grid_from_structure"](item["gold_structure_raw"])
        item["det_grid_pred"] = ns["grid_from_structure"](item["det_structure_raw"]) if item["det_crop"] is not None else {"rows": [], "columns": [], "cells": [], "valid_geometry": False}
    ns["canonical_tables"] = [hit, miss]
    ns["stage_structure_metrics"](None)
    rows = {(r["table_id"], r["path"]): r for r in ns["structure_rows"]}
    assert rows[("miss", "detected_crop")]["status"] == "detection_miss"
    assert rows[("miss", "detected_crop")]["grid_shape_exact"] is False
    assert rows[("miss", "detected_crop")]["row_count_exact"] is False
    assert rows[("hit", "detected_crop")]["grid_shape_exact"] is True
    assert rows[("miss", "gold_crop")]["grid_shape_exact"] is True
    detected_exact = sum(r["grid_shape_exact"] for r in ns["structure_rows"] if r["path"] == "detected_crop")
    assert detected_exact == 1  # never more than the one detection hit


def test_structure_counts_compare_with_the_table_grid_not_clipped_reference_boxes():
    ns = load_stages()
    full = grid_boxes(3, 2)
    clipped_refs = [o for o in full if not (o["label"] == "table row" and o["box"][1] == 0.0)]  # top row cut away by the crop
    pred = grid_boxes(2, 2)  # the model sees only two rows
    item = {"source_id": "clip", "n_rows": 3, "n_cols": 2, "det_crop": object(),
            "gold_structure_raw": full, "det_structure_raw": pred, "gold_crop_objects": full, "det_crop_objects": clipped_refs}
    item["gold_grid_pred"] = ns["grid_from_structure"](full)
    item["det_grid_pred"] = ns["grid_from_structure"](pred)
    ns["canonical_tables"] = [item]
    ns["stage_structure_metrics"](None)
    det = next(r for r in ns["structure_rows"] if r["path"] == "detected_crop")
    assert det["gold_rows"] == 3 and det["predicted_rows"] == 2
    assert det["grid_shape_exact"] is False
    fields = re.search(r'out_dir/"structure_metrics\.csv",structure_rows,\s*\[(.*?)\]', fn_src("stage_exports"), re.S).group(1)
    assert '"status"' in fields


# ---------------------------------------------------------------- TBL-m2


def test_waterfall_reports_correct_answer_counts():
    ns = load_stages()
    rows = [{"denotation_correct": i < 30, "aggregation_correct": True, "pipeline_status": "ok"} for i in range(50)]
    summary = ns["qa_summary"](rows)
    assert summary["correct"] == 30 and abs(summary["denotation_accuracy"] - 0.6) < 1e-12
    waterfall = fn_src("stage_waterfall")
    for key in ("gold_table_correct", "structure_only_correct", "end_to_end_correct", '"unit"'):
        assert key in waterfall
    assert "(out of 50" in markdown("### What to notice — waterfall")


# ---------------------------------------------------------------- TBL-M2


class _Batch(dict):
    def to(self, _device):
        return self


class _Processor:
    def __init__(self):
        self.image = None

    def __call__(self, images, return_tensors="pt"):
        self.image = images
        return _Batch()

    def post_process_object_detection(self, out, target_sizes, threshold):
        import torch

        objs = [o for o in grid_boxes(3, 2, 10, 10, self.image.width - 20, self.image.height - 20) if o["score"] >= threshold]
        ids = {"table row": 2, "table column": 1}
        return [{"scores": torch.tensor([o["score"] for o in objs]), "labels": torch.tensor([ids[o["label"]] for o in objs]),
                 "boxes": torch.tensor([o["box"] for o in objs]).reshape(-1, 4)}]


class _Model:
    class config:  # noqa: N801
        id2label = {0: "table", 1: "table column", 2: "table row"}

    def __call__(self, **_kw):
        return None


def test_activity_reruns_after_run_all_in_its_own_process():
    pytest.importorskip("torch")
    ns = load_stages()
    loads = []

    def load_structure_model():
        loads.append(1)
        return _Processor(), _Model()

    ns["load_structure_model"] = load_structure_model
    table = Image.new("RGB", (220, 120), "white")
    ns["splits"] = {"validation": [{"source_id": f"v{i}", "image": table, "n_rows": 3, "n_cols": 2} for i in range(2)]}
    # A process that already holds the model neither loads nor unloads it.
    ns["struct_processor"], ns["struct_model"] = _Processor(), _Model()
    ns["stage_activity"](types.SimpleNamespace(threshold=0.95))
    assert loads == [] and "struct_model" in ns
    # A stage process after Run all starts without the model: the rerun loads it, runs, and unloads it again.
    del ns["struct_processor"], ns["struct_model"]
    ns["stage_activity"](types.SimpleNamespace(threshold=0.98))
    assert loads == [1]
    assert "struct_model" not in ns and "struct_processor" not in ns
    assert len(ns["activity_rows"]) == 2 and all(r["at_reference"] == (3, 2) for r in ns["activity_rows"])
    with pytest.raises(ValueError, match="ACTIVITY_STRUCTURE_THRESHOLD must be in"):
        ns["stage_activity"](types.SimpleNamespace(threshold=1.5))


def test_activity_cell_passes_its_threshold_to_the_stage():
    activity = host_cell("ACTIVITY_STRUCTURE_THRESHOLD = ")
    assert 'ACTIVITY_STRUCTURE_THRESHOLD = 0.95  # @param {type:"number"}' in activity
    assert 'run_stage("activity", "--threshold", str(ACTIVITY_STRUCTURE_THRESHOLD))' in activity


def test_section19_still_unloads_structure_before_tapas_and_activity_comes_first():
    srcs = [joined(c) for c in cells()]
    activity = next(i for i, s in enumerate(srcs) if 'run_stage("activity"' in s)
    probe = next(i for i, s in enumerate(srcs) if 'run_stage("complex-probe")' in s)
    tapas = next(i for i, s in enumerate(srcs) if 'run_stage("qa")' in s)
    assert activity < probe < tapas
    assert "del struct_model,struct_processor" in fn_src("stage_complex_probe")
    assert "rerun it as often as you like after `Run all`" in markdown("### Activity — Predict")


# ---------------------------------------------------------------- TBL-M3 / TBL-m1


class _FakePandas:
    class DataFrame:
        def __init__(self, rows, columns):
            self.rows, self.columns = rows, columns

        def to_string(self, index=False):
            return "\n".join([" | ".join(self.columns)] + [" | ".join(r) for r in self.rows])


def byod_namespace(tmp_path, folder):
    ns = load_stages(tmp_path)
    rows = grid_boxes(3, 2, 10, 10, 360, 240)

    def detect_table(image, threshold=None):
        return [{"label": "table", "score": 0.99, "box": [40, 40, 360, 260]}]

    def recognize_structure(crop, threshold=None):
        return [dict(o) for o in rows]

    ns.update(
        pd=_FakePandas, USE_BYOD=True, BYOD_PATH=str(folder), OUTPUT_DIR=str(tmp_path / "outputs"),
        load_detection_model=lambda: (None, None), load_structure_model=lambda: (None, None),
        detect_table=detect_table, recognize_structure=recognize_structure,
        tapas_answer=lambda table, question: {"coordinates": [[0, 1]], "cells": ["12"], "aggregation": "SUM",
                                              "numeric_answer": 12.0, "answer": "SUM > 12"},
    )
    return ns


def make_byod(tmp_path, ocr_lines, questions=None, extra_pages=()):
    folder = tmp_path / "byod_in"
    (folder / "pages").mkdir(parents=True)
    Image.new("RGB", (400, 300), "white").save(folder / "pages" / "page001.png")
    for name in extra_pages:
        if name.endswith(".png"):
            Image.new("RGB", (400, 300), "white").save(folder / "pages" / name)
        else:
            (folder / "pages" / name).write_text("not an image\n", encoding="utf-8")
    (folder / "ocr.csv").write_text("\n".join(["page_id,text,x0,y0,x1,y1,order", *ocr_lines]) + "\n", encoding="utf-8")
    if questions is not None:
        (folder / "questions.csv").write_text("\n".join(["id,page_id,question,accepted_answer", *questions]) + "\n", encoding="utf-8")
    return folder


# Cell centres (crop frame) of the 3x2 stand-in grid: crop origin is (30, 30) for the box [40,40,360,260] padded by 10.
GOOD_OCR = [
    "page001,Name,40,40,60,50,1", "page001,Value,240,40,260,50,2",
    "page001,a,40,120,60,130,3", "page001,12,240,120,260,130,4",
    "page001,b,40,200,60,210,5", "page001,3,240,200,260,210,6",
]


def test_byod_prints_and_exports_tables_and_answers(tmp_path, capsys):
    folder = make_byod(tmp_path, GOOD_OCR, ["q1,page001,What is the total Value?,12"], extra_pages=("page002.png", "notes.txt"))
    ns = byod_namespace(tmp_path, folder)
    ns["stage_byod"](None)
    out = capsys.readouterr().out
    result = json.loads((tmp_path / "outputs" / "byod" / "byod_results.json").read_text(encoding="utf-8"))
    assert result["outputs"][0]["status"] == "ok"
    assert result["outputs"][0]["table"] == [["Name", "Value"], ["a", "12"], ["b", "3"]]
    assert {"file": "page002.png", "reason": "no_ocr_rows"} in result["skipped"]
    assert {"file": "notes.txt", "reason": "unsupported_extension"} in result["skipped"]
    with open(tmp_path / "outputs" / "byod" / "tables" / "page001.csv", encoding="utf-8", newline="") as f:
        assert list(csv.reader(f)) == [["Name", "Value"], ["a", "12"], ["b", "3"]]
    answer = result["outputs"][0]["answers"][0]
    assert answer["accepted_answer_match"] is True  # 12.0 == "12"
    assert "Name | Value" in out and "match" in out and "page002.png" in out
    assert ns["byod_result"]["pages"] == 1  # carried to the summary stage through the stage state


def test_byod_loads_tapas_only_when_there_are_questions(tmp_path):
    loads = []
    folder = make_byod(tmp_path, GOOD_OCR)
    ns = byod_namespace(tmp_path, folder)
    ns["load_tapas"] = lambda: loads.append(1)
    ns["stage_byod"](None)
    assert loads == []
    (folder / "questions.csv").write_text("id,page_id,question\nq1,page001,What?\n", encoding="utf-8")
    ns["stage_byod"](None)
    assert loads == [1]


def test_byod_removes_outputs_from_an_earlier_run(tmp_path):
    folder = make_byod(tmp_path, GOOD_OCR)
    ns = byod_namespace(tmp_path, folder)
    stale = tmp_path / "outputs" / "byod" / "tables" / "old_page.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("x\n", encoding="utf-8")
    ns["stage_byod"](None)
    assert not stale.exists()


@pytest.mark.parametrize(
    ("ocr", "expected"),
    [
        (["page001,a,twelve,1,2,3,1"], r"ocr\.csv line 2: x0,y0,x1,y1 must be numbers"),
        (["page001,a,1,1,2,3,first"], r"ocr\.csv line 2: order must be a whole number"),
        (["page001,a,5,1,2,3,1"], r"ocr\.csv line 2: box .* x0<=x1"),
        ([",a,1,1,2,3,1"], r"ocr\.csv line 2: page_id is empty"),
        (["page001,a,10,10,20,20,1", "page001,b,390,290,900,700,2"], r"ocr\.csv lines \[3\]: boxes for page001 extend beyond its 400x300 image"),
        (["other,a,1,1,2,3,1"], r"No page in .* has OCR rows"),
        ([], r"ocr\.csv has a header but no rows"),
    ],
)
def test_byod_rejects_bad_ocr_rows_with_file_line_and_field(tmp_path, ocr, expected):
    folder = make_byod(tmp_path, ocr)
    ns = byod_namespace(tmp_path, folder)
    with pytest.raises(ValueError, match=expected):
        ns["stage_byod"](None)


def test_byod_rejects_missing_columns_and_empty_path(tmp_path):
    folder = make_byod(tmp_path, GOOD_OCR)
    (folder / "ocr.csv").write_text("page_id,text,x0,y0,x1,y1\npage001,a,1,1,2,2\n", encoding="utf-8")
    ns = byod_namespace(tmp_path, folder)
    with pytest.raises(ValueError, match=r"ocr\.csv: missing column\(s\) \['order'\]"):
        ns["stage_byod"](None)
    ns["BYOD_PATH"] = "  "
    with pytest.raises(ValueError, match="BYOD_PATH is empty"):
        ns["stage_byod"](None)
    ns["BYOD_PATH"] = str(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match=r"is missing \['pages/', 'ocr\.csv'\]"):
        ns["stage_byod"](None)


def test_accepted_answer_matching_is_numeric_for_numbers():
    match = load_stages()["accepted_answer_matches"]
    total = {"aggregation": "SUM", "numeric_answer": 12.0, "cells": ["5", "7"]}
    assert match(total, "12") and match(total, "12.000")
    assert not match({**total, "numeric_answer": 12.5}, "12")
    assert not match({**total, "numeric_answer": None}, "12")
    lookup = {"aggregation": "NONE", "numeric_answer": None, "cells": ["Deep  Net"]}
    assert match(lookup, "deep net") and not match(lookup, "shallow net")


def test_byod_contract_is_stated_before_use():
    text = markdown("## 28. BYOD (optional)")
    assert "page-image pixels with the origin at the top-left corner" in text
    assert "byod_results.json" in text and "skipped" in text
    assert "sends nothing to an external service" in text


# ---------------------------------------------------------------- TBL-m3, TBL-m4


def test_split_usage_is_stated_consistently():
    text = markdown("## 8. Paper-disjoint split")
    assert "only the 21 held-out test tables are used below" not in text
    assert "23 **validation** tables" in text


def test_a_box_is_traced_through_the_frames(capsys):
    ns = load_stages()
    table = Image.new("RGB", (300, 100), "white")
    item = {"source_id": "t1", "image": table, "objects": [{"label": "table row", "box": [3.0, 30.0, 290.0, 58.0]}],
            "cells": [], "selected_detection": {"box": [125.0, 165.0, 415.0, 255.0]}}
    item["page"], item["page_gt_box"] = ns["make_page"](item)
    ns["canonical_tables"] = [item]
    ns["stage_crops"](None)
    printed = capsys.readouterr().out
    assert "'page_frame': [123.0, 190.0, 410.0, 218.0]" in printed
    assert "'detected_crop_frame': [8.0, 35.0, 295.0, 63.0]" in printed  # crop origin (115, 155)
    assert "Checkpoint: trace a box" in markdown("## 14. Build gold and detected crops")


# ---------------------------------------------------------------- TBL-m5, TBL-m6, TBL-m7


def test_resource_rows_name_their_latency_unit_and_tapas_peak_is_reset():
    resources = fn_src("stage_resources")
    assert resources.count('"latency_unit":') == 3
    assert '"peak_gpu_memory_bytes":tapas_peak' in resources
    assert "torch.cuda.reset_peak_memory_stats()" in fn_src("stage_qa")
    assert "struct_times.append(" in fn_src("timed_structure")
    assert "mean_pair_seconds" not in stage_text()


@pytest.mark.parametrize(("field", "bad"), [("CROP_PADDING", -5), ("CROP_PADDING", 2.5), ("GRID_NMS_IOU", 0),
                                            ("COMPLEX_STRUCTURE_TABLES", -1), ("COMPLEX_STRUCTURE_TABLES", 11)])
def test_configuration_controls_are_validated(field, bad):
    src = re.sub(rf"^{field} = .*$", f"{field} = {bad!r}", host_cell("USE_BYOD = False"), count=1, flags=re.M)
    with pytest.raises(ValueError, match=field):
        exec(src, {})
    exec(host_cell("USE_BYOD = False"), {})


def test_fixed_question_set_is_not_a_form_field():
    config = host_cell("USE_BYOD = False")
    assert re.search(r"^END_TO_END_TABLES = 10$", config, re.M)


def test_complex_probe_keeps_only_this_runs_panels():
    probe = fn_src("stage_complex_probe")
    assert probe.index('complex_dir.glob("*.png")') < probe.index("panel.save(")


def test_notebook_records_the_review_revision():
    meta = notebook()["metadata"]
    assert meta["workshop_revision"] == "0.3.0-candidate"
    record = meta["dimer"]["review_revisions"][0]
    assert record["base_commit"] == "c218d3119e157293949862d6972e77d4a029c6b0"
    assert {"TBL-M1", "TBL-M2", "TBL-M3"} <= set(record["fixed"])
    assert meta["dimer"]["clean_runtime_evidence"] == "pending"
