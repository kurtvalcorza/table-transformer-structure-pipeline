"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from table_transformer_structure_pipeline import (
    INPUT_SCHEMA,
    LABELS,
    MAX_DETECTIONS,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    RECOGNITION_THRESHOLD,
    evaluation_report,
    structure_summary,
    validate_inputs,
)

REFERENCES = {
    "table": [[10.0, 10.0, 720.0, 340.0]],
    "table row": [[10.0, 10.0, 720.0, 50.0], [10.0, 50.0, 720.0, 90.0]],
    "table column": [[10.0, 10.0, 365.0, 340.0], [365.0, 10.0, 720.0, 340.0]],
    "table column header": [[10.0, 10.0, 720.0, 50.0]],
}


def _image(width: int = 730, height: int = 350) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(detections: list[dict]) -> dict:
    return {"detections": detections, "threshold": RECOGNITION_THRESHOLD, "width": 730, "height": 350}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), names=["table.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["labels"] == list(LABELS)
    assert manifest["schema"]["max_detections"] == MAX_DETECTIONS
    assert manifest["inputs"] == [{"id": "table.png", "mode": "RGB", "size": [730, 350]}]
    assert manifest["threshold"] == RECOGNITION_THRESHOLD
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_threshold() -> None:
    manifest = validate_inputs(_image(), threshold=0.9)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["threshold"] == 0.9


def test_validate_inputs_rejects_like_recognize() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 64))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8))
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="threshold"):
        validate_inputs(_image(), threshold=1.5)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), names=["a", "b"])


def test_structure_summary_counts_and_grid() -> None:
    detections = [
        {"box": [0, 0, 1, 1], "label": "table", "score": 0.9},
        {"box": [0, 0, 1, 1], "label": "table row", "score": 0.9},
        {"box": [0, 0, 1, 1], "label": "table row", "score": 0.9},
        {"box": [0, 0, 1, 1], "label": "table column", "score": 0.9},
        {"box": [0, 0, 1, 1], "label": "table column header", "score": 0.9},
    ]
    summary = structure_summary(_result(detections))
    assert summary["n_rows"] == 2 and summary["n_columns"] == 1 and summary["n_cells_implied"] == 2
    assert summary["has_column_header"] is True
    assert summary["counts"]["table spanning cell"] == 0


def test_evaluation_report_not_measurable_without_reference_boxes() -> None:
    detection = {"box": [10.0, 10.0, 720.0, 340.0], "label": "table", "score": 0.99}
    report = evaluation_report(_result([detection]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_detections"] == 1
    assert report["structure_summary"]["counts"]["table"] == 1
    assert "box_iou" in report["needs"] and "GriTS" in report["needs"]
    assert report["baselines"] == []
    assert report["threshold"] == RECOGNITION_THRESHOLD
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_matches_same_label_only() -> None:
    detections = [
        {"box": [10.0, 10.0, 720.0, 340.0], "label": "table", "score": 1.0},
        {"box": [10.0, 10.0, 720.0, 50.0], "label": "table row", "score": 1.0},
        {"box": [10.0, 50.0, 720.0, 90.0], "label": "table row", "score": 1.0},
        {"box": [10.0, 10.0, 365.0, 340.0], "label": "table column", "score": 1.0},
        # the second column is missing; the header box overlaps row 0 exactly but has another label
        {"box": [10.0, 10.0, 720.0, 50.0], "label": "table column header", "score": 1.0},
    ]
    report = evaluation_report(_result(detections), REFERENCES, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity"
    by_ref = {metric["reference"]: metric for metric in report["metrics"]}
    assert by_ref["table-0"]["value"] == pytest.approx(1.0)
    assert by_ref["table row-0"]["value"] == pytest.approx(1.0) and by_ref["table row-1"][
        "value"
    ] == pytest.approx(1.0)
    assert by_ref["table column-0"]["value"] == pytest.approx(1.0)
    assert by_ref["table column-1"]["value"] < 0.2  # only the first column detection exists to match against
    assert by_ref["table column-1"]["n_detected"] == 1 and by_ref["table column-1"]["n_reference"] == 2
    assert by_ref["table column header-0"]["value"] == pytest.approx(1.0)
    assert all(metric["estimation"] for metric in report["metrics"])


def test_evaluation_report_rejects_unknown_reference_label_and_handles_zero_detections() -> None:
    with pytest.raises(ValueError, match="unknown reference label"):
        evaluation_report(_result([]), {"cell": [[0.0, 0.0, 1.0, 1.0]]})
    report = evaluation_report(_result([]), REFERENCES)
    assert report["n_detections"] == 0
    assert all(metric["value"] == 0.0 and metric["n_detected"] == 0 for metric in report["metrics"])
