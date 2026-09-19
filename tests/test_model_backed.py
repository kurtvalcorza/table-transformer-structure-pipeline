"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): the zero-shot
evaluation on synthetic ruled tables, the three-policy ladder with a one-layer unfreeze, and the artifact round
trip with head, box-head and decoder tensors. Skipped when the weights are absent."""

# ruff: noqa: E501  -- assertion lines are kept on one line

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from table_transformer_structure_pipeline import (
    ADAPT_LABELS,
    DEFAULT_WEIGHTS_DIR,
    POLICY_FROZEN,
    POLICY_ZERO_SHOT,
    WEIGHTS_FILE,
    TableTransformerStructurePipeline,
)

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)


def _table(seed: int, n_rows: int = 4, n_cols: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    width, height = 320 + seed * 7 % 60, 160 + seed * 11 % 50
    canvas = np.full((height, width), 250, dtype=np.uint8)
    x0, y0, x1, y1 = 8, 6, width - 8, height - 6
    for r in range(n_rows + 1):
        y = round(y0 + (y1 - y0) * r / n_rows)
        canvas[y : y + 2, x0:x1] = 20
    for c in range(n_cols + 1):
        x = round(x0 + (x1 - x0) * c / n_cols)
        canvas[y0:y1, x : x + 2] = 20
    for r in range(n_rows):  # a short dark "word" in every cell
        for c in range(n_cols):
            cy = round(y0 + (y1 - y0) * (r + 0.5) / n_rows)
            cx = round(x0 + (x1 - x0) * (c + 0.5) / n_cols)
            canvas[cy - 4 : cy + 4, cx - 12 : cx + 12] = rng.integers(20, 80, size=(8, 24), dtype=np.uint8)
    objects = [{"label": "table", "box": [x0, y0, x1, y1]}]
    for r in range(n_rows):
        objects.append(
            {"label": "table row", "box": [x0, y0 + (y1 - y0) * r / n_rows, x1, y0 + (y1 - y0) * (r + 1) / n_rows]}
        )
    for c in range(n_cols):
        objects.append(
            {
                "label": "table column",
                "box": [x0 + (x1 - x0) * c / n_cols, y0, x0 + (x1 - x0) * (c + 1) / n_cols, y1],
            }
        )
    return {
        "id": f"s{seed:02d}",
        "image": Image.fromarray(canvas).convert("RGB"),
        "objects": [{"label": o["label"], "box": [float(v) for v in o["box"]]} for o in objects],
    }


RECORDS = [_table(i) for i in range(12)]


@pytest.fixture(scope="module")
def pipe():
    return TableTransformerStructurePipeline.from_pretrained(device="cpu")


def test_zero_shot_evaluation_is_finite_and_labelled(pipe):
    metrics = pipe.evaluate_zero_shot(RECORDS[9:])
    assert metrics["n_images"] == 3 and metrics["adapted"] is False and metrics["policy"].startswith("zero-shot")
    assert metrics["scored_labels"] == ["table", "table column", "table row"]
    assert 0.0 <= metrics["map50"] <= 1.0 and 0.0 <= metrics["mean_best_iou"] <= 1.0
    assert metrics["per_label"]["table spanning cell"]["ap50"] is None


def test_ladder_with_one_layer_unfreeze_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(RECORDS[:9], RECORDS[9:], head_steps=40, trainable_layers=1, epochs=1, lr=1e-4)
    assert result["classes"] == list(ADAPT_LABELS)
    assert [h["stage"] for h in result["history"]][:2] == [POLICY_ZERO_SHOT, POLICY_FROZEN]
    assert result["n_trainable_head"] == 256 * 5 + 5 + 132_612 and result["n_trainable_layers"] == 1_578_752
    assert result["n_total"] == 28_828_619 and len(result["history"]) == 3 and result["best_epoch"] in (0, 1, 2)
    assert (result["policy"] == POLICY_ZERO_SHOT) == (result["best_epoch"] == 0)
    assert (result["policy"] == POLICY_FROZEN) == (result["best_epoch"] == 1)
    metrics = pipe.evaluate(RECORDS[9:])
    assert metrics["n_images"] == 3 and metrics["adapted"] is True and metrics["policy"] == result["policy"]
    assert metrics["loss"] == pytest.approx(result["history"][result["best_epoch"]]["val"]["loss"], abs=1e-4)
    recognised = pipe.recognize_adapted(RECORDS[0]["image"], threshold=0.05)
    assert recognised["classes"] == list(ADAPT_LABELS) and all(
        d["label"] in ADAPT_LABELS for d in recognised["detections"]
    )
    base = pipe.recognize(RECORDS[0]["image"], threshold=0.05)
    assert all(d["label"] in pipe.classes or "header" in d["label"] for d in base["detections"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert "head.weight" in manifest["tensors"] and "bbox_head.layers.0.weight" in manifest["tensors"]
    assert any(t.startswith("model.decoder.layers.5.") for t in manifest["tensors"]) == (result["best_epoch"] > 1)
    reloaded = TableTransformerStructurePipeline.from_artifact(artifact, device="cpu")
    assert reloaded.predict_objects(RECORDS[:2]) == pipe.predict_objects(RECORDS[:2])
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.classes == result["classes"]
