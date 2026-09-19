"""Offline tests for the structure-labelled dataset contract, the embedded SciTSR-PD sample and its provenance,
the seeded paper-level split, the per-label AP / operating-point / grid-agreement metrics with the grid prior,
BYOD loaders (directory and zip), the exact Hungarian matcher and the multi-class DETR set loss on tensors,
artifact-manifest rejections and adapt() argument validation. No model library beyond torch tensors is
loaded."""

# ruff: noqa: E501  -- assertion lines are kept on one line

from __future__ import annotations

import base64
import csv
import hashlib
import itertools
import json
import random
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from table_transformer_structure_pipeline import (
    ADAPT_LABELS,
    ARTIFACT_FORMAT,
    DECODER_LAYERS,
    LABELS,
    MODEL_ID,
    MODEL_REVISION,
    NUM_QUERIES,
    RECOGNITION_THRESHOLD,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    TableTransformerStructurePipeline,
    average_precision,
    build_sample_dataset,
    check_split_disjoint,
    dataset_digest,
    grid_prior,
    grid_prior_prediction,
    image_digest,
    load_byod_dataset,
    load_corpus,
    load_sample_dataset,
    prior_baseline,
    split_dataset,
    split_summary,
    structure_metrics,
    validate_dataset,
    write_dataset_csv,
)
from table_transformer_structure_pipeline import pipeline as pl
from table_transformer_structure_pipeline import sample_data as sd

W, H = 120, 80


def _table(seed: int, n_rows=3, n_cols=2, size=(W, H)) -> dict:
    """A synthetic table crop: ruled grid on a light background, with the derived-box convention."""
    rng = np.random.default_rng(seed)
    canvas = np.full((size[1], size[0]), 245, dtype=np.uint8)
    x0, y0, x1, y1 = 4 + seed % 3, 3 + seed % 2, size[0] - 5, size[1] - 4
    for r in range(n_rows + 1):
        y = round(y0 + (y1 - y0) * r / n_rows)
        canvas[y : y + 1, x0:x1] = 30
    for c in range(n_cols + 1):
        x = round(x0 + (x1 - x0) * c / n_cols)
        canvas[y0:y1, x : x + 1] = 30
    canvas[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2] -= rng.integers(
        0, 20, size=(y1 - y0 - 4, x1 - x0 - 4), dtype=np.uint8
    )
    objects = [{"label": "table", "box": [x0, y0, x1, y1]}]
    for r in range(n_rows):
        objects.append(
            {
                "label": "table row",
                "box": [x0, y0 + (y1 - y0) * r / n_rows, x1, y0 + (y1 - y0) * (r + 1) / n_rows],
            }
        )
    for c in range(n_cols):
        objects.append(
            {
                "label": "table column",
                "box": [x0 + (x1 - x0) * c / n_cols, y0, x0 + (x1 - x0) * (c + 1) / n_cols, y1],
            }
        )
    if seed % 4 == 0:
        objects.append({"label": "table spanning cell", "box": [x0, y0, x1, y0 + (y1 - y0) / n_rows]})
    return {
        "id": f"t{seed:02d}",
        "image": Image.fromarray(canvas).convert("RGB"),
        "objects": [{"label": o["label"], "box": [float(v) for v in o["box"]]} for o in objects],
        "group": f"paper{seed % 5}",
    }


def _records(n=12):
    return [_table(i) for i in range(n)]


def _pipeline_without_model():
    return TableTransformerStructurePipeline(lambda image, threshold: [], "cpu")


def test_embedded_sample_is_complete_and_traceable(forbid_model_imports):
    assert len(SAMPLE_RECORDS) == 94 and len(sd.SAMPLE_IMAGES_B64) == 94
    assert tuple(sd.SAMPLE_LABELS) == ADAPT_LABELS and set(ADAPT_LABELS) < set(LABELS)
    assert {f["split"] for f in sd.SOURCE_FILES} == {"train", "test"} and all(
        len(f["sha256"]) == 64 for f in sd.SOURCE_FILES
    )
    ids = [r["table_id"] for r in SAMPLE_RECORDS]
    assert len(set(ids)) == 94 and ids == sorted(ids)
    for entry in SAMPLE_RECORDS[:5]:
        data = base64.b64decode(sd.SAMPLE_IMAGES_B64[entry["file"]])
        assert len(data) == entry["png_bytes"] and hashlib.sha256(data).hexdigest() == entry["png_sha256"]
        assert entry["paper_license"].startswith(
            ("CC0", "other:http://creativecommons.org/licenses/publicdomain")
        )
        labels = [o["label"] for o in entry["objects"]]
        assert (
            labels.count("table") == 1
            and labels.count("table row") == entry["n_rows"]
            and labels.count("table column") == entry["n_cols"]
        )
        assert labels.count("table spanning cell") == entry["n_spanning"] and set(labels) <= set(ADAPT_LABELS)
        assert 0.0 < entry["alignment"]["ink_score"] < 1.0
    assert len({r["paper_id"] for r in SAMPLE_RECORDS}) == 46
    assert sum(SAMPLE_SPLIT.values()) == 46


def test_load_corpus_decodes_and_verifies(monkeypatch, forbid_model_imports):
    corpus = load_corpus()
    assert len(corpus) == 94 and all(r["image"].mode == "RGB" for r in corpus)
    first = corpus[0]
    assert first["image"].size == (SAMPLE_RECORDS[0]["width"], SAMPLE_RECORDS[0]["height"])
    assert first["objects"][0]["label"] == "table" and first["grid"] == [
        SAMPLE_RECORDS[0]["n_rows"],
        SAMPLE_RECORDS[0]["n_cols"],
    ]
    report = validate_dataset(corpus)
    assert (
        report["n_records"] == 94
        and report["objects_per_label"]["table"] == 94
        and report["objects_per_label"]["table row"] > 600
    )
    # a tampered PNG is refused by its digest
    tampered = {**SAMPLE_RECORDS[0], "png_sha256": "0" * 64}
    monkeypatch.setattr(sd, "SAMPLE_RECORDS", (tampered, *SAMPLE_RECORDS[1:]))
    monkeypatch.setattr(
        "table_transformer_structure_pipeline.samples.SAMPLE_RECORDS", (tampered, *SAMPLE_RECORDS[1:])
    )
    with pytest.raises(ValueError, match="does not match its recorded size / digest"):
        load_corpus()


def test_sample_split_is_by_paper_seeded_and_disjoint(forbid_model_imports):
    splits = load_sample_dataset()
    assert {k: len(v) for k, v in splits.items()} == {"test": 21, "validation": 23, "train": 50}
    assert check_split_disjoint(splits) == {"test": 21, "validation": 23, "train": 50}
    summary = split_summary(splits)
    assert (
        summary["test"]["papers"] == 10
        and summary["validation"]["papers"] == 7
        and summary["train"]["papers"] == 29
    )
    assert splits["train"][0]["id"] == "train-000" and splits["train"][0]["source_id"] in {
        r["table_id"] for r in SAMPLE_RECORDS
    }
    again = load_sample_dataset()
    assert [r["source_id"] for r in again["test"]] == [r["source_id"] for r in splits["test"]]
    other = load_sample_dataset(seed=1)
    assert [r["source_id"] for r in other["test"]] != [r["source_id"] for r in splits["test"]]
    with pytest.raises(ValueError, match="papers available"):
        build_sample_dataset(load_corpus(), sizes={"train": 40, "test": 10})


def test_validate_dataset_reports_and_rejects(tmp_path, forbid_model_imports):
    records = _records()
    report = validate_dataset(records)
    assert (
        report["n_records"] == 12
        and report["objects_per_label"]["table"] == 12
        and report["objects_per_label"]["table spanning cell"] == 3
    )
    assert (
        report["objects_per_image"] == {"min": 6, "max": 7}
        and len(report["digest"]) == 64
        and report["model_id"] == MODEL_ID
    )
    assert dataset_digest(records) == report["digest"] and dataset_digest(records[::-1]) != report["digest"]
    path = tmp_path / "t.png"
    records[0]["image"].save(path)
    assert validate_dataset([{**records[0], "image": str(path)}], min_records=1)["n_records"] == 1
    with pytest.raises(ValueError, match="8..2000"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="missing 'objects'"):
        validate_dataset([{"id": "a", "image": records[0]["image"]}] * 8)
    with pytest.raises(ValueError, match="not one of"):
        validate_dataset(
            [
                {
                    **records[0],
                    "objects": [
                        {"label": "table column header", "box": [1, 1, 5, 5]},
                        *records[0]["objects"],
                    ],
                }
            ]
            * 8
        )
    with pytest.raises(ValueError, match="must lie inside"):
        validate_dataset(
            [
                {
                    **records[0],
                    "objects": [{"label": "table row", "box": [0, 0, W + 1, 5]}, *records[0]["objects"]],
                }
            ]
            * 8
        )
    with pytest.raises(ValueError, match="smaller than"):
        validate_dataset(
            [{**records[0], "objects": [{"label": "table row", "box": [1, 1, 2, 5]}, *records[0]["objects"]]}]
            * 8
        )
    with pytest.raises(ValueError, match="at most one `table`"):
        validate_dataset(
            [{**records[0], "objects": [{"label": "table", "box": [1, 1, 50, 50]}, *records[0]["objects"]]}]
            * 8
        )
    with pytest.raises(ValueError, match="at least one `table row`"):
        validate_dataset(
            [{**records[0], "objects": [o for o in records[0]["objects"] if o["label"] != "table row"]}] * 8
        )
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0]] * 8)
    with pytest.raises(ValueError, match="image side outside"):
        validate_dataset([{**records[0], "image": Image.new("RGB", (8, 8))}] * 8)


def test_split_dataset_groups_by_paper_deduplicates_and_is_seeded(forbid_model_imports):
    records = _records(20) + [{**_records(1)[0], "id": "dup"}]
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == 20  # the duplicate table is dropped
    assert check_split_disjoint(splits)
    assert split_dataset(records, seed=3)["test"][0]["id"] == splits["test"][0]["id"]
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.6, test_fraction=0.5)


def test_metrics_operating_points_grid_and_prior(forbid_model_imports):
    refs = [
        [
            {"label": "table row", "box": [0, 0, 10, 10]},
            {"label": "table row", "box": [0, 10, 10, 20]},
            {"label": "table column", "box": [0, 0, 10, 20]},
        ]
    ]
    preds = [
        [
            {"box": [0, 0, 10, 10], "label": "table row", "score": 0.9},
            {"box": [0, 10, 10, 20], "label": "table row", "score": 0.8},
            {"box": [0, 0, 10, 20], "label": "table column", "score": 0.7},
            {"box": [0, 0, 10, 10], "label": "table column", "score": 0.6},  # a false positive
        ]
    ]
    assert (
        average_precision(preds, refs, "table row") == 1.0
        and average_precision(preds, refs, "table column") == 1.0
    )
    assert average_precision(preds, refs, "table") is None
    m = structure_metrics(preds, refs, labels=ADAPT_LABELS, threshold=0.65)
    assert (
        m["scored_labels"] == ["table column", "table row"]
        and m["map50"] == 1.0
        and m["per_label"]["table"]["ap50"] is None
    )
    assert (
        m["recall_at_threshold"]["table row"] == 1.0
        and m["precision_at_threshold"]["table column"] == 1.0
        and m["grid_exact_at_threshold"] == 1.0
    )
    low = structure_metrics(preds, refs, labels=ADAPT_LABELS, threshold=0.5)
    assert (
        low["precision_at_threshold"]["table column"] == 0.5 and low["grid_exact_at_threshold"] == 0.0
    )  # two columns survive
    assert m["mean_best_iou"] == 1.0
    with pytest.raises(ValueError, match="needs a 4-value"):
        structure_metrics(
            [[{"box": [0, 0, 1], "label": "table", "score": 1}]], refs, labels=ADAPT_LABELS, threshold=0.5
        )
    records = _records()
    grid = grid_prior(records)
    assert grid == {"rows": 3, "columns": 2}
    pred = grid_prior_prediction(W, H, grid)
    assert [p["label"] for p in pred] == [
        "table",
        "table row",
        "table row",
        "table row",
        "table column",
        "table column",
    ]
    prior = prior_baseline(records, records[:4], labels=ADAPT_LABELS, threshold=RECOGNITION_THRESHOLD)
    assert (
        prior["prior_grid"] == grid
        and prior["per_label"]["table"]["ap50"] == 1.0
        and prior["per_label"]["table spanning cell"]["ap50"] == 0.0
    )


def test_byod_directory_and_zip_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records(8)
    csv_path = write_dataset_csv(records, tmp_path / "structure.csv")
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert (
        rows[0]["label"] == "table"
        and rows[0]["group"] == "paper0"
        and len(rows) == sum(len(r["objects"]) for r in records)
    )
    for record in records:
        record["image"].save(tmp_path / f"{record['id']}.png")
    loaded = load_byod_dataset(tmp_path)
    assert (
        len(loaded) == 8
        and loaded[0]["group"] == "paper0"
        and [o["label"] for o in loaded[0]["objects"]] == [o["label"] for o in records[0]["objects"]]
    )
    assert validate_dataset(loaded)["digest"] == dataset_digest(records)
    zip_path = tmp_path / "byod.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, "inner/structure.csv")
        for record in records:
            archive.write(tmp_path / f"{record['id']}.png", f"inner/{record['id']}.png")
    from_zip = load_byod_dataset(zip_path)
    assert dataset_digest(from_zip) == dataset_digest(records)
    with pytest.raises(ValueError, match="must contain structure.csv"):
        with zipfile.ZipFile(tmp_path / "empty.zip", "w") as archive:
            archive.writestr("x.txt", "x")
        load_byod_dataset(tmp_path / "empty.zip")
    with pytest.raises(ValueError, match="missing columns"):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "structure.csv").write_text("id,file\n", encoding="utf-8")
        load_byod_dataset(bad)
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "nope.txt")


def test_hungarian_is_exact_on_rectangular_costs(forbid_model_imports):
    rng = random.Random(0)
    for n, m in [(1, 3), (4, 6), (5, 9), (6, 6)]:
        cost = [[rng.random() for _ in range(m)] for _ in range(n)]
        pairs = TableTransformerStructurePipeline.hungarian(cost)
        assert len(pairs) == n and len({j for _, j in pairs}) == n
        best = min(sum(cost[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        assert sum(cost[i][j] for i, j in pairs) == pytest.approx(best)
    assert TableTransformerStructurePipeline.hungarian([]) == []
    with pytest.raises(ValueError, match="rows <= columns"):
        TableTransformerStructurePipeline.hungarian([[1.0], [2.0]])


def test_matcher_and_set_loss_prefer_the_right_query_and_class():
    pipe = _pipeline_without_model()
    classes = list(ADAPT_LABELS)
    target_classes = torch.tensor([classes.index("table row"), classes.index("table column")])
    target_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])
    boxes = torch.full((NUM_QUERIES, 4), 0.5)
    boxes[3] = target_boxes[0]
    boxes[7] = target_boxes[1]
    logits = torch.zeros(NUM_QUERIES, len(classes) + 1)
    logits[3, classes.index("table row")] = 3.0
    logits[7, classes.index("table column")] = 3.0
    pairs = pipe._match(torch.softmax(logits, dim=-1), boxes, target_classes, target_boxes)
    assert sorted(pairs) == [(3, 0), (7, 1)]
    good = float(pipe._loss(logits, boxes, target_classes, target_boxes))
    swapped = logits.clone()
    swapped[3, classes.index("table row")], swapped[3, classes.index("table column")] = 0.0, 3.0
    assert good < float(pipe._loss(swapped, boxes, target_classes, target_boxes))
    giou = pipe._giou(target_boxes, target_boxes)
    assert torch.allclose(torch.diagonal(giou), torch.ones(2), atol=1e-6) and giou[0, 1] < 0.5
    many_classes = torch.zeros(40, dtype=torch.long)
    many = torch.rand(40, 4) * 0.4 + 0.3
    assert len(pipe._match(torch.softmax(logits, dim=-1), boxes, many_classes, many)) == 40
    record = _records(1)[0]
    tc, tb = pipe._targets(record, classes)
    assert (
        tc.tolist()[0] == classes.index("table")
        and tb.shape == (len(record["objects"]), 4)
        and float(tb.max()) <= 1.0
    )


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="head_steps"):
        pipe.adapt(_records(), head_steps=0)
    with pytest.raises(ValueError, match="head_lr"):
        pipe.adapt(_records(), head_lr=2.0)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=-1)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="trainable_layers"):
        pipe.adapt(_records(), trainable_layers=DECODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.evaluate_zero_shot(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    with pytest.raises(ValueError, match="no adapted heads"):
        pipe.recognize_adapted(_records()[0]["image"])
    with pytest.raises(ValueError, match="no adapted heads"):
        pipe.evaluate(_records())


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["bbox_head.layers.0.weight", "head.bias", "head.weight"],
        "adapter": {"classes": list(ADAPT_LABELS), "policy": pl.POLICY_FROZEN, "trainable_layers": 2},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)
    wrong_classes = {**manifest, "adapter": {**manifest["adapter"], "classes": ["table"]}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(wrong_classes))
    with pytest.raises(ValueError, match="not the adaptation labels"):
        pipe.load_artifact(tmp_path)


def test_image_digest_is_encoding_invariant(forbid_model_imports):
    image = _records(1)[0]["image"]
    buffer = __import__("io").BytesIO()
    image.save(buffer, "PNG")
    assert image_digest(Image.open(buffer)) == image_digest(image)


def test_load_artifact_refuses_unsupported_versions_extra_files_traversal_and_policies(
    tmp_path, forbid_model_imports
):
    pipe = _pipeline_without_model()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["head.bias", "head.weight"],
        "adapter": {"classes": list(ADAPT_LABELS), "policy": pl.POLICY_ZERO_SHOT, "trainable_layers": 0},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": 0})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "pytorch_model.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {**good["adapter"], "policy": "something else"}})
    with pytest.raises(ValueError, match="not a canonical policy"):
        pipe.load_artifact(tmp_path)
    write(
        {
            **good,
            "adapter": {**good["adapter"], "policy": pl.POLICY_UNFROZEN.format(k=3), "trainable_layers": 2},
        }
    )
    with pytest.raises(ValueError, match="not a canonical policy"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {**good["adapter"], "trainable_layers": 99}})
    with pytest.raises(ValueError, match="trainable_layers"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)


def test_byod_requires_a_group_and_consistent_rows(tmp_path, forbid_model_imports):
    records = _records(8)
    folder = tmp_path / "ungrouped"
    folder.mkdir()
    for record in records:
        record["image"].save(folder / f"{record['id']}.png")
    fields = ["id", "file", "group", "label", "x_min", "y_min", "x_max", "y_max"]

    def write_rows(rows):
        with open(folder / "structure.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    base = [
        {
            "id": r["id"],
            "file": f"{r['id']}.png",
            "group": "",
            "label": o["label"],
            "x_min": o["box"][0],
            "y_min": o["box"][1],
            "x_max": o["box"][2],
            "y_max": o["box"][3],
        }
        for r in records
        for o in r["objects"]
    ]
    write_rows(base)
    with pytest.raises(ValueError, match="has no `group`"):
        load_byod_dataset(folder)
    ungrouped = load_byod_dataset(folder, require_group=False)  # the explicit opt-out
    assert len(ungrouped) == 8 and all("group" not in r for r in ungrouped)
    grouped = [{**row, "group": "paper" + row["id"][-1]} for row in base]
    write_rows(grouped)
    assert {r["group"] for r in load_byod_dataset(folder)} == {"paper" + r["id"][-1] for r in records}
    write_rows(grouped + [{**grouped[0], "group": "other"}])
    with pytest.raises(ValueError, match="disagree on file or group"):
        load_byod_dataset(folder)
    write_rows(grouped + [{**grouped[0], "file": grouped[-1]["file"]}])
    with pytest.raises(ValueError, match="disagree on file or group"):
        load_byod_dataset(folder)


def test_split_dataset_never_lets_a_group_straddle_splits(forbid_model_imports):
    records = [{**r, "group": f"paper{i % 5}"} for i, r in enumerate(_records(30))]
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=1)
    where = {}
    for name, part in splits.items():
        for r in part:
            assert where.setdefault(r["group"], name) == name
    assert len(where) == 5 and check_split_disjoint(splits) and len(splits["test"]) == 6
