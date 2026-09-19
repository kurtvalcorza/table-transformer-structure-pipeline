"""Structure-recognition metrics for the adaptation contract, implemented here with no external scorer.

Boxes are ``[x_min, y_min, x_max, y_max]`` in pixels of the table image they belong to. A prediction is a
list of ``{"box", "label", "score"}`` entries per image — every (query, class) pair the heads emit, not just
those above a threshold, because average precision ranks them all; the reference is the list of
``{"label", "box"}`` structure objects per image.

- **AP@t per class** — average precision at IoU threshold *t* for one label: its predictions over all images
  are ranked by score, each is a true positive when it overlaps a not-yet-matched reference object of the
  same label on its image with IoU ≥ *t* (greedy in score order), the precision-recall curve is made
  monotone from the right and the area under it is summed over the recall steps (the VOC 2010+ / COCO
  "all-points" convention). A label absent from the references of a split is reported as ``None`` and left
  out of the means.
- **mAP@0.5 / mAP** — the mean over the scored labels of AP@0.5, and of the mean of AP@t over
  t = 0.50, 0.55, …, 0.95 (the COCO primary metric averaged over classes).
- **operating point** — at the pipeline's `RECOGNITION_THRESHOLD`, per-label recall / precision of the
  surviving predictions (greedy IoU ≥ 0.5 matching), and **grid agreement**: the fraction of tables whose
  surviving `table row` and `table column` counts both equal the reference counts — the structure-level
  question a caller actually asks.
- **mean best IoU** — for every reference object, the IoU of its best-overlapping prediction of the same
  label, averaged.

The **grid prior** frames the numbers: the training split's mean row and column counts, rounded, laid out as
a uniform grid over the whole image (table = the image), with a constant score — what "tables are usually
about this shape" alone buys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .pipeline import box_iou

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
MATCH_IOU = 0.5

METRIC_DEFINITIONS = {
    "ap50": "per label: average precision at IoU 0.50 over all ranked predictions of that label",
    "ap75": "per label: average precision at IoU 0.75",
    "ap": "per label: mean of the average precision at IoU 0.50, 0.55, ..., 0.95",
    "map50": "mean over the scored labels of ap50",
    "map": "mean over the scored labels of ap (the COCO primary metric averaged over classes)",
    "recall_at_threshold": (
        "per label: fraction of reference objects matched (IoU >= 0.5) by a prediction of the same label at "
        "or above the operating threshold"
    ),
    "precision_at_threshold": (
        "per label: fraction of predictions of that label at or above the operating threshold that match a "
        "reference object"
    ),
    "grid_exact_at_threshold": (
        "fraction of tables whose surviving `table row` and `table column` counts both equal the reference "
        "counts"
    ),
    "mean_best_iou": (
        "mean over reference objects of the IoU of their best-overlapping prediction of the same label, "
        "any score"
    ),
}


def _check_pairs(
    predictions: Sequence[Sequence[Mapping[str, Any]]], references: Sequence[Sequence[Mapping[str, Any]]]
) -> None:
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} prediction lists for {len(references)} reference lists")
    if not references:
        raise ValueError("at least one image is required")
    for preds in predictions:
        for p in preds:
            if "box" not in p or "score" not in p or "label" not in p or len(p["box"]) != 4:
                raise ValueError("each prediction needs a 4-value 'box', a 'label' and a 'score'")
    for refs in references:
        for r in refs:
            if "box" not in r or "label" not in r or len(r["box"]) != 4:
                raise ValueError("each reference needs a 4-value 'box' and a 'label'")


def _of_label(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Sequence[Mapping[str, Any]]],
    label: str,
) -> tuple[list[list[Mapping[str, Any]]], list[list[Sequence[float]]]]:
    return (
        [[p for p in preds if p["label"] == label] for preds in predictions],
        [[r["box"] for r in refs if r["label"] == label] for refs in references],
    )


def average_precision(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Sequence[Mapping[str, Any]]],
    label: str,
    iou_threshold: float = MATCH_IOU,
) -> float | None:
    """All-points average precision for one label over a list of images (None when it has no references)."""
    _check_pairs(predictions, references)
    preds_l, refs_l = _of_label(predictions, references, label)
    n_ref = sum(len(r) for r in refs_l)
    if n_ref == 0:
        return None
    ranked = sorted(
        (
            (float(p["score"]), i, [float(v) for v in p["box"]])
            for i, preds in enumerate(preds_l)
            for p in preds
        ),
        key=lambda t: -t[0],
    )
    if not ranked:
        return 0.0
    matched = [np.zeros(len(r), dtype=bool) for r in refs_l]
    tp = np.zeros(len(ranked))
    for k, (_score, i, box) in enumerate(ranked):
        best, best_j = 0.0, -1
        for j, ref in enumerate(refs_l[i]):
            if matched[i][j]:
                continue
            iou = box_iou(box, ref)
            if iou > best:
                best, best_j = iou, j
        if best >= iou_threshold and best_j >= 0:
            matched[i][best_j] = True
            tp[k] = 1.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = cum_tp / n_ref
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope from the right
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision))


def _operating_point(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """Per-label recall / precision at or above `threshold` (greedy IoU >= MATCH_IOU) and grid agreement."""
    per_label = {}
    per_image: list[dict[str, Any]] = [{} for _ in references]
    for label in labels:
        preds_l, refs_l = _of_label(predictions, references, label)
        n_ref = sum(len(r) for r in refs_l)
        found = kept = true_kept = 0
        for i, (preds, refs) in enumerate(zip(preds_l, refs_l, strict=True)):
            surviving = [p for p in preds if float(p["score"]) >= threshold]
            kept += len(surviving)
            used: set[int] = set()
            image_found = 0
            for p in sorted(surviving, key=lambda q: -float(q["score"])):
                best, best_j = 0.0, -1
                for j, ref in enumerate(refs):
                    if j in used:
                        continue
                    iou = box_iou(p["box"], ref)
                    if iou > best:
                        best, best_j = iou, j
                if best >= MATCH_IOU and best_j >= 0:
                    used.add(best_j)
                    image_found += 1
                    true_kept += 1
            found += image_found
            per_image[i][label] = {
                "n_reference": len(refs),
                "n_at_threshold": len(surviving),
                "found": image_found,
            }
        per_label[label] = {
            "recall": found / n_ref if n_ref else None,
            "precision": true_kept / kept if kept else None,
            "n_reference": n_ref,
            "n_at_threshold": kept,
        }
    grid_hits = 0
    for entry in per_image:
        row, col = entry.get("table row"), entry.get("table column")
        if (
            row
            and col
            and row["n_at_threshold"] == row["n_reference"]
            and col["n_at_threshold"] == col["n_reference"]
        ):
            grid_hits += 1
            entry["grid_exact"] = True
        else:
            entry["grid_exact"] = False
    return {
        "threshold": float(threshold),
        "per_label": per_label,
        "grid_exact": grid_hits / len(references),
        "per_image": per_image,
    }


def structure_metrics(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Sequence[Mapping[str, Any]]],
    *,
    labels: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """Per-label AP@0.5 / AP@0.75 / AP, their class means, the operating point at `threshold` and at 0.5, grid
    agreement and mean best IoU."""
    _check_pairs(predictions, references)
    per_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        aps = {t: average_precision(predictions, references, label, t) for t in IOU_THRESHOLDS}
        n_ref = sum(1 for refs in references for r in refs if r["label"] == label)
        if aps[0.5] is None:
            per_label[label] = {"n_reference": 0, "ap50": None, "ap75": None, "ap": None}
        else:
            per_label[label] = {
                "n_reference": n_ref,
                "ap50": aps[0.5],
                "ap75": aps[0.75],
                "ap": float(np.mean([v for v in aps.values()])),
                "ap_by_iou": {str(t): v for t, v in aps.items()},
            }
    scored = [label for label in labels if per_label[label]["ap50"] is not None]
    points = {
        t_: _operating_point(predictions, references, labels, t_) for t_ in sorted({0.5, float(threshold)})
    }
    primary = points[float(threshold)]
    best_ious = []
    per_image = []
    for i, (preds, refs) in enumerate(zip(predictions, references, strict=True)):
        image_best = []
        for ref in refs:
            same = [p for p in preds if p["label"] == ref["label"]]
            image_best.append(max((box_iou(p["box"], ref["box"]) for p in same), default=0.0))
        best_ious.extend(image_best)
        per_image.append(
            {
                "n_reference": len(refs),
                "grid_exact": primary["per_image"][i]["grid_exact"],
                "labels": {k: v for k, v in primary["per_image"][i].items() if k != "grid_exact"},
                "mean_best_iou": float(np.mean(image_best)) if image_best else 0.0,
            }
        )
    return {
        "n_images": len(references),
        "n_reference_objects": sum(len(r) for r in references),
        "labels": list(labels),
        "scored_labels": scored,
        "per_label": per_label,
        "map50": float(np.mean([per_label[label]["ap50"] for label in scored])) if scored else 0.0,
        "map75": float(np.mean([per_label[label]["ap75"] for label in scored])) if scored else 0.0,
        "map": float(np.mean([per_label[label]["ap"] for label in scored])) if scored else 0.0,
        "threshold": float(threshold),
        "recall_at_threshold": {label: primary["per_label"][label]["recall"] for label in labels},
        "precision_at_threshold": {label: primary["per_label"][label]["precision"] for label in labels},
        "grid_exact_at_threshold": primary["grid_exact"],
        "operating_points": {
            str(t_): {"per_label": point["per_label"], "grid_exact": point["grid_exact"]}
            for t_, point in points.items()
        },
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "per_image": per_image,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def grid_prior(train_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The training split's mean `table row` and `table column` counts, rounded to at least 1."""
    rows = [sum(1 for o in r["objects"] if o["label"] == "table row") for r in train_records]
    cols = [sum(1 for o in r["objects"] if o["label"] == "table column") for r in train_records]
    if not rows:
        raise ValueError("the training split has no records")
    return {"rows": max(1, round(float(np.mean(rows)))), "columns": max(1, round(float(np.mean(cols))))}


def grid_prior_prediction(width: int, height: int, grid: Mapping[str, int]) -> list[dict[str, Any]]:
    """A uniform grid over the whole image: one `table`, `rows` bands and `columns` bands, scored 1.0."""
    out = [{"box": [0.0, 0.0, float(width), float(height)], "label": "table", "score": 1.0}]
    n_rows, n_cols = int(grid["rows"]), int(grid["columns"])
    for i in range(n_rows):
        out.append(
            {
                "box": [0.0, height * i / n_rows, float(width), height * (i + 1) / n_rows],
                "label": "table row",
                "score": 1.0,
            }
        )
    for j in range(n_cols):
        out.append(
            {
                "box": [width * j / n_cols, 0.0, width * (j + 1) / n_cols, float(height)],
                "label": "table column",
                "score": 1.0,
            }
        )
    return out


def prior_baseline(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """The grid prior of the training split scored on the test records."""
    grid = grid_prior(train_records)
    predictions = [grid_prior_prediction(*record["image"].size, grid) for record in test_records]
    out = structure_metrics(
        predictions, [r["objects"] for r in test_records], labels=labels, threshold=threshold
    )
    out["baseline"] = (
        "a uniform grid of the training split's mean row and column counts over the whole image (grid prior)"
    )
    out["prior_grid"] = grid
    return out
