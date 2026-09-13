from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"
MODEL_REVISION = "7587a7ef111d9dcbf8ac695f1376ab7014340a0c"
MODEL_LICENSE = "mit"
MODEL_KEY = "table-transformer-structure-v1.1-all"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# The six structure classes the checkpoint was fine-tuned on (config.json id2label, in id order).
LABELS = (
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
)
# Recognition threshold: the value the upstream repository's inference script applies to every
# structure class (microsoft/table-transformer src/inference.py `structure_class_thresholds`, main @
# 16d124f, 2023-09-07). It gates a softmax class score over 125 DETR queries that was not calibrated
# for any document domain; the deployment owns tuning it on labelled tables.
RECOGNITION_THRESHOLD = 0.5
# The upstream inference script crops each detected table with this many pixels of padding before
# structure recognition; the tutorial reproduces that convention. It is documentation, not enforced.
UPSTREAM_CROP_PADDING = 10
# The checkpoint's DETR decoder emits exactly num_queries proposals per image (config.json), so no
# image can yield more than this many structure objects.
MAX_DETECTIONS = 125
# Input ceilings. The processor resizes so the longest edge is 800 px (preprocessor_config.json
# `size.longest_edge`), so image cost is bounded whatever the caller sends; the side ceiling only
# guards memory during decoding and resizing.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
# preprocessor_config.json says `size: {"longest_edge": 800}`; the pinned transformers image processor
# only accepts the two-key form, and capping both edges at 800 resizes every image so its longest edge
# is 800 px — the same transform (see from_pretrained).
PROCESSOR_SIZE = {"shortest_edge": 800, "longest_edge": 800}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy pixel boxes; the building block for any caller-side mAP."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be [x0, y0, x1, y1]")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("boxes must satisfy x0 <= x1 and y0 <= y1")
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"threshold must be a number in [0, 1], got {value!r}")
    return float(value)


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one table image as PIL.Image.Image (any mode, converted to RGB): a crop of a single table, "
        f"ideally with about {UPSTREAM_CROP_PADDING} px of page around it, as the upstream inference "
        "script crops tables"
    ),
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "threshold": [0.0, 1.0],
    "labels": list(LABELS),
    "max_detections": MAX_DETECTIONS,
    "preprocessing": (
        "image converted to RGB; the processor resizes so the longest edge is 800 px, normalises with "
        "ImageNet mean/std, and returned boxes are mapped back to input pixels"
    ),
}


def _check_inputs(image: Any, threshold: Any) -> tuple[Image.Image, float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``recognize`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    return validate_image(image), _check_threshold(threshold)


def validate_inputs(
    image: Image.Image,
    *,
    threshold: float = RECOGNITION_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``recognize`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, checked = _check_inputs(image, threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (recognize takes one table image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "threshold": checked,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def structure_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Count the structure objects per label and derive the grid size the rows and columns imply.

    ``n_rows`` x ``n_columns`` is the cell grid a caller would build by intersecting row and column
    boxes (the upstream ``objects_to_structures`` step); this helper only counts, it does not build cells.
    """
    counts = {label: 0 for label in LABELS}
    for det in result["detections"]:
        counts[det["label"]] += 1
    return {
        "counts": counts,
        "n_rows": counts["table row"],
        "n_columns": counts["table column"],
        "n_cells_implied": counts["table row"] * counts["table column"],
        "has_column_header": counts["table column header"] > 0,
    }


def evaluation_report(
    result: Mapping[str, Any],
    ground_truth_boxes: Mapping[str, Sequence[Sequence[float]]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``ground_truth_boxes`` (label -> xyxy reference boxes for that structure class) the report
    carries one ``box_iou`` entry per reference — the best-overlapping detection **of the same label**
    — plus per-label reference/detection counts, as sample-sanity geometry evidence; without them the
    verdict is ``not-measurable`` and the report says what labelled data would make the task measurable.
    """
    detections = list(result["detections"])
    summary = structure_summary(result)
    base = {
        "task": "table structure recognition on a table-crop image",
        "decision_rule": (
            "a DETR query survives when its softmax score for one of the six structure classes reaches "
            "the threshold; the score is a class probability under the model's own softmax, not a "
            "calibrated estimate for the deployment's tables"
        ),
        "threshold": result.get("threshold", RECOGNITION_THRESHOLD),
        "sample_kind": sample_kind,
        "n_detections": len(detections),
        "structure_summary": summary,
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not ground_truth_boxes:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth structure boxes were supplied for the evaluated table",
            "needs": (
                "labelled row/column/header/spanning-cell boxes on your own tables, scored per object with "
                "box_iou and aggregated into per-class precision/recall or the cell-level metrics (GriTS) "
                "the upstream paper uses; no such labelled set ships with this repository"
            ),
        }
    metrics = []
    for label, boxes in ground_truth_boxes.items():
        if label not in LABELS:
            raise ValueError(f"unknown reference label {label!r}; expected one of {LABELS}")
        same = [det for det in detections if det["label"] == label]
        for index, box in enumerate(boxes):
            ious = [box_iou(det["box"], box) for det in same]
            best = max(range(len(ious)), key=ious.__getitem__) if ious else None
            metrics.append(
                {
                    "id": "box_iou",
                    "label": label,
                    "reference": f"{label}-{index}",
                    "value": ious[best] if best is not None else 0.0,
                    "n_reference": len(boxes),
                    "n_detected": len(same),
                    "estimation": (
                        "one reference box per structure object on a single table, no dispersion estimate"
                    ),
                }
            )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} reference box(es) on one tutorial table; geometry sanity evidence, "
            "not a structure-recognition benchmark"
        ),
        "needs": (
            "a labelled table set from the deployment domain (publishers, scans, layouts) for any "
            "precision/recall or GriTS claim"
        ),
    }


@dataclass
class TableTransformerStructurePipeline:
    """Table structure recognition (rows, columns, headers, spanning cells) over the pinned checkpoint."""

    _runner: Callable[[Image.Image, float], list[dict[str, Any]]]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> TableTransformerStructurePipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # The pinned preprocessor_config.json declares `size: {"longest_edge": 800}`, a shape the pinned
        # transformers release's DetrImageProcessor refuses ("Size must contain ... 'shortest_edge' and
        # 'longest_edge'"). PROCESSOR_SIZE is the same resize expressed in the accepted form: with both
        # edges capped at 800 the longest edge always lands on 800 px, exactly as the upstream file says.
        processor = AutoImageProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, size=dict(PROCESSOR_SIZE), **kwargs
        )
        # This checkpoint's config carries an in-library ResNet backbone_config (use_timm_backbone=False),
        # so nothing is fetched at construction; use_pretrained_backbone=False is passed anyway so the
        # loader can never reach for ImageNet weights the checkpoint already contains.
        model = TableTransformerForObjectDetection.from_pretrained(
            source,
            revision=MODEL_REVISION,
            trust_remote_code=False,
            use_pretrained_backbone=False,
            **kwargs,
        )
        model = model.to(resolved_device).eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}

        def runner(image: Image.Image, threshold: float) -> list[dict]:
            inputs = processor(images=image, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=[image.size[::-1]]
            )[0]
            return [
                {
                    "box": [float(v) for v in box.tolist()],
                    "label": id2label[int(label)],
                    "score": float(score),
                }
                for box, label, score in zip(result["boxes"], result["labels"], result["scores"], strict=True)
            ]

        return cls(runner, resolved_device)

    def recognize(self, image: Image.Image, *, threshold: float = RECOGNITION_THRESHOLD) -> dict[str, Any]:
        """Recognise the structure of one table-crop image; boxes are xyxy pixel coordinates in the input."""
        rgb, checked = _check_inputs(image, threshold)
        detections = self._runner(rgb, checked)
        if len(detections) > MAX_DETECTIONS:
            raise RuntimeError(
                f"backend returned {len(detections)} detections > num_queries {MAX_DETECTIONS}"
            )
        for det in detections:
            if set(det) != {"box", "label", "score"} or len(det["box"]) != 4 or det["label"] not in LABELS:
                raise RuntimeError(f"backend returned a malformed detection: {det!r}")
        return {
            "detections": sorted(detections, key=lambda d: -d["score"]),
            "threshold": checked,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
