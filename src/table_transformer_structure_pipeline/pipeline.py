from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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

# ---- the adaptation contract ----
WEIGHTS_FILE = "model.safetensors"
WEIGHT_SHA256 = "9df416575a3a36ebd0129342d4f597f14d6e5170268f3d52d28584ab4466a501"  # dimer-base-manifest.json
PARAMETER_COUNT = 28_828_619
D_MODEL = 256  # the decoder's hidden size (config.json d_model)
NUM_QUERIES = MAX_DETECTIONS  # the decoder emits exactly this many (query, class, box) proposals per image
DECODER_LAYERS = 6  # config.json decoder_layers; only these are ever unfrozen
DEFAULT_TRAINABLE_LAYERS = 2
# The four checkpoint labels the adaptation contract scores and trains; SciTSR annotates no headers, so the
# restricted head has one logit per entry here plus no-object, initialised from the checkpoint's own rows.
ADAPT_LABELS: tuple[str, ...] = ("table", "table column", "table row", "table spanning cell")
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 50  # below this `evaluate` says measured-small-sample
CLASS_COST, BBOX_COST, GIOU_COST = 1.0, 5.0, 2.0  # DETR matching costs / loss weights
NO_OBJECT_WEIGHT = 0.1  # DETR's eos_coef
ARTIFACT_FORMAT = "org.valcorza.table-transformer-structure.adapter.v1"
ARTIFACT_FORMAT_VERSION = 1
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
POLICY_ZERO_SHOT = "zero-shot checkpoint (restricted heads copied from it, untrained)"
POLICY_FROZEN = "frozen backbone, encoder and decoder + restricted heads"


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
    source: str = "injected-runner"
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    _head: Any = field(default=None, repr=False)  # restricted class head (len(classes) + 1 logits)
    _bbox_head: Any = field(default=None, repr=False)  # a trained copy of the checkpoint's box head
    classes: list[str] = field(default_factory=list)
    adapter: dict[str, Any] | None = None

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

        return cls(
            runner,
            resolved_device,
            source="local-snapshot" if kwargs else "hub",
            _model=model,
            _processor=processor,
        )

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

    # ---- adaptation ----
    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._processor

    def _require_heads(self) -> tuple[Any, Any, list[str]]:
        if self._head is None or self._bbox_head is None or not self.classes:
            raise ValueError("no adapted heads: call adapt() or load an artifact first")
        return self._head, self._bbox_head, list(self.classes)

    def _trainable_names(self, trainable_layers: int) -> list[str]:
        if isinstance(trainable_layers, bool) or not isinstance(trainable_layers, int):
            raise ValueError(f"trainable_layers must be an int in 0..{DECODER_LAYERS}")
        if not 0 <= trainable_layers <= DECODER_LAYERS:
            raise ValueError(f"trainable_layers must be an int in 0..{DECODER_LAYERS}")
        if trainable_layers == 0:
            return []
        model, _ = self._require_model()
        first = DECODER_LAYERS - trainable_layers
        prefixes = tuple(f"model.decoder.layers.{k}." for k in range(first, DECODER_LAYERS))
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def _decoder_features(self, image: Image.Image, *, grad: bool = False) -> Any:
        """The (NUM_QUERIES, D_MODEL) decoder output for one validated image through the processor."""
        import torch

        model, processor = self._require_model()
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        if grad:
            outputs = model.model(pixel_values=inputs["pixel_values"], pixel_mask=inputs.get("pixel_mask"))
        else:
            with torch.no_grad():
                outputs = model.model(
                    pixel_values=inputs["pixel_values"], pixel_mask=inputs.get("pixel_mask")
                )
        hidden = outputs.last_hidden_state[0]
        if tuple(hidden.shape) != (NUM_QUERIES, D_MODEL):
            raise RuntimeError(f"decoder returned {tuple(hidden.shape)}, expected {(NUM_QUERIES, D_MODEL)}")
        return hidden

    @staticmethod
    def _xyxy_pixels(boxes_cxcywh: Any, width: int, height: int) -> list[list[float]]:
        out = []
        for cx, cy, w, h in boxes_cxcywh.tolist():
            out.append(
                [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]
            )
        return out

    @staticmethod
    def _expand(xyxy: list[list[float]], probabilities: Any, classes: Sequence[str]) -> list[dict[str, Any]]:
        """One prediction per (query, class): the ranking average precision consumes."""
        probs = probabilities.tolist()
        out = []
        for box, row in zip(xyxy, probs, strict=True):
            for c, label in enumerate(classes):
                out.append({"box": box, "label": label, "score": float(row[c])})
        return out

    def _checkpoint_class_index(self) -> list[int]:
        model, _ = self._require_model()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        missing = [label for label in ADAPT_LABELS if label not in label2id]
        if missing:
            raise RuntimeError(f"checkpoint lacks the adaptation labels {missing}")
        return [label2id[label] for label in ADAPT_LABELS]

    def evaluate_zero_shot(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Score the untouched checkpoint on validated `{id, image, objects}` records with its own heads:
        every query's softmax probability for each of the four scored labels (headers are neither scored nor
        penalised); per-label AP over all 125 queries per image, the operating point at RECOGNITION_THRESHOLD.
        """
        model, _ = self._require_model()
        import torch

        from .metrics import structure_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        index = self._checkpoint_class_index()
        started = time.perf_counter()
        predictions = []
        for record in checked:
            hidden = self._decoder_features(record["image"])
            with torch.no_grad():
                probabilities = torch.softmax(model.class_labels_classifier(hidden), dim=-1)[:, index]
                boxes = model.bbox_predictor(hidden).sigmoid()
            xyxy = self._xyxy_pixels(boxes, *record["image"].size)
            predictions.append(self._expand(xyxy, probabilities, ADAPT_LABELS))
        metrics = structure_metrics(
            predictions, [r["objects"] for r in checked], labels=ADAPT_LABELS, threshold=RECOGNITION_THRESHOLD
        )
        metrics["policy"] = "zero-shot checkpoint (its own heads, softmax over all seven classes)"
        metrics["adapted"] = False
        metrics["verdict"] = "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample"
        metrics["seconds"] = round(time.perf_counter() - started, 3)
        metrics["model_id"] = MODEL_ID
        metrics["model_revision"] = MODEL_REVISION
        return metrics

    def _query_objects(self, image: Image.Image) -> list[dict[str, Any]]:
        import torch

        head, bbox_head, classes = self._require_heads()
        hidden = self._decoder_features(image)
        with torch.no_grad():
            probabilities = torch.softmax(head(hidden), dim=-1)[:, : len(classes)]
            boxes = bbox_head(hidden).sigmoid()
        return self._expand(self._xyxy_pixels(boxes, *image.size), probabilities, classes)

    def predict_objects(self, records: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
        """Every (query, class) score and pixel box of the adapted heads, per validated record."""
        from .samples import validate_dataset

        self._require_heads()
        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        return [self._query_objects(record["image"]) for record in checked]

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Score the adapted heads on validated labelled tables: per-label AP@0.5 / AP@0.75 / AP and their
        class means, grid agreement and the operating point at RECOGNITION_THRESHOLD, plus the DETR loss the
        selection used."""
        from .metrics import structure_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        predictions = self.predict_objects(checked)
        metrics = structure_metrics(
            predictions, [r["objects"] for r in checked], labels=self.classes, threshold=RECOGNITION_THRESHOLD
        )
        metrics["loss"] = self._dataset_loss(checked)
        metrics["policy"] = self.adapter["policy"] if self.adapter else "unknown"
        metrics["adapted"] = True
        metrics["verdict"] = "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample"
        metrics["seconds"] = round(time.perf_counter() - started, 3)
        metrics["model_id"] = MODEL_ID
        metrics["model_revision"] = MODEL_REVISION
        return metrics

    def recognize_adapted(
        self, image: Image.Image, *, threshold: float = RECOGNITION_THRESHOLD
    ) -> dict[str, Any]:
        """`recognize` with the adapted heads: each query's arg-max label among the adapted classes when its
        probability reaches `threshold`; boxes in pixels."""
        rgb, checked = _check_inputs(image, threshold)
        head, bbox_head, classes = self._require_heads()
        queries = self._query_objects(rgb)
        detections = []
        for start in range(0, len(queries), len(classes)):
            best = max(queries[start : start + len(classes)], key=lambda q: q["score"])
            if best["score"] >= checked:
                detections.append({"box": best["box"], "label": best["label"], "score": best["score"]})
        return {
            "detections": sorted(detections, key=lambda d: -d["score"]),
            "threshold": checked,
            "classes": classes,
            "policy": self.adapter["policy"] if self.adapter else "unknown",
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # ---- the DETR set loss, implemented here (no scipy) ----
    @staticmethod
    def _targets(record: Mapping[str, Any], classes: Sequence[str]) -> tuple[Any, Any]:
        """(class indices, cxcywh boxes normalised to the image) for one validated record."""
        import torch

        width, height = record["image"].size
        labels, rows = [], []
        for obj in record["objects"]:
            x0, y0, x1, y1 = obj["box"]
            labels.append(list(classes).index(obj["label"]))
            rows.append(
                [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height]
            )
        return torch.tensor(labels, dtype=torch.long), torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def _giou(a: Any, b: Any) -> Any:
        """Generalised IoU between (N, 4) and (M, 4) cxcywh boxes -> (N, M)."""
        import torch

        def xyxy(box):
            return torch.stack(
                [
                    box[:, 0] - box[:, 2] / 2,
                    box[:, 1] - box[:, 3] / 2,
                    box[:, 0] + box[:, 2] / 2,
                    box[:, 1] + box[:, 3] / 2,
                ],
                dim=-1,
            )

        a, b = xyxy(a), xyxy(b)
        area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
        area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
        lt = torch.max(a[:, None, :2], b[None, :, :2])
        rb = torch.min(a[:, None, 2:], b[None, :, 2:])
        inter = (rb - lt).clamp_min(0).prod(dim=-1)
        union = area_a[:, None] + area_b[None, :] - inter
        iou = inter / union.clamp_min(1e-9)
        lt_c = torch.min(a[:, None, :2], b[None, :, :2])
        rb_c = torch.max(a[:, None, 2:], b[None, :, 2:])
        enclosing = (rb_c - lt_c).clamp_min(0).prod(dim=-1)
        return iou - (enclosing - union) / enclosing.clamp_min(1e-9)

    @staticmethod
    def hungarian(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
        """Exact minimum-cost assignment of every row of a rectangular cost matrix (rows <= columns) to a
        distinct column: the O(n^2 m) shortest-augmenting-path algorithm with dual potentials (Jonker–
        Volgenant style), in plain Python. Returns (row, column) pairs."""
        n = len(cost)
        m = len(cost[0]) if n else 0
        if n == 0:
            return []
        if n > m:
            raise ValueError(f"hungarian needs rows <= columns, got {n} x {m}")
        inf = math.inf
        u = [0.0] * (n + 1)
        v = [0.0] * (m + 1)
        way = [0] * (m + 1)  # for each column, the column it was reached from on the augmenting path
        assigned = [0] * (m + 1)  # column -> row (1-based), 0 = free
        for i in range(1, n + 1):
            assigned[0] = i
            j0 = 0
            minv = [inf] * (m + 1)
            used = [False] * (m + 1)
            while True:
                used[j0] = True
                i0 = assigned[j0]
                delta, j1 = inf, 0
                row = cost[i0 - 1]
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
                for j in range(m + 1):
                    if used[j]:
                        u[assigned[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if assigned[j0] == 0:
                    break
            while j0:
                j1 = way[j0]
                assigned[j0] = assigned[j1]
                j0 = j1
        return sorted((assigned[j] - 1, j - 1) for j in range(1, m + 1) if assigned[j])

    def _match(
        self, probabilities: Any, boxes: Any, target_classes: Any, target_boxes: Any
    ) -> list[tuple[int, int]]:
        """Minimum-cost one-to-one assignment of reference objects to queries (exact Hungarian)."""
        import torch

        with torch.no_grad():
            cost = -CLASS_COST * probabilities[:, target_classes]
            cost = (
                cost
                + BBOX_COST * torch.cdist(boxes, target_boxes, p=1)
                - GIOU_COST * self._giou(boxes, target_boxes)
            )
            matrix = cost.t().cpu().tolist()  # targets x queries
        return [(q, g) for g, q in self.hungarian(matrix)]

    def _loss(self, logits: Any, boxes: Any, target_classes: Any, target_boxes: Any) -> Any:
        """DETR set loss for one image: weighted cross-entropy over queries (no-object weight 0.1), L1 and
        GIoU on the matched boxes, normalised by the number of reference objects."""
        import torch

        device = logits.device
        target_classes = target_classes.to(device)
        target_boxes = target_boxes.to(device)
        probabilities = torch.softmax(logits, dim=-1)
        pairs = self._match(probabilities.detach(), boxes.detach(), target_classes, target_boxes)
        n_classes = logits.shape[-1] - 1
        assigned = torch.full((NUM_QUERIES,), n_classes, dtype=torch.long, device=device)
        query_index = torch.tensor([q for q, _g in pairs], dtype=torch.long, device=device)
        target_index = torch.tensor([g for _q, g in pairs], dtype=torch.long, device=device)
        assigned[query_index] = target_classes[target_index]
        weights = torch.ones(n_classes + 1, device=device)
        weights[n_classes] = NO_OBJECT_WEIGHT
        loss_ce = torch.nn.functional.cross_entropy(logits, assigned, weight=weights)
        matched_boxes = boxes[query_index]
        matched_targets = target_boxes[target_index]
        loss_bbox = torch.nn.functional.l1_loss(matched_boxes, matched_targets, reduction="sum") / max(
            len(pairs), 1
        )
        loss_giou = (1.0 - torch.diagonal(self._giou(matched_boxes, matched_targets))).sum() / max(
            len(pairs), 1
        )
        return loss_ce + BBOX_COST * loss_bbox + GIOU_COST * loss_giou

    def _dataset_loss(self, checked: Sequence[Mapping[str, Any]]) -> float:
        import torch

        head, bbox_head, classes = self._require_heads()
        total = 0.0
        for record in checked:
            hidden = self._decoder_features(record["image"])
            with torch.no_grad():
                total += float(
                    self._loss(head(hidden), bbox_head(hidden).sigmoid(), *self._targets(record, classes))
                )
        return total / max(len(checked), 1)

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        head_steps: int = 300,
        head_lr: float = 1e-3,
        trainable_layers: int = DEFAULT_TRAINABLE_LAYERS,
        epochs: int = 3,
        lr: float = 1e-4,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Three-policy bounded adaptation to the four scored labels, selected on validation.

        Epoch 0 (the **zero-shot policy**): a restricted `Linear(D_MODEL, 5)` class head whose rows are
        copied from the checkpoint's own `table` / `table column` / `table row` / `table spanning cell` /
        no-object rows, and a copy of the checkpoint's box head, scored on `val` untouched. Stage A (the
        **frozen policy**, epoch 1): both trained on the cached decoder features of the training tables with
        AdamW (`head_lr`, weight decay 1e-4) for `head_steps` full-batch steps under the DETR set loss (exact
        Hungarian matching). Stage B (the **unfrozen policy**, epochs 2.., when `trainable_layers` > 0 and
        `epochs` > 0): the last `trainable_layers` decoder layers trained with both heads end to end, one
        table per step, for `epochs` epochs (AdamW at `lr`, weight decay 0.01, gradient clipping 0.1, seeded
        order, no augmentation); the backbone, the input projection, the encoder, the query embeddings, the
        earlier decoder layers and the checkpoint's own heads stay untouched. Every epoch is scored on `val`
        and the epoch with the **lowest validation DETR loss** is kept (epoch 0 competes, so the selected
        policy can be the checkpoint itself) and its tensors restored; without `val` the final epoch is kept.
        """
        if isinstance(head_steps, bool) or not isinstance(head_steps, int) or not 1 <= head_steps <= 5_000:
            raise ValueError("head_steps must be an int in 1..5000")
        if not isinstance(head_lr, int | float) or not 0.0 < float(head_lr) <= 1e-1:
            raise ValueError("head_lr must be in (0, 0.1]")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 0 <= epochs <= 20:
            raise ValueError("epochs must be an int in 0..20")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-2:
            raise ValueError("lr must be in (0, 1e-2]")
        names = self._trainable_names(trainable_layers)
        model, _ = self._require_model()
        import torch

        from .samples import validate_dataset

        train_records = validate_dataset(train)["records"]
        val_records = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
            if val is not None
            else None
        )
        classes = list(ADAPT_LABELS)
        index = self._checkpoint_class_index()
        started = time.perf_counter()
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Epoch 0: the restricted heads copied from the checkpoint's rows, scored untouched.
        cached = [self._decoder_features(r["image"]) for r in train_records]
        targets = [self._targets(r, classes) for r in train_records]
        head = torch.nn.Linear(D_MODEL, len(classes) + 1).to(self.device)
        with torch.no_grad():
            source = model.class_labels_classifier
            rows = [*index, source.weight.shape[0] - 1]  # the checkpoint's no-object row is its last
            head.weight.copy_(source.weight[rows])
            head.bias.copy_(source.bias[rows])
        bbox_head = copy.deepcopy(model.bbox_predictor).to(self.device)
        for p in bbox_head.parameters():
            p.requires_grad_(True)
        self._head, self._bbox_head, self.classes = head, bbox_head, classes
        self.adapter = {"policy": POLICY_ZERO_SHOT}

        def brief(metrics: Mapping[str, Any] | None) -> dict[str, float] | None:
            if metrics is None:
                return None
            return {
                "n": metrics["n_images"],
                "loss": round(metrics["loss"], 6),
                "map50": round(metrics["map50"], 6),
                "map": round(metrics["map"], 6),
                "grid_exact": round(metrics["grid_exact_at_threshold"], 6),
            }

        def score() -> dict[str, float] | None:
            head.eval()
            bbox_head.eval()
            model.eval()
            return brief(self.evaluate(val_records)) if val_records is not None else None

        def snapshot() -> dict[str, Any]:
            return {
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                "bbox_head": {k: v.detach().clone() for k, v in bbox_head.state_dict().items()},
                "layers": {n: params[n].detach().clone() for n in names},
            }

        params = dict(model.named_parameters())
        history: list[dict[str, Any]] = [
            {"epoch": 0, "stage": POLICY_ZERO_SHOT, "train_loss": None, "val": score()}
        ]
        best_epoch, policy = 0, POLICY_ZERO_SHOT
        best_score = history[0]["val"]["loss"] if history[0]["val"] else math.inf
        best_state = snapshot()

        # Stage A: the restricted heads trained on the cached features (epoch 1).
        head_opt = torch.optim.AdamW(
            [*head.parameters(), *bbox_head.parameters()], lr=float(head_lr), weight_decay=1e-4
        )
        head_loss = math.nan
        for _ in range(head_steps):
            head_opt.zero_grad(set_to_none=True)
            loss = sum(
                self._loss(head(h), bbox_head(h).sigmoid(), *tg)
                for h, tg in zip(cached, targets, strict=True)
            ) / len(cached)
            loss.backward()
            head_opt.step()
            head_loss = float(loss.detach())
        self.adapter = {"policy": POLICY_FROZEN}
        val_metrics = score()
        history.append({"epoch": 1, "stage": POLICY_FROZEN, "train_loss": head_loss, "val": val_metrics})
        if progress is not None:
            progress(history[-1])
        current = val_metrics["loss"] if val_metrics else -1
        if current < best_score:
            best_epoch, best_score, policy = 1, current, POLICY_FROZEN
            best_state = snapshot()
        if names and epochs > 0:
            policy_b = f"unfrozen last {trainable_layers} decoder layers + restricted heads"
            for n in names:
                params[n].requires_grad_(True)
            optimiser = torch.optim.AdamW(
                [*head.parameters(), *bbox_head.parameters(), *(params[n] for n in names)],
                lr=float(lr),
                weight_decay=0.01,
            )
            for epoch in range(2, epochs + 2):
                model.train()
                head.train()
                bbox_head.train()
                order = list(range(len(train_records)))
                rng.shuffle(order)
                losses = []
                for i in order:
                    hidden = self._decoder_features(train_records[i]["image"], grad=True)
                    loss = self._loss(head(hidden), bbox_head(hidden).sigmoid(), *targets[i])
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [*head.parameters(), *bbox_head.parameters(), *(params[n] for n in names)], 0.1
                    )
                    optimiser.step()
                    losses.append(float(loss.detach()))
                self.adapter = {"policy": policy_b}
                val_metrics = score()
                entry = {
                    "epoch": epoch,
                    "stage": policy_b,
                    "train_loss": float(sum(losses) / len(losses)),
                    "val": val_metrics,
                }
                history.append(entry)
                if progress is not None:
                    progress(entry)
                current = val_metrics["loss"] if val_metrics else -epoch  # no val: the last epoch wins
                if current < best_score:
                    best_epoch, best_score, policy = epoch, current, policy_b
                    best_state = snapshot()
        with torch.no_grad():
            head.load_state_dict(best_state["head"])
            bbox_head.load_state_dict(best_state["bbox_head"])
            for n, value in best_state["layers"].items():
                params[n].copy_(value)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in [*head.parameters(), *bbox_head.parameters()]:
            p.requires_grad_(False)
        model.eval()
        head.eval()
        bbox_head.eval()
        self.adapter = {
            "policy": policy,
            "classes": classes,
            "head_steps": head_steps,
            "head_lr": float(head_lr),
            "head_final_loss": head_loss,
            "trainable_layers": trainable_layers,
            "n_trainable_head": sum(p.numel() for p in head.parameters())
            + sum(p.numel() for p in bbox_head.parameters()),
            "n_trainable_layers": sum(params[n].numel() for n in names),
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": (
                "lowest validation DETR loss (epoch 0 = the checkpoint's own rows untrained, epoch 1 = the "
                "restricted heads trained on frozen features, later epochs = the unfreeze)"
                if val is not None
                else "final epoch (no validation split)"
            ),
            "lr": float(lr),
            "n_train": len(train_records),
            "n_val": len(val_records) if val_records is not None else 0,
            "seed": seed,
            "history": history,
            "trainable_names": names if policy not in (POLICY_ZERO_SHOT, POLICY_FROZEN) else [],
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the restricted heads (and any trained decoder-layer tensors) as safetensors with a base
        manifest."""
        if self.adapter is None or self._head is None or self._bbox_head is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter.get("trainable_names", []))
        tensors = {f"head.{k}": v.detach().cpu().contiguous() for k, v in self._head.state_dict().items()}
        tensors.update(
            {f"bbox_head.{k}": v.detach().cpu().contiguous() for k, v in self._bbox_head.state_dict().items()}
        )
        tensors.update(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        )
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHTS_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter.get("history", []),
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest and digest **before** deserialising, rebuild the heads and overlay any
        decoder-layer tensors onto the base."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        entry = manifest["files"][0]
        weights_path = root / entry["path"]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        classes = list(manifest.get("adapter", {}).get("classes") or [])
        if tuple(classes) != ADAPT_LABELS:
            raise ValueError(f"artifact classes {classes} are not the adaptation labels {list(ADAPT_LABELS)}")
        model, _ = self._require_model()
        import torch
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != manifest["tensors"]:
            raise ValueError("artifact tensor names differ from its manifest")
        if (
            tuple(tensors.get("head.weight", torch.empty(0)).shape) != (len(classes) + 1, D_MODEL)
            or "head.bias" not in tensors
        ):
            raise ValueError("artifact class head does not match D_MODEL and the manifest's classes")
        bbox_head = copy.deepcopy(model.bbox_predictor)
        bbox_state = {k[len("bbox_head.") :]: v for k, v in tensors.items() if k.startswith("bbox_head.")}
        if set(bbox_state) != set(bbox_head.state_dict()):
            raise ValueError("artifact box head tensors do not match the checkpoint's box head")
        state = model.state_dict()
        layer_tensors = {k: v for k, v in tensors.items() if not k.startswith(("head.", "bbox_head."))}
        for key, value in layer_tensors.items():
            if key not in state or not key.startswith("model.decoder.layers."):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable decoder-layer tensor of the base"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key}: shape {tuple(value.shape)} != {tuple(state[key].shape)}"
                )
        head = torch.nn.Linear(D_MODEL, len(classes) + 1)
        head.load_state_dict({"weight": tensors["head.weight"].float(), "bias": tensors["head.bias"].float()})
        bbox_head.load_state_dict({k: v.float() for k, v in bbox_state.items()})
        head = head.to(self.device).eval()
        bbox_head = bbox_head.to(self.device).eval()
        for p in [*head.parameters(), *bbox_head.parameters()]:
            p.requires_grad_(False)
        if layer_tensors:
            with torch.no_grad():
                params = dict(model.named_parameters())
                for key, value in layer_tensors.items():
                    params[key].copy_(value.to(params[key].dtype))
            model.eval()
        self._head, self._bbox_head, self.classes = head, bbox_head, classes
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": sorted(layer_tensors),
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> TableTransformerStructurePipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
