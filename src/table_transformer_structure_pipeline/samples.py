"""Structure-labelled table dataset contract for adapting the recogniser: the embedded SciTSR-PD sample,
validation, seeded splitting by paper, BYOD loaders and CSV export.

The default dataset is **real** and public domain: 94 scientific tables from SciTSR-PD (arXiv LaTeX
tables rendered at 150 DPI whose source papers carry a CC0 or public-domain dedication), embedded in
`sample_data.py` with structure boxes derived once from the dataset's text chunks and logical cells (see
that module's docstring for the derivation and the drop rules). Column headers and projected row headers
are not annotated in SciTSR, so the sample carries four of the checkpoint's six labels — `table`,
`table column`, `table row`, `table spanning cell` — and the adaptation contract restricts itself to
them. Several tables come from the same paper, so the sample is split **by paper**, never by table.

A record is ``{id, image, objects}``: a PIL image (or a path to one) and a list of ``{label, box}``
structure objects with ``box`` as ``[x_min, y_min, x_max, y_max]`` pixels and ``label`` in
`ADAPT_LABELS`.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import random
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import ADAPT_LABELS, MAX_DETECTIONS, MAX_IMAGE_SIDE, MIN_IMAGE_SIDE, MODEL_ID
from .sample_data import (
    SAMPLE_DPI,
    SAMPLE_IMAGES_B64,
    SAMPLE_LABELS,
    SAMPLE_LICENSE,
    SAMPLE_RECORDS,
    SAMPLE_SOURCE,
    SOURCE_FILES,
)

CORPUS_NAME = "SciTSR-PD scientific tables"
CORPUS_RELEASE = SAMPLE_SOURCE
CORPUS_LICENSE = SAMPLE_LICENSE
CORPUS_DPI = SAMPLE_DPI
CORPUS_SOURCE_FILES = SOURCE_FILES
if tuple(SAMPLE_LABELS) != tuple(ADAPT_LABELS):
    raise RuntimeError("sample_data labels differ from the pipeline's adaptation labels")
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"test": 10, "validation": 7, "train": 29}  # papers (46 in the sample), drawn in this order
MIN_RECORDS = 8
MAX_RECORDS = 2_000
MAX_OBJECTS = (
    MAX_DETECTIONS  # the decoder emits at most this many boxes, so a table cannot carry more targets
)
MIN_BOX_SIDE = 2.0  # pixels: a row of a dense table at 150 DPI is about 20 px; 2 px refuses degenerate boxes
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_corpus() -> list[dict[str, Any]]:
    """Decode the embedded tables into `{id, image, objects}` records with provenance, verifying each PNG's
    byte size and SHA-256 against `SAMPLE_RECORDS` first."""
    out = []
    for entry in SAMPLE_RECORDS:
        data = base64.b64decode(SAMPLE_IMAGES_B64[entry["file"]])
        if len(data) != entry["png_bytes"] or _sha256_bytes(data) != entry["png_sha256"]:
            raise ValueError(f"{entry['table_id']}: embedded PNG does not match its recorded size / digest")
        image = Image.open(io.BytesIO(data))
        image.load()
        if image.size != (entry["width"], entry["height"]):
            raise ValueError(
                f"{entry['table_id']}: embedded PNG is {image.size}, "
                f"recorded {(entry['width'], entry['height'])}"
            )
        out.append(
            {
                "id": entry["table_id"],
                "image": image.convert("RGB"),
                "objects": [
                    {"label": o["label"], "box": [float(v) for v in o["box"]]} for o in entry["objects"]
                ],
                "paper_id": entry["paper_id"],
                "paper_title": entry["paper_title"],
                "paper_license": entry["paper_license"],
                "scitsr_split": entry["scitsr_split"],
                "grid": [entry["n_rows"], entry["n_cols"]],
            }
        )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded draw of whole papers into test / validation / train (in that order): `sizes` counts papers."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_paper.setdefault(_split_unit(record), []).append(dict(record))
    papers = sorted(by_paper)
    rng.shuffle(papers)
    needed = sum(sizes.values())
    if len(papers) < needed:
        raise ValueError(f"only {len(papers)} papers available, need {needed}")
    out: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name, count in sizes.items():
        part = [r for paper in papers[cursor : cursor + count] for r in by_paper[paper]]
        cursor += count
        rng.shuffle(part)
        out[name] = [{**r, "id": f"{name}-{i:03d}", "source_id": r["id"]} for i, r in enumerate(part)]
    return out


def load_sample_dataset(
    *, seed: int = SAMPLE_SEED, sizes: Mapping[str, int] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the embedded corpus."""
    return build_sample_dataset(load_corpus(), seed=seed, sizes=sizes)


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/image/objects")
    for key in ("id", "image", "objects"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, image, objects = record["id"], record["image"], record["objects"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label_name}: id must match {_ID_RE.pattern}")
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{label_name}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label_name}: image must be a PIL.Image.Image or a file path")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(
            f"{label_name}: image side outside {MIN_IMAGE_SIDE}..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: "
            f"{image.size}"
        )
    if (
        isinstance(objects, str | bytes)
        or not isinstance(objects, Sequence)
        or not 1 <= len(objects) <= MAX_OBJECTS
    ):
        raise ValueError(f"{label_name}: objects must be a list of 1..{MAX_OBJECTS} {{label, box}} entries")
    checked = []
    for b, obj in enumerate(objects):
        if not isinstance(obj, Mapping) or "label" not in obj or "box" not in obj:
            raise ValueError(f"{label_name}: objects[{b}] must be a mapping with label and box")
        label, box = obj["label"], obj["box"]
        if label not in ADAPT_LABELS:
            raise ValueError(f"{label_name}: objects[{b}] label {label!r} is not one of {ADAPT_LABELS}")
        if isinstance(box, str | bytes) or not isinstance(box, Sequence) or len(box) != 4:
            raise ValueError(f"{label_name}: objects[{b}] box must have four values")
        x0, y0, x1, y1 = (float(v) for v in box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"{label_name}: objects[{b}] box {[x0, y0, x1, y1]} must lie inside the {image.size} image "
                "with x0 < x1 and y0 < y1"
            )
        if x1 - x0 < MIN_BOX_SIDE or y1 - y0 < MIN_BOX_SIDE:
            raise ValueError(f"{label_name}: objects[{b}] box is smaller than {MIN_BOX_SIDE} px on a side")
        checked.append({"label": label, "box": [x0, y0, x1, y1]})
    labels = [o["label"] for o in checked]
    if labels.count("table") > 1:
        raise ValueError(f"{label_name}: a record carries at most one `table` box")
    if "table row" not in labels or "table column" not in labels:
        raise ValueError(f"{label_name}: a record needs at least one `table row` and one `table column`")
    item = {"id": rid, "image": image.convert("RGB"), "objects": checked}
    for key in ("source_id", "paper_id", "paper_title", "paper_license", "scitsr_split", "group", "grid"):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a structure-labelled table dataset; raises ValueError before any model
    import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, objects} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = [_check_record(record, index) for index, record in enumerate(records)]
    ids = [r["id"] for r in checked]
    if len(set(ids)) != len(ids):
        duplicate = next(i for i in ids if ids.count(i) > 1)
        raise ValueError(f"duplicate id {duplicate!r}")
    sides = [max(r["image"].size) for r in checked]
    n_objects = [len(r["objects"]) for r in checked]
    per_label = {label: 0 for label in ADAPT_LABELS}
    for r in checked:
        for o in r["objects"]:
            per_label[o["label"]] += 1
    return {
        "records": checked,
        "n_records": len(checked),
        "n_objects": sum(n_objects),
        "objects_per_label": per_label,
        "objects_per_image": {"min": min(n_objects), "max": max(n_objects)},
        "image_side": {"min": min(sides), "max": max(sides)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size + bytes), so a re-encoded copy of the same table matches."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [
            r["id"],
            image_digest(r["image"]),
            [[o["label"], [round(float(v), 2) for v in o["box"]]] for o in r["objects"]],
        ]
        for r in records
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _split_unit(record: Mapping[str, Any]) -> str:
    return str(record.get("group") or record.get("paper_id") or record.get("source_id") or record["id"])


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no table (by decoded-pixel digest) and no paper appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    units: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
            unit = _split_unit(record)
            if unit in units and units[unit] != name:
                raise ValueError(f"paper {unit!r} has tables in both {units[unit]} and {name}")
            units[unit] = name
    return {name: len(records) for name, records in splits.items()}


def split_summary(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Tables, objects per label and papers per split (an observation of what the split unit was)."""
    out = {}
    for name, records in splits.items():
        per_label = {label: 0 for label in ADAPT_LABELS}
        for r in records:
            for o in r["objects"]:
                per_label[o["label"]] += 1
        out[name] = {
            "tables": len(records),
            "objects": per_label,
            "papers": len({_split_unit(r) for r in records}),
        }
    return out


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test by split unit (`group` / `paper_id`, else
    the table itself) after de-duplicating tables."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            by_unit.setdefault(_split_unit(record), []).append(record)
    rng = random.Random(seed)
    units = sorted(by_unit)
    rng.shuffle(units)
    n_test = max(1, round(len(units) * test_fraction))
    n_val = round(len(units) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for name, chosen in (
        ("test", units[:n_test]),
        ("validation", units[n_test : n_test + n_val]),
        ("train", units[n_test + n_val :]),
    ):
        for unit in chosen:
            splits[name].extend(by_unit[unit])
    for part in splits.values():
        rng.shuffle(part)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path, *, require_group: bool = True) -> list[dict[str, Any]]:
    """Read `{id, image, objects}` records from a directory or a zip holding `structure.csv` (columns `id`,
    `file`, `group`, `label`, `x_min`, `y_min`, `x_max`, `y_max`; one row per structure box, pixel
    coordinates) beside the image files; images are decoded, never extracted to disk. `group` (the paper,
    document or source the table comes from) must be non-empty on every row unless `require_group=False`,
    in which case the split falls back to one unit per table and the group-disjoint guarantee is gone. Rows
    of one `id` must agree on `file` and `group`."""
    source = Path(path)
    if source.is_dir():
        table = (source / "structure.csv").read_text(encoding="utf-8")
        base_dir = source.resolve()

        def loader(name: str) -> Image.Image:
            target = (source / name).resolve()
            if base_dir not in target.parents:
                raise ValueError(f"BYOD file reference {name!r} leaves the dataset directory")
            return Image.open(target)

    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        names = [n for n in archive.namelist() if not n.endswith("/")]
        basenames = [Path(n).name for n in names]
        if len(set(basenames)) != len(basenames):
            duplicate = next(b for b in basenames if basenames.count(b) > 1)
            raise ValueError(f"BYOD zip holds more than one member named {duplicate!r}")
        members = dict(zip(basenames, names, strict=True))
        if "structure.csv" not in members:
            raise ValueError("BYOD zip must contain structure.csv")
        table = archive.read(members["structure.csv"]).decode("utf-8")
        loader = lambda name: Image.open(io.BytesIO(archive.read(members[name])))  # noqa: E731
    else:
        raise ValueError(
            "BYOD datasets must be a directory or a .zip holding structure.csv and the image files"
        )
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "file", "label", "x_min", "y_min", "x_max", "y_max"} - set(
        rows[0].keys() if rows else set()
    )
    if missing:
        raise ValueError(f"structure.csv is missing columns {sorted(missing)}")
    grouped: dict[str, dict[str, Any]] = {}
    origin: dict[str, tuple[str, str]] = {}
    for row in rows:
        group = (row.get("group") or "").strip()
        if require_group and not group:
            raise ValueError(
                f"structure.csv row for id {row['id']!r} has no `group`; every row needs the paper, document "
                "or source the table comes from so the split stays group-disjoint (pass require_group=False "
                "to split by table instead, without that guarantee)"
            )
        item = grouped.get(row["id"])
        if item is None:
            image = loader(row["file"])
            image.load()
            item = {"id": row["id"], "image": image.convert("RGB"), "objects": []}
            if group:
                item["group"] = group
            grouped[row["id"]] = item
            origin[row["id"]] = (row["file"], group)
        elif origin[row["id"]] != (row["file"], group):
            raise ValueError(
                f"structure.csv rows for id {row['id']!r} disagree on file or group "
                f"({origin[row['id']]} vs {(row['file'], group)})"
            )
        item["objects"].append(
            {
                "label": row["label"],
                "box": [float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"])],
            }
        )
    return list(grouped.values())


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the structure table of a split (one row per box, provenance) in the shape BYOD expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "file", "label", "x_min", "y_min", "x_max", "y_max", "group", "paper_id"],
        )
        writer.writeheader()
        for record in records:
            for obj in record["objects"]:
                box = obj["box"]
                writer.writerow(
                    {
                        "id": record["id"],
                        "file": f"{record.get('source_id') or record['id']}.png",
                        "label": obj["label"],
                        "x_min": round(box[0], 2),
                        "y_min": round(box[1], 2),
                        "x_max": round(box[2], 2),
                        "y_max": round(box[3], 2),
                        "group": _split_unit(record),
                        "paper_id": record.get("paper_id", ""),
                    }
                )
    return out
