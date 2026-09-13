"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "table_transformer_structure_pipeline",
    "repo_name": "table-transformer-structure-pipeline",
    "stem": "table_transformer_structure",
    "notebook_name": "table_transformer_structure_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "TableTransformerStructurePipeline",
    "weights_key": "table-transformer-structure-v1.1-all",
    "runtime_imports": ["torch", "transformers"],
    "title": "Table Transformer v1.1-all — DIMER table structure recognition tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/table-transformer-structure-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/table-transformer-structure-pipeline/blob/main/tutorials/table_transformer_structure_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2Ftable--transformer--structure--recognition--v1.1--all-ffcc4d?style=flat",
            "https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-microsoft%2Ftable--transformer-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/microsoft/table-transformer",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2303.00716-b31b1b.svg", "https://arxiv.org/abs/2303.00716"),
    ],
    "capability": "table structure recognition on a table-crop image (boxes labelled `table`, `table column`, `table row`, `table column header`, `table projected row header`, `table spanning cell`) using the pinned `microsoft/table-transformer-structure-recognition-v1.1-all` weights",
    "intro": (
        "At inference the DETR-style model reads one table image resized so its longest edge is 800 px, runs an in-library "
        "ResNet-18 backbone and a 6-layer encoder–decoder, and emits exactly 125 query proposals, each a box and a softmax "
        "over the six structure classes and *no object*; the processor keeps the queries whose class score reaches the "
        "threshold and maps their boxes back to input pixels. The rows and columns are the load-bearing output: "
        "intersecting them yields the cell grid, and the header, projected-row-header and spanning-cell boxes refine it. "
        "**No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens in "
        "this notebook — the upstream checkpoint supplies the weights and image-processor configuration, and the carried "
        "module adds snapshot verification, the input contract, a fixed output contract and the `box_iou`, "
        "`structure_summary`, `validate_inputs` and `evaluation_report` helpers. The default sample is a table rendered "
        "in code and cropped with the upstream script's 10 px padding, whose drawn row/column boxes serve as references; "
        "its `box_iou` values are demonstration (plumbing) evidence for one table, not a recognition benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, render a synthetic table crop with reference structure boxes (or upload your "
        "own table image) and validate it into an input manifest, run the supported task, read the six structure classes, "
        "the class scores and the caller-owned threshold correctly, derive the implied cell grid with `structure_summary`, "
        "exercise an optional BYOD path, produce an evaluation report that is `sample-sanity` with per-label `box_iou` only "
        "when reference boxes exist and `not-measurable` otherwise, and export machine-readable structure objects plus an "
        "annotated image and provenance."
    ),
    "exclusions": (
        "table *detection* on a full page (the sibling `table-transformer-detection-pipeline` finds the crop this notebook "
        "expects), OCR or cell text extraction, assembling the final cell grid or HTML/CSV export (the rows and columns are "
        "returned; intersecting them is left to the caller), precision/recall or GriTS evaluation (which needs a labelled "
        "table set), or any training. The model was trained on PubTables-1M and FinTabNet.c renders; scans, photographs "
        "and non-Latin layouts are outside what this notebook measures. A non-table image still yields structure objects."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 4.8 s to load and 0.12 s per `recognize` on the 730×350 synthetic crop in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 115 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union measures; that a table's cell grid is the intersection of its row and column boxes.",
        "- **Data:** the default sample is a deterministic 730×350 crop of an 8×5 ruled table rendered in code with Pillow's bundled font, with 10 px of white page around it (the upstream inference script's crop padding), so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar) showing a **single table**, ideally cropped with a little page margin, any colour mode, sides between 16 and 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Render the synthetic table crop or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference boxes: an 8×5 ruled table (a header row and "
                "seven data rows of short tokens, rendered with Pillow's bundled font) is drawn on a white page at "
                "`[70, 330, 780, 660]` and cropped with `UPSTREAM_CROP_PADDING` (10) px of page around it, the way the upstream "
                "inference script crops each detected table before structure recognition; the crop is 730×350. This is the "
                "same crop the repository's smoke run used. The drawn table, row, column and column-header boxes (in crop "
                "coordinates) are the references for the per-label `box_iou` sanity check later; they are not a labelled "
                "dataset, so nothing here is a precision/recall measurement. The image digest is printed for the record. BYOD "
                "is optional and disabled by default; when enabled, upload one table image — no reference boxes exist for it, "
                "so the evaluation report will be `not-measurable`.\n\n"
                "The recognition threshold is a **caller-owned request parameter**, not a pipeline constant: a query survives "
                "when its softmax score for one of the six structure classes reaches it. The package default "
                "(`RECOGNITION_THRESHOLD = 0.5`) is the value the upstream repository's inference script applies to every "
                "structure class, not a calibration; it is exposed here as a form parameter and passed explicitly on every "
                "call. Nothing is validated in this cell — the next section hands the image and the threshold to the "
                "pipeline's own validation stage, which is the only checker. Look for a dictionary naming the sample kind, "
                "the crop size and digest, the threshold, and the number of reference boxes per label."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "threshold = 0.5  # @param {{type:\"number\"}}\n\n\n"
                "def synthetic_table(rows=8, cols=5, x0=70, y0=330, x1=780, y1=660, pad=UPSTREAM_CROP_PADDING):\n"
                "    \"\"\"An 8x5 ruled table rendered with Pillow's bundled font, cropped with `pad` px of page around it.\"\"\"\n"
                "    page = Image.new('RGB', (850, 1100), 'white')\n"
                "    d = ImageDraw.Draw(page)\n"
                "    body = ImageFont.load_default(size=15)\n"
                "    d.rectangle([x0, y0, x1, y1], outline='black', width=2)\n"
                "    rh, cw = (y1 - y0) / rows, (x1 - x0) / cols\n"
                "    d.line([(x0, y0 + rh), (x1, y0 + rh)], fill='black', width=2)\n"
                "    for r in range(2, rows):\n"
                "        d.line([(x0, y0 + rh * r), (x1, y0 + rh * r)], fill=(120, 120, 120), width=1)\n"
                "    for c in range(1, cols):\n"
                "        d.line([(x0 + cw * c, y0), (x0 + cw * c, y1)], fill=(120, 120, 120), width=1)\n"
                "    for r in range(rows):\n"
                "        for c in range(cols):\n"
                "            token = ('Region' if c == 0 else f'Q{{c}}') if r == 0 else (f'North {{r}}' if c == 0 else f'{{(r * 7 + c * 13) % 97 + 1}},{{(r * 31 + c) % 900 + 100:03d}}')\n"
                "            d.text((x0 + cw * c + 8, y0 + rh * r + rh / 2 - 8), token, fill='black', font=body)\n"
                "    crop = page.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))\n"
                "    ox, oy = x0 - pad, y0 - pad\n"
                "    refs = {{\n"
                "        'table': [[x0 - ox, y0 - oy, x1 - ox, y1 - oy]],\n"
                "        'table row': [[x0 - ox, y0 + rh * r - oy, x1 - ox, y0 + rh * (r + 1) - oy] for r in range(rows)],\n"
                "        'table column': [[x0 + cw * c - ox, y0 - oy, x0 + cw * (c + 1) - ox, y1 - oy] for c in range(cols)],\n"
                "        'table column header': [[x0 - ox, y0 - oy, x1 - ox, y0 + rh - oy]],\n"
                "    }}\n"
                "    return crop, {{label: [[float(v) for v in box] for box in boxes] for label, boxes in refs.items()}}\n\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    drawn_boxes = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic table: no randomness, so no seed is needed and the digest is stable per Pillow build.\n"
                "    image, drawn_boxes = synthetic_table()\n"
                "    image_name = 'synthetic_table_crop_730x350.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'threshold': threshold, 'reference_boxes': None if drawn_boxes is None else {{k: len(v) for k, v in drawn_boxes.items()}}}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `recognize` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px and a threshold in `[0, 1]` — and returns an "
                "**input manifest** naming the schema (including the six labels and the 125-query ceiling on structure objects), "
                "the input's observed mode and size, the threshold, and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a threshold "
                "outside `[0, 1]` and records the pipeline's own error message as a finding. Inside the pipeline the image is "
                "converted to RGB and resized by the processor so its longest edge is 800 px; boxes are mapped back to input "
                "pixels, and nothing else is dropped or altered. The pipeline cannot tell whether the image is a single table: "
                "that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_DETECTIONS': MAX_DETECTIONS, 'LABELS': list(LABELS), 'RECOGNITION_THRESHOLD': RECOGNITION_THRESHOLD, 'UPSTREAM_CROP_PADDING': UPSTREAM_CROP_PADDING}}}})\n"
                "input_manifest = validate_inputs(image, threshold=threshold, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, threshold=1.5)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'out-of-range-threshold-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Recognise the structure and read the scores correctly\n\n"
                "`recognize` returns a dict with `detections` — a list of `{{box, label, score}}` **ordered by descending score**, "
                "`box` in xyxy pixel coordinates of the input, `label` one of the six structure classes — plus the threshold "
                "used, `width`, `height` and the model identity. At most 125 objects can ever be returned (the DETR decoder has "
                "125 queries). Each `score` is the query's **softmax class probability under the model's own head, not a "
                "calibrated estimate for your tables**. The threshold you passed is the only decision rule; the pipeline ships "
                "0.5 as a default (the upstream inference script's per-class value), not as a calibration, and the caller owns "
                "it per deployment. `structure_summary` counts the objects per label and reports the cell grid the rows and "
                "columns imply (`n_rows` × `n_columns`); building the actual cells by intersecting the boxes is the caller's next "
                "step. Inference is deterministic on a fixed device and dtype (no sampling, `torch.inference_mode`); CUDA kernel "
                "selection can move scores in the third or fourth decimal place. As recorded in the model card, the repository's "
                "CPU smoke on this same crop at threshold 0.5 returned exactly 1 `table`, 5 `table column`, 8 `table row` and "
                "1 `table column header` (15 objects, every score 1.00) — a count that matches the rendered grid — and the same "
                "15 at 0.9; that is one observation on one synthetic table, not a calibration point."
            ),
            "code": (
                "result = pipe.recognize(image, threshold=threshold)\n"
                "summary = structure_summary(result)\n"
                "print({{'n_detections': len(result['detections']), 'threshold': result['threshold'], 'device': pipe.device, 'summary': summary}})\n"
                "for rank, det in enumerate(result['detections'], start=1):\n"
                "    print(f\"{{rank:>3}}. score {{det['score']:.4f}}  label {{det['label']!r:28}}  box {{[round(v, 1) for v in det['box']]}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No recognition "
                "metric is reported by default: per-class precision/recall or the cell-level GriTS scores the upstream paper "
                "uses need a labelled table set, and this repository ships none. The repository's only metric helper is "
                "`box_iou(a, b)`; when reference boxes are supplied, keyed by label, the report carries one `box_iou` entry per "
                "reference — matched only against detections **of the same label** — together with the reference and detected "
                "counts for that label, with the verdict `sample-sanity`. On the synthetic path those references are rows and "
                "columns **you rendered yourself**, so a high IoU proves only that the input contract, forward pass and "
                "coordinate mapping round-trip. On BYOD no reference exists, the verdict is `not-measurable`, and the report "
                "states what would make the task measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, drawn_boxes, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k != 'metrics'}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['reference']:28}} iou {{metric['value']:.3f}}  (references {{metric['n_reference']}}, detected {{metric['n_detected']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference boxes exist for this input, so box_iou is not computed; inspect the annotated PNG instead.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (score-ordered structure objects with boxes and labels, the "
                "threshold), the structure summary, the evaluation report, the input manifest, the sample identity, digest and "
                "reference boxes, the notebook's source (repository, revision, embedded module digest, generator), the model "
                "identifier, the immutable model revision, the model licence, and the runtime identity (Python, `torch`, "
                "`transformers`, device). The objects are also written as CSV with explicit `image`, `rank`, `label`, `score`, "
                "`x0`, `y0`, `x1`, `y1` columns so score ordering survives downstream use, and an annotated PNG draws rows in "
                "blue, columns in green and everything else in red for visual inspection (a supplement to, not a replacement "
                "for, the machine-readable files). No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "COLOURS = {{'table row': (40, 90, 220), 'table column': (0, 160, 0)}}\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for det in result['detections']:\n"
                "    draw.rectangle(det['box'], outline=COLOURS.get(det['label'], (200, 30, 30)), width=2)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'structure_summary': summary,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'reference_boxes': drawn_boxes}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_objects.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1'])\n"
                "    for rank, det in enumerate(result['detections'], start=1):\n"
                "        writer.writerow([image_name, rank, det['label'], f\"{{det['score']:.6f}}\", *[f\"{{v:.2f}}\" for v in det['box']]])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The boxes are the structure objects the model sees in an image it assumes to be one table; the six labels are the "
        "model's vocabulary, the softmax score is not calibrated for your documents, and the threshold is a request parameter "
        "you own (the default is the upstream script's value, not a tuned operating point). On the synthetic crop the per-label "
        "`box_iou` values in the evaluation report compare objects to rows and columns you rendered yourself and the verdict "
        "is `sample-sanity`, which proves only that the input contract, forward pass and coordinate mapping work; they say "
        "nothing about scans, borderless or merged-cell tables, rotated tables, multi-line cells, or non-Latin documents, and "
        "a BYOD result is a single-table observation with the verdict `not-measurable`. **The model emits structure objects "
        "for any image**: the repository's smoke run fed it a 4096×4096 blank image and got 23 objects above 0.5, so a crop "
        "that is not a table produces confident nonsense rather than an empty result — detect first, then recognise. The "
        "pipeline provides no page-level detection, no OCR, no cell assembly, no GriTS evaluation and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** raise `threshold` to 0.9 and check the object count stays at 15 (the smoke run found it did); "
        "merge two header cells in `synthetic_table` and look for a `table spanning cell`; enable `USE_BYOD` with a real table "
        "crop, hand-label its rows and columns and pass them to `evaluation_report` to see the verdict switch to `sample-sanity`; "
        "then intersect the row and column boxes to build the cell grid and compare it with the `n_cells_implied` count.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/table-transformer-structure-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/table-transformer-structure-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/table-transformer-structure-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/microsoft/table-transformer\n"
        "- Aligning benchmark datasets for table structure recognition (Smock, Pesala, Abraham, 2023): https://arxiv.org/abs/2303.00716\n"
        "- PubTables-1M (Smock, Pesala, Abraham, 2021): https://arxiv.org/abs/2110.00061"
    ),
}
