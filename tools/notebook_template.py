"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier), E2E profile.

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline, metrics,
sample-data and dataset modules, and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package. The notebook: the pinned Table Transformer
structure snapshot is digest-verified and loaded, the embedded public-domain SciTSR-PD table corpus (94
scientific tables with structure boxes derived from the dataset's cells) is decoded, validated and split by
paper, the inference contract is exercised on a real table with reference boxes (a `sample-sanity` report per
label), a grid prior and the untouched checkpoint are scored on the test split, restricted heads copied from
the checkpoint are scored untrained (the copied-head zero-shot policy — a restricted five-way softmax over the
checkpoint's rows, not the untouched checkpoint), trained on the frozen decoder features (the
frozen policy) and then with the last decoder layers (the unfrozen policy), the policy is selected on
validation, the held-out split is scored by per-label AP and grid agreement, and the adapter is exported and
reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "table_transformer_structure_pipeline",
    "repo_name": "table-transformer-structure-pipeline",
    "stem": "table_transformer_structure",
    "notebook_name": "table_transformer_structure_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned Table Transformer structure snapshot (safetensors, 115 MB), decodes the 94 public-domain SciTSR-PD tables "
        "embedded in the carried `sample_data` module (0.9 MB of PNG, no download, no credential), validates them and draws "
        "50 / 23 / 21 training, validation and test tables by a seeded split of whole papers, runs one test table through the "
        "inference contract with an input manifest, a rejection probe and a per-label `sample-sanity` evaluation report "
        "against its reference boxes, scores a grid prior and the untouched checkpoint on the test split (the **zero-shot "
        "row**), copies the checkpoint's own rows for the four scored labels into restricted heads and scores them untrained "
        "(the **copied-head zero-shot policy**, epoch 0 — a restricted five-way softmax over the checkpoint's own rows, not the untouched seven-way checkpoint), trains them on the frozen DETR decoder features (the **frozen policy**) and then "
        "the last two decoder layers with them (the **unfrozen policy**), selects among the three by validation DETR loss, "
        "scores the held-out split by per-label AP@0.5 / AP@0.75 / AP, their class means and grid agreement with the selected "
        "model, renders structure before and after, exports the trained tensors as safetensors with a manifest, and reloads "
        "that artifact into a fresh pipeline to verify parity. The default path needs no repository clone, no DIMER worker or "
        "service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path "
        "takes about 3 minutes of model time after the install and the checkpoint download; a CUDA runtime is used "
        "automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "structure-labelled table crops as a `.zip` holding `structure.csv` (columns `id`, `file`, `label`, `x_min`, `y_min`, "
        "`x_max`, `y_max`, `group`; one row per structure box, pixel coordinates, labels among `table`, `table column`, "
        "`table row`, `table spanning cell`; `group` — the paper, document or source — must be non-empty on every row, and "
        "rows of one `id` must agree on `file` and `group`, or the loader refuses the set) beside the image files — images are decoded from the archive, "
        "never extracted to disk. They pass through the same validation, seeded group-disjoint split, prior, zero-shot "
        "scoring, three-policy ladder and selection, held-out evaluation, structure rendering, artifact export and "
        "reload-parity cells as the SciTSR-PD sample. The expected schema and the ceilings are stated in the Prerequisites "
        "and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "TableTransformerStructurePipeline",
    "weights_key": "table-transformer-structure-v1.1-all",
    "modules": ["pipeline.py", "metrics.py", "sample_data.py", "samples.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "Table Transformer v1.1-all — DIMER E2E supervised adaptation tutorial: table structure on public-domain scientific tables, zero-shot vs restricted heads vs bounded decoder unfreeze (standalone)",
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
    "capability": "table structure recognition on a table-crop image (boxes labelled `table`, `table column`, `table row`, `table column header`, `table projected row header`, `table spanning cell`) and bounded supervised adaptation of four of those labels to a new table corpus and box convention — restricted heads copied from the checkpoint, trained on the frozen DETR decoder features, with an optional unfreeze of the last decoder layers — measured by held-out per-label AP@0.5 / AP@0.75 / AP and grid agreement, using the pinned `microsoft/table-transformer-structure-recognition-v1.1-all` weights",
    "intro": (
        "At inference the DETR-style model reads one table image resized so its longest edge is 800 px, runs an in-library "
        "ResNet-18 backbone and a 6-layer encoder–decoder, and emits exactly 125 query proposals, each a box and a softmax "
        "over the six structure classes and *no object*; the processor keeps the queries whose class score reaches the "
        "threshold and maps their boxes back to input pixels. The carried pipeline module adds snapshot verification, the "
        "input contract, a fixed output contract and the `box_iou`, `structure_summary`, `validate_inputs` and "
        "`evaluation_report` helpers; `evaluation_report` becomes `sample-sanity` only when a caller supplies reference "
        "boxes per label — which this notebook, unlike its inference-only predecessor, does, on a real table.\n\n"
        "What this notebook adds to inference is **supervised adaptation of the structure vocabulary to a new table corpus "
        "and box convention, under an explicit copied-head zero-shot / frozen / unfrozen policy ladder**. The dataset is real and "
        "public domain: 94 scientific tables from SciTSR-PD (arXiv LaTeX tables rendered at 150 DPI whose source papers "
        "carry a CC0 or public-domain dedication), embedded in the carried `sample_data` module with structure boxes "
        "**derived once** from the dataset's text chunks and logical cells — rows and columns tiling an ink-bounded table "
        "box at the mid-gaps, spanning cells over their grid area — so the boxes are a stated convention, not hand "
        "annotation, and SciTSR annotates no column or row headers, so the contract scores four of the six labels. Several "
        "tables come from the same paper, so the sample is split by **paper**, never by table. The carried `metrics.py` "
        "scores a prediction set by **per-label AP@0.5 / AP@0.75 / AP** (the COCO convention, every (query, class) pair of "
        "every image ranked by score), their class means, the recall and precision per label at the pipeline's operating "
        "threshold and **grid agreement** (how often the surviving row and column counts both match the reference); a "
        "**grid prior** (the training split's mean row and column counts laid out uniformly) and the **untouched checkpoint** "
        "frame the numbers. The **copied-head zero-shot policy** copies the checkpoint's own rows for the four labels into restricted "
        "heads and scores them untrained; the **frozen policy** trains those heads on the frozen decoder features under the "
        "DETR set loss (exact Hungarian matching, cross-entropy with a 0.1 no-object weight, L1 and GIoU — implemented in the "
        "carried module, no external matcher); the **unfrozen policy** continues by training the last decoder layers with "
        "them end to end, and the epoch with the lowest validation loss — which may be the untrained copied heads — is kept. The copied heads run a five-way softmax over the checkpoint's rows, so they are a restricted copy and not the untouched seven-way checkpoint: that checkpoint is scored beside the ladder by `evaluate_zero_shot` and is never selected by it, and selecting epoch 0 still exports the restricted heads. "
        "The checkpoint's own heads and `recognize` are never trained or exported. This corpus is close to the PubTables-1M "
        "renders the model was trained on, so the adaptation question is not whether the model can learn the task but "
        "whether adapting to a new corpus and a new box convention buys anything the checkpoint does not already give — "
        "and what it costs. Nothing here is a quality claim about your tables: it is one seeded split of one small corpus."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics, sample-data and dataset modules guarantee; "
        "stage and digest-verify the immutable upstream snapshot; decode an embedded public-domain structure-labelled "
        "corpus, understand how its boxes were derived, and validate and split it by paper without leakage; run a real "
        "table through the public recognition API and read a per-label `sample-sanity` report against reference boxes; read "
        "per-label AP@0.5 / AP@0.75 / AP, class means and grid agreement beside a grid prior and the zero-shot checkpoint "
        "and understand why a score threshold is an operating point, not part of AP; train restricted heads on frozen "
        "features and a bounded decoder unfreeze with explicit hyperparameters and validation-based selection among three "
        "policies, one of which is the checkpoint's own rows under a restricted softmax; evaluate on an independent paper-disjoint test split; compare "
        "structure before and after; and export a safetensors adapter (heads plus any trained decoder layers) that reloads "
        "against the pinned base with verified parity."
    ),
    "exclusions": (
        "table *detection* on a full page (the sibling `table-transformer-detection-pipeline` finds the crop this notebook "
        "expects), OCR or cell text extraction, assembling the final cell grid or HTML/CSV export, GriTS or any cell-level "
        "metric, column-header or projected-row-header adaptation (SciTSR does not annotate them), backbone or encoder "
        "training, data augmentation, any training of the checkpoint's own heads, any PubTables-1M or FinTabNet accuracy "
        "claim, and any claim that 94 arXiv tables stand in for your documents. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; float32 on both. The build record measured about 0.1 s per table to run the decoder on CPU (2 s for the 21-table zero-shot pass), about 45 s for the restricted heads including feature extraction and two validation passes, and about 15 s per unfreeze epoch over 50 tables plus a 23-table validation pass. The pinned `torch==2.14.0` install and the 115 MB checkpoint are the large downloads of the run; the tables travel inside the notebook.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union and average precision measure and why AP does not depend on a score threshold; what a Hungarian (one-to-one) matching between predictions and references is; what validation-based selection among policies means; that a table's cell grid is the intersection of its row and column boxes.",
        "- **Data contract:** records are `{{id, image, objects}}` — a PIL image (or a path to one) with sides 16..4,096 px and a list of 1..125 `{{label, box}}` structure objects with `box` = `[x_min, y_min, x_max, y_max]` pixels inside the image (sides of at least 2 px), `label` among `table` (at most one), `table column`, `table row` (at least one each) and `table spanning cell`, ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a training set needs 8..2,000 records; tables are de-duplicated by decoded-pixel digest and split by `group` / `paper_id` so one paper never straddles splits. BYOD accepts a `.zip` (or a directory) holding `structure.csv` and the image files, and requires a non-empty `group` on every row — the notebook's automatic split is group-disjoint only because the loader refuses ungrouped rows (`load_byod_dataset(..., require_group=False)` is the explicit opt-out, without that guarantee).",
        "- **Validation is structural, not semantic:** nothing checks that a box is a row or a column — a mislabelled set is trained on without complaint; the four labels are the checkpoint's own, so a BYOD set must use exactly those strings.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing and downloads nothing beyond the Hub snapshot.",
        "- **External access (data):** none beyond the Hub. The 94 tables are embedded in the carried `sample_data` module as base64 PNGs (each verified against its recorded byte size and SHA-256 before it is decoded); they come from `bevaya/SciTSR-pd` on the Hugging Face Hub (commit `dae336ef`, two parquet files pinned by SHA-256 in `SOURCE_FILES`), whose source papers carry a CC0 or public-domain dedication, credited in the References.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Embedded corpus, validation and paper-level split\n\n"
                "`load_corpus` decodes the 94 embedded PNGs after checking each against its recorded byte size and SHA-256, and "
                "turns each into a `{{id, image, objects}}` record with its derived structure boxes and provenance (paper id, "
                "title, licence, SciTSR split, grid size). The boxes were derived once by the build script recorded in the "
                "repository: the dataset's text chunks were placed on the trimmed image by an ink-coverage offset search, "
                "matched to the logical cells by text, and turned into a PubTables-style structure — an ink-bounded `table` "
                "box, `table row` / `table column` boxes tiling it at the mid-gaps, and a `table spanning cell` over the grid "
                "area of every multi-row or multi-column cell; 14 of the 108 SciTSR-PD tables were dropped by stated rules "
                "(unmatchable cells, overlapping rows or columns, a row or column without a text cell). `build_sample_dataset` "
                "draws 10 / 7 / 29 whole **papers** by a seeded shuffle into 21 / 23 / 50 test, validation and training "
                "tables; `validate_dataset` then checks every record against the contract, `check_split_disjoint` asserts no "
                "table (by decoded-pixel digest) and no paper appears in two splits, `split_summary` reports tables, objects "
                "per label and papers per split, and the training structure table is written to `outputs/{stem}_train.csv` "
                "in the shape BYOD expects.\n\n"
                "Look for: 94 tables from 46 papers, splits 50 / 23 / 21, three digests, and four refusal probes — a duplicate "
                "id, a box outside its image, a label outside the four, and an oversized image — each rejected before `torch` "
                "does anything. A few seconds."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import time\n\n"
                "from PIL import ImageDraw\n\n"
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'SPLIT_SEED = 42  # @param {{type:"integer"}}\n\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                "t0 = time.perf_counter()\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_count = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = load_corpus()\n"
                "    raw_count = {{'tables': len(corpus), 'papers': len({{r['paper_id'] for r in corpus}}), 'objects': sum(len(r['objects']) for r in corpus), 'spanning_cells': sum(1 for r in corpus for o in r['objects'] if o['label'] == 'table spanning cell')}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "load_seconds = round(time.perf_counter() - t0, 1)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "summary = split_summary(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw': raw_count, 'splits': disjoint, 'split_summary': summary, 'load_seconds': load_seconds, 'source_files': [f['path'] for f in CORPUS_SOURCE_FILES]}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'objects': manifest['objects_per_label'], 'objects_per_image': manifest['objects_per_image'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{k: example[k] for k in ('id', 'source_id', 'paper_id', 'paper_title', 'paper_license', 'grid') if k in example}}, 'size': example['image'].size, 'objects': [(o['label'], [round(v, 1) for v in o['box']]) for o in example['objects'][:4]]}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'box outside image': [{{**train_records[0], 'objects': [{{'label': 'table row', 'box': [0, 0, train_records[0]['image'].width + 5, 20]}}, *train_records[0]['objects']]}}, *train_records[1:8]],\n"
                "    'label outside the four': [{{**train_records[0], 'objects': [{{'label': 'table column header', 'box': [1, 1, 40, 20]}}, *train_records[0]['objects']]}}, *train_records[1:8]],\n"
                "    'oversized image': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 16))}}, *train_records[1:8]],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Recognise through the inference contract, with reference boxes\n\n"
                "Before any adaptation, the inference contract is exercised as it always was, on one test table. "
                "`validate_inputs` applies exactly the checks `recognize` applies — image type, sides `MIN_IMAGE_SIDE`.."
                "`MAX_IMAGE_SIDE` px, a threshold in `[0, 1]` — and returns an input manifest; a deliberately invalid "
                "threshold is validated too and its rejection recorded as a finding. `recognize` returns the queries at or "
                "above `RECOGNITION_THRESHOLD` with their six-class labels, and `structure_summary` counts them into the grid "
                "they imply. Because this table carries reference boxes, `evaluation_report` can for the first time return "
                "**`sample-sanity`**: one `box_iou` entry per reference, matched only against detections of the same label, with "
                "the reference and detected counts per label (the build record's probe table: every row, column and the table "
                "matched at IoU 0.8 or better). A rendered preview (reference boxes in green, detections coloured by label) is "
                "displayed. Read the header rows: the checkpoint may label the first row a `table column header` — a label the "
                "reference set does not carry, which the adaptation vocabulary leaves out rather than penalises."
            ),
            "code": (
                "COLOURS = {{'table': (60, 60, 220), 'table row': (0, 160, 0), 'table column': (230, 30, 30), 'table spanning cell': (200, 0, 200), 'table column header': (240, 140, 0), 'table projected row header': (0, 170, 170)}}\n\n"
                "def draw_objects(image, reference, detections, width=2):\n"
                "    canvas = image.convert('RGB').copy()\n"
                "    pen = ImageDraw.Draw(canvas)\n"
                "    for obj in reference:\n"
                "        pen.rectangle([round(v) for v in obj['box']], outline=(0, 200, 0), width=1)\n"
                "    for det in detections:\n"
                "        pen.rectangle([round(v) for v in det['box']], outline=COLOURS.get(det['label'], (120, 120, 120)), width=width)\n"
                "    return canvas\n\n"
                "def references_by_label(record):\n"
                "    out = {{}}\n"
                "    for obj in record['objects']:\n"
                "        out.setdefault(obj['label'], []).append(obj['box'])\n"
                "    return out\n\n"
                "probe_record = test_records[0]\n"
                "image = probe_record['image']\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_DETECTIONS': MAX_DETECTIONS, 'LABELS': list(LABELS), 'ADAPT_LABELS': list(ADAPT_LABELS), 'RECOGNITION_THRESHOLD': RECOGNITION_THRESHOLD}}, 'contract': {{'NUM_QUERIES': NUM_QUERIES, 'D_MODEL': D_MODEL, 'DECODER_LAYERS': DECODER_LAYERS, 'PARAMETER_COUNT': PARAMETER_COUNT}}}})\n"
                "input_manifest = validate_inputs(image, threshold=RECOGNITION_THRESHOLD, names=[probe_record['id']])\n"
                "try:\n"
                "    validate_inputs(image, threshold=1.5)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'threshold-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "result = pipe.recognize(image, threshold=RECOGNITION_THRESHOLD)\n"
                "recognize_seconds = round(time.perf_counter() - started, 3)\n"
                "checks = {{\n"
                "    'at_most_num_queries': len(result['detections']) <= MAX_DETECTIONS,\n"
                "    'labels_in_vocabulary': all(d['label'] in LABELS for d in result['detections']),\n"
                "    'scores_descending': all(a['score'] >= b['score'] for a, b in zip(result['detections'], result['detections'][1:])),\n"
                "    'boxes_inside_image': all(0 <= d['box'][0] <= d['box'][2] <= image.width + 1 and 0 <= d['box'][1] <= d['box'][3] <= image.height + 1 for d in result['detections']),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'recognize output failed a sanity check: {{checks}}')\n"
                "grid = structure_summary(result)\n"
                "report = evaluation_report(result, references_by_label(probe_record), sample_kind='one SciTSR-PD test table' if not USE_BYOD else 'one BYOD test table')\n"
                "print({{'probe_id': probe_record['id'], 'source_id': probe_record.get('source_id'), 'reference_grid': probe_record.get('grid'), 'reference_objects': len(probe_record['objects']), 'detections_at_threshold': len(result['detections']), 'implied_grid': [grid['n_rows'], grid['n_columns']], 'counts': grid['counts'], 'seconds': recognize_seconds, 'device': pipe.device, 'checks': checks, 'findings': len(input_manifest['findings'])}})\n"
                "print({{'verdict': report['verdict'], 'reason': report.get('reason')}})\n"
                "for metric in report['metrics']:\n"
                "    print({{'reference': metric['reference'], 'box_iou': round(metric['value'], 3), 'n_reference': metric['n_reference'], 'n_detected': metric['n_detected']}})\n"
                "assert report['verdict'] == 'sample-sanity'\n"
                "try:\n"
                "    from IPython.display import display\n"
                "    display(draw_objects(image, probe_record['objects'], result['detections']))\n"
                "except ImportError:\n"
                "    print({{'preview': 'IPython display unavailable; the preview PNG is written in Section 9'}})"
            ),
        },
        {
            "md": (
                "## 6. The grid prior, the zero-shot checkpoint and the frozen policy\n\n"
                "Three rows frame the adaptation, all on the 21 test tables and all threshold-free: per-label AP@0.5, AP@0.75 "
                "and AP rank every (query, class) pair of every image by score, so the recogniser is judged on its ordering, and "
                "the per-label recall at the operating threshold (0.5, the pipeline's default) and the **grid agreement** — the "
                "fraction of tables whose surviving row and column counts both match — are printed beside them. The **grid "
                "prior** lays the training split's mean row and column counts out uniformly over each image — what \"tables "
                "are usually about this shape\" alone buys. The **zero-shot checkpoint** (`evaluate_zero_shot`) scores every "
                "query by its own softmax probability for each of the four labels, headers neither scored nor penalised. "
                "The **frozen policy** is `adapt` with `trainable_layers=0`: it first copies the checkpoint's own rows for the "
                "four labels and no-object into a restricted head and copies the box head, scores them untrained on validation "
                "(epoch 0, the **copied-head zero-shot policy** — its softmax runs over five logits, so its numbers differ from the untouched seven-way checkpoint scored just above), then trains both on the cached decoder features of the 50 training "
                "tables under the DETR set loss for `HEAD_STEPS` full-batch steps (epoch 1) and keeps whichever has the lower "
                "validation loss, scored on the test split by `evaluate`. The build record: prior mAP@0.5 38.7 % (grid "
                "agreement 4.8 %), zero-shot 88.7 % / AP@0.75 73.4 % / mAP 63.5 % (columns 100 %, rows 86.5 %, spanning cells "
                "74.8 %; grid agreement 85.7 %), and the trained heads 85.2 % / 76.1 % / 71.5 % (validation loss 0.295 against "
                "0.771 for the copied rows, so the frozen policy was kept over the copied-head zero-shot one) — the checkpoint already "
                "recognises the structure, and training the heads mostly moves the boxes onto the derived convention (mean "
                "best IoU 0.81 → 0.91) at a cost on spanning cells. About a minute on CPU."
            ),
            "code": (
                'HEAD_STEPS = 300  # @param {{type:"integer"}}\n'
                'HEAD_LR = 1e-3  # @param {{type:"number"}}\n\n'
                "def brief(m):\n"
                "    return {{'map50': round(m['map50'], 4), 'map75': round(m['map75'], 4), 'map': round(m['map'], 4), 'ap50_per_label': {{label: (None if v['ap50'] is None else round(v['ap50'], 4)) for label, v in m['per_label'].items()}}, 'grid_exact_at_0.5': round(m['grid_exact_at_threshold'], 4), 'recall_at_0.5': {{label: (None if v is None else round(v, 4)) for label, v in m['recall_at_threshold'].items()}}, 'mean_best_iou': round(m['mean_best_iou'], 4), 'n': m['n_images']}}\n\n"
                "prior = prior_baseline(train_records, test_records, labels=ADAPT_LABELS, threshold=RECOGNITION_THRESHOLD)\n"
                "print({{'grid_prior': brief(prior), 'baseline': prior['baseline'], 'prior_grid': prior['prior_grid']}})\n"
                "t0 = time.perf_counter()\n"
                "zero_shot_test = pipe.evaluate_zero_shot(test_records)\n"
                "print({{'zero_shot': brief(zero_shot_test), 'policy': zero_shot_test['policy'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "probe_result = pipe.adapt(train_records, val_records, head_steps=HEAD_STEPS, head_lr=HEAD_LR, trainable_layers=0)\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_policy_call': probe_result['policy'], 'best_epoch': probe_result['best_epoch'], 'head_final_loss': round(probe_result['head_final_loss'], 4), 'validation': {{entry['epoch']: entry['val'] for entry in probe_result['history']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'frozen_policy_test': brief(frozen_test), 'loss': round(frozen_test['loss'], 4), 'verdict': frozen_test['verdict']}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "assert frozen_test['map50'] > prior['map50'] and probe_result['policy'] in (POLICY_ZERO_SHOT, POLICY_FROZEN)"
            ),
        },
        {
            "md": (
                "## 7. The unfrozen policy: a bounded decoder unfreeze selected against the heads and the checkpoint\n\n"
                "`adapt` with `TRAINABLE_LAYERS` > 0 repeats epochs 0 and 1 of the ladder (the copied rows untrained, then the "
                "restricted heads on the frozen features), then unfreezes the last `TRAINABLE_LAYERS` decoder layers — two "
                "by default, 3,157,504 of 28,828,619 parameters; the backbone, the input projection, the encoder, the query "
                "embeddings, the earlier decoder layers and the checkpoint's own heads stay frozen — and trains them with "
                "both heads end to end, one table per step, for `EPOCHS` epochs (AdamW at `LEARNING_RATE`, weight decay 0.01, "
                "gradient clipping 0.1, seeded order, no augmentation) under the same DETR set loss: exact Hungarian matching "
                "of the 125 queries to the reference objects (the O(n²m) shortest-augmenting-path algorithm in the carried "
                "module, no external solver), cross-entropy over the five logits with a 0.1 no-object weight, L1 and GIoU box "
                "terms. Every epoch is scored on validation, and the epoch with the **lowest validation loss** is kept — the "
                "checkpoint's own rows (epoch 0) and the heads alone (epoch 1) compete on equal terms, so the selected policy "
                "can be any of the three. mAP@0.5, mAP and grid agreement are printed beside the loss at every epoch.\n\n"
                "Watch the validation loss: in the build record it fell from 0.771 (copied rows, untrained) to 0.295 (heads), then "
                "0.310 → 0.282 → 0.286 over three unfreeze epochs at 1e-4, so the second unfreeze epoch was selected; at "
                "3e-4 the unfreeze never beat the heads (0.390 → 0.356 → 0.347) and the frozen policy was kept; with all six decoder layers unfrozen it reached 0.288 at the second unfreeze epoch (selected) for 85.6 % mAP@0.5 / 73.5 % mAP on the test split and a 38 MB adapter — no better than two layers. "
                "Note that DETR's train-mode dropout makes the per-table training loss sit above the full-batch head loss."
            ),
            "code": (
                'EPOCHS = 3  # @param {{type:"integer"}}\n'
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'TRAINABLE_LAYERS = 2  # @param {{type:"integer"}}\n\n'
                "def report_epoch(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'stage': entry['stage'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_loss'] = round(entry['val']['loss'], 4)\n"
                "        row['val_map50'] = round(entry['val']['map50'], 4)\n"
                "        row['val_map'] = round(entry['val']['map'], 4)\n"
                "        row['val_grid_exact'] = round(entry['val']['grid_exact'], 4)\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, head_steps=HEAD_STEPS, head_lr=HEAD_LR, trainable_layers=TRAINABLE_LAYERS, epochs=EPOCHS, lr=LEARNING_RATE, progress=report_epoch)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "report_epoch(adapt_result['history'][0])\n"
                "print({{'selected_policy': adapt_result['policy'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'trainable_heads': adapt_result['n_trainable_head'], 'trainable_layers': adapt_result['n_trainable_layers'], 'total_parameters': adapt_result['n_total'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or policy selection, and no paper in it appears in the training "
                "or validation splits. The selected model is scored exactly as the frozen policy was in Section 6, and the "
                "four rows are put side by side: grid prior, zero-shot checkpoint, frozen policy, selected policy — per label "
                "and as class means, with grid agreement and the set loss. Read the policy first: if validation kept the "
                "heads, the last two rows are the same model; if it kept the untrained copied rows, the selected row is the "
                "checkpoint under a restricted softmax; if it chose the unfreeze, the delta is what the unfreeze bought on "
                "21 tables — the build record: 86.8 % mAP@0.5 / 77.5 % AP@0.75 / 72.2 % mAP against 85.2 % / 76.1 % / 71.5 % "
                "for the heads, and against 88.7 % / 73.4 % / 63.5 % for the untouched checkpoint: adaptation tightened the "
                "boxes onto the derived convention (mAP +8.7 points over the checkpoint) and lost AP@0.5 on spanning cells "
                "(74.8 % → 62.6 %, from 56 training instances) and grid agreement at 0.5 (85.7 % → 76.2 %). The cell asserts "
                "the selected model beats the grid prior on mAP@0.5; it does **not** assert a gain over the heads or the "
                "checkpoint, because that is the question, not the answer. 21 tables with 300 objects from one seeded split "
                "of one corpus give no dispersion estimate — one table is about five points of grid agreement."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "rows = {{'grid_prior': prior, 'zero_shot': zero_shot_test, 'frozen_policy': frozen_test, 'selected_policy': adapted_test}}\n"
                "comparison = {{metric: {{name: round(m[metric], 4) for name, m in rows.items()}} for metric in ('map50', 'map75', 'map', 'mean_best_iou', 'grid_exact_at_threshold')}}\n"
                "for label in ADAPT_LABELS:\n"
                "    comparison[f'ap50[{{label}}]'] = {{name: (None if m['per_label'][label]['ap50'] is None else round(m['per_label'][label]['ap50'], 4)) for name, m in rows.items()}}\n"
                "    comparison[f'recall_at_0.5[{{label}}]'] = {{name: (None if m['recall_at_threshold'][label] is None else round(m['recall_at_threshold'][label], 4)) for name, m in rows.items()}}\n"
                "comparison['loss'] = {{'frozen_policy': round(frozen_test['loss'], 4), 'selected_policy': round(adapted_test['loss'], 4)}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('map50', 'map75', 'map')}}\n"
                "comparison['delta_vs_zero_shot'] = {{metric: round(adapted_test[metric] - zero_shot_test[metric], 4) for metric in ('map50', 'map75', 'map')}}\n"
                "comparison['selected_policy'] = adapt_result['policy']\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'split_summary': summary,\n"
                "    'single_image_report': report,\n"
                "    'baselines': {{'grid_prior': prior, 'zero_shot': zero_shot_test}},\n"
                "    'frozen_policy': {{'adaptation': {{k: v for k, v in probe_result.items() if k not in ('history', 'trainable_names')}}, 'history': probe_result['history'], 'test': frozen_test}},\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['map50'] > prior['map50']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Recognise before and after, export the adapter and reload it\n\n"
                "Three test tables are run through `recognize_adapted` with the selected model at the 0.5 threshold — each "
                "query's arg-max label among the four adapted classes when its probability reaches the threshold — and "
                "rendered beside the frozen policy's objects (from a fresh pipeline with heads trained the same way — after an "
                "unfreeze the decoder inside `pipe` has moved, so the frozen column needs its own decoder) and the reference "
                "boxes: reference in thin green, objects coloured by label (rows green, columns red, table blue, spanning "
                "cells magenta); `outputs/{stem}_preview.png` holds the sheet. `recognize` — the checkpoint's own six-class "
                "heads — still answers in its own label space on the same tables, headers included.\n\n"
                "`save_artifact` writes the restricted class head and box head and, when the unfrozen policy was selected, the "
                "trained decoder-layer tensors — about 0.5 MB for the heads alone, 13.2 MB with two decoder layers — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, "
                "the digest of the base `model.safetensors`, the classes, the selected policy, the tensor names, the file size "
                "and SHA-256, the training configuration and the epoch history (OUT8). "
                "`TableTransformerStructurePipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest "
                "and digest **before** deserialising, rebuilds the heads from the manifest, refuses any tensor that is not a "
                "decoder-layer tensor of the base, and overlays the tensors onto a freshly loaded base — a new object from "
                "files, not the in-memory model (VER2). The cell asserts identical (query, class) scores and boxes on three "
                "tables and an identical test mAP@0.5 (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "show = test_records[:3]\n"
                "frozen_pipe = TableTransformerStructurePipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "frozen_pipe.adapt(train_records, val_records, head_steps=HEAD_STEPS, head_lr=HEAD_LR, trainable_layers=0)\n"
                "before_after = []\n"
                "panels = []\n"
                "for record in show:\n"
                "    after = pipe.recognize_adapted(record['image'], threshold=RECOGNITION_THRESHOLD)\n"
                "    before = frozen_pipe.recognize_adapted(record['image'], threshold=RECOGNITION_THRESHOLD)\n"
                "    base = pipe.recognize(record['image'], threshold=RECOGNITION_THRESHOLD)\n"
                "    reference_counts = {{label: sum(1 for o in record['objects'] if o['label'] == label) for label in ADAPT_LABELS}}\n"
                "    def counts(detections):\n"
                "        return {{label: sum(1 for d in detections if d['label'] == label) for label in ADAPT_LABELS}}\n"
                "    before_after.append({{'id': record['id'], 'source_id': record.get('source_id'), 'reference': reference_counts, 'frozen': counts(before['detections']), 'selected': counts(after['detections']), 'checkpoint': structure_summary(base)['counts']}})\n"
                "    print(before_after[-1])\n"
                "    left = draw_objects(record['image'], record['objects'], before['detections'])\n"
                "    right = draw_objects(record['image'], record['objects'], after['detections'])\n"
                "    panel = Image.new('RGB', (left.width * 2 + 8, left.height), (255, 255, 255))\n"
                "    panel.paste(left, (0, 0))\n"
                "    panel.paste(right, (left.width + 8, 0))\n"
                "    panels.append(panel)\n"
                "sheet = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels) + 8 * (len(panels) - 1)), (255, 255, 255))\n"
                "y = 0\n"
                "for panel in panels:\n"
                "    sheet.paste(panel, (0, y))\n"
                "    y += panel.height + 8\n"
                "sheet.save('outputs/{stem}_preview.png')\n"
                "try:\n"
                "    from IPython.display import display\n"
                "    display(sheet)\n"
                "except ImportError:\n"
                "    pass\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'policy': artifact_manifest['adapter']['policy'], 'classes': artifact_manifest['adapter']['classes'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = TableTransformerStructurePipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "parity = {{'queries_identical': reloaded.predict_objects(show) == pipe.predict_objects(show), 'map50_in_memory': round(adapted_test['map50'], 6), 'map50_reloaded': round(reloaded_test['map50'], 6), 'classes_identical': reloaded.classes == pipe.classes}}\n"
                "print({{'reload_parity': parity, 'reloaded_policy': reloaded.adapter['policy'], 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['queries_identical'] and parity['classes_identical'] and abs(adapted_test['map50'] - reloaded_test['map50']) < 1e-9\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHTS_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(MANIFEST['files']), 'total_bytes': MANIFEST['totalBytes'], 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'source_files': list(CORPUS_SOURCE_FILES), 'tables': len(SAMPLE_RECORDS), 'license': CORPUS_LICENSE, 'labels': list(ADAPT_LABELS), 'dpi': CORPUS_DPI}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'probe_id': probe_record['id'], 'implied_grid': grid, 'single_image_report': report, 'seconds': recognize_seconds}},\n"
                "    'comparison': comparison,\n"
                "    'before_after': before_after,\n"
                "    'preview_file': 'outputs/{stem}_preview.png',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'policy': artifact_manifest['adapter']['policy']}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 21 held-out scientific tables the grid prior scores 38.7 % mAP@0.5, the untouched checkpoint 88.7 % (mAP 63.5 %, "
        "grid agreement 85.7 %), the restricted heads trained on its frozen features 85.2 % (mAP 71.5 %), and the last two "
        "decoder layers unfrozen with them 86.8 % (mAP 72.2 %, AP@0.75 77.5 %), selected at the second of three unfreeze "
        "epochs by validation loss; the adapter reloads to identical query scores. That is the claim and the finding: on a "
        "corpus close to the model's training renders, adaptation does not buy recognition the checkpoint lacks — it buys "
        "**agreement with a box convention** (mean best IoU 0.81 → 0.90, mAP +8.7 points over the checkpoint) and pays for it "
        "on the rarest label (spanning cells, 56 training instances: AP@0.5 74.8 % → 62.6 %) and on the structure-level "
        "question a caller asks (grid agreement at 0.5: 85.7 % → 76.2 %). The ladder ran all three policies end to end on a "
        "real structure-labelled corpus with the set loss and an exact Hungarian matcher implemented in the open, chose "
        "among them on validation rather than by assumption — and the validation loss, which is a set loss under the "
        "derived convention, preferred the trained heads to the copied checkpoint rows by a wide margin (0.295 vs 0.771) "
        "while the checkpoint kept the higher mAP@0.5 on the test split: the selection criterion and the headline metric "
        "disagree, and the notebook reports both rather than hiding one.\n\n"
        "The test split is 21 tables with 300 structure objects from one seeded split of one small corpus with no dispersion "
        "estimate — one table is about five points of grid agreement, so a few points of AP is noise. The reference boxes are "
        "a derivation from SciTSR's cells (rows tiling an ink-bounded table at the mid-gaps), stated in full in the carried "
        "`sample_data` module; a metric against them measures agreement with that convention as much as recognition, which "
        "is exactly why the untouched checkpoint's AP@0.75 rises after adaptation while its AP@0.5 does not. AP ranks every "
        "(query, class) pair by the head's score; the pipeline's operating threshold of 0.5 is the upstream script's value, "
        "not a calibration, and the per-label recall at it is reported beside the AP — a deployment must choose its own "
        "threshold on its own labelled tables. When the unfrozen policy is selected it changes the last decoder layers, which "
        "every query shares, so `recognize` — which keeps the checkpoint's heads — reads a moved decoder afterwards; the "
        "artifact records which policy won.\n\n"
        "Three things to carry to real data. **Baselines first:** the grid prior and the zero-shot checkpoint on *your* "
        "tables are the numbers to read before any trained head's — if the checkpoint already agrees with your boxes, "
        "adaptation is a convention change, not a capability change. **Leakage:** split by paper, document or source (the "
        "contract splits by `group` / `paper_id`, never by table). **Selection vs headline:** a loss-based selection and an "
        "AP-based headline can disagree; report both and say which one you optimised.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, decode and digest-verify an embedded "
        "structure-labelled corpus, validate the demonstrated dataset contract without leakage, execute the inference "
        "contract with a per-label `sample-sanity` report against reference boxes, run a three-policy adaptation ladder with "
        "validation-based selection, evaluate by per-label AP and grid agreement against a prior and the zero-shot checkpoint "
        "on an independent split, and emit the shown machine-readable artifacts — without the repository being reachable. It "
        "does **not** establish benchmark superiority, PubTables-1M or FinTabNet accuracy, GriTS, a usable acceptance "
        "threshold, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** raise `TRAINABLE_LAYERS` to 6 (in the build record all six layers were selected at their second epoch for 85.6 % mAP@0.5 / 73.5 % mAP — within the grain of two layers, with a 38 MB adapter); raise `LEARNING_RATE` to 3e-4 (the unfreeze never beat the heads on validation and the frozen policy was kept, 85.2 %); set `EPOCHS = 0` to keep the frozen "
        "policy and read the before/after sheet as a heads-only result; raise `HEAD_STEPS`; or bring your own "
        "structure-labelled tables through BYOD and read the prior and the zero-shot row before any policy.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/table-transformer-structure-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/table-transformer-structure-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/table-transformer-structure-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/microsoft/table-transformer\n"
        "- Aligning benchmark datasets for table structure recognition (Smock, Pesala, Abraham, 2023): https://arxiv.org/abs/2303.00716\n"
        "- PubTables-1M (Smock, Pesala, Abraham, 2021): https://arxiv.org/abs/2110.00061\n"
        "- End-to-End Object Detection with Transformers (DETR; the set loss and Hungarian matching, Carion et al., 2020): https://arxiv.org/abs/2005.12872\n"
        "- SciTSR: Complicated Table Structure Recognition (Chi et al., 2019): https://arxiv.org/abs/1908.04729 — the public-domain subset SciTSR-PD: https://huggingface.co/datasets/bevaya/SciTSR-pd\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
