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
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
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
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell (it first puts the pipeline back to the pinned base) to supply your own structure-labelled table crops: at "
        "least **12 tables in 12 groups** (or fewer groups with several tables each — the training split needs 8 tables after "
        "20 % + 20 % of the groups are held out), as a `.zip` or folder holding `structure.csv` (columns `id`, `file`, `label`, `x_min`, `y_min`, "
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
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see how a pretrained table-structure recogniser is adapted to a new corpus and box convention with a small labelled set — copied heads, trained heads on frozen features, and a bounded unfreeze — and how to tell from held-out numbers what adaptation bought and what it cost. No prior experience with DETR or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. CPU is adequate (about three minutes of model time); a GPU is faster.\n\n**Input → Model → Output.**\n\n| | Recognition (checkpoint heads) | Adapted recognition | Bounded adaptation |\n|---|---|---|---|\n| Input | one table crop, sides 16..4,096 px, and a score threshold | one table crop | structure-labelled tables (50 training and 23 validation in the sample), split by paper |\n| Model | Table Transformer: ResNet-18 backbone, DETR encoder-decoder with 125 queries, six structure classes and *no object* | the same decoder with a restricted five-way class head and a copied box head | copied-head, frozen and unfrozen policies; validation loss chooses |\n| Output | labelled boxes (table, rows, columns, headers, spanning cells) with softmax scores — not calibrated | boxes for the four adapted labels | a safetensors adapter, and held-out per-label AP and grid agreement beside a grid prior and the zero-shot row |\n\n**How to use this notebook.** Choose a runtime (CPU works; a GPU is faster), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package (including the embedded tables) and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer that names the run it quotes — the Kaggle T4 release run of 19 September 2026 or the CPU build record. Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the embedded corpus, validation and a paper-level split *(evaluation practice)* → 5 the inference contract with reference boxes *(core concept: what a structure recogniser returns)* → 6 the grid prior, the zero-shot checkpoint and the frozen policy *(evaluation practice)* → 7 the unfrozen policy and validation selection *(core concept)* → 8 held-out evaluation → 9 structure before and after, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            )
        ]
    },
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
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with DETR or fine-tuning. The notebook explains DETR queries, the set loss with Hungarian matching and GIoU, per-label AP, grid agreement, the copied-head, frozen and unfrozen policies and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path runs on CPU and uses CUDA automatically when available; float32 on both. The build record measured about 0.1 s per table to run the decoder on CPU (2 s for the 21-table zero-shot pass), about 45 s for the restricted heads including feature extraction and two validation passes, and about 15 s per unfreeze epoch over 50 tables plus a 23-table validation pass. The pinned `torch==2.14.0` install and the 115 MB checkpoint are the large downloads of the run; the tables travel inside the notebook.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union and average precision measure and why AP does not depend on a score threshold; what a Hungarian (one-to-one) matching between predictions and references is; what validation-based selection among policies means; that a table's cell grid is the intersection of its row and column boxes.",
        "- **Data contract:** records are `{id, image, objects}` — a PIL image (or a path to one) with sides 16..4,096 px and a list of 1..125 `{label, box}` structure objects with `box` = `[x_min, y_min, x_max, y_max]` pixels inside the image (sides of at least 2 px), `label` among `table` (at most one), `table column`, `table row` (at least one each) and `table spanning cell`, ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a training set needs 8..2,000 records, so with the default 20 % + 20 % group hold-out the effective BYOD minimum is **12 tables in 12 groups** (`min_byod_records()` computes it); tables are de-duplicated by decoded-pixel digest and split by `group` / `paper_id` so one paper never straddles splits. BYOD accepts a `.zip` (or a directory) holding `structure.csv` and the image files, and requires a non-empty `group` on every row — the notebook's automatic split is group-disjoint only because the loader refuses ungrouped rows (`load_byod_dataset(..., require_group=False)` is the explicit opt-out, without that guarantee).",
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
                "does anything. A few seconds.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or "
                "a folder holding `structure.csv` and the images, as a path in this runtime — this works on Colab, Kaggle and "
                "Jupyter) or leave `BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** "
                "from this cell. This cell first puts the pipeline back to the pinned base, so the zero-shot row and every policy "
                "start from the untouched checkpoint. The effective minimum is 12 tables in 12 groups.\n\n"
                "**Predict before running:** several tables come from the same paper. Why does the split keep a whole paper on "
                "one side, and which label do you expect to be rarest in the training split?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import time\n\n"
                "from PIL import ImageDraw\n\n"
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                'SPLIT_SEED = 42  # @param {{type:"integer"}}\n\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                "# A re-run after Sections 6-7 (BYOD, or a new split): the zero-shot row and every policy must start from the pinned base.\n"
                "had_adapter = pipe.adapter is not None\n"
                "restored_layers = pipe.restore_base()\n"
                "if had_adapter or restored_layers:\n"
                "    print({{'restored_pinned_base': len(restored_layers), 'note': 'the restricted heads and adapted decoder layers were removed; Sections 5-7 start from the checkpoint again'}})\n"
                "t0 = time.perf_counter()\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding structure.csv and the images.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding structure.csv and the images.')\n"
                "        byod_path = Path('work') / 'byod.zip'\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_count = {{'byod': len(records), 'duplicate_tables_dropped': len(records) - sum(len(part) for part in splits.values()), 'effective_minimum': min_byod_records()['total']}}\n"
                "    if len(splits['test']) < 10:\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test tables: AP and grid agreement move in large steps and carry no dispersion estimate; add tables before reading them\"}})\n"
                "else:\n"
                "    corpus = load_corpus()\n"
                "    raw_count = {{'tables': len(corpus), 'papers': len({{r['paper_id'] for r in corpus}}), 'objects': sum(len(r['objects']) for r in corpus), 'spanning_cells': sum(1 for r in corpus for o in r['objects'] if o['label'] == 'table spanning cell')}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "load_seconds = round(time.perf_counter() - t0, 1)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "# The training split must hold MIN_RECORDS; validation and test only need one table each (split_dataset checks that).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
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
                "**What to notice:** 94 tables from 46 papers, 50 / 23 / 21 tables in 29 / 7 / 10 papers, the per-label counts in `split_summary` — especially `table spanning cell` per split — and the four refusals.\n\n<details><summary>Check your reasoning</summary>Tables of one paper share a typesetting style, column layout and often a header pattern; split by table, the test set would hold near-copies of training tables and the score would measure recall of one paper's style. Splitting by paper keeps every paper on one side; a review probe confirmed no paper is shared. With the default seed the training split holds only **19** spanning cells, against 39 in validation and 13 in test — the rarest label is rarer in training than in validation, which matters in Section 8.</details>"
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
                "reference set does not carry, which the adaptation vocabulary leaves out rather than penalises.\n\n"
                "*Core concept.* A DETR recogniser answers with a fixed set of 125 query slots, each a box with a class "
                "distribution; the threshold decides which slots you see. The score is a softmax over the checkpoint's own "
                "classes, not a calibrated confidence.\n\n"
                "**Predict before running:** the checkpoint was trained on PubTables-1M renders, which resemble these arXiv "
                "tables. Will its rows and columns match the derived reference boxes closely (IoU above 0.7) on this table?"
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
                "if report['verdict'] != 'sample-sanity':  # a contract check on the report itself, not a quality claim\n"
                "    raise RuntimeError(f\"evaluation_report returned verdict {{report['verdict']!r}} for a record with reference boxes\")\n"
                "try:  # `display` is provided by the isolated worker (and by any IPython kernel); the worker has no IPython\n"
                "    display(draw_objects(image, probe_record['objects'], result['detections']))\n"
                "except NameError:\n"
                "    print({{'preview': 'display unavailable; the preview PNG is written in Section 9'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the `box_iou` per reference, the implied grid against `reference_grid`, and whether a `table column header` appears.\n\n<details><summary>Check your reasoning</summary>Yes, mostly. In the local pre-flight record (19 September 2026, CPU) the probe table `test-000` (a 10 × 7 grid) gave `sample-sanity` with the table at IoU 0.93, rows 0.71–0.78 and columns 0.77–0.97, and an implied grid of 10 × 7. The checkpoint already recognises this kind of table; what it does not know is the derived box convention (rows tiled at the mid-gaps), which is why the row IoUs sit lower than the columns'.</details>"
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
                "validation loss, scored on the test split by `evaluate`. **What to look for:** the zero-shot row against the "
                "prior and against the trained heads at each IoU threshold separately, the mean best IoU, and the spanning-cell "
                "AP. About a minute on CPU.\n\n"
                "*Evaluation practice.* The cell reports a **verdict** — whether the frozen policy beats the grid prior on "
                "mAP@0.5 — and records it; when the prior is competitive (regular, evenly spaced tables such as forms or "
                "financial statements) that is a finding, and the notebook continues. The policy identity stays a hard check: "
                "`trainable_layers=0` must give the copied-head or frozen policy.\n\n"
                "**Predict before running:** order the three rows by mAP@0.5 — the grid prior, the untouched checkpoint "
                "(zero-shot), and heads trained on its frozen decoder features. Will the order be the same for AP@0.75?"
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
                "if probe_result['policy'] not in (POLICY_ZERO_SHOT, POLICY_FROZEN):  # a contract check: trainable_layers=0 never unfreezes\n"
                "    raise RuntimeError(f\"adapt(trainable_layers=0) returned policy {{probe_result['policy']!r}}\")\n"
                "# A reported verdict, not an assertion: on your tables the grid prior may be competitive, and that is a finding.\n"
                "frozen_verdict = 'frozen policy above the grid prior on mAP@0.5' if frozen_test['map50'] > prior['map50'] else 'frozen policy NOT above the grid prior on mAP@0.5: your tables may be regular enough for a uniform grid; read the prior before the recogniser'\n"
                "print({{'frozen_vs_prior': frozen_verdict}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three rows' mAP@0.5, AP@0.75 and mAP, the mean best IoU, the spanning-cell AP@0.5 and the grid agreement.\n\n<details><summary>Check your reasoning</summary>At mAP@0.5 the order is prior, then trained heads, then the untouched checkpoint; at AP@0.75 and mAP the heads overtake the checkpoint. In the Kaggle T4 release run (19 September 2026) the grid prior scored 38.68 % mAP@0.5 (grid agreement 4.76 %), the checkpoint 88.73 % / AP@0.75 73.39 % / mAP 63.52 % (spanning cells 74.8 %, grid agreement 85.71 %), and the frozen policy 86.30 % / 76.16 % / 71.97 % (spanning cells 60.16 %, grid agreement 80.95 %). Training the heads mostly moves the boxes onto the derived convention — mean best IoU 0.81 → 0.91 — at a cost on spanning cells and on grid agreement.</details>"
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
                "Note that DETR's train-mode dropout makes the per-table training loss sit above the full-batch head loss.\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: decoder layers an earlier call (or "
                "an artifact) changed are restored before the features are cached, so every policy starts from the checkpoint "
                "and re-running Sections 6–8 with a changed field repeats the comparison validly. To compare a change side by "
                "side without replacing the default exports, use Section 10.\n\n"
                "**Predict before running:** validation loss is a set loss under the derived box convention. Will it prefer "
                "the untrained copied rows (epoch 0) or the trained heads, and will that preference agree with the test mAP@0.5?"
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
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'trainable_layers': TRAINABLE_LAYERS, 'head_steps': HEAD_STEPS, 'head_lr': HEAD_LR}}\n"
                "if settings != {{'epochs': 3, 'lr': 1e-4, 'trainable_layers': 2, 'head_steps': 300, 'head_lr': 1e-3}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, head_steps=HEAD_STEPS, head_lr=HEAD_LR, trainable_layers=TRAINABLE_LAYERS, epochs=EPOCHS, lr=LEARNING_RATE, progress=report_epoch)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "report_epoch(adapt_result['history'][0])\n"
                "print({{'selected_policy': adapt_result['policy'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'trainable_heads': adapt_result['n_trainable_head'], 'trainable_layers': adapt_result['n_trainable_layers'], 'total_parameters': adapt_result['n_total'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the validation loss per epoch, which epoch was selected, and whether the epoch with the lowest loss also has the highest validation mAP@0.5.\n\n<details><summary>Check your reasoning</summary>The trained heads, by a wide margin — and no, the two criteria disagree. In the local pre-flight record (19 September 2026, CPU) validation loss went 0.7705 (copied rows, untrained) → 0.2945 (heads) → 0.3104 → 0.2818 → 0.2860 over three unfreeze epochs, so the second unfreeze epoch (`best_epoch` 3) was kept, while validation mAP@0.5 was highest for the untrained copied rows (0.970, against 0.922–0.924 afterwards). The Kaggle T4 release run selected the same policy. The loss measures agreement with the derived box convention; mAP@0.5 mostly measures finding the objects at all.</details>'
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
                "21 tables. **What to look for:** the selected row against the checkpoint at AP@0.5 and at AP@0.75 / mAP "
                "separately, the spanning-cell AP@0.5, and grid agreement. The training split holds few spanning cells (19 "
                "with the default seed — Section 4's `split_summary` prints the count for your split) and the test split 13, "
                "so one spanning cell is about eight points of that label's recall. The cell records verdicts — whether the "
                "selected model beats the grid prior, and whether it improved on the frozen heads and on the checkpoint at "
                "mAP@0.5 and at mAP — instead of asserting them, because a policy that does not help is a finding, not an "
                "error, and Section 9 still exports, reloads and writes the result. 21 tables with 300 objects from one "
                "seeded split of one corpus give no dispersion estimate — one table is about five points of grid "
                "agreement.\n\n"
                "**Predict before running:** will the selected policy beat the untouched checkpoint on mAP@0.5? On AP@0.75? On "
                "spanning cells?"
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
                "def direction(new, old):\n"
                "    return 'improved' if new > old else ('no gain' if new == old else 'worse')\n"
                "# Reported verdicts, not assertions: a policy that does not help is a result to record, and export and reload still run.\n"
                "comparison['verdicts'] = {{\n"
                "    'frozen_vs_prior': frozen_verdict,\n"
                "    'selected_above_prior_map50': bool(adapted_test['map50'] > prior['map50']),\n"
                "    'selected_vs_frozen_map50': direction(adapted_test['map50'], frozen_test['map50']),\n"
                "    'selected_vs_frozen_map': direction(adapted_test['map'], frozen_test['map']),\n"
                "    'selected_vs_zero_shot_map50': direction(adapted_test['map50'], zero_shot_test['map50']),\n"
                "    'selected_vs_zero_shot_map': direction(adapted_test['map'], zero_shot_test['map']),\n"
                "}}\n"
                "comparison['spanning_cells_per_split'] = {{name: sum(1 for r in part for o in r['objects'] if o['label'] == 'table spanning cell') for name, part in splits.items()}}\n"
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
                "print({{'verdicts': comparison['verdicts']}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the selected policy, the four rows on each metric, `delta_vs_zero_shot`, the spanning-cell row and the `verdicts`.\n\n<details><summary>Check your reasoning</summary>In the Kaggle T4 release run (19 September 2026) validation chose the unfrozen policy, which scored 87.28 % mAP@0.5 / 76.75 % AP@0.75 / 72.55 % mAP: below the untouched checkpoint at mAP@0.5 (88.73 %) but well above it at AP@0.75 and mAP (+3.4 and +9.0 points). Spanning-cell AP@0.5 fell from 74.8 % to 61.9 % and grid agreement from 85.7 % to 76.2 %. Adaptation bought agreement with the derived box convention and paid for it on the rarest label — 19 spanning cells in training — and on the structure-level question. The selection criterion (validation set loss) and the headline metric (mAP@0.5) disagree; the notebook reports both.</details>'
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
                "tables and an identical test mAP@0.5 (VER4) — a contract check, so it stays a hard check.\n\n"
                "**Predict before running:** on these three tables, will the selected model's row and column counts match the "
                "reference more often than the frozen heads'?"
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
                "try:  # `display` is provided by the isolated worker (and by any IPython kernel)\n"
                "    display(sheet)\n"
                "except NameError:\n"
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
        {
            "md": (
                "**What to notice:** the `reference`, `frozen`, `selected` and `checkpoint` counts in `before_after`, the preview sheet, and the reload parity line.\n\n<details><summary>Check your reasoning</summary>Three tables cannot rank the policies: in the local pre-flight record all three tables' row and column counts matched the reference under every policy. Use the sheet to see *what* the recogniser does — where the rows sit, whether a spanning cell is found — not to choose a policy; that is Section 8's job on 21 tables. Reload parity held in the release run: identical (query, class) scores and boxes, and test mAP@0.5 0.8728 in memory and reloaded.</details>"
            ),
        },
        {
            "md": (
                "## 10. Change one thing: unfreeze all six decoder layers (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default all six decoder layers "
                "train instead of two — and run this cell after Sections 4–9. The experiment loads its **own** pipeline from the "
                "verified snapshot, so it starts from the checkpoint and never touches the default `pipe`; it writes only to "
                "`outputs/{stem}_experiment/`, prints the default and the changed run side by side, and checks that the default "
                "exports (adapter, evaluation report, result) are byte-identical afterwards. A few minutes on CPU.\n\n"
                "**Predict:** with three times the trainable decoder layers, will the held-out mAP rise, and will spanning cells "
                "recover?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_TRAINABLE_LAYERS = 6  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the checkpoint and the default pipe is untouched.\n"
                "    experiment_pipe = TableTransformerStructurePipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, head_steps=HEAD_STEPS, head_lr=HEAD_LR, trainable_layers=EXPERIMENT_TRAINABLE_LAYERS, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, progress=report_epoch)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records)\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'trainable_layers': adapt_result['trainable_layers'], 'epochs': adapt_result['epochs'], 'lr': adapt_result['lr']}}, 'experiment': {{'trainable_layers': EXPERIMENT_TRAINABLE_LAYERS, 'epochs': EXPERIMENT_EPOCHS, 'lr': EXPERIMENT_LEARNING_RATE}}}},\n"
                "        'selected_policy': {{'default': adapt_result['policy'], 'experiment': experiment_result['policy']}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'validation_loss_by_epoch': {{'default': [round(e['val']['loss'], 4) for e in adapt_result['history'] if e.get('val')], 'experiment': [round(e['val']['loss'], 4) for e in experiment_result['history'] if e.get('val')]}},\n"
                "        'test': {{metric: {{'zero_shot': round(zero_shot_test[metric], 4), 'default': round(adapted_test[metric], 4), 'experiment': round(experiment_test[metric], 4)}} for metric in ('map50', 'map75', 'map', 'grid_exact_at_threshold')}},\n"
                "        'spanning_cell_ap50': {{name: (None if m['per_label']['table spanning cell']['ap50'] is None else round(m['per_label']['table spanning cell']['ap50'], 4)) for name, m in (('zero_shot', zero_shot_test), ('default', adapted_test), ('experiment', experiment_test))}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history']}}, handle, indent=2, ensure_ascii=False, default=str)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the two validation-loss curves, the `test` rows and the spanning-cell row.\n\n"
                "<details><summary>Check your reasoning</summary>Epochs 0 and 1 of both runs are the copied rows and the trained "
                "heads, so the two curves start from the same models. In the CPU build record all six layers reached validation "
                "loss 0.288 at the second unfreeze epoch (selected) for 85.6 % mAP@0.5 / 73.5 % mAP on the test split, with a "
                "38 MB adapter — no better than two layers. More capacity does not add spanning-cell examples: with 19 in "
                "training, the rarest label stays the weak one. No experiment run is recorded on the release runtime; on 21 "
                "tables, read a difference of a point or two as noise.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 21 held-out scientific tables, in the Kaggle T4 release run (19 September 2026), the grid prior scores 38.7 % "
        "mAP@0.5, the untouched checkpoint 88.7 % (mAP 63.5 %, grid agreement 85.7 %), the restricted heads trained on its "
        "frozen features 86.3 % (mAP 72.0 %), and the last two decoder layers unfrozen with them 87.3 % (mAP 72.6 %, AP@0.75 "
        "76.8 %), selected by validation loss; the adapter reloads to identical query scores. That is the claim and the finding: on a "
        "corpus close to the model's training renders, adaptation does not buy recognition the checkpoint lacks — it buys "
        "**agreement with a box convention** (mean best IoU 0.81 → 0.91, mAP +9.0 points over the checkpoint) and pays for it "
        "on the rarest label (spanning cells, 19 instances in the training split with the default seed and 13 in the test "
        "split: AP@0.5 74.8 % → 61.9 %) and on the structure-level question a caller asks (grid agreement at 0.5: 85.7 % → "
        "76.2 %). The ladder ran all three policies end to end on a "
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
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 unfreezes all six "
        "decoder layers in its own pipeline and prints it beside the default run — change `EXPERIMENT_TRAINABLE_LAYERS`, "
        "`EXPERIMENT_EPOCHS` or `EXPERIMENT_LEARNING_RATE` (at 3e-4 the build record's unfreeze never beat the heads on "
        "validation) there and run that cell again. Setting `EPOCHS = 0` or changing `HEAD_STEPS` / `TRAINABLE_LAYERS` / "
        "`LEARNING_RATE` and choosing **Run after** from Section 6 also starts from the pinned base — every `adapt` restores "
        "it first — but replaces the default results and exports. BYOD: `USE_BYOD` and `BYOD_PATH` in Section 4, then **Run "
        "after** from Section 4, and read the prior and the zero-shot row before any policy.\n\n"
        "## Troubleshooting\n\n"
'- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering `files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and choose **Run all**; leave the optional experiment off on a small runtime.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again; the snapshot comes from `huggingface.co`.\n- **Section 4 reports a size or SHA-256 mismatch for an embedded table** — the notebook file was altered after download; open a fresh copy from the Colab link.\n- **Out of memory** — the model is small; restart the session and choose **Run all**, and leave Section 10 off on a small runtime.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip in the runtime (or attach it as a dataset) and set `BYOD_PATH`.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: "structure.csv line N (file …): names an image that is not in the dataset"** — fix the `file` column of that row, or add the image to the zip.\n- **BYOD: "… has no `group`"** — every row needs the paper, document or source the table comes from.\n- **BYOD: a label outside the four** — use exactly `table`, `table column`, `table row` or `table spanning cell`.\n- **BYOD: "split leaves … supply at least 12 tables in as many groups"** — add tables or groups.\n'
        "## Glossary\n\n"
        "- **DETR query** — one of 125 learned slots the decoder fills with a box and a class distribution; most answer "
        "\"no object\".\n"
        "- **Set loss / Hungarian matching** — predictions are matched one-to-one to reference boxes at the lowest total "
        "cost before the loss is computed.\n"
        "- **GIoU** — generalised IoU, a box-overlap term that still gives a gradient when boxes do not overlap.\n"
        "- **IoU** — intersection over union of two boxes.\n"
        "- **Per-label AP@0.5 / AP@0.75 / AP** — average precision over the ranked predictions of one label, counting a "
        "match at IoU ≥ 0.5, ≥ 0.75, and averaged over 0.5..0.95; threshold-free. mAP is the mean over labels.\n"
        "- **Grid agreement** — the fraction of tables whose surviving row and column counts both match the reference.\n"
        "- **Operating point** — a score threshold and the recall and precision it gives; chosen per deployment.\n"
        "- **Grid prior** — the training split's mean row and column counts laid out uniformly over each table.\n"
        "- **Zero-shot row** — the untouched checkpoint scored on the four labels.\n"
        "- **Copied-head / frozen / unfrozen policy** — the checkpoint's own rows in a restricted head, untrained; the "
        "same heads trained on frozen decoder features; the same plus the last decoder layers trained.\n"
        "- **Derived box convention** — the reference boxes were computed from SciTSR's cells by a stated rule, not drawn "
        "by hand.\n"
        "- **Validation selection** — choosing the policy and epoch by validation loss, never by the test split.\n"
        "- **Group-disjoint split** — every paper (or document, source) on one side of the split.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline gives identical scores.\n"
        "- **BYOD** — bring your own data: your structure-labelled tables through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ on ___ test tables with ___ structure objects (___ spanning cells).\n"
        "- The grid prior scored mAP@0.5 ___, the zero-shot row ___ and the frozen policy ___ (mAP ___).\n"
        "- Validation selected the ___ policy at epoch ___; on the test split it scored mAP@0.5 ___ / AP@0.75 ___ / mAP ___ "
        "and grid agreement ___, so against the checkpoint it ___.\n"
        "- What I would need before claiming adaptation helps: ___ (for example more test tables, several seeds).\n\n"
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
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
