---
license: mit
model_card_spec: "1.1"
pipeline_tag: object-detection
base_model: microsoft/table-transformer-structure-recognition-v1.1-all
date_published: "2023-11-18"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/microsoft/table-transformer-structure-recognition-v1.1-all); the v1.1 models accompany the 2023-03 paper arXiv:2303.00716, but this Transformers-format checkpoint is the 2023-11 Hub release"
---

# Table Transformer Structure Recognition v1.1-all (DIMER package v0.1.0) — Table Structure Recognition (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2Ftable--transformer--structure--recognition--v1.1--all-ffcc4d?style=flat)](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-microsoft%2Ftable--transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/table-transformer)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2303.00716-b31b1b.svg)](https://arxiv.org/abs/2303.00716)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/table-transformer-structure-pipeline/blob/main/tutorials/table_transformer_structure_colab.ipynb) [`table_transformer_structure_colab.ipynb`](https://github.com/kurtvalcorza/table-transformer-structure-pipeline/blob/main/tutorials/table_transformer_structure_colab.ipynb)  
  *Structure recognition on a table crop rendered in code with the pinned `microsoft/table-transformer-structure-recognition-v1.1-all` weights: score-ordered boxes over six structure labels under a caller-owned `threshold`, the implied row×column grid, and per-label `box_iou` against the rendered rows and columns as sanity evidence only — no precision/recall or GriTS.*

---

#### Description

`microsoft/table-transformer-structure-recognition-v1.1-all` is the Transformers-format release of the Table Transformer (TATR) v1.1 structure-recognition model trained on the *aligned* PubTables-1M and FinTabNet.c datasets described in "Aligning benchmark datasets for table structure recognition" (Smock, Pesala and Abraham, arXiv:2303.00716), converted by the Hugging Face team and pinned here to revision `7587a7ef111d9dcbf8ac695f1376ab7014340a0c`. The snapshot `config.json` declares `TableTransformerForObjectDetection` with an in-library ResNet backbone (`backbone_config`: `model_type` resnet, `depths` 2/2/2/2, `hidden_sizes` 64–512 — the ResNet-18 shape; `use_timm_backbone` false): a DETR whose backbone feeds a 6-layer encoder and 6-layer decoder (`d_model` 256, 8 heads, feed-forward width 2048) with DETR's "normalize before" layer-norm placement, and whose decoder holds exactly 125 learned object queries (`num_queries`). At inference the model reads one table image resized so its longest edge is 800 px (`preprocessor_config.json`, `DetrImageProcessor`, ImageNet mean/std) and emits, per query, a normalised box and a softmax over seven outcomes — the six structure classes `table`, `table column`, `table row`, `table column header`, `table projected row header`, `table spanning cell` (`id2label`) and *no object*; the processor keeps the queries whose class score reaches a threshold and maps their boxes back to input pixels. Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `TableTransformerStructurePipeline.from_pretrained` (verified local loading with `trust_remote_code=False`, `use_pretrained_backbone=False`, and the image-processor `size` re-expressed as `{"shortest_edge": 800, "longest_edge": 800}` because the pinned transformers release refuses the checkpoint's one-key form — the same resize), `recognize` (input validation, threshold checks, sorted pixel-space output), `structure_summary` (per-label counts and the implied grid), the `validate_inputs` and `evaluation_report` stage helpers, and `box_iou`.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is table structure recognition: input one image of a single table (`PIL.Image.Image`, any mode, converted to RGB; ideally the detected table cropped with about 10 px of surrounding page, `UPSTREAM_CROP_PADDING`, as the upstream inference script does) and a threshold; output a list of at most 125 structure objects, each an xyxy pixel box, a `label` among the six classes, and the model's softmax `score`, sorted by score, plus `structure_summary`'s per-label counts and the `n_rows × n_columns` grid the row and column boxes imply. Envisioned applications are the second stage of table extraction from born-digital PDFs and reports — after the sibling `table-transformer-detection-pipeline` has located a table, decomposing it into rows, columns, headers and spanning cells so that a caller can intersect the boxes into a cell grid and pair cells with OCR or PDF text — plus layout analysis of financial statements (the FinTabNet.c half of the training data) and scientific tables. Within DIMER the pipeline is an inference component and a zero-configuration baseline for table structure, not a certified extractor for any specific document family.

###### Primary Intended Users

Intended users are machine-learning engineers, document-processing developers, and data analysts integrating table structure recognition into research prototypes, internal enterprise document tooling, or the DIMER workbench. A user is expected to understand that the input must already be a single table (the model emits structure objects for *any* image, including a blank one — it has no notion of "no table here"), that the score is the model's own softmax over six classes and "no object" — a ranking signal within one image, not a calibrated probability on their documents — that the threshold trades recall against spurious rows and columns and must be tuned per document family, that the model was trained on rendered scientific and financial PDF tables so scans, photographs, borderless tables, merged cells and non-Latin scripts are distribution shifts, that building the cell grid (intersecting rows and columns, resolving spanning cells) and reading cell text are the caller's work, and that precision/recall or GriTS can only be measured on a labelled table set they supply. Users who need cell text, an HTML or CSV table, or page-level detection are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no page-level table detection (use the sibling detection pipeline first), no OCR or cell text, no cell-grid assembly, HTML or CSV export, no reading order, no handling of nested tables or of a crop that contains more than one table. `table projected row header` and `table spanning cell` are returned as boxes only; the pipeline does not resolve what they span.
2. **Input boundary:** `recognize` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, and thresholds outside `[0, 1]` (`ValueError`). One image per call; the backend can return at most `MAX_DETECTIONS = 125` objects because the decoder has 125 queries, so a table with more rows plus columns than that is under-described by construction. Every image is resized so its longest edge is 800 px, so rows or columns that are only a few pixels tall or wide at that scale are unlikely to be found.
3. **Input boundary:** the training corpora are PubTables-1M and FinTabNet.c — tables rendered from PubMed Central articles and from annual reports. Scanned or photographed tables (skew, shading, noise), spreadsheet screenshots, hand-drawn tables, tables with heavy merged-cell structure beyond what the six classes express, and non-Latin scripts fall outside what the upstream authors evaluated and what this repository measured; results on them are undefined, not merely degraded. An image that is not a table at all still produces objects (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on extracted table values — automated financial-statement analysis feeding a lending or audit decision, clinical-trial result extraction, regulatory reporting — without a human reviewing the recognised structure and the values read from it, and a locally measured precision/recall on the deployment's own tables.

#### Factors

###### Groups

This pipeline is not human-centric by design: it localises rows, columns and headers in a table image and never classifies, identifies or scores people. The training data (PubTables-1M from PubMed Central Open Access articles; FinTabNet.c from S&P 500 annual reports, per the papers) contains no evaluation groups in the demographic sense, and neither the upstream authors nor this repository audited it for anything of the kind. What does vary is the table population: the corpora are English-language scientific and financial typesetting, so tables from other languages, scripts, publishers, eras or production tools (right-to-left layouts, hand-ruled scans, borderless designer tables, government forms) are the groups whose recall is unknown, not known to be equal. Where tables carry personal data — patient-level results, payroll, customer lists — the pipeline's output makes that data easier to extract; the operator who processes such documents is responsible for a fairness and privacy audit on their own table set, stratified by document family, before relying on the output.

###### Instrumentation

The upstream training data was produced by rendering PDF articles and reports to images and aligning structure annotations from the publishers' XML or from FinTabNet's annotations (with the paper's canonicalisation corrections), so the "instrument" is a PDF renderer over born-digital typesetting: crisp glyphs, straight rules, consistent margins, no sensor noise. Inference images arrive from whatever produced them — a PDF renderer at some DPI, a scanner, a phone camera, or a crop cut by a detector with some padding — and resolution, skew, JPEG artefacts, crop tightness and shading all change the visual evidence; the longest-edge-800 resize (`preprocessor_config.json`, bilinear, ImageNet mean/std) discards detail below that scale on every image regardless of its source. The pipeline validates type and size only; it cannot detect a low-DPI render, a skewed scan, a crop that cut off a column, or a crop that contains two tables. The synthetic tutorial crop (Pillow's bundled font, ruled lines, 10 px margin) is itself a rendering instrument whose glyphs and rule placement differ from PubMed Central or annual-report typesetting — visibly so: the model's column boxes on it sit about 47 px left of the drawn rules, splitting columns by text alignment rather than by the lines.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible but was not exercised for this card. No timm is needed: the backbone is the config's in-library ResNet. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 4-file, 115 MB snapshot 0.06 s, load 4.78 s, a 730×350 rendered table crop 0.12 s, a 4096×4096 blank image 0.37 s — cost is dominated by the fixed 800 px working resolution and the 115 MB model, not by the caller's pixel count. Data environment: the model assumes the image is one rendered table with ruled or aligned rows and columns, as in scientific or financial PDFs, cropped with a little page margin; the synthetic tutorial crop satisfies that assumption and is where the measured behaviour holds. Scans, photographs, borderless or heavily merged tables, over-tight or over-wide crops, and non-Latin scripts violate it to degrees this repository did not measure, and the pipeline reports no signal when they do — nor when the image is not a table at all.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. Each object carries `score`, the query's softmax probability for its winning structure class under the model's own seven-way head — a ranking signal within one image, not a probability that the row or column is real on the deployment's tables and not a measure of correctness. `structure_summary` counts objects per label and reports the `n_rows × n_columns` grid they imply; that is a description of the output, not a metric. The repository ships `box_iou(a, b)`, the intersection-over-union of two xyxy boxes, because it is the primitive every detection metric is built from; per-class precision/recall, mean average precision and the cell-level GriTS scores the upstream paper uses are not implemented, since they need a labelled table set with matching conventions that the caller must choose. To evaluate, the caller supplies ground-truth boxes per label and computes precision/recall at their chosen IoU with `box_iou`, or assembles cells and applies GriTS. The public `evaluation_report(result, ground_truth_boxes=None)` stage returns that report in machine-readable form: one `box_iou` entry per supplied reference box (its best-overlapping detection **of the same label**, plus the reference and detected counts for that label) with the verdict `sample-sanity`, or the verdict `not-measurable` naming the labelled set that would be required when no reference is supplied. The upstream paper's GriTS and AP numbers are upstream-reported and this pipeline does not reproduce or claim them.

###### Decision thresholds

One threshold is applied and exposed as a module constant: `RECOGNITION_THRESHOLD = 0.5` keeps a query only if its softmax score for one of the six structure classes is at least 0.5; below it the query is discarded, and there is no non-maximum suppression beyond what DETR's set prediction already provides. This is the value the upstream repository's inference script applies to every structure class (`structure_class_thresholds` in `src/inference.py`, `main` @ `16d124f`, 2023-09-07); it was not tuned by this repository and is not calibrated for any document family. It can be overridden per call (`recognize(..., threshold=)`), and the smoke run shows the synthetic crop is insensitive to it (the same 15 objects at 0.5 and 0.9, every score above 0.99) while a blank image is not (23 objects at 0.5). A deployment owns tuning it on its own labelled tables: lower the threshold when a missed row or column corrupts the whole grid downstream (a missing row shifts every cell below it), raise it when spurious rows split real ones, and re-tune whenever the document source or the crop convention changes.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, object counts, boxes and scores on one synthetic crop, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable but not guaranteed bitwise-identical across machines, and the synthetic crop's own bytes depend on the Pillow build's bundled font. The `score` is the model's softmax, not a calibrated confidence: on the synthetic crop every one of the 15 objects scored 0.999 or above and so did 23 objects on a blank image, which is exactly what an uncalibrated score looks like. On the synthetic crop the row boxes tracked the drawn rules (`box_iou` 0.88–0.91 for interior rows, 0.65/0.68 for the first and last), the column boxes were offset (`box_iou` 0.53–0.63) and the table box dropped the last column (0.87) — annotation-convention effects between PubTables-1M/FinTabNet.c boxes and hand-drawn rules, not quantities the pipeline estimates. A caller who needs calibrated confidences must fit a calibration map on their own labelled tables; a caller who needs an uncertainty estimate for a metric must supply labelled tables and compute it over many tables or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README states that the model was trained on PubTables-1M and FinTabNet.c; the papers describe those corpora as roughly one million tables from PubMed Central Open Access articles and the FinTabNet tables from S&P 500 annual reports, both re-annotated with the alignment corrections of arXiv:2303.00716, distributed under the articles' open-access licences and FinTabNet's terms. Scientific and financial tables can contain patient-level, survey or personnel data, so personal data in the training corpus is not ruled out; it is not enumerated by the upstream authors and was not audited here. This repository distributes code, tests, and documentation; it does not distribute the 115,437,156-byte `model.safetensors`, which is staged locally under `weights/table-transformer-structure-v1.1-all/` and git-ignored, and it ships no sample documents — the tutorial table is rendered in code. The operator must audit the tables they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will decompose a payroll table as readily as a results table.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — reading clinical-trial result tables for automated evidence synthesis, financial-statement tables for lending, audit or trading decisions, HR or legal tables for screening — would be admissible only with human review of the recognised structure and of every value extracted through it (a missed row silently misaligns every value below it), a locally measured precision/recall on the deployment's own labelled tables, a documented threshold and crop policy, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 4 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False` and `use_pretrained_backbone=False`, and the smoke run loaded with `HF_HUB_OFFLINE=1`. No pickle checkpoint exists at the pinned revision. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported.
- **Input integrity:** the public `validate_inputs(image, *, threshold)` stage applies exactly the checks `recognize` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; thresholds outside `[0, 1]` (and booleans) are rejected; `recognize` raises on a malformed backend object, an unknown label, or more than 125 objects; `evaluation_report` rejects reference labels outside the six classes.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; the image-processor size override is a module constant (`PROCESSOR_SIZE`) named in the card and the notebook; every result carries `model_id`, `model_revision` and the threshold used.
- **Refusals:** no batching, no download without the explicit flag, no threshold default hidden inside the runner, no pickle deserialisation, no attempt to guess whether the input is a table.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Structure on non-table input:** the model emits objects for any image — a blank 4096×4096 image returned 23 objects above the default threshold in the smoke run; an operator who feeds an undetected crop, a figure or a form gets a confident row/column grid of nothing, and every downstream value read through it is fabricated. The pipeline cannot detect this; detect first, then recognise.
- **Missed or split rows and columns:** a missed row shifts every cell below it, a spurious column splits values; the downstream consumer (analytics, compliance extraction) bears the harm; likely on borderless, merged-cell or scanned tables.
- **Convention drift:** the model's boxes follow PubTables-1M/FinTabNet.c annotation conventions (observed: columns split by text alignment, table box dropping a column); a caller who assumes ruled-line boxes gets systematically offset crops.
- **Under-description by construction:** at most 125 objects; a very large table loses rows or columns silently.
- **Automation bias:** clean grids with 0.999 scores invite trust that an uncalibrated softmax has not earned.
- **Privacy exposure:** tables containing personal or confidential data are processed without any content check and made easier to extract.
- **Bias amplification:** any table style the scientific/financial corpora under-represent (non-Latin scripts, non-Western publishers, historical typesetting, government forms) is reproduced as uneven recall, undetected because no per-family evaluation exists.
- **Resource use:** small model (115 MB, ~0.1 s per table on the reference CPU); a document stream can still saturate a shared host.

###### Use cases

Prohibited even where the model would work: decomposing tables in order to extract personal data for surveillance, profiling, social scoring, or unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access; extracting data from documents the operator has no right to process or from paywalled or licence-restricted publications in breach of their terms; deceptive uses that present recognised structure or values read through it as verified document facts or as evidence; and any use that violates the upstream MIT licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the documents processed. Autonomous high-consequence actions triggered by unreviewed extracted values are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `microsoft/table-transformer-structure-recognition-v1.1-all`
- Revision: `7587a7ef111d9dcbf8ac695f1376ab7014340a0c`
- Snapshot manifest: `weights/table-transformer-structure-v1.1-all/dimer-base-manifest.json`, 4 files, `totalBytes` 115515347
- `model.safetensors` SHA-256: `9df416575a3a36ebd0129342d4f597f14d6e5170268f3d52d28584ab4466a501` (115,437,156 bytes)
- `config.json` SHA-256: `17a8a6edfb9e394263fa6ba9b82176ebccdfcc5d6cd29121ec91572c7d6be22c` (76,761 bytes; carries the in-library ResNet `backbone_config`)
- Weight format: SafeTensors; loader `TableTransformerForObjectDetection.from_pretrained(<dir>, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, use_pretrained_backbone=False)` with `AutoImageProcessor` (`DetrImageProcessor`, `size={"shortest_edge": 800, "longest_edge": 800}`) from the same directory. No pickle checkpoint exists at this revision.

## Input/output contract

- `TableTransformerStructurePipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`.
- `recognize(image, *, threshold=0.5) -> dict` with keys `detections` (list of `{"box": [x0, y0, x1, y1], "label": <one of LABELS>, "score": float}` in input-pixel coordinates, sorted by descending score, at most 125 entries), `threshold`, `width`, `height`, `model_id`, `model_revision`.
- `structure_summary(result) -> dict` with `counts` per label, `n_rows`, `n_columns`, `n_cells_implied`, `has_column_header`.
- Ceilings and constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_DETECTIONS = 125`, `LABELS` (six classes in `id2label` order), `RECOGNITION_THRESHOLD = 0.5`, `UPSTREAM_CROP_PADDING = 10`, `PROCESSOR_SIZE = {"shortest_edge": 800, "longest_edge": 800}`.
- `box_iou(a, b) -> float` on xyxy boxes; `validate_inputs(image, *, threshold, names) -> dict`; `evaluation_report(result, ground_truth_boxes=None, *, sample_kind) -> dict` where `ground_truth_boxes` maps a label to its xyxy reference boxes; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing resize so the longest edge is 800 px (expressed to the pinned `DetrImageProcessor` as `{"shortest_edge": 800, "longest_edge": 800}` because it refuses the checkpoint's `{"longest_edge": 800}` form; both caps at 800 make every aspect ratio land on longest edge 800), bilinear, ImageNet mean/std.
- Measured 2026-09-13 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.06 s (4 files, 115 MB); load 4.78 s; `recognize` on a synthetic 730×350 crop (an 8×5 ruled table rendered with Pillow's bundled font, header row plus seven data rows of short tokens, cropped with 10 px of white page) at the default threshold 0.5 → 15 objects in 0.12 s: 1 `table` [17.3, 22.6, 680.1, 330.2], 5 `table column`, 8 `table row`, 1 `table column header`, every score ≥ 0.999; `box_iou` against the drawn boxes: table 0.87, rows 0.65 / 0.90 / 0.91 / 0.91 / 0.89 / 0.89 / 0.88 / 0.68, columns 0.63 / 0.53 / 0.54 / 0.54 / 0.54, column header 0.65 (column boxes offset ~47 px left of the drawn rules; table box drops the last column); same crop at threshold 0.9 → identical 15 objects in 0.07 s; 4096×4096 blank image → 23 objects in 0.37 s.
- Tutorial execution: `tutorials/table_transformer_structure_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 20.3 s, snapshot staged from the Hub by the carried `stage_missing_files`); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA path, half precision, any precision/recall or GriTS measurement against labelled tables, scans or photographs, tables with spanning cells or projected row headers (the synthetic crop has neither, and the smoke run returned zero of each).

## References

- Smock, Pesala, Abraham. Aligning benchmark datasets for table structure recognition. ICDAR 2023. https://arxiv.org/abs/2303.00716
- Smock, Pesala, Abraham. PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents. CVPR 2022. https://arxiv.org/abs/2110.00061
- Carion et al. End-to-End Object Detection with Transformers (DETR). ECCV 2020. https://arxiv.org/abs/2005.12872
- Upstream code (inference script with the per-class thresholds and crop padding): https://github.com/microsoft/table-transformer
- Upstream card: https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all
- Transformers `TableTransformer` documentation: https://huggingface.co/docs/transformers/model_doc/table-transformer
