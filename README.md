# Table Transformer v1.1-all table structure recognition pipeline

DIMER inference wrapper for **Table Transformer — structure recognition v1.1-all** (`microsoft/table-transformer-structure-recognition-v1.1-all`), the DETR model with a ResNet-18 backbone that Microsoft trained on PubTables-1M and FinTabNet.c to decompose a table image into rows, columns, headers and spanning cells, pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline returns pixel-space boxes over six structure labels with the model's softmax score and a summary of the cell grid they imply; it does not detect tables on a page and does not read text.

## Upstream alignment

- Model: `microsoft/table-transformer-structure-recognition-v1.1-all`
- Revision: `7587a7ef111d9dcbf8ac695f1376ab7014340a0c`
- Upstream weight license: MIT
- Upstream task: object detection over six table-structure classes on a table-crop image
- Repository adaptation: bounded supervised adaptation on `{id, image, objects}` records (`adapt`) — the checkpoint's own rows for `table` / `table column` / `table row` / `table spanning cell` copied into restricted heads and scored untouched (zero-shot policy), trained on the frozen DETR decoder features (frozen policy), and an optional unfreeze of the last decoder layers (unfrozen policy), selected among the three by validation set loss — measured by held-out per-label AP@0.5 / AP@0.75 / AP and grid agreement (`evaluate`, `evaluate_zero_shot`, `prior_baseline`); `recognize` and the checkpoint's own heads are unchanged; the base weights are never redistributed and the adapter is a tutorial output

## Quick start

```python
from PIL import Image
from table_transformer_structure_pipeline import TableTransformerStructurePipeline, structure_summary

pipe = TableTransformerStructurePipeline.from_pretrained()   # stages + verifies weights/table-transformer-structure-v1.1-all first
result = pipe.recognize(Image.open("table_crop.png"))          # one table, cropped with ~10 px of page around it
for obj in result["detections"]:                               # sorted by score, boxes are [x0, y0, x1, y1] pixels
    print(obj["label"], obj["box"], round(obj["score"], 3))
print(structure_summary(result))                               # per-label counts, n_rows x n_columns

# the threshold is the upstream inference script's per-class value; override per call
result = pipe.recognize(Image.open("table_crop.png"), threshold=0.9)

# adaptation: the embedded SciTSR-PD tables, a grid prior, the zero-shot row, the three-policy ladder
from table_transformer_structure_pipeline import ADAPT_LABELS, RECOGNITION_THRESHOLD, load_sample_dataset, prior_baseline

splits = load_sample_dataset()                                            # 50 / 23 / 21 {id, image, objects} tables, split by paper
print(prior_baseline(splits["train"], splits["test"], labels=ADAPT_LABELS, threshold=RECOGNITION_THRESHOLD)["map50"])
print(pipe.evaluate_zero_shot(splits["test"])["map50"])                    # the checkpoint's own heads, four labels scored
pipe.adapt(splits["train"], splits["validation"], trainable_layers=0)        # zero-shot rows vs restricted heads on frozen features
print(pipe.adapter["policy"], pipe.evaluate(splits["test"])["map50"])
pipe.adapt(splits["train"], splits["validation"])                          # + the last two decoder layers, selected on validation
print(pipe.adapter["policy"], pipe.evaluate(splits["test"])["map50"])
print(pipe.recognize_adapted(splits["test"][0]["image"])["detections"][:2])
pipe.save_artifact("outputs/tts_adapter")                                  # adapter.safetensors + manifest.json
reloaded = TableTransformerStructurePipeline.from_artifact("outputs/tts_adapter")
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed; 2 model-backed tests run when the snapshot is staged). On a fresh clone the manifest is committed but the weights are not: `TableTransformerStructurePipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/table-transformer-structure-v1.1-all/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (4 files)
  config.json                # 76 KB: carries the in-library ResNet backbone_config
  preprocessor_config.json
  model.safetensors          # git-ignored, 115,437,156 bytes
  README.md
```

## Input ceilings and threshold

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_DETECTIONS = 125` (the checkpoint's `num_queries`), `LABELS = ("table", "table column", "table row", "table column header", "table projected row header", "table spanning cell")`; `RECOGNITION_THRESHOLD = 0.5`; `UPSTREAM_CROP_PADDING = 10` (documentation of the upstream crop convention, not enforced). One table image per call; the processor resizes it so its longest edge is 800 px. See `MODEL_CARD.md` for who owns tuning the threshold, the processor-size note, and the measured CPU timings.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/table-transformer-structure-pipeline/blob/main/tutorials/table_transformer_structure_colab.ipynb)

`tutorials/table_transformer_structure_colab.ipynb` is declared `E2E` (mode `GUIDED`) under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the four pipeline modules (`pipeline.py`, `metrics.py`, `sample_data.py` — the 94 embedded public-domain SciTSR-PD tables with their derived structure boxes — and `samples.py`), the model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`; see `tutorials/README.md`). Its default path stages the missing `model.safetensors` with `stage_missing_files(..., allow_download=True)` and digest-verifies the snapshot, decodes and digest-verifies the embedded tables and draws 50 / 23 / 21 by a seeded split of whole papers with `validate_dataset`, `check_split_disjoint` and `split_summary`, runs one test table through the inference contract with a per-label `sample-sanity` `evaluation_report` against its reference boxes, scores a grid prior and the untouched checkpoint on the test split (mAP@0.5 38.7 % and 88.7 % in the recorded run), runs the three-policy ladder — the checkpoint's own rows untouched, restricted heads on the frozen decoder features (85.2 %), the last two decoder layers unfrozen with them — selected by validation set loss (selected `unfrozen last 2 decoder layers + restricted heads`, 86.8 % / mAP 72.2 %), renders structure before and after, and exports the trained tensors as a safetensors adapter that reloads to identical (query, class) scores. Six `outputs/` artifacts are written. BYOD (a `.zip` with `structure.csv` beside the image files) is optional and gated off by default.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. One local fresh-kernel execution is recorded in `docs/release-verification.md` as pre-flight; the supported-runtime run is pending. Complete that record against the exact release revision before calling the notebook release-grade.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output and adaptation contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, hosting notes and the embedded adaptation corpus.
- `docs/release-verification.md` — the release gate and recorded notebook executions.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are MIT; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
