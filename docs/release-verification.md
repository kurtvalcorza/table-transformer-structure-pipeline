# Release verification

`tutorials/table_transformer_structure_colab.ipynb` (`E2E`, **standalone** carrier) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.0. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `2.0`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on
  the primary path; four cells tagged `embedded_module` equal to
  `src/table_transformer_structure_pipeline/{pipeline,metrics,sample_data,samples}.py` after the generator's
  documented rewrites (in dependency order, relative imports removed); the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `TableTransformerStructurePipeline.from_pretrained(weights_dir=...)`, `load_corpus`, `build_sample_dataset`,
  `validate_dataset`, `check_split_disjoint`, `split_summary`, `write_dataset_csv`, `validate_inputs`, `recognize`,
  `structure_summary`, `evaluation_report`, `prior_baseline`, `evaluate_zero_shot`, `adapt` with
  `trainable_layers=0` and with `TRAINABLE_LAYERS` / `LEARNING_RATE`, `evaluate`, `recognize_adapted`,
  `save_artifact`, `from_artifact` with the parity assertion), the ceiling print (`MIN_IMAGE_SIDE`,
  `MAX_IMAGE_SIDE`, `MAX_DETECTIONS`, `NUM_QUERIES`, `D_MODEL`, `DECODER_LAYERS`, `PARAMETER_COUNT`), the six
  exports, the learner-facing statements (copied-head zero-shot, frozen and unfrozen policy, lowest validation loss, grid
  prior, untouched checkpoint, per-label AP, grid agreement, `sample-sanity`, no dispersion estimate, the
  checkpoint's heads never trained, capability exclusions, the corpus licence) and the gated-off BYOD default
  listed in the validator; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a
  mutable `revision='main'`, direct `from transformers import` / `TableTransformerForObjectDetection`
  / `AutoImageProcessor` / `from huggingface_hub import` / `from safetensors` / `torch.optim` /
  `.backward(` / `last_hidden_state` / `pred_boxes` / `scipy` / `base64` / `pyarrow` / `pipe._model` use
  **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_adaptation.py`,
`tests/test_notebook_parity.py`; injected runner, synthetic tables, the embedded sample, no weights;
`tests/test_model_backed.py` skips without the staged snapshot). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy, Pillow or torch that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/table-transformer-structure-v1.1-all/` (the standalone path writes the manifest itself and stages
   the missing file from the Hub; the 94 tables travel inside the notebook, so nothing else is fetched);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `HEAD_STEPS = 300`, `HEAD_LR = 1e-3`, `EPOCHS = 3`,
   `LEARNING_RATE = 1e-4`, `TRAINABLE_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0` (an interpreter restart after the install is expected
   where the runtime's preinstalled torch, numpy or Pillow differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the four carried module cells execute (defining `TableTransformerStructurePipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `structure_summary`, `box_iou`, `structure_metrics`,
     `prior_baseline`, `SAMPLE_RECORDS`, `SAMPLE_IMAGES_B64`, `load_corpus`, `build_sample_dataset`, `validate_dataset`,
     `check_split_disjoint`, `split_summary`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv` and the
     ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `microsoft/table-transformer-structure-recognition-v1.1-all` at the immutable revision, and `verify_snapshot`
     returning its dict (4 files); `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory;
   - Section 4: `load_corpus` decoding the 94 embedded tables after their 94 digest checks, and the seeded draw of
     10 / 7 / 29 whole papers into 21 / 23 / 50 test, validation and training tables (300 / 351 / 636 objects) with
     `check_split_disjoint` reporting no shared table and no shared paper, `split_summary` printed and the three
     dataset digests `d189a035…` / `20cf1d03…` / `4deb0f90…`; `outputs/…_train.csv` written; the four
     dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_DETECTIONS` 125) and the contract
     (`NUM_QUERIES` 125, `D_MODEL` 256, `DECODER_LAYERS` 6, `PARAMETER_COUNT` 28,828,619) surfaced;
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding
     from the out-of-range-threshold probe); `recognize` on one test table at `RECOGNITION_THRESHOLD` with the
     four sanity checks and `structure_summary`; `evaluation_report` against the table's reference boxes with
     verdict **`sample-sanity`** and one `box_iou` entry per reference label; the rendered preview displayed;
   - Section 6: the grid prior (≈ 38.7 % mAP@0.5) on the 21 test tables, `pipe.evaluate_zero_shot` on the
     untouched checkpoint (≈ 88.7 % mAP@0.5, ≈ 63.5 % mAP, grid agreement ≈ 85.7 %), then
     `pipe.adapt(..., trainable_layers=0)` scoring the checkpoint's own rows untouched (validation loss ≈ 0.771)
     and training the restricted heads for 300 steps (≈ 0.295; policy `frozen backbone, encoder and decoder +
     restricted heads`) and `pipe.evaluate` (≈ 85.2 % mAP@0.5, ≈ 71.5 % mAP) with per-image rows and the cell's
     assertion that the heads beat the prior;
   - Section 7: `pipe.adapt` printing epoch 0 (the copied rows, untrained) and epoch 1 (the heads) on validation, then 3
     epochs of the last two decoder layers (3,157,504 + 133,897 trainable of 28,828,619 parameters) with validation
     loss / mAP@0.5 / mAP / grid agreement each epoch (0.771 → 0.295 → 0.310 → 0.282 → 0.286 in the recorded run)
     and the selected policy `unfrozen last 2 decoder layers + restricted heads` (`best_epoch` 3);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison (prior, zero-shot,
     frozen policy, selected policy) per label and as class means, grid agreement, per-image rows and
     `outputs/…_evaluation_report.json` written (the cell asserts the selected model beats the prior — on the
     sample ≈ 86.8 % versus 38.7 % mAP@0.5; the deltas over the heads and the checkpoint are reported, not
     asserted);
   - Section 9: three test tables rendered with `recognize`, the heads-only structure (a fresh pipeline adapted
     with `trainable_layers=0`) and the selected model's `recognize_adapted` at 0.5, `outputs/…_preview.png`
     written; `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors, manifest.json}` (60 tensors,
     about 13.2 MB with two trained decoder layers — 8 tensors, about 0.5 MB when the heads or the untouched rows
     are selected; `policy` recorded) and `TableTransformerStructurePipeline.from_artifact` reloading it with
     identical (query, class) scores on three tables and an identical test mAP@0.5 (the cell asserts both);
     `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the snapshot block
     (`weight_format`, `weight_sha256`), the `corpus` block, the inference-contract items with the single-table
     report and implied grid, the comparison, the before/after rows, the artifact digest and policy, the reload
     parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model identifier
   and immutable revision, whether the model cache and the weights directory were clean, outcome, produced outputs,
   the observed metrics and the selected policy (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `table_transformer_structure_colab.ipynb` (`E2E`) | `c88210b` / `dea5747f` | 2026-10-09 | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (isolated `uv` CPython 3.12.12 environment, `cuda:0`) | **PASSED** — 14/14 code cells, 0 errors, no restart, 448.0 s; evidence `docs/execution-evidence/2026-10-09-c88210b/`; REL12 BYOD not exercised |
| `table_transformer_structure_colab.ipynb` (`E2E`) | `4c56fa0` / `4686cf7d` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-table-transformer-structure` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED only after a manual restart** — 12/12 code cells ok (1 restart after install cell), so not a one-pass `Run all` and not promotion evidence (REL2/RUN10; review finding TTS-M1); 10 files, 116 MB staged from the Hub into a clean cache; comparison {map50: {grid_prior: 0.3868, zero_shot: 0.8873, frozen_policy: 0.863, selected_policy: 0.8728}, map75: {grid_prior: 0.2514, zero_shot: 0.7339, frozen_policy: 0.7616, selected_policy: 0.7675}, map: {grid_prior: 0.2621, zero_shot: 0.6352, frozen_policy: 0.7197, selected_policy: 0.7255}, mean_best_iou: {grid_prior: 0.5173, zero_shot: 0.8127, frozen_policy: 0.9112, selected_policy: 0.9081}, grid_exact_at_threshold: {grid_prior: 0.0476, zero_shot: 0.8571, frozen_policy: 0.8095, selected_policy: 0.7619}, ap50[table]: {grid_prior: 0.9184, zero_shot: 0.9365, frozen_policy: 1, selected_policy: 1}, recall_at_0.5[table]: {grid_prior: 0.9524, zero_shot: 0.9524, frozen_policy: 1, selected_policy: 1}, ap50[table column]: {grid_prior: 0.3525, zero_shot: 0.9998, frozen_policy: 1, selected_policy: 1}, recall_at_0.5[table column]: {grid_prior: 0.5048, zero_shot: 1, frozen_policy: 1, selected_policy: 1}, ap50[table row]: {grid_prior: 0.2761, zero_shot: 0.8647, frozen_policy: 0.8505, selected_policy: 0.872}, recall_at_0.5[table row]: {grid_prior: 0.4596, zero_shot: 0.8758, frozen_policy: 0.9006, selected_policy: 0.913}, ap50[table spanning cell]: {grid_prior: 0, zero_shot: 0.748, frozen_policy: 0.6016, selected_policy: 0.6191}, recall_at_0.5[table spanning cell]: {grid_prior: 0, zero_shot: 0.6154, frozen_policy: 0.5385, selected_policy: 0.5385}, loss: {frozen_policy: 0.5084, selected_policy: 0.5121}, delta_vs_frozen: {map50: 0.0097, map75: 0.0059, map: 0.0058}, delta_vs_zero_shot: {map50: -0.0145, map75: 0.0336, map: 0.0903}, selected_policy: unfrozen last 2 decoder layers + restricted heads}; reload parity {queries_identical: True, map50_in_memory: 0.8728, map50_reloaded: 0.8728, classes_identical: True}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-table-transformer-structure/v2/evidence/` in the workspace |
| `table_transformer_structure_colab.ipynb` (`E2E`) | `5d70d67` / `39b2b5fc` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `table_transformer_structure_colab.ipynb` (`TASK-INFERENCE`, superseded) | `2f07026` / `fe2059b8bffa` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-table-transformer-structure` v1) | PASSED — 8/8 code cells, 211.8 s, 10 files, 116 MB staged; evidence for the earlier inference-only notebook — not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/table_transformer_structure_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/table_transformer_structure_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-09 | `c88210b` / `dea5747f` | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4; kernel Python 3.13.15, isolated `uv` CPython 3.12.12 environment from the 47-package hash lock (setup 62 s), `cuda:0` | Default sample path, one pass from a fresh runtime, no repository checkout (blob verified before execution); BYOD and Section 10 off | 448.0 s | **PASSED** — 14/14 code cells, 0 errors, no restart; split 50 / 23 / 21 tables (spanning cells 19 / 39 / 13); probe `sample-sanity`, implied grid 10 × 7; mAP@0.5 grid prior 0.3868 / checkpoint 0.8873 / frozen 0.8630 / selected 0.8728; AP@0.75 0.2514 / 0.7339 / 0.7616 / 0.7675; mAP 0.2621 / 0.6352 / 0.7197 / 0.7255; spanning-cell AP@0.5 0 / 0.748 / 0.6016 / 0.6191; grid agreement 0.0476 / 0.8571 / 0.8095 / 0.7619; selected `unfrozen last 2 decoder layers + restricted heads` at `best_epoch` 2 (adapt 120.6 s), `started_from` `pinned base`; reload parity exact (0.872763 both ways); evidence `docs/execution-evidence/2026-10-09-c88210b/` (executed notebook SHA-256 `7b83af6e5022…`). Optional journeys not exercised |
| 2026-09-19 | `4c56fa0` / `4686cf7d` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-table-transformer-structure` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 553.1 s | **PASSED only after a manual restart** — 12/12 code cells ok (1 restart after install cell), so not a one-pass `Run all` and not promotion evidence (REL2/RUN10; review finding TTS-M1); 10 files, 116 MB staged from the Hub into a clean cache; comparison {map50: {grid_prior: 0.3868, zero_shot: 0.8873, frozen_policy: 0.863, selected_policy: 0.8728}, map75: {grid_prior: 0.2514, zero_shot: 0.7339, frozen_policy: 0.7616, selected_policy: 0.7675}, map: {grid_prior: 0.2621, zero_shot: 0.6352, frozen_policy: 0.7197, selected_policy: 0.7255}, mean_best_iou: {grid_prior: 0.5173, zero_shot: 0.8127, frozen_policy: 0.9112, selected_policy: 0.9081}, grid_exact_at_threshold: {grid_prior: 0.0476, zero_shot: 0.8571, frozen_policy: 0.8095, selected_policy: 0.7619}, ap50[table]: {grid_prior: 0.9184, zero_shot: 0.9365, frozen_policy: 1, selected_policy: 1}, recall_at_0.5[table]: {grid_prior: 0.9524, zero_shot: 0.9524, frozen_policy: 1, selected_policy: 1}, ap50[table column]: {grid_prior: 0.3525, zero_shot: 0.9998, frozen_policy: 1, selected_policy: 1}, recall_at_0.5[table column]: {grid_prior: 0.5048, zero_shot: 1, frozen_policy: 1, selected_policy: 1}, ap50[table row]: {grid_prior: 0.2761, zero_shot: 0.8647, frozen_policy: 0.8505, selected_policy: 0.872}, recall_at_0.5[table row]: {grid_prior: 0.4596, zero_shot: 0.8758, frozen_policy: 0.9006, selected_policy: 0.913}, ap50[table spanning cell]: {grid_prior: 0, zero_shot: 0.748, frozen_policy: 0.6016, selected_policy: 0.6191}, recall_at_0.5[table spanning cell]: {grid_prior: 0, zero_shot: 0.6154, frozen_policy: 0.5385, selected_policy: 0.5385}, loss: {frozen_policy: 0.5084, selected_policy: 0.5121}, delta_vs_frozen: {map50: 0.0097, map75: 0.0059, map: 0.0058}, delta_vs_zero_shot: {map50: -0.0145, map75: 0.0336, map: 0.0903}, selected_policy: unfrozen last 2 decoder layers + restricted heads}; reload parity {queries_identical: True, map50_in_memory: 0.8728, map50_reloaded: 0.8728, classes_identical: True}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-table-transformer-structure/v2/evidence/` in the workspace |
| 2026-09-19 | `5d70d67` / `39b2b5fc` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → four carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 4 entries because the snapshot was pre-staged → `verify_snapshot` 4 files → `from_pretrained` on CPU → `load_corpus` decoded the 94 embedded tables after their digest checks → 21 / 23 / 50 drawn by whole papers with `check_split_disjoint` clean, 300 / 351 / 636 objects, digests `4deb0f90…` / `20cf1d03…` / `d189a035…` → four dataset refusals → input manifest with the threshold refusal → `recognize` on `test-000` (SciTSR `1702.02925v1.7`, a 10 × 7 grid, 0.13 s) with all four sanity checks `True`, 19 objects at 0.5 including one `table column header`, implied grid 10 × 7 → `evaluation_report` **`sample-sanity`**, 18 `box_iou` entries (table 0.93, rows 0.71–0.78, columns 0.77–0.97) → grid prior → zero-shot → the zero-shot-vs-heads call (54.3 s) → the full ladder (84.3 s) → validation + test evaluation → before/after render + preview → adapter export → reload parity) | 212.9 s | **PASSED** — 12/12 code cells; grid prior mAP@0.5 0.3868 / mAP 0.2621 (grid agreement 0.0476); zero-shot 0.8873 / AP@0.75 0.7339 / mAP 0.6352 (columns 1.000, rows 0.865, spanning cells 0.748; grid agreement 0.857; mean best IoU 0.813); frozen policy 0.8525 / 0.7613 / 0.7148 (validation loss 0.2945 against 0.7705 for the copied rows; set loss 0.518); `adapt`: 133,897 head + 3,157,504 decoder-layer parameters of 28,828,619, validation loss 0.7705 (copied rows, untrained) → 0.2945 (heads) → 0.3104 → 0.2818 → 0.2860 with mAP@0.5 0.970 → 0.922 → 0.924 → 0.923 → 0.923, selected `unfrozen last 2 decoder layers + restricted heads` at `best_epoch` 3; **selected policy on the test split 0.8684 / 0.7748 / 0.7222 (Δ +0.016 mAP@0.5, +0.014 AP@0.75, +0.007 mAP vs the heads; −0.019 / +0.041 / +0.087 vs the checkpoint), grid agreement 0.762, mean best IoU 0.901**; three test tables' row and column counts matched the reference under every policy; adapter 13,172,204 B / 60 tensors, SHA-256 `baca5641…`; reload parity exact ((query, class) scores identical, mAP@0.5 0.868382 both ways); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `2f07026` / `fe2059b8bffa` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-table-transformer-structure` v1) | Default sample path of the inference-only notebook: synthetic 730×350 crop, `stage_missing_files` fetching the four manifest entries from the Hub, `verify_snapshot`, `recognize` → 15 objects, `evaluation_report` `sample-sanity` with per-label `box_iou`, 5 exports | 211.8 s | **PASSED** — 8/8 code cells, 10 files, 116 MB staged; history only |
| 2026-09-13 | working tree of the initial build (blob `36e39919473b`, commit `0c5c4f4`; `TASK-INFERENCE`, superseded) | Local Windows-venv harness, Python 3.12, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells | 20.3 s | PASS — pre-flight only; history |

## Current status

**Candidate.** The 2026-09-19 Kaggle T4 run of `4c56fa0` / blob `4686cf7d` needed a manual restart after the install cell, so it is not a one-pass `Run all` and not promotion evidence (review finding TTS-M1), so the Release-grade label that rested on it is withdrawn. For the 2026-10-05 review-fix revision, the current blob `dea5747fafbe` (commit `c88210b`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-09 (Colab CLI sequential execution, 14/14 code cells, 448.0 s; isolated `uv` environment, 47 locked packages; test mAP@0.5 checkpoint 0.8873 / frozen 0.8630 / selected 0.8728, mAP 0.6352 / 0.7197 / 0.7255; reload parity exact). The REL12 BYOD journeys have not been run on a hosted runtime; until they are, the status stays Candidate. Any later change to the carried modules or to the notebook produces a new blob that needs its own recorded run.

## Supplemental table intelligence workshop — `tutorials/DIMER_Table_Intelligence_Workshop.ipynb`

This entry applies only to the supplemental workshop notebook, not the primary tutorial executions above.

### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_Table_Intelligence_Workshop.ipynb) and authorized merging PR #8 (merge commit `f1886fe`). The file is archived byte-for-byte, SHA-256 `c8fed54721db86a76bb3f591fcf37de4f05bd6fc7866756d63cc7502cda16dd5`. All 29 code cells have execution counts, 49 saved outputs and zero saved errors. Code-cell sources match commit `50446bee9a233d0c93b88567bbaa9af002c5ebc0`, tutorial blob `3b570ea3f4bd116a51baccbbb4d28cf000638bd6`, apart from Colab-inserted `# @title` lines. Later commits on `main` that touch the notebook (`a7cf06b` (AI Use Disclosure)) change only markdown cells; its code cells are identical to the executed revision. This evidence commit does not change tutorial code.

Scope: Default path: 10 canonical tables and 50 questions through Table Transformer detection, Table Transformer structure recognition, grid reconstruction and TAPAS question answering. BYOD was not exercised.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, Transformers 4.57.6, timm 1.0.30, datasets 4.1.1, NumPy 2.1.3 (preloaded by the host kernel, retained), CUDA Tesla T4. Execution reaches the final completion summary. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Saved counts run sequentially from 1 to 29; runtime freshness and absence of manual restarts/reruns are not independently established by the artifact.

Results (sample-sanity measures on the built-in data, not general model rankings): Detection AP50 0.900, hit@0.50 0.900, mean IoU 0.778; exact structure grids 10/10 on gold crops and 9/10 on detected crops; valid reconstructed tables 10/10 (gold) and 8/10 (detected); TAPAS denotation accuracy 0.800 on gold tables, 0.800 structure-only and 0.600 end to end.

Status remains **Candidate**. Merge approval and this successful default-path run do not close the optional-path (FULL/BYOD) or REL12 qualification gates, and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored (editing it would change the verified blob).

### Correction after the 2026-10-02 notebook review (TBL-M1)

The 2026-09-26 run's "exact structure grids … 9/10 on detected crops" counted the page with no detection (`1108.4723v1.4`, `det_crop_pred_shape [0, 0]`) as an exact grid, because an empty prediction matched an empty set of reference boxes. Its saved per-table output shows 8 exact grids among the 9 detected crops (`1611.09238v1.1` gained a column). Under the corrected scoring in revision `0.2.0-candidate` that run would read **8/10** exact detected-crop grids; no other recorded number changes. The fix changes notebook code, so the 2026-09-26 run is not evidence for `0.2.0-candidate`: a new hosted run on the fix head is required, covering the default path, the threshold activity rerun after `Run all`, and BYOD with one valid folder and one rejected input (REL12). See [`reviews/2026-10-02-notebook-review/`](reviews/2026-10-02-notebook-review/DIMER_Table_Intelligence_Workshop_Review.md).

### 2026-10-03 uv isolated environment (revision 0.3.0-candidate)

Applies only to `DIMER_Table_Intelligence_Workshop.ipynb`: notebook blob `52b465d4a3b842b85a75136a0dde014be69e5234` (revision 0.2.0-candidate, branch head `a043211`) → `179d8620f0ed0607585a3ecf43dcd99787282575` (revision 0.3.0-candidate).

- **Change.** The in-kernel `pip install` and its "Restart session, then Run all" guard are removed. The notebook carries `tools/table_intelligence_workshop.py` (every stage; the former cell code moved unchanged apart from call-time defaults and state carried between stage processes) and `tools/table-intelligence-workshop-requirements.lock` (`uv pip compile --generate-hashes` for CPython 3.12 on `x86_64-manylinux_2_28`) in carrier cell `uvcarrier`, written by `tools/build_table_intelligence_workshop.py` in pieces of at most 1,000 characters (longest notebook line 1,183 characters). The runtime cell verifies a pinned uv 0.12.15 wheel by size and SHA-256, builds a CPython 3.12.12 venv (`--managed-python`) and installs with `--require-hashes --only-binary :all:`; every stage runs as a separate process with that venv's interpreter, `MPLBACKEND=Agg`, and `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`/HF tokens dropped. The kernel only displays the stage functions and what each stage prints. A warm runtime reuses the venv when its recorded lock hash matches.
- **Pins.** Unchanged: torch 2.14.0, torchvision 0.29.0, torchaudio 2.11.0, transformers 4.57.6, timm 1.0.30, safetensors 0.8.0, numpy 2.5.3, pillow 11.3.0, huggingface-hub 0.36.2, datasets 4.1.1, pyarrow 25.0.1, pandas 2.2.3. Every package resolves to a cp312 manylinux or pure-Python wheel; no sdist.
- **User-visible.** Linux x86_64 only (Google Colab, Kaggle, Linux Jupyter); Windows and macOS kernels are refused. No restart. No metric, split, seed, model or output file changes.
- **Local verification (not clean-runtime evidence).** A CPU harness on Windows executed all 30 code cells in order with the real pinned model snapshots and SciTSR-PD shards. Stand-ins: the platform check was patched, and a local venv with the lock's direct pins (CPU builds of torch 2.14.0 / torchvision 0.29.0) replaced the uv venv. The environment stage checked a lock listing that venv's versions. The default path reproduced the 2026-09-26 Colab values with the TBL-M1 correction: the same 10 canonical tables; detection AP50 0.900, hit@0.50 0.900, mean IoU 0.778; exact grids 10/10 gold-crop and **8/10** detected-crop (`1108.4723v1.4` detection miss, `1611.09238v1.1` one extra column); valid reconstructions 10/10 and 8/10; TAPAS denotation 0.800 / 0.800 / 0.600 (40 / 40 / 30 of 50). `pytest` 88 passed / 1 skipped before, 107 passed / 1 skipped after.
- **Hosted default-path run: passed; optional paths pending.** The hosted Colab T4 run of this blob passed on the default path, including the uv bootstrap and the hash-locked install (see [Colab CLI execution of revision 0.3.0-candidate — 2026-10-03](#colab-cli-execution-of-revision-030-candidate--2026-10-03)). Still pending: the threshold activity rerun after `Run all`, and BYOD with one valid folder and one rejected input (REL12). Status remains **Candidate**.

### Colab CLI execution of revision 0.3.0-candidate — 2026-10-03

- **Evidence.** [`execution-evidence/2026-10-03/DIMER_Table_Intelligence_Workshop_4e19c6e_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_Table_Intelligence_Workshop_4e19c6e_colab-cli-t4.ipynb), archived byte-for-byte, SHA-256 `fd4eb32fb123086ad8adb5c4a019ec5e640c7418ce153244d82e326fbe15ebe6`.
- **Revision.** Branch head `4e19c6ec1d83a48ae07d53a1fd45516fbf3317c5`, notebook blob `179d8620f0ed0607585a3ecf43dcd99787282575` (0.3.0-candidate), downloaded from GitHub at the PR head and blob-verified before the session.
- **Executor.** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/30 … 30/30, sequential and complete). Session stopped after the run.
- **Path exercised.** Default path only: 10 canonical tables, 50 questions. The threshold activity ran once inside the run (validation exact grids 23/23 at 0.50, 20/23 at 0.95, 3 tables changed), but the rerun after `Run all` was not run; BYOD was not run (`BYOD disabled on canonical Run all path.`).
- **Environment.** The uv bootstrap succeeded on the hosted runtime: carried stage file and hash lock verified; an isolated CPython 3.12.12 venv at `/content/work/env` was built and the hashed lock installed (`lock_pins_checked` 66) with `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `transformers 4.57.6`, `timm 1.0.30`, `datasets 4.1.1`, `numpy 2.5.3` (from the hash lock), `pandas 2.2.3`, `pillow 11.3.0`, `cuda:0` Tesla T4. No restart. The saved outputs carry no per-cell timings; the venv build and install fall inside the 226.0 s wall time of the whole run.
- **Outcome.** **PASSED** — 30/30 code cells, no errors; 226.0 s wall.
- **Results.** Same 10 canonical tables as the 2026-09-26 run, in the same order; split 21 / 23 / 50 tables (10 / 7 / 29 papers); carrier geometry SHA-256 `d6d524c3…`. Detection AP50 0.900, hit@0.50 0.900, mean IoU 0.778; exact grids 10/10 gold-crop and **8/10** detected-crop (`1108.4723v1.4` detection miss, `1611.09238v1.1` predicted 4 × 6 against 4 × 5); valid reconstructions 10/10 and 8/10; TAPAS denotation 0.800 / 0.800 / 0.600 (40 / 40 / 30 of 50).
- **Comparison with the 2026-09-26 run.** Every recorded value is equal, including the per-table rows, the threshold activity and the gold/detected mean row and column IoU, except the detected-crop exact-grid count: 9/10 there, **8/10** here. The difference is the TBL-M1 grading fix (a detection miss no longer counts as an exact grid), not a model change; the 2026-09-26 run reads 8/10 under the corrected scoring, as stated above. Mean detection latency differs (0.057 s here against 0.091 s per page), as expected between sessions.
- **Boundary.** Saved outputs were inspected; the journeys not exercised (threshold activity rerun after `Run all`, BYOD/REL12) remain open. Status remains **Candidate**; `metadata.dimer.clean_runtime_evidence` is unchanged.
