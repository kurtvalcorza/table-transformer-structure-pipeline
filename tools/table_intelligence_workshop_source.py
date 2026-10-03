# ruff: noqa: E501,I001
"""Source cells for the DIMER Table Intelligence workshop."""
CELLS = [
  {
    "kind": "markdown",
    "source": "# DIMER Table Intelligence Workshop\n## Table Detection → Structure Recognition → Table Reconstruction → TAPAS QA\n\n**Notebook profile:** `TASK-INFERENCE`  \n**Pedagogical mode:** `WORKSHOP`  \n**DIMER Notebook Specification:** `2.1`  \n**Workflow scope:** `COMPOSED-PIPELINE`  \n**Standalone:** yes  \n**Canonical workflow:** frozen inference only\n\nThis notebook treats Table Intelligence as a system: three frozen models joined by notebook-local composition rules.\n\n### Who this notebook is for\n\nLearners who can run cells in Colab or Jupyter and read short Python functions, and who are new to composed document-intelligence pipelines. No prior experience with Table Transformer or TAPAS is assumed. A Tesla T4-class GPU is recommended; the notebook also runs on CPU, more slowly (TAPAS Large dominates the run time). It runs on **Linux x86_64 only** (Google Colab, Kaggle or a Linux Jupyter server), because it builds its own isolated Python environment there; Windows and macOS kernels are refused with a message.\n\n### How to use this notebook\n\n1. Select a T4 GPU runtime (Colab: *Runtime → Change runtime type → T4 GPU*).\n2. Choose *Runtime → Run all*. The defaults in the Configuration form (§3) are the reference settings; change a setting only in the activity marked **Predict → Change one thing** (after §18). The first run spends several minutes building an isolated environment (§4); nothing is installed into the notebook's own Python, and `Run all` needs no restart.\n3. While the notebook runs, read the concept cells and write down your prediction whenever a cell asks for one, before its output appears.\n4. Each section is labelled by role:\n   - **Core concept** — what the models do and how the stages fit together;\n   - **Evaluation practice** — how each stage is measured and how to read the numbers;\n   - **Infrastructure** — runtime setup, pinned downloads, integrity checks, data plumbing and exports. You may run these cells without studying their implementation; their code is collapsed where your notebook viewer supports it.\n5. Checkpoint questions have a collapsed **Sample answer**. Answer first, then open it.\n\n### Task at a glance\n\n```text\nInput:  a document page containing one scientific table + a natural-language question\n  → Table Transformer Detection       where is the table on the page?\n  → crop\n  → Table Transformer Structure       where are its rows and columns?\n  → grid reconstruction + cell text   which string belongs in which cell?\n  → TAPAS Large WTQ                   which cells answer the question, with which aggregation?\nOutput: an answer (cell strings or a number) + per-stage metrics showing where errors entered\n```\n\nModels:\n\n1. **Table Transformer Detection**\n2. **Table Transformer Structure Recognition v1.1-all**\n3. **TAPAS Large WTQ**\n\nThe canonical corpus is the public-domain **SciTSR-PD** scientific-table dataset. Its logical cell text is used as a controlled text provider so this notebook can isolate table geometry and QA without mixing in OCR quality.\n\n### Roadmap\n\n1. Distinguish the three table-intelligence tasks and predict where errors will enter (§1–2).\n2. Set up the runtime, models and SciTSR-PD inputs (§3–10, mostly Infrastructure).\n3. Run table detection and measure it (§11–13).\n4. Recognize structure on gold and detected crops, reconstruct grids and measure them (§14–18); then change the structure threshold on validation tables (activity).\n5. Probe tables with spanning cells (§19).\n6. Ask 50 questions through three paths with TAPAS and score them (§20–22).\n7. Read the error waterfall and trace failures to their stage (§23–24).\n8. Inspect resources, panels and exports (§25–28).\n9. Write an evidence-based conclusion (§29); troubleshooting and a glossary follow.\n\n### Learning objectives\n\nBy the end you should be able to:\n\n- **distinguish** table detection, structure recognition and table QA, naming each one's input and output;\n- **reconstruct** a rectangular cell grid from predicted row and column boxes;\n- **trace** a box through the page → crop → table coordinate frames;\n- **explain** why TAPAS needs cell strings rather than boxes, and where OCR would supply them in production;\n- **compare** gold-table, structure-only and end-to-end QA accuracy on the same questions;\n- **diagnose** a wrong answer by locating the stage in the error waterfall where it first went wrong;\n- **predict, then test,** how one structure-threshold change moves predicted row and column counts.\n\n> **Important boundary:** Table Transformer models recover geometry, not text. The canonical workflow uses SciTSR annotation text, not OCR inference. OCR / Document Extraction is the next notebook in the Document Intelligence track.\n\n**AI Use Disclosure:** Generative AI assisted with this notebook’s code and instructional content under maintainer direction. The maintainer remains responsible for review, validation, and release decisions. AI-generated material may contain errors; validation claims are limited to documented runs and configurations. AI use does not imply independent verification, provider endorsement, or release approval."
  },
  {
    "kind": "markdown",
    "source": "## 1. Three different tasks\n\n> **Core concept**\n\n### Table detection\n**Where is the table on the page?**\n\n### Structure recognition\n**Where are rows, columns and spanning cells inside the table?**\n\n### Table QA\n**What does the reconstructed table say?**\n\nA correct final answer depends on all upstream interfaces."
  },
  {
    "kind": "markdown",
    "source": "## 2. Error waterfall\n\n> **Core concept**\n\nThe same TAPAS questions are evaluated through three paths:\n\n```text\nPath 1:\ngold logical table → TAPAS\n\nPath 2:\ngold table crop → predicted structure → reconstructed table → TAPAS\n\nPath 3:\npage → predicted table crop → predicted structure → reconstructed table → TAPAS\n```\n\nThis separates QA-model error from structure error and table-detection error.\n\n### Before you run: make predictions\n\nWrite down short answers now; §24 returns to them with the measured waterfall and sample answers.\n\n1. Which stage is most likely to reduce end-to-end accuracy?\n2. Can perfect table detection guarantee correct structure?\n3. Can perfect structure guarantee correct QA?\n4. Why does TAPAS need strings rather than row/column boxes?\n5. Where would OCR enter a production version of this workflow?\n6. What happens if one predicted row is missing?\n7. What happens if the first reconstructed row is mistaken?"
  },
  {
    "kind": "markdown",
    "source": "## 3. Configuration\n\n> **Configuration** — the form below holds the reference settings. Keep them for `Run all`; the activity after §18 is the place to experiment.\n\nThe defaults define the canonical `Run all` path."
  },
  {
    "kind": "code",
    "source": "USE_BYOD = False  # @param {type:\"boolean\"}\nBYOD_PATH = \"\"  # @param {type:\"string\"}\n\nDETECTION_THRESHOLD = 0.90  # @param {type:\"number\"}\nSTRUCTURE_THRESHOLD = 0.50  # @param {type:\"number\"}\nCROP_PADDING = 10  # @param {type:\"integer\"}\nGRID_NMS_IOU = 0.50  # @param {type:\"number\"}\n\n# The QA set is fixed: ten test tables, five questions each (50 questions). These two are not form fields.\nEND_TO_END_TABLES = 10\nQUESTIONS_PER_TABLE = 5\n\nRUN_COMPLEX_STRUCTURE_PROBE = True  # @param {type:\"boolean\"}\nCOMPLEX_STRUCTURE_TABLES = 3  # @param {type:\"integer\"}\n\nOUTPUT_DIR = \"outputs/table_intelligence\"\nWORK_DIR = \"work\"  # isolated environment, carried stage file, stage logs and state\n\nif not 0 <= DETECTION_THRESHOLD <= 1:\n    raise ValueError(\"DETECTION_THRESHOLD must be in [0,1]\")\nif not 0 <= STRUCTURE_THRESHOLD <= 1:\n    raise ValueError(\"STRUCTURE_THRESHOLD must be in [0,1]\")\nif not (isinstance(CROP_PADDING,int) and 0 <= CROP_PADDING <= 100):\n    raise ValueError(f\"CROP_PADDING must be a whole number of pixels in [0,100], got {CROP_PADDING!r}\")\nif not 0 < GRID_NMS_IOU <= 1:\n    raise ValueError(f\"GRID_NMS_IOU must be in (0,1], got {GRID_NMS_IOU!r}\")\nif not (isinstance(COMPLEX_STRUCTURE_TABLES,int) and 0 <= COMPLEX_STRUCTURE_TABLES <= 10):\n    raise ValueError(f\"COMPLEX_STRUCTURE_TABLES must be a whole number in [0,10], got {COMPLEX_STRUCTURE_TABLES!r}\")\nif END_TO_END_TABLES != 10:\n    raise ValueError(\"END_TO_END_TABLES is fixed at 10: the canonical QA set is ten test tables and 50 questions\")\nprint({\n    \"detection_threshold\":DETECTION_THRESHOLD,\n    \"structure_threshold\":STRUCTURE_THRESHOLD,\n    \"crop_padding\":CROP_PADDING,\n    \"tables\":END_TO_END_TABLES,\n})"
  },
  {
    "kind": "markdown",
    "source": "## 4. Runtime\n\n> **Infrastructure** — environment, pinned downloads, integrity checks and data plumbing. You may run the next cell without studying its implementation; its code is collapsed where your notebook viewer supports it.\n\nThe model code does not run in this notebook's own Python. The next two cells:\n\n1. write two carried files under `work/runner/` and check their SHA-256: `table_intelligence_workshop.py`, which holds every function and stage of this workshop, and `requirements.lock.txt`, a hash lock of every package (the same versions this notebook pinned before: torch 2.14.0, transformers 4.57.6, timm 1.0.30, numpy 2.5.3, datasets 4.1.1, pandas 2.2.3 and the rest);\n2. download a pinned `uv` (0.12.15, checked by size and SHA-256), create a CPython 3.12.12 virtual environment in `work/env` and install the lock with `uv pip install --require-hashes --only-binary :all:` — only wheels whose hashes are in the lock.\n\nEach later section runs one stage of the carried file as a separate process with that environment's Python (`run_stage`), and shows the functions it runs (`show_source`). Objects that pass from one section to the next (tables, crops, predictions) are saved under `work/state/` between stages. Nothing is installed into this kernel, so `Run all` needs no restart, and a second `Run all` reuses the environment.\n\nThis works on **Linux x86_64 only** (Google Colab, Kaggle, a Linux Jupyter server). `datasets` and `pyarrow` are used only to read the small pinned SciTSR-PD parquet files.\n\nA Tesla T4 is recommended because TAPAS Large is approximately 1.35 GB."
  },
  {
    "kind": "code",
    "id": "uvcarrier",
    "carrier": True,
    "infrastructure": True,
    "source": ""
  },
  {
    "kind": "code",
    "source": "import ast\nimport hashlib\nimport io\nimport json\nimport os\nimport platform\nimport subprocess\nimport time\nimport urllib.error\nimport urllib.request\nimport zipfile\n\nif platform.system() != \"Linux\" or platform.machine() != \"x86_64\":\n    raise RuntimeError(\"This notebook builds a Linux x86_64 environment: use Google Colab, Kaggle \"\n                       \"or a Linux x86_64 Jupyter server (Windows and macOS kernels are not supported).\")\ntry:\n    gpu = subprocess.run([\"nvidia-smi\", \"--query-gpu=name\", \"--format=csv,noheader\"], capture_output=True, text=True)\n    print(\"GPU:\", gpu.stdout.strip() if gpu.returncode == 0 else \"none detected\")\nexcept FileNotFoundError:\n    print(\"GPU: none detected; the canonical runtime is a T4 (CPU works but is much slower).\")\n\nENV_ROOT = WORK_ROOT / \"env\"\nPYTHON = ENV_ROOT / \"bin\" / \"python\"\nLOCK_PATH = RUNNER_ROOT / \"requirements.lock.txt\"\nSTAGE_SCRIPT = RUNNER_ROOT / \"table_intelligence_workshop.py\"\nENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN=\"1\", HF_HUB_DISABLE_TELEMETRY=\"1\", DO_NOT_TRACK=\"1\")\n# The kernel's own settings must not leak into the isolated environment: hosted kernels export an\n# inline plotting backend and a PYTHONPATH that exist only in the kernel.\nfor name in (\"HF_TOKEN\", \"HUGGING_FACE_HUB_TOKEN\", \"PYTHONPATH\", \"PYTHONHOME\", \"PYTHONSTARTUP\"):\n    ENV.pop(name, None)\nENV[\"MPLBACKEND\"] = \"Agg\"\n\nbuilt_from = ENV_ROOT / \".dimer-lock-sha256\"\nif PYTHON.is_file() and built_from.is_file() and built_from.read_text().strip() == CARRIED_HASHES[\"requirements.lock.txt\"]:\n    print(\"Reusing the isolated environment built from this lock:\", ENV_ROOT)\nelse:\n    UV_URL = \"https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl\"\n    UV_SHA256 = \"aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60\"\n    for attempt in range(3):\n        try:\n            with urllib.request.urlopen(UV_URL, timeout=90) as response:\n                wheel = response.read(20081405)\n            break\n        except (urllib.error.URLError, TimeoutError, ConnectionError):\n            if attempt == 2:\n                raise\n            time.sleep(2 ** attempt)\n    if len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != UV_SHA256:\n        raise RuntimeError(\"uv wheel size/hash mismatch\")\n    UV = WORK_ROOT / \"uv\"\n    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:\n        member = next(n for n in archive.namelist() if n.endswith(\".data/scripts/uv\"))\n        UV.write_bytes(archive.read(member))\n    UV.chmod(0o700)\n    print(\"Creating the isolated Python 3.12.12 environment...\", flush=True)\n    subprocess.run([str(UV), \"venv\", \"--clear\", \"--managed-python\", \"--python\", \"3.12.12\", str(ENV_ROOT)], env=ENV, check=True)\n    print(\"Installing the hashed dependency lock (several minutes; CUDA wheels are large)...\", flush=True)\n    subprocess.run([str(UV), \"pip\", \"install\", \"--python\", str(PYTHON), \"--require-hashes\", \"--only-binary\", \":all:\",\n                    \"--index-url\", \"https://pypi.org/simple\", \"-r\", str(LOCK_PATH)], env=ENV, check=True)\n    built_from.write_text(CARRIED_HASHES[\"requirements.lock.txt\"])\n\n\ndef run_stage(stage, *options):\n    \"\"\"Run one stage of the carried file with the isolated interpreter; stream and keep its log.\"\"\"\n    config = {\n        \"use_byod\": USE_BYOD, \"byod_path\": BYOD_PATH,\n        \"detection_threshold\": DETECTION_THRESHOLD, \"structure_threshold\": STRUCTURE_THRESHOLD,\n        \"crop_padding\": CROP_PADDING, \"grid_nms_iou\": GRID_NMS_IOU,\n        \"end_to_end_tables\": END_TO_END_TABLES, \"questions_per_table\": QUESTIONS_PER_TABLE,\n        \"run_complex_structure_probe\": RUN_COMPLEX_STRUCTURE_PROBE,\n        \"complex_structure_tables\": COMPLEX_STRUCTURE_TABLES,\n        \"output_dir\": OUTPUT_DIR, \"work_dir\": str(WORK_ROOT),\n    }\n    config_path = WORK_ROOT / \"stage_config.json\"\n    config_path.write_text(json.dumps(config, indent=2), encoding=\"utf-8\")\n    log = WORK_ROOT / \"logs\" / (stage + \".log\")\n    log.parent.mkdir(parents=True, exist_ok=True)\n    command = [str(PYTHON), \"-u\", str(STAGE_SCRIPT), \"--config\", str(config_path), \"--stage\", stage, *options]\n    with log.open(\"w\", encoding=\"utf-8\") as output:\n        process = subprocess.Popen(command, env=ENV, cwd=os.getcwd(), stdout=subprocess.PIPE,\n                                   stderr=subprocess.STDOUT, text=True)\n        for line in process.stdout:\n            output.write(line)\n            output.flush()\n            print(line.rstrip(\"\\n\"), flush=True)\n        process.wait()\n    if process.returncode:\n        # The stage's own output, including its traceback, is printed above; its last line names the problem.\n        last = [line for line in log.read_text(encoding=\"utf-8\").splitlines() if line.strip()][-1:] or [\"no output\"]\n        raise RuntimeError(f\"{stage} failed with exit {process.returncode}: {last[0]} (full log: {log})\")\n\n\nfrom IPython.display import Markdown, display\n\n\ndef show_source(*names):\n    \"\"\"Display functions of the carried stage file (they run in the isolated environment, not here).\"\"\"\n    text = STAGE_SCRIPT.read_text(encoding=\"utf-8\")\n    found = {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body\n             if isinstance(node, ast.FunctionDef)}\n    display(Markdown(\"```python\\n\" + \"\\n\\n\".join(found[name] for name in names) + \"\\n```\"))\n\n\nrun_stage(\"environment\")\n",
    "infrastructure": True
  },
  {
    "kind": "markdown",
    "source": "## 5. Immutable model provenance\n\n> **Infrastructure** — environment, pinned downloads, integrity checks and data plumbing. You may run the next cell without studying its implementation; its code is collapsed where your notebook viewer supports it.\n\nEvery model is staged only from its exact immutable revision and verified against the DIMER manifest before loading.\n\nNo upstream `.bin` file is used."
  },
  {
    "kind": "code",
    "source": "# Stage each model from its immutable revision and verify every file's size and SHA-256 against the DIMER manifest.\nrun_stage(\"models\")",
    "infrastructure": True
  },
  {
    "kind": "markdown",
    "source": "## 6. SciTSR-PD provenance\n\n> **Infrastructure** — environment, pinned downloads, integrity checks and data plumbing. You may run the next cell without studying its implementation; its code is collapsed where your notebook viewer supports it.\n\nPinned dataset:\n\n`bevaya/SciTSR-pd @ dae336efa7af07d69194a510ff5568980e8ef253`\n\nThe dataset contains 108 public-domain scientific tables (89 train / 19 test) with:\n\n- table image;\n- logical cells;\n- text chunks;\n- paper identity;\n- source license metadata.\n\nThe two source parquet files are verified by byte size and SHA-256 before they are read."
  },
  {
    "kind": "code",
    "source": "# Download the two pinned SciTSR-PD parquet shards and verify their size and SHA-256.\nrun_stage(\"dataset\")",
    "infrastructure": True
  },
  {
    "kind": "markdown",
    "source": "## 7. Build the table-geometry/text companion from SciTSR-PD\n\n> **Infrastructure** — environment, pinned downloads, integrity checks and data plumbing. You may run the next cell without studying its implementation; its code is collapsed where your notebook viewer supports it.\n\nThe source dataset stores logical cells (with their text) and PDF-coordinate text chunks, but no pixel boxes. The notebook combines two things:\n\n- **Pinned carrier geometry.** The DIMER structure carrier derived pixel-space `table` / `table row` / `table column` / `table spanning cell` boxes for 94 of the 108 SciTSR-PD tables, once, and recorded the whole-pixel offset that places each table's text chunks on its image. The carried stage file (§4) embeds that geometry and its SHA-256; every stage checks the digest when it starts, and the next cell prints it.\n- **Annotation text.** Each logical cell's text is matched to its PDF text chunks, and the chunk boxes are placed on the image with the carrier's offset. This gives every cell a pixel box, its row/column span and its text.\n\nThis is a deterministic data transformation, not model inference and not OCR. The notebook stops if any carrier table fails to rebuild, if its pixels differ from the carrier's, or if its cell spans disagree with the carrier's row and column counts."
  },
  {
    "kind": "code",
    "source": "# The pinned carrier geometry (94 tables) and its SHA-256 are embedded in the carried stage file; every stage\n# checks the digest on start. This prints the table count and the digest.\nrun_stage(\"geometry\")",
    "infrastructure": True
  },
  {
    "kind": "code",
    "source": "# Rebuild the table-geometry/text companion from the verified shards (stage_companion in the carried file).\nrun_stage(\"companion\")",
    "infrastructure": True
  },
  {
    "kind": "markdown",
    "source": "## 8. Paper-disjoint split\n\n> **Evaluation practice**\n\nThe corpus is split by paper, not by individual table: tables from one paper never appear in two splits.\n\nThe notebook reproduces the carrier's split exactly (seed 42, papers drawn in the order test → validation → train):\n\n| split | papers | tables |\n|---|---:|---:|\n| test | 10 | 21 |\n| validation | 7 | 23 |\n| train | 29 | 50 |\n\nThe notebook performs no training, so the 50 training tables are never used. The ten canonical QA tables (§9) and the spanning-cell probe (§19) come from the 21 held-out **test** tables. The threshold activity after §18 uses the 23 **validation** tables, so experimenting with it never touches the test results. The next cell stops if the counts differ."
  },
  {
    "kind": "code",
    "source": "show_source(\"stage_split\")\nrun_stage(\"split\")"
  },
  {
    "kind": "markdown",
    "source": "## 9. Select ten QA-eligible rectangular tables\n\n> **Evaluation practice**\n\nThe full end-to-end QA workflow intentionally excludes spanning-cell tables.\n\nEligibility:\n\n- no spanning cells;\n- 3–20 rows;\n- 2–12 columns;\n- complete first row with unique non-empty headers;\n- complete first body column with unique row identifiers;\n- at least one numeric body column;\n- cells ≤200 characters.\n\nSelection is deterministic and independent of model output."
  },
  {
    "kind": "code",
    "source": "show_source(\"parse_number\", \"eligible\", \"stage_select\")\nrun_stage(\"select\")"
  },
  {
    "kind": "markdown",
    "source": "## 10. Create deterministic document pages\n\n> **Evaluation practice**\n\nThe real SciTSR table pixels are placed unchanged on a simple white page.\n\nThis creates measurable page-level table-detection ground truth without introducing a second table."
  },
  {
    "kind": "code",
    "source": "show_source(\"make_page\", \"stage_pages\")\nrun_stage(\"pages\")"
  },
  {
    "kind": "markdown",
    "source": "## 11. Common detection/structure metrics\n\n> **Evaluation practice**\n\nBounding-box comparisons use pixel-space IoU.\n\nAP is Pascal-VOC-style all-point AP over score-ranked predictions with one-to-one matching."
  },
  {
    "kind": "code",
    "source": "# These metric functions run inside the isolated environment; they are shown here and used from §13 on.\nshow_source(\"box_iou\", \"ap_for_label\")"
  },
  {
    "kind": "markdown",
    "source": "## 12. Stage 1 — Table Transformer Detection\n\n> **Core concept**\n\nThe detector runs on the synthetic page. The highest-scoring surviving `table` / `table rotated` detection becomes the target crop.\n\nA miss is a real pipeline failure and propagates downstream.\n\n**Question tested:** does a detector trained on document pages find a real scientific table placed on a plain synthetic page?\n\n**Predict:** how many of the ten pages will yield a `table` detection at threshold 0.90? Will the best-box IoU be close to 1.0?"
  },
  {
    "kind": "code",
    "source": "show_source(\"load_detection_model\", \"detect_table\", \"stage_detection\")\nrun_stage(\"detection\")"
  },
  {
    "kind": "markdown",
    "source": "## 13. Detection metrics\n\n> **Evaluation practice**\n\nEach page has exactly one ground-truth table box.\n\nThe tutorial reports AP50/AP75, hit rate and best-box IoU across the ten pages."
  },
  {
    "kind": "code",
    "source": "show_source(\"stage_detection_metrics\")\nrun_stage(\"detection-metrics\")"
  },
  {
    "kind": "markdown",
    "source": "### What to notice — detection\n\n- `hit50` is the fraction of pages whose best detection overlaps the placed table with IoU ≥ 0.50. A page with no detection above the threshold is a **miss**; every later stage records it as `detection_miss` instead of dropping it, so it still counts against end-to-end accuracy.\n- Best-box IoU stays below 1.0 even for a good detection. The reference box is the whole pasted SciTSR image, including its white margin, while the detector tends to hug the ruled table. The red boxes in the §26 panels show the difference.\n- AP50 and AP75 are close when every hit is tightly localized; a large gap between them would mean loose boxes."
  },
  {
    "kind": "markdown",
    "source": "## 14. Build gold and detected crops\n\n> **Core concept**\n\nThe structure recognizer is evaluated twice:\n\n- **gold crop** — exact known table location, with 10 px page padding;\n- **detected crop** — predicted box, with the same padding.\n\nGround-truth structure boxes are transformed into each crop's coordinate frame.\n\n### Three coordinate frames\n\nEvery stage works in its own frame, so a box has to be translated as it moves through the pipeline:\n\n- **table frame**: pixels of the original SciTSR table image, where the carrier's reference boxes live;\n- **page frame**: the synthetic page; §10 pastes the table at `(120, 160)`, so page = table + (120, 160);\n- **crop frame**: the padded crop given to the structure model; crop = page − (crop left, crop top).\n\nThe next cell builds both crops and prints one reference row box in all of these frames.\n\n**Checkpoint: trace a box.** A row box is `[3, 30, 560, 58]` in the table frame, and the detected crop starts at page pixel `(112, 150)`. Where is the box in the page frame and in the detected-crop frame? Work it out, then compare your method with the printed trace.\n\n<details>\n<summary>Sample answer</summary>\n\nPage frame: add the paste offset, `[3+120, 30+160, 560+120, 58+160] = [123, 190, 680, 218]`. Detected-crop frame: subtract the crop origin, `[123−112, 190−150, 680−112, 218−150] = [11, 40, 568, 68]`. A box that ends up partly outside the crop is clipped to it, which is how a tight detection can cut a row or column out of the structure model's view.\n\n</details>"
  },
  {
    "kind": "code",
    "source": "show_source(\"padded_crop\", \"translate_gold_objects\", \"to_crop_frame\", \"stage_crops\")\nrun_stage(\"crops\")"
  },
  {
    "kind": "markdown",
    "source": "## 15. Stage 2 — Table Transformer Structure Recognition\n\n> **Core concept**\n\nRaw model output is preserved.\n\nRows and columns are later subjected to deterministic same-label NMS for grid reconstruction. NMS is a composition rule of this notebook, not part of the model.\n\n**Question tested:** does structure recognition depend on the crop it is given?\n\n**Predict:** will the gold-crop path or the detected-crop path produce more exact grids (row count and column count both right)?"
  },
  {
    "kind": "code",
    "source": "show_source(\"load_structure_model\", \"recognize_structure\", \"timed_structure\", \"stage_structure\")\nrun_stage(\"structure\")"
  },
  {
    "kind": "markdown",
    "source": "## 16. Grid reconstruction\n\n> **Core concept**\n\nRows are sorted top-to-bottom; columns left-to-right.\n\nSame-label duplicate boxes are removed with NMS at IoU 0.50 before grid construction."
  },
  {
    "kind": "code",
    "source": "show_source(\"nms_items\", \"grid_from_structure\", \"stage_grid\")\nrun_stage(\"grid\")"
  },
  {
    "kind": "markdown",
    "source": "## 17. Assign SciTSR source cell text to predicted cells\n\n> **Core concept**\n\nFor each gold logical cell, its pixel center is transformed into the current crop and located in the predicted row and column.\n\nNo gold row/column index is supplied to the reconstruction algorithm."
  },
  {
    "kind": "code",
    "source": "show_source(\"source_cells_in_crop\", \"contains\", \"reconstruct_table\", \"stage_reconstruct\")\nrun_stage(\"reconstruct\")"
  },
  {
    "kind": "markdown",
    "source": "## 18. Structure and reconstruction metrics\n\n> **Evaluation practice**\n\nRows/columns are scored independently on gold crops and detected crops.\n\nHeader predictions are not penalized because SciTSR-PD does not annotate the two PubTables header classes used by the structure checkpoint."
  },
  {
    "kind": "code",
    "source": "show_source(\"refs_for\", \"preds_for\", \"stage_structure_metrics\")\nrun_stage(\"structure-metrics\")"
  },
  {
    "kind": "markdown",
    "source": "### What to notice — structure\n\n- `grid_exact` is the gate for reconstruction: when the row or column count is wrong, text is assigned to the wrong cells no matter how good the boxes look.\n- Mean row/column IoU compares each reference row or column with its best-matching prediction. Exact counts with IoU well below 1.0 are normal here: the carrier's reference rows and columns tile the whole table from mid-gap to mid-gap (the PubTables-1M convention), while predicted boxes follow the model's own boundaries. Boundary offsets matter only when a cell's center falls on the wrong side of a boundary.\n- The detected-crop path can lose a table entirely or gain or lose a row or column when the crop clips or adds margin. A page with no detection is scored as a failed detected-crop grid (`status` = `detection_miss` in `structure_metrics.csv`), so the detected-crop exact-grid count can never exceed the number of detection hits.\n- Column-header and projected-row-header predictions are not scored, because SciTSR-PD does not annotate them.\n\n### Checkpoint — structure\n\n1. A predicted grid has the right number of columns but one row too few. What happens to the text of the cells in the missing row?\n2. Why can a detected crop change the structure result even when its detection IoU is above 0.8?\n\n<details>\n<summary>Sample answer</summary>\n\n1. The centers of the missing row's cells fall into a neighbouring predicted row, and their text is merged into that row's cells, or into no row at all, and they become unmapped. Either way the reconstructed table no longer says what the source table says, so questions about that row can fail although TAPAS itself is unchanged.\n2. The structure model sees a different image. The crop's margins, and therefore its scale after resizing to 800 px, change, and a clipped rule or header line removes evidence. IoU measures overlap with the reference box, not whether every rule and header line is inside the crop.\n\n</details>\n\n### Activity — Predict → Change one thing → Run → Observe → Explain\n\nThis activity uses the 23 **validation** tables only, so the canonical test-table results above and the exported files are unchanged.\n\n1. **Predict.** If the structure threshold rises from 0.50 to `ACTIVITY_STRUCTURE_THRESHOLD`, will the model report more rows and columns, fewer, or the same? Which kinds of tables will change?\n2. **Change one thing.** Set `ACTIVITY_STRUCTURE_THRESHOLD` in the next cell (default 0.95; then try 0.98 and 0.30).\n3. **Run** the cell. It recognizes structure on each validation table's gold crop at both thresholds. You can rerun it as often as you like after `Run all`: each run is a separate process that loads the structure model from the verified snapshot and unloads it when it finishes.\n4. **Observe** the exact-grid counts and the tables whose row/column counts changed.\n5. **Explain** the tradeoff: what does a higher threshold remove, and what does that cost?\n\n<details>\n<summary>Sample answer</summary>\n\nA higher threshold keeps only the more confident boxes, so it can only remove predictions. On these clean, ruled scientific tables the model is confident about nearly every real row and column, so lowering the threshold changes little, while raising it toward 1.0 starts removing real rows or columns — typically short or sparse ones — and the grid comes out a row or column short. On noisier documents a lower threshold can also admit spurious or duplicated boxes that NMS must then suppress. Because reconstruction needs exact counts, the useful threshold is the one that gets counts right most often on held-out validation tables. That is why the experiment runs on validation tables, not on the test tables the notebook reports.\n\n</details>"
  },
  {
    "kind": "code",
    "source": "# Change one thing: the structure threshold, on validation tables only (canonical outputs are unaffected).\nACTIVITY_STRUCTURE_THRESHOLD = 0.95  # @param {type:\"number\"}\nif not 0 <= ACTIVITY_STRUCTURE_THRESHOLD <= 1:\n    raise ValueError(\"ACTIVITY_STRUCTURE_THRESHOLD must be in [0,1]\")\n\n# Each run is its own process: it loads the verified structure snapshot, runs both thresholds and unloads it.\nshow_source(\"grid_counts\", \"stage_activity\")\nrun_stage(\"activity\", \"--threshold\", str(ACTIVITY_STRUCTURE_THRESHOLD))"
  },
  {
    "kind": "markdown",
    "source": "## 19. Complex spanning-cell probe\n\n> **Core concept**\n\nThe first three held-out spanning-cell tables are run through structure recognition on their gold crops. Each panel in `complex_probe/` shows the gold spanning cells beside the predicted rows, columns and spanning cells.\n\nThey are intentionally excluded from rectangular TAPAS reconstruction because simple row × column intersections cannot faithfully represent their semantics."
  },
  {
    "kind": "code",
    "source": "show_source(\"outline\", \"stage_complex_probe\")\nrun_stage(\"complex-probe\")"
  },
  {
    "kind": "markdown",
    "source": "## 20. Generate 50 table-QA questions from the gold logical tables\n\n> **Evaluation practice**\n\nEach table contributes:\n\n- 2 lookup (`NONE`) questions;\n- 1 `SUM`;\n- 1 `AVERAGE`;\n- 1 `COUNT`.\n\nGeneration uses only the gold logical table and happens independently of model predictions."
  },
  {
    "kind": "code",
    "source": "show_source(\"gold_table_dict\", \"numeric_columns\", \"stage_questions\")\nrun_stage(\"questions\")"
  },
  {
    "kind": "markdown",
    "source": "## 21. Stage 3 — TAPAS\n\n> **Core concept**\n\nTAPAS receives a string table and a question. It selects cells at threshold 0.5 and predicts one aggregation operator.\n\nFor `SUM`, `AVERAGE` and `COUNT`, the notebook computes the numeric answer from selected cells exactly as the DIMER reference carrier does.\n\n**Question tested:** how much QA accuracy is lost when TAPAS reads a reconstructed table instead of the gold one?\n\n**Predict:** rank the three paths (gold table, structure-only, end-to-end) by denotation accuracy, and estimate the gap between the first and the last."
  },
  {
    "kind": "code",
    "source": "# TAPAS is loaded by the QA stage in the next cell, in the same process that answers the questions.\nshow_source(\"load_tapas\", \"compute_numeric\", \"table_to_df\", \"tapas_answer\")"
  },
  {
    "kind": "markdown",
    "source": "## 22. QA scoring\n\n> **Evaluation practice**\n\nDenotation correctness is the headline downstream metric:\n\n- lookup: same multiset of normalized selected cell strings;\n- numeric aggregation: numeric answer matches within `1e-6`.\n\nGold-table evaluation also reports aggregation and exact selected-coordinate accuracy."
  },
  {
    "kind": "code",
    "source": "show_source(\"norm_cell\", \"denotation_correct\", \"score_path\", \"gold_provider\", \"struct_provider\", \"full_provider\", \"qa_summary\", \"stage_qa\")\nrun_stage(\"qa\")"
  },
  {
    "kind": "markdown",
    "source": "### What to notice — QA\n\n- The gold-table path is TAPAS's own ceiling on these questions. Its errors are QA-model errors (wrong cells or wrong aggregation), not pipeline errors.\n- The structure-only path matches the gold path only when every gold-crop reconstruction is valid and puts each cell's text in the right place; any drop below the gold path was introduced by structure recognition or reconstruction.\n- The end-to-end path additionally pays for detection misses and detected-crop structure errors. Questions whose table could not be reconstructed count as wrong (`pipeline_status` records why), so all three paths share the same 50-question denominator.\n- A lookup answer counts as correct when the selected cells are right, even if TAPAS also attached an aggregation operator, so aggregation accuracy can sit below denotation accuracy."
  },
  {
    "kind": "markdown",
    "source": "## 23. Pipeline waterfall\n\n> **Evaluation practice**\n\nThe waterfall counts failures rather than dropping them from downstream denominators."
  },
  {
    "kind": "code",
    "source": "show_source(\"stage_waterfall\")\nrun_stage(\"waterfall\")"
  },
  {
    "kind": "markdown",
    "source": "### What to notice — waterfall\n\nRead the waterfall top to bottom. Detection, structure and reconstruction are counts of tables (out of 10); the QA block gives each path's accuracy and its number of correct answers (out of 50, five per table). Along the end-to-end path, the counts generally shrink from detection hits, to exact detected-crop grids, to valid detected-crop tables, to correct answers. The step with the largest drop is where most of the end-to-end loss entered. Compare each gold-crop count with its detected-crop count to separate structure errors from errors caused by the crop."
  },
  {
    "kind": "markdown",
    "source": "## 24. Checkpoint — trace the losses\n\n> **Evaluation practice**\n\nReturn to the predictions you wrote down in §2. The next cell prints, for each canonical table, the detection IoU, whether each path produced a valid table, the predicted grid shapes, and the cell-assignment accuracy on each crop. Compare them with the waterfall above.\n\n1. Which stage reduced end-to-end accuracy the most in this run?\n2. Can perfect table detection guarantee correct structure?\n3. Can perfect structure guarantee correct QA?\n4. Why does TAPAS need strings rather than row/column boxes?\n5. Where would OCR enter a production version of this workflow?\n6. What happens if one predicted row is missing?\n7. What happens if the first reconstructed row is mistaken?\n\n<details>\n<summary>Sample answer</summary>\n\n1. Look for the largest step in the waterfall. When the gold-crop path is nearly perfect, the loss enters with detection (misses) and with detected-crop structure errors (a row or column gained or lost through the crop).\n2. No. Structure recognition sees only the crop; margins, clipped rules and faint rows can still change the predicted grid.\n3. No. The gold-table path is the ceiling: TAPAS still selects wrong cells or the wrong aggregation on some questions with a perfect table.\n4. TAPAS is a language model over a table of strings: it embeds cell text together with row/column position ids. Boxes carry no words, so something must place text into the grid first.\n5. Between structure recognition and reconstruction: OCR or PDF text extraction supplies the words and their boxes, which are then assigned to predicted cells. This notebook uses SciTSR annotation text in that position.\n6. The texts of that row merge into a neighbouring row or become unmapped; the table changes meaning, and questions about the row fail.\n7. The first row becomes the header. With a wrong or duplicate header the table is invalid for TAPAS here; with a plausible but wrong header, questions that name a column select the wrong one.\n\n</details>"
  },
  {
    "kind": "code",
    "source": "show_source(\"stage_checkpoint\")\nrun_stage(\"checkpoint\")"
  },
  {
    "kind": "markdown",
    "source": "## 25. Resource comparison\n\n> **Evaluation practice**\n\nModels are loaded sequentially. The table records model size, load time, stage latency and peak accelerator memory where available.\n\nLatency units differ by stage, so compare them with care: detection is timed per page, structure per crop and TAPAS per question (`latency_unit`). Peak memory is reset before each stage is measured. These are measurements from this runtime only (the printed `device` and `gpu_name`)."
  },
  {
    "kind": "code",
    "source": "show_source(\"median\", \"stage_resources\")\nrun_stage(\"resources\")"
  },
  {
    "kind": "markdown",
    "source": "## 26. Qualitative four-stage panels\n\n> **Evaluation practice**\n\nEach canonical table gets a panel in `examples/` showing:\n\n1. synthetic document page + predicted table box;\n2. detected crop + row/column structure;\n3. reconstructed table text;\n4. one representative TAPAS result.\n\nVisualizations supplement machine-readable outputs."
  },
  {
    "kind": "code",
    "source": "show_source(\"draw_boxes\", \"stage_panels\")\nrun_stage(\"panels\")"
  },
  {
    "kind": "markdown",
    "source": "## 27. Machine-readable exports\n\n> **Infrastructure** — environment, pinned downloads, integrity checks and data plumbing. You may run the next cell without studying its implementation; its code is collapsed where your notebook viewer supports it.\n\nOutputs include stage metrics, reconstructed tables, QA records, waterfall, resource metrics and provenance."
  },
  {
    "kind": "code",
    "source": "# Write the CSV/JSON exports, per-table JSON and provenance under OUTPUT_DIR (stage_exports in the carried file).\nrun_stage(\"exports\")",
    "infrastructure": True
  },
  {
    "kind": "markdown",
    "source": "## 28. BYOD (optional)\n\n> **Optional** — skipped on the canonical `Run all` path (`USE_BYOD = False`). To use it, set `USE_BYOD = True` and `BYOD_PATH` in §3, then choose `Run all` again; the sample path runs first.\n\nA production table-intelligence workflow needs text to populate predicted cells. This notebook does not run OCR itself, so BYOD takes page images plus OCR/text boxes you produced elsewhere, and optional questions.\n\nFolder layout:\n\n```text\ndataset/\n  pages/\n    page001.png        # .png .jpg .jpeg .webp .tif .tiff .bmp; each side 16..4096 px\n  ocr.csv\n  questions.csv        # optional\n```\n\n`ocr.csv`, one row per word or text box:\n\n```text\npage_id,text,x0,y0,x1,y1,order\npage001,Method,130.0,172.5,196.0,190.0,1\n```\n\n- `page_id` is the image file name without its extension (`page001` for `pages/page001.png`).\n- `x0,y0,x1,y1` are **page-image pixels with the origin at the top-left corner**, as the image is stored (no rescaling). Convert PDF points or a bottom-left origin first: a box outside the image is rejected with the page and lines named.\n- `order` is an integer reading order; words that fall in the same cell are joined in this order.\n\n`questions.csv` (optional) has `id,page_id,question` and an optional `accepted_answer`. A numeric accepted answer is compared numerically (within `1e-6`); anything else is compared as normalized text.\n\nWhat happens: each page goes through the same detection, crop, structure, grid-reconstruction and TAPAS steps as the sample path, and OCR words are assigned to predicted cells by word-centre containment. Every row of both CSV files is validated before a model runs, and an error names the file, line and field. A page without OCR rows, or with an unsupported file type, is listed as skipped with its reason. The cell prints each reconstructed table and answer, and writes `outputs/table_intelligence/byod/byod_results.json` plus one CSV per reconstructed table under `byod/tables/`; earlier BYOD outputs are removed first.\n\n### Privacy\n\nYour files stay inside this notebook runtime: this branch sends nothing to an external service and writes only under `outputs/table_intelligence/byod/`. Tables may contain financial, personal, employee, medical or proprietary information. Do not upload restricted or regulated documents to a hosted runtime unless you are authorized to process them there."
  },
  {
    "kind": "code",
    "source": "# Optional BYOD: runs only when USE_BYOD is True (§3). Every CSV row is validated before any model loads.\nshow_source(\"accepted_answer_matches\", \"read_byod_csv\", \"stage_byod\")\nrun_stage(\"byod\")"
  },
  {
    "kind": "markdown",
    "source": "## 29. Interpretation and limitations\n\n### Table Intelligence is a pipeline\n\nDetection, structure, text extraction, reconstruction and QA are separable failure points.\n\n### Geometry is not text\n\nTable Transformer predicts boxes. A real deployment still needs PDF text extraction, OCR or another document-extraction model.\n\n### Gold-table QA is an upper-stage diagnostic\n\nStrong TAPAS performance on the gold logical table does not imply strong end-to-end performance.\n\n### Structure errors alter semantics\n\nMissing or duplicated rows/columns can turn a correct source table into a different structured table before TAPAS sees it.\n\n### Complex tables need richer reconstruction\n\nSpanning cells, projected headers and nested structure require more than simple row × column intersection.\n\n### Tutorial evidence is not a benchmark\n\nThis notebook uses a small, deterministic public-domain scientific-table sample and one seeded composition policy. Results are measurements from this notebook, not production-quality claims."
  },
  {
    "kind": "markdown",
    "source": "## Your conclusion\n\nComplete this template from the outputs above, citing the printed number or exported file you used for each claim. Do not generalise beyond ten scientific tables, 50 generated questions and one run.\n\n- **Task.** This notebook composed ___ → ___ → ___ to answer questions about tables on document pages.\n- **Principal result.** End-to-end denotation accuracy was ___ on 50 questions, against ___ on the gold table (the reference) and ___ on the structure-only path.\n- **Where the loss entered.** The largest drop occurred between ___ and ___ (evidence: ___ in `waterfall.json` or `reconstruction_metrics.csv`), mainly because ___.\n- **Failure mode or uncertainty.** One important failure was ___. With ten tables, a single table moves a per-table rate by 10 percentage points and five questions.\n- **Limitations.** Cell text came from annotations rather than OCR; each synthetic page held one table; only rectangular tables were used for QA; one paper-disjoint sample and one composition policy were measured.\n\n<details>\n<summary>Sample conclusion (illustrative; your numbers may differ)</summary>\n\nOn ten held-out SciTSR-PD tables, TAPAS answered 80% of 50 questions correctly from the gold tables and from gold-crop reconstructions, but 60% end to end. Every gold crop produced an exact grid, so structure recognition on a well-framed crop was not the bottleneck here. The loss entered at detection (one page had no detection above 0.90, which made its five questions unanswerable) and on one detected crop whose grid gained a column. These are measurements on ten tables with annotation text; they do not predict accuracy on scanned documents, OCR text, or tables with spanning cells.\n\n</details>"
  },
  {
    "kind": "markdown",
    "source": "## Troubleshooting\n\n| Symptom | Likely cause | What to do |\n|---|---|---|\n| Run is very slow; `device` prints `cpu` | no GPU runtime selected | select a T4 GPU and `Run all` again (CPU works, but TAPAS takes about a second per question) |\n| `This notebook builds a Linux x86_64 environment` | a Windows or macOS kernel | use Google Colab, Kaggle or a Linux x86_64 Jupyter server |\n| `uv wheel size/hash mismatch`, or a `uv pip install` error | package index unreachable or a truncated download | rerun the Runtime cell (§4); no restart is needed |\n| `Installed versions differ from the hash lock` | packages in `work/env` were changed by hand | delete the `work/env` folder and rerun the Runtime cell (§4) |\n| `<stage> failed with exit …` | an error inside the isolated environment | read the log tail printed above it (`work/logs/<stage>.log`) |\n| `… is missing: run the earlier notebook cells first` or a `NameError` in a stage | a cell was run before the cells it depends on | run the notebook from the top (`Run all`) |\n| model size or SHA-256 mismatch | incomplete or changed download | delete that model's folder under `weights/` and rerun; never bypass the check |\n| SciTSR-PD shard size or SHA-256 mismatch | incomplete download | delete `weights/scitsr-pd` and rerun |\n| `Embedded carrier geometry does not match its SHA-256`, or `Carried file integrity failure` | the carried stage file was edited | restore the notebook from the published copy; never bypass the check |\n| `carrier tables failed to derive` or `source pixels differ` | a different dataset revision was read | keep the pinned `SCITSR_REVISION`; do not relax the check |\n| split or eligibility `RuntimeError` | seed, derivation or eligibility rules were edited | restore the defaults; never borrow train/validation tables for the QA set |\n| CUDA out of memory | another notebook shares the GPU | close the other notebook and rerun the failed cell; each stage process loads one model at a time and releases it when it exits |\n| every reconstruction invalid | Configuration thresholds were changed | restore the defaults and experiment in the activity cell instead |\n| BYOD: `BYOD_PATH is empty` or `is missing ['pages/', …]` | path not set, or wrong folder layout | set `BYOD_PATH` in §3 to the folder that holds `pages/` and `ocr.csv` (layout in §28) |\n| BYOD: `ocr.csv line N: …` or `questions.csv line N: …` | that row breaks the file's contract | fix the named line and field |\n| BYOD: `boxes … extend beyond its W×H image` | OCR boxes in PDF points, or with a bottom-left origin | convert them to page-image pixels with a top-left origin |\n| BYOD: a page listed under `skipped` | no OCR rows for that `page_id`, or an unsupported file type | make the `page_id` equal the image file name without its extension; use a listed image type |"
  },
  {
    "kind": "markdown",
    "source": "## Glossary\n\n| Term | Meaning |\n|---|---|\n| **Table detection** | Finding the bounding box of each table on a page |\n| **Structure recognition** | Finding the rows, columns, headers and spanning cells inside a table crop |\n| **Spanning cell** | A cell that covers more than one row or column |\n| **Crop / coordinate frame** | A sub-image; boxes must be shifted between page, crop and table coordinates |\n| **IoU** | Intersection over union of two boxes: 1.0 is identical, 0 is disjoint |\n| **AP50 / AP75** | Average precision counting a detection as correct at IoU ≥ 0.50 / ≥ 0.75 |\n| **NMS** | Non-maximum suppression: drop a box that overlaps a higher-scoring box of the same label |\n| **Grid reconstruction** | Intersecting predicted rows and columns to form cells |\n| **Cell-text assignment** | Placing each text item into the predicted cell that contains its center |\n| **TAPAS** | A BERT-style model that answers questions over a table of strings by selecting cells and an aggregation |\n| **WTQ** | WikiTableQuestions, the QA dataset TAPAS Large WTQ was fine-tuned on |\n| **Aggregation operator** | `NONE`, `SUM`, `AVERAGE` or `COUNT` applied to the selected cells |\n| **Denotation** | The final answer value (cell strings or a number), compared with the gold answer |\n| **Paper-disjoint split** | Tables from one paper never appear in two splits |\n| **Carrier** | The DIMER repository whose pinned model and data this notebook reproduces |\n| **Error waterfall** | Stage-by-stage counts showing where end-to-end failures entered |"
  },
  {
    "kind": "markdown",
    "source": "## 30. Terminal summary\n\nThe final cell verifies required outputs and prints the stage waterfall without declaring a winner."
  },
  {
    "kind": "code",
    "source": "run_stage(\"summary\")"
  }
]
