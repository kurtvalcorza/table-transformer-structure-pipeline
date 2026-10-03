# ruff: noqa: E501
"""Contract tests for the Table Intelligence workshop's uv isolated environment (revision 0.3.0-candidate).

The notebook must not install anything into its kernel or ask for a restart; it carries the stage file and a
hash lock, builds a CPython 3.12.12 venv with a pinned uv, installs only hashed wheels, and runs every stage
with that venv's interpreter.
"""

import ast
import hashlib
import json
import pickle
import re
import subprocess
import sys
import types

import pytest

from table_intelligence_workshop_support import (
    LOCK_FILE,
    REQUIREMENTS_IN,
    ROOT,
    STAGE_FILE,
    carried_files,
    cell_by_id,
    cells,
    host_cell,
    joined,
    load_stages,
    notebook,
    stage_text,
)

BOOT_ID = "dimer-table-workshop-06"
STDLIB = set(sys.stdlib_module_names)
HOST_THIRD_PARTY = {"IPython"}
MODEL_LIBRARIES = {"torch", "torchvision", "torchaudio", "transformers", "timm", "huggingface_hub", "safetensors",
                   "datasets", "pyarrow", "pandas"}


def host():
    return [c for c in cells() if c["cell_type"] == "code" and c["id"] != "uvcarrier"]


def test_no_kernel_install_and_no_restart_guard():
    for cell in host():
        text = joined(cell)
        assert '"-m", "pip"' not in text and "'-m', 'pip'" not in text and '"-m","pip"' not in text, cell["id"]
        assert "pip install" not in text and "pip\",\"install" not in text, cell["id"]
        assert "Restart session" not in text and "NUMPY_PRELOADED" not in text, cell["id"]
    whole = "\n".join(joined(c) for c in cells())
    assert "Restart the Python session" not in whole
    assert "pip" not in stage_text().replace("pipeline", "")


def test_host_cells_import_only_stdlib_and_display_libraries():
    for cell in host():
        for node in ast.walk(ast.parse(joined(cell))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                top = name.partition(".")[0]
                assert top in STDLIB or top in HOST_THIRD_PARTY, (cell["id"], name)
                assert top not in MODEL_LIBRARIES, (cell["id"], name)


def test_stage_file_imports_model_libraries_only_inside_stages():
    tree = ast.parse(stage_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                assert name.partition(".")[0] not in MODEL_LIBRARIES, name


def test_carrier_cell_carries_the_repository_files_byte_for_byte():
    carried = carried_files()
    assert carried["table_intelligence_workshop.py"] == STAGE_FILE.read_text(encoding="utf-8")
    assert carried["requirements.lock.txt"] == LOCK_FILE.read_text(encoding="utf-8")
    hashes = notebook()["metadata"]["dimer"]["carried_files"]
    for name, text in carried.items():
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert hashes[name]["sha256"] == digest
        assert f"{name!r}: {digest!r}" in cell_by_id("uvcarrier")
        assert b"\r" not in text.encode("utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "tools/build_table_intelligence_workshop.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_carrier_cell_writes_and_verifies_the_files(tmp_path):
    ns = {"WORK_DIR": str(tmp_path / "work")}
    exec(compile(cell_by_id("uvcarrier"), "uvcarrier", "exec"), ns)
    for name, text in ns["CARRIED_FILES"].items():
        written = (tmp_path / "work" / "runner" / name).read_bytes()
        assert hashlib.sha256(written).hexdigest() == ns["CARRIED_HASHES"][name]
        assert written == text.encode("utf-8")


def test_lock_file_pins_every_requirement_with_hashes():
    text = LOCK_FILE.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+==)", text)
    entries = [b for b in blocks if re.match(r"[A-Za-z0-9_.-]+==", b)]
    assert len(entries) >= 50
    for block in entries:
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", block), block.split("==", 1)[0]
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", text, re.M))
    # The exact versions of the former in-kernel install (section 4 PINS), unchanged.
    expected = {"torch": "2.14.0", "torchvision": "0.29.0", "torchaudio": "2.11.0", "transformers": "4.57.6",
                "timm": "1.0.30", "safetensors": "0.8.0", "numpy": "2.5.3", "pillow": "11.3.0",
                "huggingface-hub": "0.36.2", "datasets": "4.1.1", "pyarrow": "25.0.1", "pandas": "2.2.3"}
    assert {k: pins.get(k) for k in expected} == expected
    assert dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", REQUIREMENTS_IN.read_text(encoding="utf-8"), re.M)) == expected
    assert "--python-platform x86_64-manylinux_2_28" in text and "--generate-hashes" in text and "--only-binary :all:" in text
    assert "former" in text.split("\n# This file was autogenerated", 1)[0]  # the header names the carried-over pins


def test_install_requires_hashes_and_binary_wheels_only():
    boot = cell_by_id(BOOT_ID)
    for token in ('"--require-hashes"', '"--only-binary", ":all:"', '"--index-url", "https://pypi.org/simple"',
                  '"--managed-python", "--python", "3.12.12"', "UV_SHA256", "uv-0.12.15-"):
        assert token in boot, token
    assert "len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != UV_SHA256" in boot
    assert 'platform.system() != "Linux" or platform.machine() != "x86_64"' in boot
    assert 'CARRIED_HASHES["requirements.lock.txt"]' in boot  # the env is rebuilt when the lock changes


def test_stages_run_with_the_venv_interpreter_and_a_clean_environment():
    boot = cell_by_id(BOOT_ID)
    assert 'PYTHON = ENV_ROOT / "bin" / "python"' in boot
    assert 'command = [str(PYTHON), "-u", str(STAGE_SCRIPT), "--config"' in boot
    assert 'ENV["MPLBACKEND"] = "Agg"' in boot
    assert '("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")' in boot
    stages = load_stages()["STAGES"]
    used = set()
    for cell in host():
        used |= set(re.findall(r'run_stage\("([a-z-]+)"', joined(cell)))
    assert used == set(stages), (used ^ set(stages))
    shown = set()
    names = {n.name for n in ast.parse(stage_text()).body if isinstance(n, ast.FunctionDef)}
    for cell in host():
        for call in re.findall(r"show_source\(([^)]*)\)", joined(cell)):
            shown |= set(re.findall(r'"([A-Za-z_]+)"', call))
    assert shown and shown <= names, shown - names


def test_run_stage_passes_every_control_the_stage_file_reads():
    boot = cell_by_id(BOOT_ID)
    config_keys = set(re.findall(r'"([a-z_]+)": ', boot.split("def run_stage", 1)[1].split("config_path", 1)[0]))
    read_keys = set(re.findall(r'config\["([a-z_]+)"\]', stage_text()))
    assert config_keys == read_keys
    # The stage file's defaults are the notebook's reference settings (section 3).
    controls = {}
    exec(host_cell("USE_BYOD = False"), controls)
    ns = load_stages()
    for name in ("USE_BYOD", "BYOD_PATH", "DETECTION_THRESHOLD", "STRUCTURE_THRESHOLD", "CROP_PADDING", "GRID_NMS_IOU",
                 "END_TO_END_TABLES", "QUESTIONS_PER_TABLE", "RUN_COMPLEX_STRUCTURE_PROBE", "COMPLEX_STRUCTURE_TABLES",
                 "OUTPUT_DIR"):
        assert ns[name] == controls[name], name


def test_controls_reach_defaults_resolved_at_call_time():
    # The controls arrive after import, so no default argument may freeze an import-time value.
    ns = load_stages()
    ns["configure"]({"use_byod": False, "byod_path": "", "detection_threshold": 0.7, "structure_threshold": 0.6,
                     "crop_padding": 25, "grid_nms_iou": 0.4, "end_to_end_tables": 10, "questions_per_table": 5,
                     "run_complex_structure_probe": True, "complex_structure_tables": 3,
                     "output_dir": "o", "work_dir": "w"})
    from PIL import Image

    page = Image.new("RGB", (400, 300), "white")
    _, box = ns["padded_crop"](page, [100, 100, 200, 200])
    assert box == [75, 75, 225, 225]
    a = {"label": "table row", "score": 0.9, "box": [0, 0, 100, 10]}
    b = {"label": "table row", "score": 0.8, "box": [0, 0, 100, 8]}  # IoU 0.8 with a
    ns["GRID_NMS_IOU"] = 0.85
    assert len(ns["nms_items"]([a, b])) == 2
    ns["GRID_NMS_IOU"] = 0.5
    assert len(ns["nms_items"]([a, b])) == 1
    for name in ("padded_crop", "nms_items", "detect_table", "recognize_structure"):
        node = next(n for n in ast.parse(stage_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert all(isinstance(d, ast.Constant) and d.value is None for d in node.args.defaults), name


def test_no_notebook_line_exceeds_2000_characters():
    for cell in cells():
        for line in joined(cell).splitlines():
            assert len(line) <= 2000, (cell["id"], len(line))


def test_environment_stage_refuses_versions_that_differ_from_the_lock(tmp_path, monkeypatch):
    ns = load_stages(tmp_path)
    runner = tmp_path / "runner"
    runner.mkdir()
    stage_file = runner / "table_intelligence_workshop.py"
    stage_file.write_text("")
    (runner / "requirements.lock.txt").write_text("numpy==0.0.1 \\\n    --hash=sha256:" + "0" * 64 + "\n")
    import importlib.metadata as metadata

    ns["__file__"] = str(stage_file)
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(__version__="x"))
    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace(__version__="x"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="x"))
    ns["pd"] = types.SimpleNamespace(__version__="x")
    monkeypatch.setattr(metadata, "version", lambda name: "9.9.9")
    with pytest.raises(RuntimeError, match="differ from the hash lock"):
        ns["stage_environment"](None)


def _config(tmp_path):
    config = {"use_byod": False, "byod_path": "", "detection_threshold": 0.9, "structure_threshold": 0.5,
              "crop_padding": 10, "grid_nms_iou": 0.5, "end_to_end_tables": 10, "questions_per_table": 5,
              "run_complex_structure_probe": True, "complex_structure_tables": 3,
              "output_dir": str(tmp_path / "outputs"), "work_dir": str(tmp_path / "work")}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_stage_cli_runs_a_model_free_stage_in_a_separate_process(tmp_path):
    result = subprocess.run([sys.executable, str(STAGE_FILE), "--config", str(_config(tmp_path)), "--stage", "geometry"],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "'carrier_tables': 94" in result.stdout


def test_state_written_by_one_stage_process_is_read_by_the_next(tmp_path):
    # Fabricated state, as the stages before the waterfall leave it; the waterfall stage runs as its own process.
    ns = load_stages(tmp_path)
    ok = {"valid": True}
    tables = [{"source_id": f"t{i}", "gold_recon": ok, "det_recon": {"valid": i != 5}} for i in range(10)]
    qa = [{"denotation_correct": i < 40, "aggregation_correct": True, "pipeline_status": "ok"} for i in range(50)]
    ns.update(canonical_tables=tables, qa_records=[{}] * 50, gold_qa=qa, structure_qa=qa, end_to_end_qa=qa[:30] + [dict(q, denotation_correct=False) for q in qa[30:]],
              det_rows=[{"hit50": i != 5, "hit75": i != 5} for i in range(10)],
              structure_rows=[{"path": p, "grid_shape_exact": not (p == "detected_crop" and i in (5, 6))} for p in ("gold_crop", "detected_crop") for i in range(10)])
    ns["persist_state"]()
    result = subprocess.run([sys.executable, str(STAGE_FILE), "--config", str(_config(tmp_path)), "--stage", "waterfall"],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "work" / "state" / "state.pkl", "rb") as f:
        state = pickle.load(f)
    waterfall = state["waterfall"]
    assert waterfall["detection"]["hit50"] == 9
    assert waterfall["structure_detected_crop"]["grid_exact"] == 8
    assert waterfall["reconstruction"]["valid_detected_crop"] == 9
    assert waterfall["qa"]["gold_table_correct"] == 40 and waterfall["qa"]["end_to_end_correct"] == 30
    assert '"grid_exact": 8' in result.stdout


def test_a_failing_stage_keeps_what_it_changed_and_names_the_error(tmp_path):
    # As in a kernel: state changed before an error is kept; the CLI exits non-zero with the error as its last line.
    ns = load_stages(tmp_path)
    ns.update(canonical_tables=[], qa_records=[])
    ns["persist_state"]()
    result = subprocess.run([sys.executable, str(STAGE_FILE), "--config", str(_config(tmp_path)), "--stage", "waterfall"],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0
    assert result.stderr.strip().splitlines()[-1].startswith("NameError")
    boot = cell_by_id(BOOT_ID)
    assert 'raise RuntimeError(f"{stage} failed with exit {process.returncode}: {last[0]} (full log: {log})")' in boot


def test_environment_stage_starts_a_fresh_state(tmp_path):
    ns = load_stages(tmp_path)
    ns["canonical_tables"] = ["stale"]
    ns["persist_state"]()
    calls = []
    ns["stage_environment"] = lambda args: calls.append(1)
    ns["stage"]("environment-test", fresh=True)(lambda args: calls.append(ns.get("canonical_tables")))(None)
    ns["STAGES"]["environment-test"](None)
    assert not (tmp_path / "work" / "state" / "state.pkl").read_bytes().count(b"stale")


def test_markdown_and_metadata_state_the_platform_and_no_restart():
    title = joined(cells()[0])
    assert "Linux x86_64 only" in title and "needs no restart" in title
    runtime = next(joined(c) for c in cells() if c["cell_type"] == "markdown" and joined(c).startswith("## 4. Runtime"))
    assert "--require-hashes --only-binary :all:" in runtime and "Linux x86_64 only" in runtime
    assert "no restart" in runtime
    meta = notebook()["metadata"]
    dimer = meta["dimer"]
    assert dimer["runtime_environment"]["platform"].startswith("Linux x86_64 only")
    assert meta["workshop_revision"] == "0.3.0-candidate"
    revision = dimer["review_revisions"][-1]
    assert revision["revision"] == "0.3.0-candidate" and revision["base_commit"].startswith("a043211")
    assert dimer["clean_runtime_evidence"] == "pending"


def test_cell_ids_are_stable_and_the_carrier_precedes_the_bootstrap():
    ids = [c["id"] for c in cells()]
    assert len(ids) == len(set(ids))
    assert [i for i in ids if i != "uvcarrier"] == [f"dimer-table-workshop-{n:02d}" for n in range(67)]
    assert ids.index("dimer-table-workshop-04") < ids.index("uvcarrier") == ids.index(BOOT_ID) - 1
    assert cells()[ids.index("uvcarrier")]["metadata"] == {"jupyter": {"source_hidden": True}}
