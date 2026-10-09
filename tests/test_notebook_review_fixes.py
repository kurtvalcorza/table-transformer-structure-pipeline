"""Regression tests for the 2026-10-05 notebook review findings (TTS-M1..M4, TTS-m1, TTS-m2).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed, and restore_base() is exercised on a small torch module, not the checkpoint. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from table_transformer_structure_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "table_transformer_structure_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "table_transformer_structure_pipeline" / "pipeline.py"
STEM = "table_transformer_structure"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- TTS-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_tts_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_tts_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


def test_tts_s5_declares_notebook_spec_2_2(notebook):
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"


def test_tts_m1_routed_cells_use_the_worker_display_not_ipython(notebook):
    """Every cell after Section 1 runs in the isolated environment, which has no IPython: a routed cell that imports
    IPython.display would silently lose its figure. The worker injects `display` into the cell namespace instead."""
    routed = [c["source"] for c in _code_cells(notebook) if "# dimer: kernel cell" not in c["source"]]
    assert routed and not [s for s in routed if re.search(r"^\s*(from|import) IPython", s, re.M)]
    assert sum("display(" in s for s in routed) >= 2
    kernel = _cell(notebook, "# dimer: kernel cell")
    assert "_main.__dict__.update(__builtins__=builtins, display=display)" in kernel


@pytest.mark.skipif(sys.platform != "linux", reason="the worker protocol uses Linux pass_fds (as in rtdetr-detection-pipeline 0feefe5)")
def test_tts_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- TTS-M2: every adaptation starts from the pinned base -------------------------------------------------------


def test_tts_m2_adapt_and_load_artifact_restore_the_base_first():
    """The decoder is restored before the decoder features are cached; a failed call puts back what it found."""
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def save_artifact(")]
    order = [adapt.index(m) for m in ("previous_layers = {n: current[n].detach().clone() for n in self._base_layers}", "restored = self.restore_base()", "self._remember_base(names)", "cached = [self._decoder_features(", "for epoch in range(2, epochs + 2):")]
    assert order == sorted(order)
    failure = adapt[adapt.index("except BaseException:") :]
    assert failure.index("for n, value in initial_layers.items():") < failure.index("for n, value in previous_layers.items():") < failure.index("raise")
    assert '"started_from": "pinned base"' in adapt
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("params[key].copy_(")
    restore = text[text.index("    def restore_base(") : text.index("    @classmethod")]
    assert "params[name].copy_(self._base_layers[name])" in restore and "self._head, self._bbox_head, self.classes, self.adapter = None, None, [], None" in restore


def test_tts_m2_restore_base_undoes_every_earlier_change_on_torch_tensors():
    """restore_base() on real torch parameters (a two-layer stand-in module, not the checkpoint); CI has torch."""
    torch = pytest.importorskip("torch")
    from table_transformer_structure_pipeline import TableTransformerStructurePipeline

    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    base = {name: value.detach().clone() for name, value in module.named_parameters()}
    pipe = TableTransformerStructurePipeline(_runner=lambda image, threshold: [], device="cpu", _model=module, _processor=object())
    assert pipe.restore_base() == []
    pipe._remember_base(["1.weight", "1.bias"])
    with torch.no_grad():
        for value in module.parameters():
            value.add_(1.0)
    pipe._head, pipe._bbox_head, pipe.classes, pipe.adapter = object(), object(), ["table"], {"policy": "x"}
    assert pipe.restore_base() == ["1.bias", "1.weight"]
    params = dict(module.named_parameters())
    assert torch.equal(params["1.weight"], base["1.weight"]) and torch.equal(params["1.bias"], base["1.bias"])
    assert torch.equal(params["0.weight"], base["0.weight"] + 1.0)  # never adapted, never touched
    assert (pipe._head, pipe._bbox_head, pipe.classes, pipe.adapter) == (None, None, [], None)
    # A repeat restore (or the restore at the start of the next adapt) changes nothing, so it reports nothing:
    # the count is of tensors that differed from the base, not of every remembered name (t5-base 93a578f).
    assert pipe.restore_base() == []
    with torch.no_grad():
        params["1.bias"].add_(0.5)
    assert pipe.restore_base() == ["1.bias"]


def test_tts_m2_byod_rerun_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("restored_layers = pipe.restore_base()") < section_4.index("if USE_BYOD:")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = TableTransformerStructurePipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)" in experiment
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)


# --- TTS-M3: quality outcomes are reported verdicts -------------------------------------------------------------


def test_tts_m3_no_quality_assert_remains(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = re.findall(r"(?m)^\s*assert .*$", code)
    assert asserts == ["assert parity['queries_identical'] and parity['classes_identical'] and abs(adapted_test['map50'] - reloaded_test['map50']) < 1e-9"]
    assert not re.search(r"assert .*> prior\['map50'\]", code)


LABELS4 = ("table", "table column", "table row", "table spanning cell")


def _m(map50, map75=0.5, m=0.4):
    return {
        "map50": map50, "map75": map75, "map": m, "mean_best_iou": 0.5, "n_images": 4, "loss": 0.5, "grid_exact_at_threshold": 0.1,
        "per_label": {label: {"ap50": map50} for label in LABELS4}, "recall_at_threshold": {label: 0.5 for label in LABELS4},
        "baseline": "stand-in", "prior_grid": {"rows": 3, "columns": 3}, "policy": "stand-in", "verdict": "measured-small-sample", "definitions": {},
    }


def test_tts_m3_a_competitive_prior_is_recorded_and_does_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """Sections 6 and 8 executed with stand-ins where the grid prior beats every policy (regular BYOD tables): both
    cells complete and record the verdicts (stand-in evidence, no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scores = iter([_m(0.2), _m(0.1, m=0.3), _m(0.12)])

    class StandIn:
        def evaluate_zero_shot(self, records):
            return _m(0.15)

        def adapt(self, *a, **k):
            return {"policy": "frozen backbone, encoder and decoder + restricted heads", "best_epoch": 1, "head_final_loss": 1.0, "history": [{"epoch": 0, "val": None}], "trainable_names": []}

        def evaluate(self, records):
            return next(scores)

    ns = {
        "pipe": StandIn(), "train_records": [], "val_records": [], "test_records": [], "time": __import__("time"), "json": json,
        "prior_baseline": lambda *a, **k: _m(0.3), "RECOGNITION_THRESHOLD": 0.5, "ADAPT_LABELS": LABELS4,
        "POLICY_FROZEN": "frozen backbone, encoder and decoder + restricted heads", "POLICY_ZERO_SHOT": "copied-head zero-shot",
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "MODEL_KEY": "stand-in", "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d"}},
        "disjoint": {}, "summary": {}, "report": {}, "splits": {"train": [], "validation": [], "test": []},
        "adapt_result": {"policy": "unfrozen last 2 decoder layers + restricted heads", "history": []}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "prior = prior_baseline("), ns)
    assert ns["frozen_verdict"].startswith("frozen policy NOT above the grid prior")
    exec(_cell(notebook, "adapted_test = pipe.evaluate("), ns)
    comparison = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))["comparison"]
    verdicts = comparison["verdicts"]
    assert verdicts["selected_above_prior_map50"] is False and verdicts["selected_vs_frozen_map"] == "worse" and verdicts["selected_vs_zero_shot_map50"] == "worse"
    assert comparison["spanning_cells_per_split"] == {"train": 0, "validation": 0, "test": 0}


def test_tts_m3_contract_checks_stay_hard_checks(notebook):
    section_6 = _cell(notebook, "prior = prior_baseline(")
    assert "if probe_result['policy'] not in (POLICY_ZERO_SHOT, POLICY_FROZEN):" in section_6 and "raise RuntimeError" in section_6
    section_5 = _cell(notebook, "result = pipe.recognize(image")
    assert "if report['verdict'] != 'sample-sanity':" in section_5


# --- TTS-M4: guided layer and infrastructure labelling ----------------------------------------------------------


def test_tts_m4_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_tts_m4_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    assert len([c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]) == 4
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_tts_m4_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- TTS-m1: BYOD contract --------------------------------------------------------------------------------------


def _png(i: int) -> bytes:
    from PIL import Image

    image = Image.fromarray(np.random.default_rng(i).integers(0, 255, (120, 160, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _zip(path: Path, n: int, *, drop: str | None = None, extra: dict[str, bytes] | None = None, bom: bool = False) -> Path:
    header = "id,file,group,label,x_min,y_min,x_max,y_max\n"
    rows = "".join(
        f"t{i},img{i}.png,g{i},table,5,5,150,110\nt{i},img{i}.png,g{i},table row,5,5,150,50\nt{i},img{i}.png,g{i},table column,5,5,70,110\n"
        for i in range(n)
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("structure.csv", ("﻿" if bom else "") + header + rows)
        for i in range(n):
            if f"img{i}.png" != drop:
                archive.writestr(f"img{i}.png", _png(i))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_tts_m1_stated_minimum_is_what_the_split_accepts(tmp_path):
    assert sm.min_byod_records() == {"total": 12, "train": 8, "validation": 2, "test": 2}
    split = sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "ok.zip", 12)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 2, "validation": 2, "train": 8}
    with pytest.raises(ValueError, match=r"split leaves 7 training records from 11 group\(s\).*supply at least 12 tables"):
        sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "small.zip", 11)), seed=42)
    markdown = _markdown(notebook_json())
    assert "**12 tables in 12 groups**" in markdown and "8..2,000 records\" in the BYOD" not in markdown


def notebook_json() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_tts_m1_missing_image_bom_header_only_and_litter(tmp_path):
    with pytest.raises(ValueError, match=r"structure.csv line 11 \(file 'img3.png'\): names an image that is not in the dataset"):
        sm.load_byod_dataset(_zip(tmp_path / "missing.zip", 14, drop="img3.png"))
    assert len(sm.load_byod_dataset(_zip(tmp_path / "mac.zip", 14, extra={"__MACOSX/._img0.png": b"\0", ".DS_Store": b"\0"}))) == 14
    assert len(sm.load_byod_dataset(_zip(tmp_path / "bom.zip", 14, bom=True))) == 14
    with zipfile.ZipFile(tmp_path / "header.zip", "w") as archive:
        archive.writestr("structure.csv", "id,file,group,label,x_min,y_min,x_max,y_max\n")
    with pytest.raises(ValueError, match="no data rows"):
        sm.load_byod_dataset(tmp_path / "header.zip")


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    from table_transformer_structure_pipeline import metrics as mt
    from table_transformer_structure_pipeline import pipeline as pl
    from table_transformer_structure_pipeline import sample_data as sd

    ns = {}
    for module in (sd, pl, mt, sm):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
    pipe = types.SimpleNamespace(adapter={"policy": "x"}, restore_base=lambda: restored.append(True) or ["a"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_tts_m1_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _zip(tmp_path / "mine.zip", 14)
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.zip"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the pipeline back to the pinned base first"
    assert ns["raw_count"] == {"byod": 14, "duplicate_tables_dropped": 0, "effective_minimum": 12}
    assert "only 3 held-out test tables" in out
    assert (tmp_path / "outputs" / f"{STEM}_train.csv").is_file()


def test_tts_m1_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.zip' does not exist"):
        exec(_section_4(notebook, "nowhere.zip"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))


# --- TTS-m2: the spanning-cell count and the data contract ------------------------------------------------------


def test_tts_m2_prose_quotes_the_training_spanning_cell_count_the_split_produces(notebook):
    splits = sm.build_sample_dataset(sm.load_corpus(), seed=42)
    counts = {name: sum(1 for r in part for o in r["objects"] if o["label"] == "table spanning cell") for name, part in splits.items()}
    assert counts == {"train": 19, "validation": 39, "test": 13}
    markdown = _markdown(notebook)
    assert "56 training instances" not in markdown
    assert "19 instances in the training split" in markdown and "13 in the test split" in markdown
    assert "**19** spanning cells" in markdown


def test_tts_m2_no_escaped_braces_in_the_data_contract(notebook):
    markdown = _markdown(notebook)
    assert "{{" not in markdown and "}}" not in markdown
    assert "`[A-Za-z0-9_.:-]{1,64}`" in markdown and "`{id, image, objects}`" in markdown and "`{label, box}`" in markdown
