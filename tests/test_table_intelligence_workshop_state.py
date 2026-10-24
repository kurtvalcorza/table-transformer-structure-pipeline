# ruff: noqa: E501
"""The workshop's stage state is saved as JSON, PNG files and an npz, never as a pickle (ARTIFACT_SPEC 2.0 SER7)."""

import importlib.util
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from table_intelligence_workshop_support import ROOT, load_stages, stage_text


def _validator():
    spec = importlib.util.spec_from_file_location("validate_release_assets_state", ROOT / "tools" / "validate_release_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(tmp_path):
    ns = load_stages(tmp_path)
    image = Image.new("RGB", (12, 7), "white")
    image.putpixel((3, 2), (10, 20, 30))
    items = [{"source_id": f"t{i}", "paper_id": f"p{i % 2}", "image": image if i == 0 else Image.new("RGB", (5, 4), (i, i, i)),
              "grid": [["a", "1"], ["b", "2"]], "objects": [{"label": "table", "box": [0.0, 1.5, 9.0, 6.0]}],
              "det_crop": None, "det_crop_box": (1.0, 2.0, 3.0, 4.0)} for i in range(4)]
    ns.update(
        derived=items, splits={"test": items[:2], "train": items[2:]}, canonical_tables=[items[1]],
        failed=["x"], SOURCE_TABLES=108, det_peak=None, det_times=[0.25, float("nan")], struct_peak=float("inf"),
        scitsr_paths={"train": Path("weights/scitsr-pd/train.parquet")},
        det_metrics={"mean_best_iou_detected": np.float64(0.75), "per_page": np.arange(3, dtype=np.float32)},
        waterfall={"counts": Counter({"ok": 3}), "by_paper": defaultdict(list, {"p0": [1]}), 7: {"__ref__": "escaped"}},
        RUNTIME={"started": datetime(2026, 10, 24, tzinfo=UTC), "pins": {"numpy", "pillow"}},
        qa_records=[{"gold_coordinates": [(0, 1)], "words": frozenset({"a"})}],
    )
    return ns


def test_round_trip_restores_values_types_images_and_shared_objects(tmp_path):
    ns = _state(tmp_path)
    ns["persist_state"]()
    out = load_stages(tmp_path)
    out["restore_state"]()
    assert out["splits"]["test"][1] is out["derived"][1] is out["canonical_tables"][0]  # one object, as in the kernel
    assert out["derived"][0]["image"].tobytes() == ns["derived"][0]["image"].tobytes()
    assert out["derived"][0]["image"].mode == "RGB" and out["derived"][0]["image"].size == (12, 7)
    assert out["derived"][2]["det_crop_box"] == (1.0, 2.0, 3.0, 4.0) and out["derived"][2]["det_crop"] is None
    assert out["det_times"][0] == 0.25 and math.isnan(out["det_times"][1]) and out["struct_peak"] == float("inf")
    assert out["scitsr_paths"]["train"] == Path("weights/scitsr-pd/train.parquet")
    assert isinstance(out["det_metrics"]["mean_best_iou_detected"], np.float64)
    assert out["det_metrics"]["per_page"].dtype == np.float32 and out["det_metrics"]["per_page"].tolist() == [0.0, 1.0, 2.0]
    assert out["waterfall"]["counts"] == Counter({"ok": 3}) and type(out["waterfall"]["counts"]) is Counter
    assert out["waterfall"]["by_paper"]["p1"] == [] and out["waterfall"][7] == {"__ref__": "escaped"}
    assert out["RUNTIME"]["started"] == datetime(2026, 10, 24, tzinfo=UTC) and out["RUNTIME"]["pins"] == {"numpy", "pillow"}
    assert out["qa_records"][0] == {"gold_coordinates": [(0, 1)], "words": frozenset({"a"})}
    assert out["SOURCE_TABLES"] == 108 and out["failed"] == ["x"] and out["det_peak"] is None


def test_the_state_is_pure_json_png_and_npz_with_no_pickle(tmp_path):
    ns = _state(tmp_path)
    ns["persist_state"]()
    root = tmp_path / "work" / "state"

    def refuse(name):
        raise ValueError(f"non-standard JSON constant {name}")

    manifest = json.loads((root / "state.json").read_text(encoding="utf-8"), parse_constant=refuse)
    assert manifest["format"] == "dimer.table-intelligence-workshop.state.v2"
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert files == sorted(["state.json", "arrays.npz", *(f"images/{k}.png" for k in manifest["images"])])
    assert len(manifest["images"]) == 4  # one file per distinct image
    with np.load(root / "arrays.npz", allow_pickle=False) as data:
        assert sorted(data.files) == ["a0", "a1"]
    assert not list(tmp_path.rglob("*.pkl"))


def test_a_tampered_image_or_array_is_refused_before_it_is_read(tmp_path, monkeypatch):
    ns = _state(tmp_path)
    ns["persist_state"]()
    root = tmp_path / "work" / "state"
    for image_path in (root / "images").iterdir():  # whichever image is read first must be refused
        image_path.write_bytes(image_path.read_bytes() + b"x")
    out = load_stages(tmp_path)

    def never(*_args, **_kwargs):
        raise AssertionError("opened before the digest check")

    monkeypatch.setattr(out["Image"], "open", never)
    with pytest.raises(RuntimeError, match="does not match the digest"):
        out["restore_state"]()

    monkeypatch.undo()
    ns["persist_state"]()  # rewrite a clean state, then tamper with the arrays
    for path in (root / "images").iterdir():
        path.unlink()
    ns["persist_state"]()
    (root / "arrays.npz").write_bytes((root / "arrays.npz").read_bytes() + b"x")
    out = load_stages(tmp_path)
    monkeypatch.setattr(out["np"], "load", never)
    with pytest.raises(RuntimeError, match="arrays.npz does not match"):
        out["restore_state"]()


def test_a_value_without_a_safe_form_is_refused_not_pickled(tmp_path):
    ns = load_stages(tmp_path)
    ns["derived"] = [{"model": object()}]
    with pytest.raises(TypeError, match=r"state derived\[0\]\.model: builtins\.object cannot be saved without pickle"):
        ns["persist_state"]()
    ns["derived"] = [np.array([object()], dtype=object)]
    with pytest.raises(TypeError, match="object arrays"):
        ns["persist_state"]()
    ns["derived"] = [Image.new("CMYK", (2, 2))]
    with pytest.raises(TypeError, match="image mode"):
        ns["persist_state"]()
    assert not list(tmp_path.rglob("*.pkl"))


def test_the_stage_file_has_no_pickle_and_the_validator_rejects_one():
    validator = _validator()
    assert validator.pickle_export_hits(stage_text()) == []
    legacy = "import pickle\ndef state_file():\n    return WORK_ROOT/'state'/'state.pkl'\n        pickle.dump(state, f)\n"
    assert validator.pickle_export_hits(legacy) == ["import pickle", "state.pkl", "pickle.dump("]
    assert validator.pickle_export_hits("torch.save(model, path)") == ["torch.save("]
