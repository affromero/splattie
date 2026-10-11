"""Pinned LHM snapshot selection with real offline HF caches and a GPU boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from splattie.methods.lhm import runtime
from splattie.methods.lhm.assets import MODEL_REVISION, model_snapshot


def cached_snapshot(vendor: Path, revision: str) -> Path:
    cache = vendor / "pretrained_models" / "huggingface" / "models--3DAIGC--LHM-500M"
    snapshot = cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    return snapshot


def mutable_main(vendor: Path) -> Path:
    snapshot = cached_snapshot(vendor, "0" * 40)
    refs = snapshot.parents[1] / "refs"
    refs.mkdir()
    (refs / "main").write_text("0" * 40)
    return snapshot


def test_pinned_snapshot_ignores_a_different_cached_main(tmp_path: Path) -> None:
    pinned = cached_snapshot(tmp_path, MODEL_REVISION)
    other = mutable_main(tmp_path)
    assert model_snapshot(tmp_path) == pinned
    assert pinned != other


def test_cached_main_cannot_replace_a_missing_pinned_snapshot(tmp_path: Path) -> None:
    mutable_main(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"setup-gpu\.sh"):
        model_snapshot(tmp_path)


@pytest.mark.parametrize("fails", [False, True])
def test_inferrer_receives_pinned_path_and_restores_process_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fails: bool,
) -> None:
    pinned = cached_snapshot(tmp_path, MODEL_REVISION)
    mutable_main(tmp_path)
    monkeypatch.setattr(runtime, "VENDOR_LHM", tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    vendor_module = ModuleType("LHM.runners.infer.human_lrm")
    received: list[tuple[tuple[str, ...], Path]] = []

    class GPUInferrer:
        pose_estimator = None
        facedetect = None
        parsingnet = None

        def __init__(self) -> None:
            received.append((tuple(sys.argv), Path.cwd()))
            if fails:
                msg = "GPU construction failed"
                raise RuntimeError(msg)

    vendor_module.HumanLRMInferrer = GPUInferrer
    monkeypatch.setitem(sys.modules, "LHM.runners.infer.human_lrm", vendor_module)
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    original_argv = sys.argv
    original_cwd = Path.cwd()
    try:
        if fails:
            with pytest.raises(RuntimeError, match="GPU construction"):
                runtime.load_inferrer()
        else:
            inferrer = runtime.load_inferrer()
            assert runtime.load_inferrer() is inferrer
        argv, cwd = received[0]
        assert "model_name=LHM-500M" in argv
        assert f"model_path={pinned}" in argv
        assert cwd == tmp_path
        assert sys.argv is original_argv
        assert Path.cwd() == original_cwd
    finally:
        runtime.unload_model()
