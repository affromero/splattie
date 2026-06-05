"""Provisioning checks: every generation pipeline is wired and its weights are on disk.

A half-provisioned GPU box (CUDA present but a checkpoint missing) otherwise only fails
deep inside inference. These assert the concrete files each pipeline's loader reads, so a
broken setup fails fast and obviously. The weight-file checks are skipped off a GPU runner,
where the weights aren't downloaded (setup-gpu.sh); the registry/coverage check runs
everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from splattie.methods.lam.method import MODEL_ZOO, VENDOR_LAM
from splattie.methods.lhm.runtime import VENDOR_LHM
from splattie.methods.object.runtime import (
    PUPPETEER_MICHELANGELO,
    PUPPETEER_PARTFIELD,
    PUPPETEER_SKELETON_WEIGHTS,
    PUPPETEER_SKINNING_MICHELANGELO,
    PUPPETEER_SKINNING_WEIGHTS,
    TRIPOSPLAT_FLOW_MODEL,
    VENDOR_PUPPETEER,
    VENDOR_TRELLIS,
    VENDOR_TRIPOSPLAT,
)
from splattie.methods.quadruped_mammal.runtime import DLC_PYTHON, SMAL_PKL
from splattie.methods.registry import registry
from tests.gpu import GPU_TEST_SKIP_REASON, gpu_tests_enabled

_LHM_SNAPSHOTS = VENDOR_LHM / "pretrained_models" / "huggingface" / "models--3DAIGC--LHM-500M" / "snapshots"

# (pipeline id, asset name, path, kind) — the files each pipeline's loader reads at
# inference time. `lhm-checkpoint` globs the HF snapshot dir (the hash varies by release).
_PIPELINE_WEIGHTS: list[tuple[str, str, Path, str]] = [
    ("lam", "config", VENDOR_LAM / "configs" / "inference" / "lam-20k-8gpu.yaml", "file"),
    (
        "lam",
        "checkpoint",
        MODEL_ZOO / "lam_models" / "releases" / "lam" / "lam-20k" / "step_045500" / "model.safetensors",
        "file",
    ),
    ("lam", "faceboxes", MODEL_ZOO / "flame_tracking_models" / "FaceBoxesV2.pth", "file"),
    ("lam", "human-parametric-models", MODEL_ZOO / "human_parametric_models", "dir"),
    ("lhm", "checkpoint", _LHM_SNAPSHOTS, "lhm-checkpoint"),
    ("lhm", "human-model-files", VENDOR_LHM / "pretrained_models" / "human_model_files", "dir"),
    ("lhm", "smplx", VENDOR_LHM / "pretrained_models" / "human_model_files" / "smplx", "dir"),
    ("trellis-puppeteer", "trellis-package", VENDOR_TRELLIS / "trellis", "dir"),
    ("trellis-puppeteer", "puppeteer-skeleton-code", VENDOR_PUPPETEER / "skeleton" / "demo.py", "file"),
    ("trellis-puppeteer", "puppeteer-skinning-code", VENDOR_PUPPETEER / "skinning" / "main.py", "file"),
    ("trellis-puppeteer", "puppeteer-skeleton", PUPPETEER_SKELETON_WEIGHTS, "file"),
    ("trellis-puppeteer", "puppeteer-skinning", PUPPETEER_SKINNING_WEIGHTS, "file"),
    ("trellis-puppeteer", "michelangelo-skeleton", PUPPETEER_MICHELANGELO, "file"),
    ("trellis-puppeteer", "michelangelo-skinning", PUPPETEER_SKINNING_MICHELANGELO, "file"),
    ("trellis-puppeteer", "partfield", PUPPETEER_PARTFIELD, "file"),
    # Default backend is TripoSplat (cleaner animal faces); TRELLIS is the optional fallback.
    ("trellis-smal-quadruped", "triposplat-code", VENDOR_TRIPOSPLAT / "triposplat.py", "file"),
    ("trellis-smal-quadruped", "triposplat-flow-model", TRIPOSPLAT_FLOW_MODEL, "file"),
    ("trellis-smal-quadruped", "smal-model", SMAL_PKL, "file"),
    ("trellis-smal-quadruped", "deeplabcut-python", DLC_PYTHON, "file"),
]


def test_every_registered_pipeline_has_weight_checks() -> None:
    """All pipelines in place: every registered method is covered here.

    Adding a new pipeline forces adding its weight checks (and removing one is caught too).
    """
    registered = {m.id for m in registry.list_available()}
    covered = {pipeline for pipeline, _, _, _ in _PIPELINE_WEIGHTS}
    assert registered == covered, (
        f"pipelines missing weight checks: {registered - covered}; stale checks: {covered - registered}"
    )


@pytest.mark.skipif(not gpu_tests_enabled(), reason=GPU_TEST_SKIP_REASON)
@pytest.mark.parametrize(
    ("pipeline", "name", "path", "kind"),
    _PIPELINE_WEIGHTS,
    ids=[f"{pipeline}-{name}" for pipeline, name, _, _ in _PIPELINE_WEIGHTS],
)
def test_pipeline_weight_present(pipeline: str, name: str, path: Path, kind: str) -> None:
    """Each pipeline's weights/assets are on disk and non-empty."""
    if kind == "lhm-checkpoint":
        weights = list(path.glob("*/model.safetensors"))
        assert weights, f"{pipeline} {name}: no model.safetensors under {path}"
        assert all(w.stat().st_size > 0 for w in weights), f"{pipeline} {name}: empty checkpoint"
        return

    assert path.exists(), f"{pipeline} {name} missing: {path}"
    if kind == "dir":
        assert path.is_dir(), f"{pipeline} {name} is not a directory: {path}"
    else:
        assert path.is_file(), f"{pipeline} {name} is not a file: {path}"
        assert path.stat().st_size > 0, f"{pipeline} {name} is empty: {path}"
