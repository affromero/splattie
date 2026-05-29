"""Asset generation method tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from splattie.methods.base import AssetGenerationMethod
from splattie.methods.lam.method import LAMMethod
from splattie.methods.lhm.method import LHMMethod
from splattie.methods.registry import registry
from splattie.types import AssetType

# A committed demo portrait — LAM's FLAME tracking needs a real face.
_FACE_IMAGE = Path(__file__).resolve().parents[2] / "apps" / "web" / "public" / "demos" / "thumbs" / "3762763.jpg"


def cuda_available() -> bool:
    """Return True when torch and a CUDA device are present (real GPU runner)."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def test_lam_implements_protocol() -> None:
    method = LAMMethod()
    assert isinstance(method, AssetGenerationMethod)


def test_lam_info() -> None:
    method = LAMMethod()
    assert method.info.id == "lam"
    assert "SIGGRAPH" in method.info.name


def test_lam_is_a_head_method() -> None:
    assert LAMMethod().info.asset_type is AssetType.HEAD
    assert LAMMethod().info.asset_type == "head"


def test_lam_capabilities() -> None:
    method = LAMMethod()
    caps = method.capabilities
    assert caps.supports_single_image is True
    assert caps.max_output_gaussians > 0


@pytest.mark.skipif(cuda_available(), reason="GPU present — covered by produces-bundle test")
def test_lam_generate_raises_without_gpu() -> None:
    """No-fallback contract: with no CUDA/weights, generation raises (no demo bundle)."""
    method = LAMMethod()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.bool_)
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure is acceptable; the point is it does NOT fall back
        method.generate(image, mask)


@pytest.mark.skipif(not cuda_available(), reason="LAM inference requires CUDA + weights")
@pytest.mark.skipif(not _FACE_IMAGE.exists(), reason="demo portrait not present")
def test_lam_generate_produces_bundle() -> None:
    """On a real GPU, generation produces a widget-loadable `.splattie` bundle."""
    from PIL import Image

    method = LAMMethod()
    method.load()

    image = np.array(Image.open(_FACE_IMAGE).convert("RGB"))
    mask = np.ones(image.shape[:2], dtype=np.bool_)

    result = method.generate(image, mask)
    assert result.method_id == "lam"
    assert result.num_gaussians > 0
    assert result.spz_url.endswith(".splattie")
    assert result.rig_params_url.endswith(".splattie")

    method.unload()


def test_registry_has_lam() -> None:
    methods = registry.list_available()
    ids = [m.id for m in methods]
    assert "lam" in ids


def test_registry_get() -> None:
    method = registry.get("lam")
    assert method.info.id == "lam"


def test_registry_default() -> None:
    assert registry.default_method_id == "lam"


def test_lhm_implements_protocol() -> None:
    assert isinstance(LHMMethod(), AssetGenerationMethod)


def test_lhm_is_a_body_method() -> None:
    assert LHMMethod().info.asset_type is AssetType.BODY
    assert LHMMethod().info.asset_type == "body"


def test_lhm_registered() -> None:
    assert registry.get("lhm").info.id == "lhm"
    assert registry.get("lhm").info.asset_type == "body"


@pytest.mark.skipif(cuda_available(), reason="GPU present — covered by produces-bundle test")
def test_lhm_generate_raises_without_gpu() -> None:
    """No-fallback contract: body generation raises without CUDA/weights."""
    method = LHMMethod()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.bool_)
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure is fine; no silent fallback
        method.generate(image, mask)
