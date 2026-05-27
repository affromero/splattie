"""Head generation method tests."""

from __future__ import annotations

import numpy as np

from mirada.methods.base import HeadGenerationMethod
from mirada.methods.lam.method import LAMMethod
from mirada.methods.registry import registry


def test_lam_implements_protocol() -> None:
    method = LAMMethod()
    assert isinstance(method, HeadGenerationMethod)


def test_lam_info() -> None:
    method = LAMMethod()
    assert method.info.id == "lam"
    assert "SIGGRAPH" in method.info.name


def test_lam_capabilities() -> None:
    method = LAMMethod()
    caps = method.capabilities
    assert caps.supports_single_image is True
    assert caps.max_output_gaussians > 0


def test_lam_generate() -> None:
    method = LAMMethod()
    method.load()

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.bool_)

    result = method.generate(image, mask)
    assert result.method_id == "lam"
    assert result.num_gaussians == 20_000
    assert result.spz_url.endswith((".spz", ".zip"))
    assert result.flame_params_url.endswith((".json", ".zip"))

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
