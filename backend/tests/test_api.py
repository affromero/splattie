"""API endpoint tests."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from httpx import AsyncClient
from PIL import Image

# A committed demo portrait — LAM's FLAME tracking needs a real face.
_FACE_IMAGE = Path(__file__).resolve().parents[2] / "apps" / "web" / "public" / "demos" / "thumbs" / "3762763.jpg"


def _face_png_bytes() -> io.BytesIO:
    buf = io.BytesIO()
    Image.open(_FACE_IMAGE).convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf


def cuda_available() -> bool:
    """Return True when torch and a CUDA device are present (real GPU runner)."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "lam" in data["methodsLoaded"]
    assert isinstance(data["gpu"], dict)
    assert "available" in data["gpu"]
    assert "modelLoaded" in data["gpu"]


async def test_models(client: AsyncClient) -> None:
    response = await client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["default"] == "lam"
    by_id = {m["id"]: m for m in data["methods"]}
    assert by_id["lam"]["assetType"] == "head"
    assert by_id["lhm"]["assetType"] == "body"


async def test_segment(client: AsyncClient) -> None:
    img = Image.new("RGB", (200, 200), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = await client.post(
        "/segment",
        files={"image": ("test.png", buf, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "maskUrl" in data
    assert "previewUrl" in data
    assert len(data["bbox"]) == 4


async def test_generate_from_upload_without_gpu_errors(client: AsyncClient) -> None:
    """No-fallback contract at the API layer: with no GPU, /generate-from-upload errors."""
    if cuda_available():
        pytest.skip("GPU present — success path covered by test_generate_from_upload_produces_bundle")

    img = Image.new("RGB", (200, 200), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure is fine; the point is NO silent fallback
        await client.post("/generate-from-upload", files={"image": ("test.png", buf, "image/png")})


@pytest.mark.skipif(not cuda_available(), reason="LAM inference requires CUDA + weights")
@pytest.mark.skipif(not _FACE_IMAGE.exists(), reason="demo portrait not present")
async def test_generate_from_upload_produces_bundle(client: AsyncClient) -> None:
    response = await client.post(
        "/generate-from-upload",
        files={"image": ("face.png", _face_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "modelId" in data
    assert data["splattieUrl"].endswith(".splattie")


@pytest.mark.skipif(not cuda_available(), reason="LAM inference requires CUDA + weights")
@pytest.mark.skipif(not _FACE_IMAGE.exists(), reason="demo portrait not present")
async def test_generate(client: AsyncClient) -> None:
    seg_response = await client.post(
        "/segment",
        files={"image": ("face.png", _face_png_bytes(), "image/png")},
    )
    seg_data = seg_response.json()

    gen_response = await client.post(
        "/generate",
        json={
            "image_url": seg_data["preview_url"].replace("/preview.png", "/original.png"),
            "mask_url": seg_data["mask_url"],
        },
    )
    assert gen_response.status_code == 200

    text = gen_response.text
    assert "event: progress" in text
    assert "event: complete" in text
    assert '"modelId"' in text
    assert '"splattieUrl"' in text
