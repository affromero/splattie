"""API endpoint tests."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from httpx import AsyncClient
from PIL import Image

from tests.gpu import GPU_TEST_SKIP_REASON, gpu_tests_enabled

# A committed demo portrait — LAM's FLAME tracking needs a real face.
_FACE_IMAGE = Path(__file__).resolve().parents[3] / "apps" / "web" / "public" / "demos" / "heads" / "h1.jpg"


def _face_png_bytes() -> io.BytesIO:
    buf = io.BytesIO()
    Image.open(_FACE_IMAGE).convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf


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


@pytest.mark.parametrize(
    ("mode", "color", "expected_rgb"),
    [
        ("RGB", (120, 80, 40), (120, 80, 40)),
        ("RGBA", (120, 80, 40, 100), (120, 80, 40)),
        ("L", 100, (100, 100, 100)),
        ("P", 1, (120, 80, 40)),
    ],
)
async def test_segmentation_preserves_colors_and_returns_masked_pngs(
    client: AsyncClient,
    mode: str,
    color: int | tuple[int, ...],
    expected_rgb: tuple[int, int, int],
) -> None:
    img = Image.new(mode, (200, 200), color)
    if mode == "P":
        img.putpalette([0, 0, 0, 120, 80, 40] + [0] * 762)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = await client.post(
        "/segment",
        files={"image": ("test.png", buf, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["bbox"] == [50, 50, 100, 100]

    mask_response = await client.get(data["maskUrl"])
    preview_response = await client.get(data["previewUrl"])
    assert mask_response.status_code == 200
    assert preview_response.status_code == 200

    with Image.open(io.BytesIO(mask_response.content)) as mask:
        assert mask.format == "PNG"
        assert mask.size == (200, 200)
        assert mask.getpixel((100, 100)) == 255
        assert mask.getpixel((0, 0)) == 0

    with Image.open(io.BytesIO(preview_response.content)) as preview:
        assert preview.format == "PNG"
        assert preview.size == (200, 200)
        assert preview.getpixel((100, 100)) == expected_rgb
        assert preview.getpixel((0, 0)) == tuple(int(value * 0.3) for value in expected_rgb)


async def test_generate_from_upload_propagates_load_failure(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No silent fallback at the API layer: a model-load failure surfaces as an error.

    Forces the GPU/weights boundary to fail so it runs on any machine (GPU or not),
    proving /generate-from-upload does not serve a demo bundle when the model can't load.
    """
    import splattie.methods.lam.method as lam_method

    def _boom() -> object:
        msg = "simulated model-load failure (no GPU / weights)"
        raise RuntimeError(msg)

    monkeypatch.setattr(lam_method, "_load_model", _boom)

    img = Image.new("RGB", (200, 200), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure is fine; the point is NO silent fallback
        await client.post("/generate-from-upload", files={"image": ("test.png", buf, "image/png")})


@pytest.mark.skipif(not gpu_tests_enabled(), reason=GPU_TEST_SKIP_REASON)
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


@pytest.mark.skipif(not gpu_tests_enabled(), reason=GPU_TEST_SKIP_REASON)
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
            "image_url": seg_data["previewUrl"].replace("/preview.png", "/original.png"),
            "mask_url": seg_data["maskUrl"],
        },
    )
    assert gen_response.status_code == 200

    text = gen_response.text
    assert "event: progress" in text
    assert "event: complete" in text
    assert '"modelId"' in text
    assert '"splattieUrl"' in text
