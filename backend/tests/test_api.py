"""API endpoint tests."""

from __future__ import annotations

import io

from httpx import AsyncClient
from PIL import Image


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "lam" in data["methods_loaded"]


async def test_models(client: AsyncClient) -> None:
    response = await client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["default"] == "lam"
    assert len(data["methods"]) == 1
    assert data["methods"][0]["id"] == "lam"


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
    assert "mask_url" in data
    assert "preview_url" in data
    assert len(data["bbox"]) == 4


async def test_generate_from_upload(client: AsyncClient) -> None:
    img = Image.new("RGB", (200, 200), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = await client.post(
        "/generate-from-upload",
        files={"image": ("test.png", buf, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "modelId" in data
    assert "zipUrl" in data


async def test_generate(client: AsyncClient) -> None:
    img = Image.new("RGB", (200, 200), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    seg_response = await client.post(
        "/segment",
        files={"image": ("test.png", buf, "image/png")},
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
    assert '"spzUrl"' in text
