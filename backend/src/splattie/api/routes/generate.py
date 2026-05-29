"""Head generation endpoint with SSE progress streaming."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator

import numpy as np
from fastapi import APIRouter, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from splattie.methods.registry import registry
from splattie.types import AssetType

router = APIRouter()


@router.post("/generate")
async def generate(request: Request) -> StreamingResponse:
    """Generate a 3DGS asset from image/mask URLs. Returns an SSE stream.

    ``assetType`` in the body selects the category (head/body/object); the registry
    resolves the method behind it.
    """
    body = await request.json()
    image_url: str = body["image_url"]
    mask_url: str = body["mask_url"]
    asset_type = AssetType(body.get("assetType", AssetType.HEAD.value))

    return StreamingResponse(
        _generation_stream(image_url, mask_url, asset_type),
        media_type="text/event-stream",
    )


@router.post("/generate-from-upload")
async def generate_from_upload(
    image: UploadFile,
    asset_type: Annotated[AssetType, Query(alias="assetType")] = AssetType.HEAD,
) -> JSONResponse:
    """Generate a 3DGS asset directly from an uploaded image. Returns JSON.

    ``?assetType=<head|body|object>`` selects the asset *category*; the registry
    resolves the method behind it (head→LAM, body→LHM), so the method can change
    without altering the URL.
    """
    import io

    contents = await image.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    mask = np.ones(img.shape[:2], dtype=np.bool_)

    gen = registry.for_asset_type(asset_type)
    gen.load()

    start = time.monotonic()
    result = gen.generate(img, mask)
    elapsed = time.monotonic() - start

    return JSONResponse(
        {**result.model_dump(by_alias=True), "inferenceSeconds": round(elapsed, 2)},
    )


async def _generation_stream(
    image_url: str,
    mask_url: str,
    asset_type: AssetType,
) -> AsyncGenerator[str, None]:
    yield _sse("progress", {"stage": "loading_model", "pct": 10})

    method = registry.for_asset_type(asset_type)
    method.load()

    yield _sse("progress", {"stage": "loading_image", "pct": 20})

    image_path = _url_to_local_path(image_url)
    mask_path = _url_to_local_path(mask_url)

    img = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127

    yield _sse("progress", {"stage": "generating", "pct": 50})

    start = time.monotonic()
    result = method.generate(img, mask)
    elapsed = time.monotonic() - start

    yield _sse("progress", {"stage": "done", "pct": 100})
    yield _sse(
        "complete",
        {**result.model_dump(by_alias=True), "inferenceSeconds": round(elapsed, 2)},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _url_to_local_path(url: str) -> str:
    """Convert a /storage/{id}/file URL to a local filesystem path."""
    from pathlib import Path

    parts = url.strip("/").split("/")
    if parts[0] == "storage":
        for base in [Path("data/uploads"), Path("data/generations")]:
            path = base / parts[1] / parts[2]
            if path.exists():
                return str(path)
    return url
