"""Head generation endpoint with SSE progress streaming."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from splattie.methods.base import AssetGenerationMethod
from splattie.methods.registry import registry
from splattie.types import AssetType, ReconstructBackend

router = APIRouter()

# Module-level so FastAPI resolves it via get_type_hints (a `Query(...)` written
# directly in the signature becomes an unresolvable string under
# `from __future__ import annotations`).
AssetTypeQuery = Annotated[AssetType, Query(alias="assetType")]
BackendQuery = Annotated[ReconstructBackend | None, Query(alias="backend")]


def _method_for(asset_type: AssetType, backend: ReconstructBackend | None) -> AssetGenerationMethod:
    """Resolve the method for a category, honoring a per-request reconstruction backend.

    Only the quadruped pipeline currently offers the TripoSplat/TRELLIS choice; a non-None backend
    returns a fresh instance of the same method configured with it. Constructing from the resolved
    singleton's class (rather than importing the concrete method) keeps this route decoupled from a
    specific method and avoids perturbing registration order. Other categories reuse the singleton.
    """
    method = registry.for_asset_type(asset_type)
    if backend is not None and asset_type is AssetType.quadruped_mammal:
        return type(method)(backend=backend)
    return method


@router.post("/generate")
async def generate(request: Request) -> StreamingResponse:
    """Generate a 3DGS asset from image/mask URLs. Returns an SSE stream.

    ``assetType`` in the body selects the category (head/body/object); the registry
    resolves the method behind it.
    """
    body = await request.json()
    image_url: str = body["image_url"]
    mask_url: str = body["mask_url"]
    asset_type = AssetType(body.get("assetType", AssetType.head.value))
    backend = ReconstructBackend(body["backend"]) if body.get("backend") else None

    return StreamingResponse(
        _generation_stream(image_url, mask_url, asset_type, backend),
        media_type="text/event-stream",
    )


@router.post("/generate-from-upload")
async def generate_from_upload(
    image: UploadFile,
    asset_type: AssetTypeQuery = AssetType.head,
    backend: BackendQuery = None,
) -> JSONResponse:
    """Generate a 3DGS asset directly from an uploaded image. Returns JSON.

    ``?assetType=<head|body|object|quadruped_mammal>`` selects the asset *category*; the registry
    resolves the method behind it. ``?backend=<triposplat|trellis>`` optionally selects the quadruped
    reconstruction backend (ignored by the other categories).
    """
    import io

    contents = await image.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    mask = np.ones(img.shape[:2], dtype=np.bool_)

    gen = _method_for(asset_type, backend)
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
    backend: ReconstructBackend | None,
) -> AsyncGenerator[str, None]:
    yield _sse("progress", {"stage": "loading_model", "pct": 10})

    method = _method_for(asset_type, backend)
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


def _sse(event: str, data: object) -> str:
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
