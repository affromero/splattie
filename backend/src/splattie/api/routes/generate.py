"""Head generation endpoint with SSE progress streaming."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator

import numpy as np
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from splattie.methods.registry import registry

router = APIRouter()


@router.post("/generate")
async def generate(request: Request) -> StreamingResponse:
    """Generate a 3DGS head from image/mask URLs. Returns SSE stream."""
    body = await request.json()
    image_url: str = body["image_url"]
    mask_url: str = body["mask_url"]
    method_id: str = body.get("method", registry.default_method_id or "lam")

    return StreamingResponse(
        _generation_stream(image_url, mask_url, method_id),
        media_type="text/event-stream",
    )


@router.post("/generate-from-upload")
async def generate_from_upload(image: UploadFile, method: str | None = None) -> JSONResponse:
    """Generate a 3DGS asset directly from an uploaded image. Returns JSON.

    The ``?method=<id>`` query param selects the generation method (e.g. ``lam``
    for heads, ``lhm`` for bodies); defaults to the registry's default method.
    """
    import io

    contents = await image.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    mask = np.ones(img.shape[:2], dtype=np.bool_)

    method_id = method or registry.default_method_id or "lam"
    gen = registry.get(method_id)
    gen.load()

    start = time.monotonic()
    result = gen.generate(img, mask)
    elapsed = time.monotonic() - start

    return JSONResponse(
        {
            "modelId": result.model_id,
            "zipUrl": result.spz_url,
            "zipSizeBytes": result.spz_size_bytes,
            "numGaussians": result.num_gaussians,
            "methodId": result.method_id,
            "inferenceSeconds": round(elapsed, 2),
        }
    )


async def _generation_stream(
    image_url: str,
    mask_url: str,
    method_id: str,
) -> AsyncGenerator[str, None]:
    yield _sse("progress", {"stage": "loading_model", "pct": 10})

    method = registry.get(method_id)
    method.load()

    yield _sse("progress", {"stage": "loading_image", "pct": 20})

    image_path = _url_to_local_path(image_url)
    mask_path = _url_to_local_path(mask_url)

    img = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127

    yield _sse("progress", {"stage": "generating_head", "pct": 50})

    start = time.monotonic()
    result = method.generate(img, mask)
    elapsed = time.monotonic() - start

    yield _sse("progress", {"stage": "done", "pct": 100})
    yield _sse(
        "complete",
        {
            "modelId": result.model_id,
            "spzUrl": result.spz_url,
            "spzSizeBytes": result.spz_size_bytes,
            "numGaussians": result.num_gaussians,
            "methodId": result.method_id,
            "rigParamsUrl": result.rig_params_url,
            "inferenceSeconds": round(elapsed, 2),
        },
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
