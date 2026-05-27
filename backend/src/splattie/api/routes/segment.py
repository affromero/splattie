"""Segmentation endpoint (server-side fallback for SAM 3)."""

from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, UploadFile
from PIL import Image

from splattie.segmentation.sam3 import segment_head
from splattie.storage.local import save_upload
from splattie.types import SegmentationResult

router = APIRouter()


@router.post("/segment")
async def segment(image: UploadFile) -> SegmentationResult:
    """Segment the head from an uploaded photo."""
    contents = await image.read()
    file_id, _file_path = save_upload(contents)

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)
    mask = segment_head(img_array)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_path = _file_path.parent / "mask.png"
    mask_img.save(mask_path)

    preview = img.copy()
    overlay = np.array(preview)
    overlay[~mask] = (overlay[~mask] * 0.3).astype(np.uint8)
    preview_img = Image.fromarray(overlay)
    preview_path = _file_path.parent / "preview.png"
    preview_img.save(preview_path)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    bbox = (int(cols[0]), int(rows[0]), int(cols[-1] - cols[0]), int(rows[-1] - rows[0]))

    return SegmentationResult(
        mask_url=f"/storage/{file_id}/mask.png",
        preview_url=f"/storage/{file_id}/preview.png",
        bbox=bbox,
    )
