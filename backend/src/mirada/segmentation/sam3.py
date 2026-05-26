"""SAM 3 segmentation wrapper (server-side fallback)."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def segment_head(image: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Segment the head from an image using SAM 3.

    This is the server-side fallback for browsers without WebGPU.
    Primary segmentation happens client-side via SAM 3 ONNX.
    """
    h, w = image.shape[:2]
    logger.info("SAM 3 segmentation (stub): %dx%d image", w, h)

    mask = np.zeros((h, w), dtype=np.bool_)
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask[(x - cx) ** 2 + (y - cy) ** 2 <= radius**2] = True

    return mask
