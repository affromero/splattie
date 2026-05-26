"""PLY → SPZ compression."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compress_ply_to_spz(ply_bytes: bytes) -> bytes:
    """Compress a PLY file to SPZ format.

    When the spz package is available (GPU environment), uses Niantic's
    SPZ encoder for ~10x compression. Falls back to passthrough in dev.
    """
    try:
        import spz

        gaussians = spz.load_ply(ply_bytes)
        return spz.save_spz(gaussians)
    except ImportError:
        logger.warning("spz package not available — returning raw bytes (dev mode)")
        return ply_bytes
