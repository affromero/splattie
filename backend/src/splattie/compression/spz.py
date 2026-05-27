"""PLY → SPZ compression via splat-transform (npm CLI)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def compress_ply_to_spz(ply_path: Path, spz_path: Path) -> Path:
    """Compress a PLY file to SPZ using @playcanvas/splat-transform.

    Falls back to returning the PLY path if splat-transform is not available.
    """
    try:
        subprocess.run(
            ["npx", "@playcanvas/splat-transform", str(ply_path), str(spz_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.warning("splat-transform not available, returning raw PLY")
        return ply_path
    else:
        logger.info("Compressed %s → %s (%d KB)", ply_path.name, spz_path.name, spz_path.stat().st_size // 1024)
        return spz_path
