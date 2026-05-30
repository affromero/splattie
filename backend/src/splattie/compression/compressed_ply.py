"""PLY → SPZ compression via splat-transform (npm CLI)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from klogr import get_logger

logger = get_logger()


def compress_ply_to_spz(ply_path: Path, spz_path: Path) -> Path:
    """Compress a PLY file to SPZ using @playcanvas/splat-transform.

    Hard-fails (raises RuntimeError) if splat-transform is unavailable or produces
    no output. Never returns the raw PLY: a bundle stamped format="spz" that
    actually holds a PLY would silently break the widget's loader.
    """
    try:
        subprocess.run(
            ["npx", "@playcanvas/splat-transform", str(ply_path), str(spz_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        msg = "splat-transform not found; install @playcanvas/splat-transform (npm)"
        raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        msg = f"splat-transform failed for {ply_path.name}: {exc.stderr or exc.stdout}"
        raise RuntimeError(msg) from exc

    if not spz_path.exists() or spz_path.stat().st_size == 0:
        msg = f"splat-transform produced no SPZ output at {spz_path}"
        raise RuntimeError(msg)

    logger.info(f"Compressed {ply_path.name} → {spz_path.name} ({spz_path.stat().st_size // 1024} KB)")
    return spz_path
