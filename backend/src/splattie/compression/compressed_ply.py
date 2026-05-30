"""PLY -> compressed PLY via splat-transform (npm CLI).

splat-transform's .compressed.ply is a quantized, chunked PLY that keeps the 'ply'
magic, so Spark loads it through its (correct) PLY reader and renders it faithfully
-- unlike SPZ, which Spark's SpzReader decodes with wrong rotations. ~4x smaller
than raw PLY with no visible quality loss.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from klogr import get_logger

logger = get_logger()


def compress_ply(ply_path: Path, out_path: Path) -> Path:
    """Compress a PLY to PlayCanvas compressed PLY via splat-transform.

    `out_path` must end in `.compressed.ply` (splat-transform picks the format from
    the extension). Hard-fails (raises RuntimeError) if splat-transform is missing
    or produces no output -- never returns the raw PLY, so a bundle can't silently
    ship uncompressed.
    """
    try:
        subprocess.run(
            ["npx", "@playcanvas/splat-transform", str(ply_path), str(out_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        msg = "splat-transform not found; install @playcanvas/splat-transform (npm)"
        raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        msg = f"splat-transform failed for {ply_path.name}: {exc.stderr or exc.stdout}"
        raise RuntimeError(msg) from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        msg = f"splat-transform produced no output at {out_path}"
        raise RuntimeError(msg)

    logger.info(f"Compressed {ply_path.name} -> {out_path.name} ({out_path.stat().st_size // 1024} KB)")
    return out_path
