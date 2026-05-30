#!/usr/bin/env python3
"""Shrink a FLAME expression-basis .bin from float32 (EXPR) to float16 (EXPH).

Halves the download (~12 MB -> ~6 MB) with no change to vertex/expression counts.
Runs anywhere (no GPU). The widget loader decodes the halves back to float32 at
load time, so in-memory behaviour is unchanged within half-float tolerance.

Usage:
    python backend/scripts/shrink_expression_basis.py \\
        --basis-path apps/web/public/demos/expression_basis.bin
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import tyro
from klogr import get_logger

logger = get_logger()


def main(basis_path: Path, output: Path | None = None) -> None:
    """Transcode an expression-basis .bin to float16.

    Args:
        basis_path: Path to the float32 (EXPR) expression_basis.bin.
        output: Where to write the float16 (EXPH) .bin. Defaults to overwriting basis_path.

    """
    out = output or basis_path
    data = basis_path.read_bytes()
    magic = data[:4]
    if magic == b"EXPH":
        logger.info(f"{basis_path.name} is already float16 (EXPH); nothing to do.")
        return
    if magic != b"EXPR":
        msg = f"Unexpected magic {magic!r} in {basis_path} (expected EXPR)"
        raise ValueError(msg)

    num_verts, num_expr = struct.unpack("<II", data[4:12])
    basis32 = np.frombuffer(data, dtype="<f4", offset=12).reshape(num_verts, num_expr, 3)
    basis16 = basis32.astype(np.float16)

    with out.open("wb") as f:
        f.write(b"EXPH")
        f.write(struct.pack("<II", num_verts, num_expr))
        f.write(basis16.tobytes())

    before, after = len(data), out.stat().st_size
    logger.info(
        f"{out.name}: {before // 1024} KB (f32) -> {after // 1024} KB (f16), {num_verts} verts x {num_expr} expr"
    )

    # Keep the sidecar JSON in sync (the widget reads its labels).
    json_path = basis_path.with_suffix(".json")
    if json_path.exists():
        meta = json.loads(json_path.read_text())
        meta["bytes"] = int(after)
        meta["format"] = "float16_le, shape (num_vertices, num_expressions, 3)"
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Updated sidecar: {out.with_suffix('.json').name}")


if __name__ == "__main__":
    tyro.cli(main)
