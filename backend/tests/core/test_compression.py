"""Compression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from splattie.compression.compressed_ply import compress_ply


def test_compress_hard_fails_on_invalid_input(tmp_path: Path) -> None:
    """Invalid input (or a missing tool) raises rather than silently returning the PLY.

    A silent fallback would let a bundle ship uncompressed (or mislabeled), so
    compression must hard-fail.
    """
    ply_path = tmp_path / "test.ply"
    ply_path.write_bytes(b"fake ply data")
    out_path = tmp_path / "test.compressed.ply"

    with pytest.raises(RuntimeError):
        compress_ply(ply_path, out_path)

    assert not out_path.exists() or out_path.stat().st_size == 0
