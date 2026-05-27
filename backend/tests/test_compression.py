"""Compression tests."""

from __future__ import annotations

from pathlib import Path

from splattie.compression.spz import compress_ply_to_spz


def test_compress_fallback(tmp_path: Path) -> None:
    """Without splat-transform, compression returns the original PLY path."""
    ply_path = tmp_path / "test.ply"
    ply_path.write_bytes(b"fake ply data")
    spz_path = tmp_path / "test.spz"

    result = compress_ply_to_spz(ply_path, spz_path)
    assert result == ply_path
