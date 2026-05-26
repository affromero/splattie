"""Compression tests."""

from __future__ import annotations

from mirada.compression.spz import compress_ply_to_spz


def test_compress_fallback() -> None:
    """Without the spz package, compression returns raw bytes."""
    data = b"fake ply data"
    result = compress_ply_to_spz(data)
    assert result == data
