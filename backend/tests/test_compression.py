"""Compression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from splattie.compression.spz import compress_ply_to_spz


def test_compress_hard_fails_on_invalid_input(tmp_path: Path) -> None:
    """Invalid input (or a missing tool) raises rather than silently returning the PLY.

    A silent PLY fallback would let a bundle be stamped format="spz" while holding a
    PLY, which breaks the widget loader — so compression must hard-fail.
    """
    ply_path = tmp_path / "test.ply"
    ply_path.write_bytes(b"fake ply data")
    spz_path = tmp_path / "test.spz"

    with pytest.raises(RuntimeError):
        compress_ply_to_spz(ply_path, spz_path)

    assert not spz_path.exists() or spz_path.stat().st_size == 0
