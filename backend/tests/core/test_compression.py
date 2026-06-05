"""Compression tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from splattie.compression.bundle import compress_bundle
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


def _manifest_bytes(splat_file: str) -> bytes:
    """Build a minimal valid `.splattie` manifest pointing at `splat_file`."""
    manifest = {
        "format": "splattie",
        "formatVersion": "0.3.1",
        "assetType": "object",
        "avatar": {"splat": {"file": splat_file, "format": "ply", "numGaussians": 3}},
        "animation": {"skeleton": {"file": "skeleton.json"}, "weights": {"file": "lbs.bin"}},
        "widget": {"config": "states.json"},
    }
    return json.dumps(manifest, indent=2).encode("utf-8")


def _write_bundle(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)


def test_compress_bundle_swaps_only_the_ply_and_preserves_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recompress swaps only the inner PLY and leaves all sibling entries byte-identical."""
    src = tmp_path / "demo.splattie"
    manifest = _manifest_bytes("avatar.ply")
    skeleton = b'{"names":["neck"]}'
    states = b'{"gaze":true}'
    raw_ply = b"ply\nraw full-precision positions for an animal head"
    _write_bundle(
        src,
        [("manifest.json", manifest), ("avatar.ply", raw_ply), ("skeleton.json", skeleton), ("states.json", states)],
    )

    compressed = b"ply\npacked_position quantized chunk payload"

    def _fake_compress(_ply_path: Path, out_path: Path) -> Path:
        out_path.write_bytes(compressed)
        return out_path

    monkeypatch.setattr("splattie.compression.bundle.compress_ply", _fake_compress)

    dst = tmp_path / "out.splattie"
    result = compress_bundle(src, dst)

    assert result.already_compressed is False
    assert result.splat_entry == "avatar.ply"
    with zipfile.ZipFile(str(dst)) as zf:
        assert zf.read("avatar.ply") == compressed  # the only entry that changes
        assert zf.read("manifest.json") == manifest  # format stays "ply"; filename unchanged
        assert zf.read("skeleton.json") == skeleton
        assert zf.read("states.json") == states
        assert zf.namelist() == ["manifest.json", "avatar.ply", "skeleton.json", "states.json"]


def test_compress_bundle_is_idempotent_on_already_compressed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass an already-compressed bundle through untouched without invoking the compressor."""
    src = tmp_path / "demo.splattie"
    manifest = _manifest_bytes("avatar.ply")
    already = b"ply\npacked_position already-quantized data"
    _write_bundle(src, [("manifest.json", manifest), ("avatar.ply", already), ("states.json", b"{}")])

    def _must_not_run(_ply_path: Path, _out_path: Path) -> Path:
        msg = "compress_ply must not run on an already-compressed bundle"
        raise AssertionError(msg)

    monkeypatch.setattr("splattie.compression.bundle.compress_ply", _must_not_run)

    dst = tmp_path / "out.splattie"
    result = compress_bundle(src, dst)

    assert result.already_compressed is True
    assert result.size_after == result.size_before
    with zipfile.ZipFile(str(dst)) as zf:
        assert zf.read("avatar.ply") == already
        assert zf.read("manifest.json") == manifest
