"""Rig-aware bundle compression tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from beartype import beartype
from jaxtyping import Float, jaxtyped

from splattie.compression import bundle as compression_bundle
from splattie.compression.bundle import compress_bundle_rigged, compressed_ply_permutation, reorder_sparse_weights
from splattie.methods.object.bundle import (
    BinaryPly,
    SparseLbsWeights,
    read_lbs_weights_binary,
    write_binary_ply,
    write_lbs_weights_binary,
)

FloatPoints = Float[npt.NDArray[np.float32], "points 3"]


@jaxtyped(typechecker=beartype)
def _points() -> FloatPoints:
    return np.asarray(
        [
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


@jaxtyped(typechecker=beartype)
def _write_positions_ply(path: Path, points: FloatPoints) -> None:
    rows = np.zeros(len(points), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]))
    rows["x"] = points[:, 0]
    rows["y"] = points[:, 1]
    rows["z"] = points[:, 2]
    write_binary_ply(path, BinaryPly(vertices=rows, properties=[("x", "float"), ("y", "float"), ("z", "float")]))


def _weights() -> SparseLbsWeights:
    return SparseLbsWeights(
        num_gaussians=4,
        joint_count=8,
        k=2,
        indices=[0, 1, 2, 3, 4, 5, 6, 7],
        weights=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
    )


def _bundle(path: Path, *, topology: str = "object-auto") -> Path:
    work = path.parent
    ply_path = work / "toy.ply"
    weights_path = work / "lbs_weights.bin"
    _write_positions_ply(ply_path, _points())
    write_lbs_weights_binary(weights_path, _weights())
    manifest = {
        "format": "splattie",
        "formatVersion": "0.3.1",
        "assetType": "object",
        "avatar": {"splat": {"file": "toy.ply", "format": "ply", "numGaussians": 4, "topology": topology}},
        "animation": {
            "type": "lbs",
            "skeleton": {"file": "skeleton.json", "rig": "puppeteer-object"},
            "weights": {"file": "lbs_weights.bin", "format": "lbsw-v1"},
            "expression": None,
        },
        "widget": {"config": "states.json"},
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.write(ply_path, "toy.ply")
        zf.writestr("skeleton.json", b'{"names":["root"]}\n')
        zf.write(weights_path, "lbs_weights.bin")
        zf.writestr("states.json", b'{"state":"idle"}\n')
    return path


def test_compressed_ply_permutation_matches_playcanvas_morton_order() -> None:
    recovered = compressed_ply_permutation(_points())

    assert recovered.tolist() == [1, 3, 2, 0]


def test_compressed_ply_permutation_recurses_large_equal_morton_buckets() -> None:
    near_origin = np.linspace(0.0003, 0.0, 300, dtype=np.float32)
    points = np.zeros((301, 3), dtype=np.float32)
    points[:300, 0] = near_origin
    points[300, 0] = 1.0

    recovered = compressed_ply_permutation(points)

    assert recovered[:300].tolist() == list(range(299, -1, -1))
    assert recovered[300] == 300


def test_reorder_sparse_weights_applies_gaussian_major_permutation() -> None:
    reordered = reorder_sparse_weights(_weights(), np.asarray([1, 3, 2, 0], dtype=np.int64))

    assert reordered.indices == [2, 3, 6, 7, 4, 5, 0, 1]
    assert np.allclose(reordered.weights, [0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.1, 0.9])


def test_compress_bundle_rigged_reorders_lbsw_and_preserves_archive_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = _bundle(tmp_path / "toy.splattie")
    dst = tmp_path / "toy.compressed.splattie"

    def fake_compress(_ply_path: Path, out_path: Path) -> Path:
        out_path.write_bytes(b"ply\ncomment packed_position\nend_header\ncompressed")
        return out_path

    monkeypatch.setattr(compression_bundle, "compress_ply", fake_compress)

    result = compress_bundle_rigged(src, dst)

    assert result.splat_entry == "toy.ply"
    assert result.weights_entry == "lbs_weights.bin"
    assert result.already_compressed is False
    with zipfile.ZipFile(src) as original, zipfile.ZipFile(dst) as compressed:
        assert compressed.namelist() == original.namelist()
        assert compressed.read("manifest.json") == original.read("manifest.json")
        assert compressed.read("skeleton.json") == original.read("skeleton.json")
        assert compressed.read("states.json") == original.read("states.json")
        assert b"packed_position" in compressed.read("toy.ply")
        weights_path = tmp_path / "reordered.bin"
        weights_path.write_bytes(compressed.read("lbs_weights.bin"))

    weights = read_lbs_weights_binary(weights_path)
    assert weights.indices == [2, 3, 6, 7, 4, 5, 0, 1]
    assert np.allclose(weights.weights, [0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.1, 0.9], atol=5e-4)


def test_compress_bundle_rigged_refuses_flame_topology(tmp_path: Path) -> None:
    src = _bundle(tmp_path / "head.splattie", topology="flame-20k")

    with pytest.raises(ValueError, match="flame-20k"):
        compress_bundle_rigged(src, tmp_path / "head.compressed.splattie")
