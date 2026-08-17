"""Tests for maintaining existing `.splattie` bundles."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from splattie.cli.bundle_tools import build_legacy_manifest, rebundle
from splattie.types import AssetType

_MINIMAL_PLY = b"""ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
end_header
0 0 0
"""


def test_head_manifest_references_bundled_expression_basis() -> None:
    manifest = build_legacy_manifest(
        stem="head",
        splat_entry="head.ply",
        splat_format="ply",
        num_gaussians=3,
        thumb_path=None,
        widget_version="0.3.1",
        asset_type=AssetType.head,
        names={
            "head.ply",
            "bone_tree.json",
            "lbs_weight_20k.json",
            "expression_basis.bin",
            "states.json",
        },
    )

    assert manifest["animation"]["expression"] == {
        "system": "flame-pca",
        "basis": "expression_basis.bin",
    }


def test_rebundle_repairs_current_manifest_missing_expression_basis(tmp_path: Path) -> None:
    bundle = tmp_path / "head.splattie"
    manifest = build_legacy_manifest(
        stem="head",
        splat_entry="head.ply",
        splat_format="ply",
        num_gaussians=1,
        thumb_path=None,
        widget_version="0.3.1",
        asset_type=AssetType.head,
        names={"head.ply", "expression_basis.bin", "states.json"},
    )
    manifest["animation"]["expression"]["basis"] = None
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("head.ply", _MINIMAL_PLY)
        zf.writestr("expression_basis.bin", b"basis")
        zf.writestr("states.json", "{}")

    status = rebundle(bundle, tmp_path, "0.3.1", AssetType.head)

    assert status.startswith("rebundled")
    with zipfile.ZipFile(bundle) as zf:
        repaired = json.loads(zf.read("manifest.json"))
    assert repaired["animation"]["expression"]["basis"] == "expression_basis.bin"


def _make_head_bundle(path: Path, num_gaussians: int, *, basis_magic: bytes = b"EXPH") -> None:
    """Write a synthetic head .splattie with consistent PLY/LBS/basis rows."""
    import numpy as np

    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4")]
    vertices = np.zeros(num_gaussians, dtype=np.dtype(props))
    vertices["x"] = np.arange(num_gaussians, dtype=np.float32)
    ply = (
        b"ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {num_gaussians}\n".encode()
        + b"property float x\nproperty float y\nproperty float z\nproperty float opacity\n"
        + b"end_header\n"
        + vertices.tobytes()
    )
    weights = [[float(i), 0.0, 0.0, 0.0, 0.0] for i in range(num_gaussians)]
    num_expr = 2
    basis_dtype = "<f2" if basis_magic == b"EXPH" else "<f4"
    basis = np.arange(num_gaussians * num_expr * 3).reshape(num_gaussians, num_expr, 3)
    basis_bytes = (
        basis_magic + np.asarray([num_gaussians, num_expr], dtype="<u4").tobytes() + basis.astype(basis_dtype).tobytes()
    )
    manifest = {
        "format": "splattie",
        "formatVersion": "0.3.3",
        "assetType": "head",
        "avatar": {"splat": {"file": "head.ply", "format": "ply", "numGaussians": num_gaussians}},
        "animation": {
            "type": "lbs",
            "expression": {"system": "flame-pca", "basis": "expression_basis.bin"},
            "skeleton": {"file": "bone_tree.json", "rig": "flame"},
            "weights": {"file": "lbs_weight_20k.json"},
        },
        "widget": {"config": "states.json"},
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("head.ply", ply)
        zf.writestr("lbs_weight_20k.json", json.dumps(weights))
        zf.writestr("expression_basis.bin", basis_bytes)
        zf.writestr("bone_tree.json", "{}")
        zf.writestr("states.json", "{}")


def test_downsample_subsets_ply_and_weights_and_drops_basis(tmp_path: Path) -> None:
    import numpy as np

    from splattie.cli.bundle_tools import downsample

    src = tmp_path / "head.splattie"
    out = tmp_path / "head-mobile.splattie"
    _make_head_bundle(src, 100)

    downsample(src, out, max_gaussians=10)

    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        names = set(zf.namelist())
        ply = zf.read("head.ply")
        weights = json.loads(zf.read("lbs_weight_20k.json"))

    kept = manifest["avatar"]["splat"]["numGaussians"]
    assert kept == 10
    assert "expression_basis.bin" not in names
    assert manifest["animation"]["expression"]["basis"] is None
    assert len(weights) == kept

    # PLY vertex ids (x property) must line up with the LBS weight rows.
    header_end = ply.index(b"end_header\n") + len(b"end_header\n")
    verts = np.frombuffer(
        ply[header_end:], dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4")])
    )
    assert len(verts) == kept
    assert [row[0] for row in weights] == verts["x"].tolist()


def test_downsample_keeps_and_subsets_expression_basis(tmp_path: Path) -> None:
    import numpy as np

    from splattie.cli.bundle_tools import downsample

    src = tmp_path / "head.splattie"
    out = tmp_path / "head-mobile.splattie"
    _make_head_bundle(src, 50)

    downsample(src, out, max_gaussians=5, keep_expression_basis=True)

    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        data = zf.read("expression_basis.bin")
        ply = zf.read("head.ply")

    assert manifest["animation"]["expression"]["basis"] == "expression_basis.bin"
    assert data[:4] == b"EXPH"
    num_verts, num_expr = np.frombuffer(data, dtype="<u4", count=2, offset=4).tolist()
    kept = manifest["avatar"]["splat"]["numGaussians"]
    assert (num_verts, num_expr) == (kept, 2)

    # Basis rows must correspond to the surviving vertex ids.
    header_end = ply.index(b"end_header\n") + len(b"end_header\n")
    verts = np.frombuffer(
        ply[header_end:], dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4")])
    )
    basis = np.frombuffer(data, dtype="<f2", offset=12).reshape(num_verts, num_expr, 3)
    expected_first_component = verts["x"] * num_expr * 3
    assert np.allclose(basis[:, 0, 0], expected_first_component)


def test_downsample_noop_when_under_target_still_drops_basis(tmp_path: Path) -> None:
    from splattie.cli.bundle_tools import downsample

    src = tmp_path / "head.splattie"
    out = tmp_path / "head-mobile.splattie"
    _make_head_bundle(src, 8)

    downsample(src, out, max_gaussians=100)

    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        weights = json.loads(zf.read("lbs_weight_20k.json"))

    assert manifest["avatar"]["splat"]["numGaussians"] == 8
    assert len(weights) == 8
    assert manifest["animation"]["expression"]["basis"] is None
