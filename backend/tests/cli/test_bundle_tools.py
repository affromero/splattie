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

    props = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
    ]
    vertices = np.zeros(num_gaussians, dtype=np.dtype(props))
    vertices["x"] = np.arange(num_gaussians, dtype=np.float32)
    vertices["opacity"] = 2.0  # equal importance unless a test overrides
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {num_gaussians}\n".encode()
        + b"".join(f"property float {name}\n".encode() for name, _ in props)
        + b"end_header\n"
    )
    ply = header + vertices.tobytes()
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
        ply[header_end:],
        dtype=np.dtype([(n, "<f4") for n in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")]),
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
        ply[header_end:],
        dtype=np.dtype([(n, "<f4") for n in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")]),
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


def test_downsample_rejects_non_head_bundle(tmp_path: Path) -> None:
    import pytest

    from splattie.cli.bundle_tools import downsample

    src = tmp_path / "obj.splattie"
    _make_head_bundle(src, 10)
    with zipfile.ZipFile(src) as zf:
        payload = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(payload["manifest.json"])
    manifest["assetType"] = "object"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        for name, data in payload.items():
            if name != "manifest.json":
                zf.writestr(name, data)

    with pytest.raises(ValueError, match="head bundles"):
        downsample(src, tmp_path / "out.splattie", max_gaussians=5)


def test_downsample_uses_manifest_declared_splat_entry(tmp_path: Path) -> None:
    from splattie.cli.bundle_tools import downsample

    src = tmp_path / "head.splattie"
    out = tmp_path / "out.splattie"
    _make_head_bundle(src, 20)
    # Prepend a decoy PLY that sorts first in zip order.
    with zipfile.ZipFile(src) as zf:
        payload = {n: zf.read(n) for n in zf.namelist()}
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("aaa-decoy.ply", _MINIMAL_PLY)
        for name, data in payload.items():
            zf.writestr(name, data)

    downsample(src, out, max_gaussians=5)

    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        weights = json.loads(zf.read("lbs_weight_20k.json"))
        decoy = zf.read("aaa-decoy.ply")

    assert manifest["avatar"]["splat"]["numGaussians"] == 5
    assert len(weights) == 5
    assert decoy == _MINIMAL_PLY  # untouched


def test_downsample_transcodes_expr_basis_to_exph(tmp_path: Path) -> None:
    from splattie.cli.bundle_tools import downsample

    src = tmp_path / "head.splattie"
    out = tmp_path / "out.splattie"
    _make_head_bundle(src, 30, basis_magic=b"EXPR")

    downsample(src, out, max_gaussians=6, keep_expression_basis=True)

    with zipfile.ZipFile(out) as zf:
        data = zf.read("expression_basis.bin")
    # The widget only loads EXPH float16, so EXPR float32 input must transcode.
    assert data[:4] == b"EXPH"


def test_downsample_handles_null_expression_and_missing_weights(tmp_path: Path) -> None:
    import pytest

    from splattie.cli.bundle_tools import downsample

    src = tmp_path / "head.splattie"
    _make_head_bundle(src, 10)
    with zipfile.ZipFile(src) as zf:
        payload = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(payload["manifest.json"])
    manifest["animation"]["expression"] = None
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        for name, data in payload.items():
            if name != "manifest.json":
                zf.writestr(name, data)

    out = tmp_path / "out.splattie"
    downsample(src, out, max_gaussians=4)
    with zipfile.ZipFile(out) as zf:
        assert json.loads(zf.read("manifest.json"))["avatar"]["splat"]["numGaussians"] == 4

    manifest["animation"]["weights"] = None
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        for name, data in payload.items():
            if name != "manifest.json":
                zf.writestr(name, data)
    with pytest.raises(ValueError, match="weights"):
        downsample(src, tmp_path / "out2.splattie", max_gaussians=4)


def test_downsample_keeps_high_importance_gaussians(tmp_path: Path) -> None:
    import numpy as np

    from splattie.cli.bundle_tools import downsample

    src = tmp_path / "head.splattie"
    out = tmp_path / "out.splattie"
    _make_head_bundle(src, 10)

    # Rewrite the PLY so splats 3 and 7 are near-invisible (tiny opacity+scale).
    with zipfile.ZipFile(src) as zf:
        payload = {n: zf.read(n) for n in zf.namelist()}
    header_end = payload["head.ply"].index(b"end_header\n") + len(b"end_header\n")
    dtype = np.dtype([(n, "<f4") for n in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")])
    verts = np.frombuffer(payload["head.ply"][header_end:], dtype=dtype).copy()
    for low in (3, 7):
        verts["opacity"][low] = -10.0
        verts[["scale_0", "scale_1", "scale_2"]][low] = (-10.0, -10.0, -10.0)
    with zipfile.ZipFile(src, "w") as zf:
        for name, data in payload.items():
            zf.writestr(name, data if name != "head.ply" else payload["head.ply"][:header_end] + verts.tobytes())

    downsample(src, out, max_gaussians=8)

    with zipfile.ZipFile(out) as zf:
        ply = zf.read("head.ply")
    kept = np.frombuffer(ply[ply.index(b"end_header\n") + len(b"end_header\n") :], dtype=dtype)
    assert sorted(kept["x"].tolist()) == [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0]


def test_downsample_scale_compensation_grows_survivors(tmp_path: Path) -> None:
    import numpy as np

    from splattie.cli.bundle_tools import downsample

    src = tmp_path / "head.splattie"
    out = tmp_path / "out.splattie"
    _make_head_bundle(src, 10)

    downsample(src, out, max_gaussians=10, scale_compensation=2.0)

    with zipfile.ZipFile(out) as zf:
        ply = zf.read("head.ply")
    dtype = np.dtype([(n, "<f4") for n in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")])
    kept = np.frombuffer(ply[ply.index(b"end_header\n") + len(b"end_header\n") :], dtype=dtype)
    # Log-scales: a 2x linear factor is an additive ln(2) shift from the 0.0 baseline.
    assert np.allclose(kept["scale_0"], np.log(2.0), atol=1e-6)
