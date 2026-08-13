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
