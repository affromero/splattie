"""Tests for the shared `.splattie` bundler (CPU, no GPU needed)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from splattie.methods.bundle_common import (
    DEFAULT_STATES_HEAD,
    HEAD_RIG,
    WIDGET_PUBLIC,
    build_manifest,
    bundle_splattie,
    count_ply_vertices,
    read_widget_version,
)
from splattie.types import AssetType

_MINIMAL_PLY = b"""ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0 0 0
1 0 0
0 1 0
"""


@pytest.fixture
def fake_ply(tmp_path: Path) -> Path:
    p = tmp_path / "fixture.ply"
    p.write_bytes(_MINIMAL_PLY)
    return p


def test_count_ply_vertices(fake_ply: Path) -> None:
    assert count_ply_vertices(fake_ply) == 3


def test_build_manifest_head_shape() -> None:
    manifest = build_manifest(
        splat_filename="x.ply",
        num_gaussians=3,
        widget_version="9.9.9",
        asset_type=AssetType.head,
        rig=HEAD_RIG,
        generator_tool="test",
    )
    # assetType serializes to the plain string, not the enum repr.
    assert json.loads(json.dumps(manifest))["assetType"] == "head"
    assert manifest["formatVersion"] == "9.9.9"
    assert manifest["avatar"]["splat"]["format"] == "ply"
    assert manifest["avatar"]["splat"]["topology"] == "flame-20k"
    assert manifest["animation"]["skeleton"] == {"file": "bone_tree.json", "rig": "flame"}
    assert manifest["animation"]["weights"]["file"] == "lbs_weight_20k.json"


@pytest.mark.skipif(
    not (WIDGET_PUBLIC / "bone_tree.json").exists(),
    reason="shared FLAME rig assets not present (widget submodule not initialized)",
)
def test_head_bundle_is_widget_loadable(fake_ply: Path, tmp_path: Path) -> None:
    """A head bundle round-trips to the exact layout the widget loader requires."""
    out = tmp_path / "head.splattie"
    manifest = build_manifest(
        splat_filename="head.ply",
        num_gaussians=count_ply_vertices(fake_ply),
        widget_version=read_widget_version(),
        asset_type=AssetType.head,
        rig=HEAD_RIG,
        generator_tool="test",
    )
    bundle_splattie(output_path=out, splat_path=fake_ply, manifest=manifest, states=DEFAULT_STATES_HEAD.jsonable())

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        # Every file the widget loader dereferences must be present.
        assert names == {"manifest.json", "head.ply", "bone_tree.json", "lbs_weight_20k.json", "states.json"}
        loaded = json.loads(zf.read("manifest.json"))
        assert loaded["format"] == "splattie"
        assert loaded["assetType"] == "head"
        assert loaded["formatVersion"] == read_widget_version()
        assert loaded["widget"]["config"] == "states.json"
        # states.json parses and carries the head idle/hover/click states.
        states = json.loads(zf.read("states.json"))
        assert set(states["states"]) == {"idle", "hover", "click"}


def test_bundle_errors_on_missing_rig_file(fake_ply: Path, tmp_path: Path) -> None:
    """If a manifest references a rig file that isn't provided, bundling fails loudly."""
    out = tmp_path / "bad.splattie"
    manifest = build_manifest(
        splat_filename="x.ply",
        num_gaussians=3,
        widget_version="9.9.9",
        asset_type=AssetType.head,
        rig=HEAD_RIG,
        generator_tool="test",
    )
    # Point the skeleton at a file that exists in neither rig_files nor WIDGET_PUBLIC.
    manifest["animation"]["skeleton"]["file"] = "does_not_exist_bone_tree.json"
    with pytest.raises(FileNotFoundError):
        bundle_splattie(output_path=out, splat_path=fake_ply, manifest=manifest, states={})
