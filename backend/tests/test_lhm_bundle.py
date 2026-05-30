"""Tests for the LHM body `.splattie` bundle adapter (CPU, no GPU needed).

The GPU-gated `test_lhm_generate_produces_body` exercises the real rig extraction
(`extract_body_rig` against the loaded SMPL-X model). Here we cover the bundle
*shape* — the body manifest (`assetType=body`, smplx rig) and a widget-loadable zip
built from synthetic skeleton/weights — which needs no model.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from splattie.methods.bundle_common import (
    build_manifest,
    bundle_splattie,
    count_ply_vertices,
    read_widget_version,
)
from splattie.methods.lhm.bundle import (
    BODY_RIG,
    DEFAULT_STATES_BODY,
    JOINTS_NAME,
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


def _synthetic_rig(tmp_path: Path, n_gaussians: int = 3) -> tuple[Path, Path]:
    """Write a minimal skeleton.json + lbs_weights.json in the body bundle format."""
    n_joints = len(JOINTS_NAME)
    skeleton = {
        "rig": "smplx",
        "jointCount": n_joints,
        "names": list(JOINTS_NAME),
        "parents": [-1, *range(n_joints - 1)],
        "restPositions": [[0.0, 0.0, 0.0] for _ in range(n_joints)],
    }
    k = 4
    weights = {
        "numGaussians": n_gaussians,
        "jointCount": n_joints,
        "k": k,
        "indices": [0, 1, 2, 3] * n_gaussians,
        "weights": [0.7, 0.1, 0.1, 0.1] * n_gaussians,
    }
    skel = tmp_path / "skeleton.json"
    wts = tmp_path / "lbs_weights.json"
    skel.write_text(json.dumps(skeleton))
    wts.write_text(json.dumps(weights))
    return skel, wts


def test_joints_name_is_smplx_55() -> None:
    assert len(JOINTS_NAME) == 55
    assert JOINTS_NAME[0] == "Pelvis"
    assert {"Head", "Neck", "Spine_3"} <= set(JOINTS_NAME)


def test_default_states_body_tracks_head_and_torso() -> None:
    """Body states mirror the head's idle/hover/click but track head/torso look-at."""
    assert set(DEFAULT_STATES_BODY["states"]) == {"idle", "hover", "click"}
    tracking = DEFAULT_STATES_BODY["states"]["idle"]["tracking"]
    assert "head" in tracking
    assert "torso" in tracking
    assert "eyes" not in tracking  # bodies look-at, they don't have eye-tracking


def test_build_manifest_body_shape() -> None:
    manifest = build_manifest(
        splat_filename="x.ply",
        num_gaussians=3,
        widget_version="9.9.9",
        asset_type=AssetType.BODY,
        rig=BODY_RIG,
        generator_method="lhm",
        generator_tool="test",
    )
    assert json.loads(json.dumps(manifest))["assetType"] == "body"
    assert manifest["animation"]["skeleton"] == {"file": "skeleton.json", "rig": "smplx"}
    assert manifest["animation"]["weights"]["file"] == "lbs_weights.json"
    assert manifest["avatar"]["splat"]["topology"] == "smplx-voxel"


def test_body_bundle_is_widget_loadable(fake_ply: Path, tmp_path: Path) -> None:
    """A body bundle round-trips to the layout the widget body loader requires."""
    skel, wts = _synthetic_rig(tmp_path)
    out = tmp_path / "body.splattie"
    manifest = build_manifest(
        splat_filename="body.ply",
        num_gaussians=count_ply_vertices(fake_ply),
        widget_version=read_widget_version(),
        asset_type=AssetType.BODY,
        rig=BODY_RIG,
        generator_method="lhm",
        generator_tool="test",
    )
    bundle_splattie(
        output_path=out,
        splat_path=fake_ply,
        manifest=manifest,
        states=DEFAULT_STATES_BODY,
        rig_files={"skeleton.json": skel, "lbs_weights.json": wts},
    )

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert names == {"manifest.json", "body.ply", "skeleton.json", "lbs_weights.json", "states.json"}
        loaded = json.loads(zf.read("manifest.json"))
        assert loaded["assetType"] == "body"
        assert loaded["animation"]["type"] == "lbs"
        skeleton = json.loads(zf.read("skeleton.json"))
        assert skeleton["rig"] == "smplx"
        assert len(skeleton["names"]) == 55
        assert len(skeleton["parents"]) == 55
