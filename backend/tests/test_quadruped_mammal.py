"""CPU tests for the quadruped_mammal pipeline stages (no GPU / reconstruction needed)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from beartype import beartype
from jaxtyping import Float, jaxtyped

if TYPE_CHECKING:
    from pathlib import Path

from splattie.methods.quadruped_mammal import runtime
from splattie.methods.quadruped_mammal.bind import (
    _MAX_CENTER_YAW,
    _canonical_transform,
    quadruped_widget_config,
)
from splattie.methods.quadruped_mammal.fit import _initial_alignment, _swap_lr, _umeyama
from splattie.methods.quadruped_mammal.gaussians import viewmat
from splattie.methods.quadruped_mammal.keypoints import _reproj_error, _triangulate_one
from splattie.methods.quadruped_mammal.schemas import Keypoints3D, NotAQuadrupedMammalError


def test_umeyama_recovers_known_similarity() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3)).astype(np.float32)
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], np.float32)
    scale, translation = 1.8, np.array([0.5, -1.0, 2.0], np.float32)
    dst = (scale * (src @ rotation.T) + translation).astype(np.float32)

    got_scale, got_rotation, got_translation, residual = _umeyama(src, dst)
    assert abs(got_scale - scale) < 1e-3
    assert np.allclose(got_rotation, rotation, atol=1e-3)
    assert np.allclose(got_translation, translation, atol=1e-3)
    assert residual < 1e-3


def test_swap_lr_flips_sides_only() -> None:
    assert _swap_lr("front_left_paw") == "front_right_paw"
    assert _swap_lr("back_right_knee") == "back_left_knee"
    assert _swap_lr("nose") == "nose"
    assert _swap_lr("tail_base") == "tail_base"


def test_triangulation_recovers_3d_point() -> None:
    point = np.array([0.1, -0.2, 0.05], np.float32)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    center = np.zeros(3, np.float32)
    intrinsics = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], np.float32)
    rows = []
    for azimuth in (0.0, 1.2):
        eye = center + 2.0 * np.array([np.sin(azimuth), 0.25, np.cos(azimuth)], np.float32)
        proj = intrinsics @ viewmat(eye.astype(np.float32), center, up)[:3, :4]
        projected = proj @ np.append(point, 1.0)
        rows.append((proj, float(projected[0] / projected[2]), float(projected[1] / projected[2]), 1.0))

    recovered = _triangulate_one(rows)
    assert np.allclose(recovered, point, atol=1e-3)
    assert _reproj_error(recovered, rows[0][0], rows[0][1], rows[0][2]) < 1e-2


def test_keypoints3d_lookup_drops_low_support_and_nan() -> None:
    keypoints = Keypoints3D(
        bodyparts=["nose", "tail", "paw"],
        positions=np.array([[0, 0, 1], [np.nan, 0, 0], [1, 1, 1]], np.float32),
        support=np.array([5, 9, 1], np.int32),
        mean_confidence=0.5,
    )
    lookup = keypoints.lookup(min_support=3)
    assert set(lookup) == {"nose"}  # tail has NaN, paw support < 3


def test_detection_gate_rejects_too_few_keypoints() -> None:
    """Non-mammals triangulate too few anchors -> the fit must refuse (no fallback)."""
    template = np.zeros((33, 3), np.float32)
    lookup = {"nose": np.array([0, 0, 1], np.float32), "tail_end": np.array([0, 0, -1], np.float32)}
    with pytest.raises(NotAQuadrupedMammalError):
        _initial_alignment(template, lookup)


def test_quadruped_widget_config_enables_head_follow() -> None:
    config = quadruped_widget_config()
    assert config.defaults.gaze.max_neck_yaw > 0
    assert config.defaults.gaze.intensity > 0
    assert config.states.idle.tracking.head > 0


def test_require_quadruped_runtime_executes() -> None:
    """Exercise the runtime check body so call-time NameErrors are caught by CI.

    The default backend is TripoSplat, so readiness needs SMAL + DeepLabCut + TripoSplat
    (vendored code + flow-model ckpt) — not TRELLIS.
    """
    from splattie.methods.quadruped_mammal.reconstruct import ReconstructBackend

    triposplat_ready = (
        runtime.SMAL_PKL.exists()
        and runtime.DLC_PYTHON.exists()
        and (runtime.VENDOR_TRIPOSPLAT / "triposplat.py").exists()
        and runtime.TRIPOSPLAT_FLOW_MODEL.exists()
    )
    if triposplat_ready:
        runtime.require_quadruped_runtime()  # default backend, must not raise when everything is present
    else:
        with pytest.raises(FileNotFoundError):
            runtime.require_quadruped_runtime()

    # Backend-aware: the TRELLIS backend checks the TRELLIS package (incl. the flexicubes mesher) instead
    # of the TripoSplat ckpts.
    flexicubes = runtime.VENDOR_TRELLIS / "trellis" / "representations" / "mesh" / "flexicubes" / "flexicubes.py"
    trellis_ready = (
        runtime.SMAL_PKL.exists()
        and runtime.DLC_PYTHON.exists()
        and (runtime.VENDOR_TRELLIS / "trellis").exists()
        and flexicubes.exists()
    )
    if trellis_ready:
        runtime.require_quadruped_runtime(ReconstructBackend.trellis)
    else:
        with pytest.raises(FileNotFoundError):
            runtime.require_quadruped_runtime(ReconstructBackend.trellis)


@pytest.mark.skipif(not runtime.SMAL_PKL.exists(), reason="SMAL weights not present")
def test_smal_forward_neutral_matches_template() -> None:
    import torch

    from splattie.methods.quadruped_mammal.smal import SMAL

    smal = SMAL(str(runtime.SMAL_PKL), device="cpu")
    vertices, joints = smal(torch.zeros(smal.B), torch.zeros(smal.K, 3))
    assert vertices.shape == (smal.V, 3)
    assert joints.shape == (smal.K, 3)
    assert float((vertices - smal.v_template).abs().max()) < 1e-5  # neutral == shaped template


@jaxtyped(typechecker=beartype)
def _synthetic_quadruped(
    head_yaw_deg: float, rng: np.random.Generator
) -> tuple[Float[npt.NDArray[np.float32], "joints 3"], Float[npt.NDArray[np.float32], "points 3"]]:
    """SMAL-indexed joints + head gaussians in a canonical frame (body forward +Z, up +Y).

    The head gaussians (muzzle + asymmetric forward-leaning ears + skull) are turned about +Y by
    ``head_yaw_deg`` around the head joint, so the muzzle's true facing is a known angle. The ears probe
    that the muzzle estimate's Y-gating ignores high features.
    """
    j = np.zeros((33, 3), np.float32)
    j[0], j[6], j[15], j[16] = (0, 0.0, 0.0), (0, 0.10, 0.30), (0, 0.20, 0.55), (0, 0.30, 0.75)
    j[25] = (0, -0.05, -0.45)  # tail base
    j[10], j[14], j[20], j[24] = (0.2, -0.5, 0.3), (-0.2, -0.5, 0.3), (0.2, -0.5, -0.3), (-0.2, -0.5, -0.3)
    muzzle = j[16] + np.column_stack(
        [rng.normal(0, 0.025, 400), rng.normal(-0.03, 0.025, 400), rng.uniform(0.06, 0.30, 400)]
    )
    ears = j[16] + np.column_stack(
        [rng.uniform(-0.14, -0.04, 120), rng.uniform(0.20, 0.32, 120), rng.uniform(0.04, 0.16, 120)]
    )
    skull = j[16] + rng.normal(0, 0.05, (250, 3))
    head = np.vstack([muzzle, ears, skull]).astype(np.float32)
    th = math.radians(head_yaw_deg)
    ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]], np.float32)
    head = ((head - j[16]) @ ry.T + j[16]).astype(np.float32)
    return j, head


@pytest.mark.parametrize("head_yaw", [0.0, 15.0, -18.0, 45.0])
def test_canonical_transform_uprights_body_and_centers_head(head_yaw: float) -> None:
    rng = np.random.default_rng(7)
    joints, head = _synthetic_quadruped(head_yaw, rng)
    # Place the animal in an arbitrary world frame (azimuth + tilt), as a reconstruction would.
    a, b = 1.1, 0.25
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    world = (rz @ rx).astype(np.float32)
    jw, hw = (joints @ world.T).astype(np.float32), (head @ world.T).astype(np.float32)

    matrix = np.asarray(_canonical_transform(jw, hw).matrix, np.float32)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-4)  # proper rotation
    assert abs(float(np.linalg.det(matrix)) - 1.0) < 1e-4

    jf, pts = jw @ matrix.T, hw @ matrix.T
    up = jf[[0, 6]].mean(0) - jf[[10, 14, 20, 24]].mean(0)
    assert (up / np.linalg.norm(up))[1] > 0.97  # body stands upright (+Y)

    near = pts[np.linalg.norm(pts - jf[16], axis=1) < 0.45]
    low = near[near[:, 1] <= jf[16][1] + 0.06]
    muzzle = low if len(low) >= 50 else near
    fwd = muzzle[muzzle[:, 2] >= np.percentile(muzzle[:, 2], 75)].mean(0) - jf[16]
    residual = abs(math.degrees(math.atan2(float(fwd[0]), float(fwd[2]))))
    expected = max(0.0, abs(head_yaw) - math.degrees(_MAX_CENTER_YAW))  # clamp leaves a residual past the cap
    assert abs(residual - expected) < 7.0


def test_reconstruct_backend_selector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The quadruped reconstruct dispatches to TripoSplat by default, TRELLIS on request.

    Plain strings coerce to the enum so the SPLATTIE_QUADRUPED_BACKEND env var (a string) selects it.
    """
    from splattie.methods.quadruped_mammal import reconstruct as recon
    from splattie.methods.quadruped_mammal.reconstruct import ReconstructBackend

    trellis_ply, tripo_ply = tmp_path / "trellis.ply", tmp_path / "tripo.ply"
    monkeypatch.setattr(
        recon, "reconstruct_object_with_trellis", lambda **_: type("R", (), {"gaussian_ply": trellis_ply})()
    )
    monkeypatch.setattr(recon, "_reconstruct_with_triposplat", lambda **_: tripo_ply)

    common = {"image_path": tmp_path / "x.png", "output_dir": tmp_path, "model_id": "m"}
    assert recon.reconstruct_gaussian_splat(**common) == tripo_ply  # default = triposplat
    assert recon.reconstruct_gaussian_splat(**common, backend=ReconstructBackend.triposplat) == tripo_ply
    assert recon.reconstruct_gaussian_splat(**common, backend=ReconstructBackend.trellis) == trellis_ply
    assert recon.reconstruct_gaussian_splat(**common, backend="trellis") == trellis_ply  # env-string coerces


def test_resolve_backend_env_selects_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    """SPLATTIE_QUADRUPED_BACKEND picks the backend; an invalid value fails fast with a clear error."""
    from splattie.methods.quadruped_mammal.method import _resolve_backend
    from splattie.methods.quadruped_mammal.reconstruct import ReconstructBackend

    monkeypatch.delenv("SPLATTIE_QUADRUPED_BACKEND", raising=False)
    assert _resolve_backend() is ReconstructBackend.triposplat  # default when unset

    monkeypatch.setenv("SPLATTIE_QUADRUPED_BACKEND", "trellis")
    assert _resolve_backend() is ReconstructBackend.trellis

    monkeypatch.setenv("SPLATTIE_QUADRUPED_BACKEND", "bogus")
    with pytest.raises(ValueError, match="SPLATTIE_QUADRUPED_BACKEND"):
        _resolve_backend()
