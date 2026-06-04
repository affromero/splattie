"""CPU tests for the quadruped_mammal pipeline stages (no GPU / reconstruction needed)."""

from __future__ import annotations

import numpy as np
import pytest

from splattie.methods.quadruped_mammal import runtime
from splattie.methods.quadruped_mammal.bind import quadruped_widget_config
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


@pytest.mark.skipif(not runtime.SMAL_PKL.exists(), reason="SMAL weights not present")
def test_smal_forward_neutral_matches_template() -> None:
    import torch

    from splattie.methods.quadruped_mammal.smal import SMAL

    smal = SMAL(str(runtime.SMAL_PKL), device="cpu")
    vertices, joints = smal(torch.zeros(smal.B), torch.zeros(smal.K, 3))
    assert vertices.shape == (smal.V, 3)
    assert joints.shape == (smal.K, 3)
    assert float((vertices - smal.v_template).abs().max()) < 1e-5  # neutral == shaped template
