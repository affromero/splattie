"""Bind the fitted SMAL rig onto the gaussians and emit a gaze-enabled `.splattie`.

Per-gaussian LBS weights come from the nearest fitted SMAL vertex (top-4). The neck joint is
named "neck" so the widget's object look-at drives head-follow; the bundle ships a gaze-enabled
widget config (vs the inert object default).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from splattie.methods.object.bundle import (
    CameraConfig,
    GazeConfig,
    GhostConfig,
    ObjectExpressionConfig,
    ObjectPoseConfig,
    ObjectStateSet,
    ObjectTrackingConfig,
    ObjectTransformName,
    ObjectTransitions,
    ObjectViewerTransform,
    ObjectWidgetConfig,
    RigSkeleton,
    SaccadeConfig,
    SparseLbsWeights,
    StateDefinition,
    TransitionConfig,
    WidgetDefaults,
    build_object_splattie,
)
from splattie.methods.quadruped_mammal.fit import HEAD_JOINT, NECK_JOINT, NOSE_JOINT
from splattie.methods.quadruped_mammal.gaussians import GaussianSplat
from splattie.methods.quadruped_mammal.schemas import QuadrupedFit
from splattie.methods.quadruped_mammal.smal import SMAL

GENERATOR_METHOD = "trellis-smal-quadruped"
GENERATOR_VERSION = "quadruped-rig-v1"
CATEGORY = "quadruped_mammal"
RIG_NAME = "smal-quadruped"
_LBS_K = 4


def quadruped_widget_config() -> ObjectWidgetConfig:
    """Object widget config with head-follow ENABLED (vs the inert object default)."""
    camera = CameraConfig(theta=0, phi=82, radius=1.7, look_at="auto", fov=45)
    gaze = GazeConfig(
        intensity=1.0,
        smoothing_tau=0.16,
        deadzone=0.04,
        # No eye gaze: a gaussian splat has no eyeball to rotate, so eye-bone rotation only swings the
        # flat eye patch off the head. The head turning toward the cursor IS the natural gaze.
        max_eye_yaw=0.0,
        max_eye_pitch=0.0,
        # Subtle: a quadruped's neck turns far less than a human's; 45deg tore the neck. ~22/18deg.
        max_neck_yaw=math.pi / 8,
        max_neck_pitch=math.pi / 10,
        saccade=SaccadeConfig(enabled=False, amplitude=0, interval_ms=(3000, 6500), move_ms=90),
    )

    def state(amplitude: float, frequency: float, wobble: float) -> StateDefinition:
        return StateDefinition(
            ghost=GhostConfig(amplitude=amplitude, frequency=frequency, wobble=wobble),
            expression=ObjectExpressionConfig(),
            camera=camera,
            rotation=(0, 0, 0),
            tracking=ObjectTrackingConfig(head=1.0, torso=0.0),
            pose=ObjectPoseConfig(),
        )

    return ObjectWidgetConfig(
        defaults=WidgetDefaults(camera=camera, gaze=gaze),
        states=ObjectStateSet(
            idle=state(0.001, 0.25, 0.08),
            hover=state(0.002, 0.40, 0.12),
            click=state(0.001, 0.60, 0.05),
        ),
        transitions=ObjectTransitions(
            idle_hover=TransitionConfig(duration=0.3, easing="easeOutCubic"),
            hover_idle=TransitionConfig(duration=0.4, easing="easeOutCubic"),
            any_click=TransitionConfig(duration=0.15, easing="easeOutCubic"),
        ),
    )


_LBS_NEIGHBORS = 12  # SMAL vertices blended per gaussian


def _blend_weights(splat: GaussianSplat, smal_vertices: np.ndarray, smal_weights: np.ndarray) -> np.ndarray:
    """Dense per-gaussian joint weights from a distance-weighted blend of nearby SMAL vertices."""
    distances, neighbors = cKDTree(smal_vertices).query(splat.xyz, k=_LBS_NEIGHBORS)
    sigma = float(np.median(distances[:, 0])) * 2.0 + 1e-8
    rbf = np.exp(-(distances**2) / (2.0 * sigma**2)).astype(np.float32)
    rbf /= np.clip(rbf.sum(1, keepdims=True), 1e-8, None)
    blended = np.zeros((len(splat), smal_weights.shape[1]), np.float32)
    for neighbor in range(_LBS_NEIGHBORS):
        blended += rbf[:, neighbor : neighbor + 1] * smal_weights[neighbors[:, neighbor]]
    return blended


def _to_sparse(weights: np.ndarray, num_gaussians: int) -> SparseLbsWeights:
    """Keep the top-4 joints per gaussian (renormalised) in the widget's sparse LBS format."""
    top = np.argpartition(-weights, _LBS_K - 1, axis=1)[:, :_LBS_K]
    values = np.take_along_axis(weights, top, axis=1)
    order = np.argsort(-values, axis=1)
    top = np.take_along_axis(top, order, axis=1).astype(np.int64)
    values = np.take_along_axis(values, order, axis=1)
    values = values / np.clip(values.sum(1, keepdims=True), 1e-8, None)
    return SparseLbsWeights(
        num_gaussians=num_gaussians,
        joint_count=int(weights.shape[1]),
        k=_LBS_K,
        indices=top.reshape(-1).tolist(),
        weights=values.reshape(-1).astype(float).tolist(),
    )


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _head_aware_weights(
    splat: GaussianSplat, smal_vertices: np.ndarray, smal_weights: np.ndarray, joints: np.ndarray
) -> SparseLbsWeights:
    """Bind the head as a RIGID unit (on the head joint) with a smooth neck falloff.

    LBS-blending the head makes it melt when the neck rotates; instead the head is bound fully to the
    head joint (rigid), with a smoothstep band at the neck-head junction and far gaussians (paws)
    excluded, so a head turn moves the head cleanly without tearing the neck or dragging the legs.
    """
    base = _blend_weights(splat, smal_vertices, smal_weights)
    neck, head = joints[NECK_JOINT], joints[HEAD_JOINT]
    axis = head - neck
    axis_len = float(np.linalg.norm(axis)) + 1e-8
    axis = axis / axis_len
    along = (splat.xyz - (neck + head) / 2.0) @ axis  # signed distance past the junction, toward the head
    head_weight = _smoothstep(along / (0.5 * axis_len))
    near_head = np.linalg.norm(splat.xyz - head, axis=1) < 2.0 * axis_len
    head_weight = (head_weight * near_head).astype(np.float32)
    base = base * (1.0 - head_weight)[:, None]
    base[:, HEAD_JOINT] += head_weight
    return _to_sparse(base, len(splat))


# Target viewer frame (columns: anterior, lateral, up) every animal is rotated into, so they all
# stand upright + face the same way (vs the fixed transform, which left each at TRELLIS's azimuth).
# Tunable against the live widget: AZIMUTH spins the facing, the up column should stay vertical.
_TARGET_VIEWER_FRAME = np.column_stack(
    [
        np.array([0.0, 0.0, 1.0], np.float32),  # anterior -> +Z (faces the camera; flip sign if it faces away)
        np.array([1.0, 0.0, 0.0], np.float32),  # lateral  -> +X
        np.array([0.0, 1.0, 0.0], np.float32),  # up       -> +Y (upright)
    ]
)

# SMAL joint indices for the anatomical frame (posed joints, robust to the articulated head/tail).
# Body forward comes from STABLE torso anchors (neck base -> tail base), never the weakly-fit nose
# joint: the SMAL head pose is poorly constrained, and using the nose as "anterior" swings the whole
# body broadside when the head is mis-posed. The head's own facing is measured separately (gaussians).
_TAIL_BASE = 25
_PAWS = (10, 14, 20, 24)
_SPINE = (0, 6)
# Cap how far the head-centering may tilt the body. Heads turned up to this much center fully; beyond
# it the head keeps a small residual turn rather than standing the whole animal at a distracting angle.
_MAX_CENTER_YAW = math.radians(22.0)


def _muzzle_yaw(joints: np.ndarray, splat_xyz: np.ndarray, body_rot: np.ndarray) -> float:
    """Yaw (about viewer-up) of the head's forward axis, measured from the gaussian MUZZLE.

    The reconstructed head is often turned at rest (the photo was not head-on), which makes the gaze
    one-sided. The SMAL nose joint is too weakly constrained to use, so the facing is taken from the
    rendered gaussians. Crucially the cluster is restricted to the head's LOWER band (the muzzle/jaw)
    before taking the forward-most points, so high features that also lean forward -- ears, horns,
    antlers, forehead -- cannot pull the estimate sideways (the dominant error of a plain
    forward-most-gaussian estimate). Returns the angle to rotate OUT so the head faces +Z.
    """
    pts = (splat_xyz @ body_rot.T).astype(np.float32)  # gaussians in the body-canonical frame
    head = body_rot @ joints[HEAD_JOINT]
    scale = float(np.linalg.norm(joints[HEAD_JOINT] - joints[NECK_JOINT])) + 1e-6
    near = pts[np.linalg.norm(pts - head, axis=1) < 1.8 * scale]
    if len(near) < 50:
        return 0.0
    lower = near[near[:, 1] <= head[1] + 0.25 * scale]  # muzzle/jaw band: drop the skull cap, ears, horns
    band = lower if len(lower) >= 50 else near
    muzzle = band[band[:, 2] >= np.percentile(band[:, 2], 75.0)]  # forward-most of the band
    forward = muzzle.mean(0) - head
    return float(np.arctan2(float(forward[0]), float(forward[2])))  # angle from +Z toward +X


def _canonical_transform(joints: np.ndarray, splat_xyz: np.ndarray) -> ObjectViewerTransform:
    """Rotate the world so the animal stands upright, faces the camera, and its HEAD is centered.

    Up = paws->spine; body forward = tail-base->neck-base (stable torso anchors, NOT the nose joint),
    so the body faces the camera even when the head is mis-posed. A second, clamped yaw then turns the
    head's measured forward axis onto +Z, so the rest pose looks straight ahead and the widget's neck
    yaw reads as a symmetric head-turn (an animal shot at an angle keeps its body slightly turned).
    """
    up = joints[list(_SPINE)].mean(0) - joints[list(_PAWS)].mean(0)
    up = up / np.linalg.norm(up)
    body_fwd = joints[NECK_JOINT] - joints[_TAIL_BASE]
    body_fwd = body_fwd - (body_fwd @ up) * up
    body_fwd = body_fwd / np.linalg.norm(body_fwd)
    lateral = np.cross(up, body_fwd)
    lateral = lateral / np.linalg.norm(lateral)
    world_frame = np.column_stack([body_fwd, lateral, up]).astype(np.float32)  # cols: anterior, lateral, up
    body_rot = (_TARGET_VIEWER_FRAME @ world_frame.T).astype(np.float32)
    # Center the head about viewer-up (+Y), clamped so a mis-posed head can't tilt the whole body far.
    yaw = float(np.clip(_muzzle_yaw(joints, splat_xyz, body_rot), -_MAX_CENTER_YAW, _MAX_CENTER_YAW))
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    center = np.array([[cos_y, 0.0, -sin_y], [0.0, 1.0, 0.0], [sin_y, 0.0, cos_y]], np.float32)
    matrix = (center @ body_rot).astype(np.float32)
    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
    quat_wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
    return ObjectViewerTransform(name=ObjectTransformName.FIT_CANONICAL, matrix=matrix, quaternion_wxyz=quat_wxyz)


def bind_and_bundle(
    smal: SMAL,
    splat: GaussianSplat,
    fit: QuadrupedFit,
    *,
    ply_path: Path,
    output_dir: Path,
    model_id: str,
    source_image_path: Path | None = None,
) -> tuple[Path, int]:
    """Skin the gaussians to the fitted SMAL rig and write the gaze-enabled `.splattie`."""
    with torch.no_grad():
        vertices, joints = smal(fit.betas, fit.pose, trans=fit.trans, scale=fit.scale)
    vertices_np = vertices.cpu().numpy()
    joints_np = joints.cpu().numpy()
    smal_weights = smal.weights.cpu().numpy()

    names = [f"joint{i}" for i in range(smal.K)]
    # The widget rotates the bone named "neck". Name the NECK-BASE joint "neck" (not the head joint):
    # forward kinematics then swings the whole head subtree (neck -> head -> nose) about the neck base
    # on a long lever, so the head visibly turns toward the cursor instead of pivoting in place at its
    # own base (a 22deg turn of a round head right at the head joint barely reads as motion).
    names[NECK_JOINT], names[HEAD_JOINT], names[NOSE_JOINT] = "neck", "head", "nose"
    skeleton = RigSkeleton(
        names=names,
        parents=[int(parent) if parent >= 0 else -1 for parent in smal.parents],
        rest_positions=[tuple(float(coord) for coord in joints_np[i]) for i in range(smal.K)],
        rig=RIG_NAME,
        joint_count=smal.K,
    )
    lbs_weights = _head_aware_weights(splat, vertices_np, smal_weights, joints_np)

    return build_object_splattie(
        ply_path=ply_path,
        output_dir=output_dir,
        model_id=model_id,
        skeleton=skeleton,
        lbs_weights=lbs_weights,
        source_image_path=source_image_path,
        transform=_canonical_transform(joints_np, splat.xyz),
        widget_config=quadruped_widget_config(),
        generator_method=GENERATOR_METHOD,
        generator_method_version=GENERATOR_VERSION,
        category=CATEGORY,
    )
