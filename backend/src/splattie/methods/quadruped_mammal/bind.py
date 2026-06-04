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

from splattie.methods.object.bundle import (
    OBJECT_VIEWER_TRANSFORM,
    CameraConfig,
    GazeConfig,
    GhostConfig,
    ObjectExpressionConfig,
    ObjectPoseConfig,
    ObjectStateSet,
    ObjectTrackingConfig,
    ObjectTransitions,
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
        max_eye_yaw=0.0,
        max_eye_pitch=0.0,
        max_neck_yaw=math.pi / 4,
        max_neck_pitch=math.pi / 6,
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


def _lbs_from_nearest_vertex(
    splat: GaussianSplat, smal_vertices: np.ndarray, smal_weights: np.ndarray
) -> SparseLbsWeights:
    """Top-4 LBS weights per gaussian, taken from its nearest fitted SMAL vertex."""
    _, nearest = cKDTree(smal_vertices).query(splat.xyz, k=1)
    gaussian_weights = smal_weights[nearest]
    top = np.argpartition(-gaussian_weights, _LBS_K - 1, axis=1)[:, :_LBS_K]
    values = np.take_along_axis(gaussian_weights, top, axis=1)
    order = np.argsort(-values, axis=1)
    top = np.take_along_axis(top, order, axis=1).astype(np.int64)
    values = np.take_along_axis(values, order, axis=1)
    values = values / np.clip(values.sum(1, keepdims=True), 1e-8, None)
    return SparseLbsWeights(
        num_gaussians=len(splat),
        joint_count=int(smal_weights.shape[1]),
        k=_LBS_K,
        indices=top.reshape(-1).tolist(),
        weights=values.reshape(-1).astype(float).tolist(),
    )


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
    names[NECK_JOINT], names[HEAD_JOINT], names[NOSE_JOINT] = "neck", "head", "nose"
    skeleton = RigSkeleton(
        names=names,
        parents=[int(parent) if parent >= 0 else -1 for parent in smal.parents],
        rest_positions=[tuple(float(coord) for coord in joints_np[i]) for i in range(smal.K)],
        rig=RIG_NAME,
        joint_count=smal.K,
    )
    lbs_weights = _lbs_from_nearest_vertex(splat, vertices_np, smal_weights)

    return build_object_splattie(
        ply_path=ply_path,
        output_dir=output_dir,
        model_id=model_id,
        skeleton=skeleton,
        lbs_weights=lbs_weights,
        source_image_path=source_image_path,
        transform=OBJECT_VIEWER_TRANSFORM,
        widget_config=quadruped_widget_config(),
        generator_method=GENERATOR_METHOD,
        generator_method_version=GENERATOR_VERSION,
        category=CATEGORY,
    )
