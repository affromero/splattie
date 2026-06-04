"""Keypoint-anchored SMAL fit.

Pin SMAL joints to the triangulated 3D SuperAnimal landmarks (Umeyama init, then chamfer +
3D-anchor loss) so the anatomy can't scramble (head=head, legs=legs, L/R correct), unlike a
chamfer-only fit. L/R convention is resolved by lower Umeyama residual.
"""

from __future__ import annotations

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import matrix_to_axis_angle

from splattie.methods.quadruped_mammal.gaussians import GaussianSplat
from splattie.methods.quadruped_mammal.keypoints import DEVICE, sample_points
from splattie.methods.quadruped_mammal.schemas import (
    FitDiagnostics,
    Keypoints3D,
    NotAQuadrupedMammalError,
    QuadrupedFit,
)
from splattie.methods.quadruped_mammal.smal import SMAL

# SMAL is a quadruped-mammal model with fixed topology, so joint<->keypoint correspondence
# is a constant. neck=15 is named "neck" downstream so the widget drives head-follow.
NECK_JOINT, HEAD_JOINT, NOSE_JOINT = 15, 16, 32
_CORRESPONDENCE = {
    32: "nose",
    15: "neck_base",
    25: "tail_base",
    31: "tail_end",
    7: "front_left_thai",
    9: "front_left_knee",
    10: "front_left_paw",
    11: "front_right_thai",
    13: "front_right_knee",
    14: "front_right_paw",
    17: "back_left_thai",
    19: "back_left_knee",
    20: "back_left_paw",
    21: "back_right_thai",
    23: "back_right_knee",
    24: "back_right_paw",
}
_LR_PAIRS = (("front_left", "front_right"), ("back_left", "back_right"))
_MIN_ANCHORS = 4


def _swap_lr(name: str) -> str:
    for left, right in _LR_PAIRS:
        if name.startswith(left):
            return name.replace(left, right, 1)
        if name.startswith(right):
            return name.replace(right, left, 1)
    return name


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Similarity (scale, R, t) with ``dst ~= s R src + t`` (Kabsch + scale) + rms residual."""
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    x_src, x_dst = src - mu_src, dst - mu_dst
    covariance = x_dst.T @ x_src / len(src)
    u, diag, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1
    rotation = u @ correction @ vt
    variance = (x_src**2).sum() / len(src)
    scale = float(np.trace(np.diag(diag) @ correction) / variance)
    translation = mu_dst - scale * rotation @ mu_src
    residual = float(np.sqrt(((dst - (scale * (src @ rotation.T) + translation)) ** 2).sum(1).mean()))
    return scale, rotation, translation, residual


def _corresponded(
    template: np.ndarray, lookup: dict[str, np.ndarray], *, swap: bool
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return (joint indices, template xyz, target xyz) for SMAL joints with a matched keypoint."""
    joints, src, dst = [], [], []
    for joint, name in _CORRESPONDENCE.items():
        key = _swap_lr(name) if swap else name
        if key in lookup:
            joints.append(joint)
            src.append(template[joint])
            dst.append(lookup[key])
    return joints, np.array(src, np.float32), np.array(dst, np.float32)


def _initial_alignment(
    template: np.ndarray, lookup: dict[str, np.ndarray]
) -> tuple[float, bool, list[int], float, np.ndarray]:
    """Pick the L/R convention with the lower Umeyama residual; return its alignment."""
    candidates = []
    for swap in (False, True):
        joints, src, dst = _corresponded(template, lookup, swap=swap)
        if len(joints) < _MIN_ANCHORS:
            continue
        scale0, rot0, _, residual = _umeyama(src, dst)
        candidates.append((residual, swap, joints, scale0, rot0))
    if not candidates:
        msg = f"only {len(lookup)} reliable keypoints triangulated — not a recognizable quadruped mammal"
        raise NotAQuadrupedMammalError(msg)
    return min(candidates, key=lambda item: item[0])


def fit_smal(smal: SMAL, splat: GaussianSplat, keypoints: Keypoints3D) -> QuadrupedFit:
    """Fit SMAL (shape+pose+global) to the splat, anchored to the 3D keypoints."""
    points = sample_points(splat)
    lookup = keypoints.lookup()
    with torch.no_grad():
        _, template = smal(torch.zeros(smal.B, device=DEVICE), torch.zeros(smal.K, 3, device=DEVICE))
    template = template.cpu().numpy()
    residual, swap, joints, scale0, rot0 = _initial_alignment(template, lookup)
    _, _, target = _corresponded(template, lookup, swap=swap)
    anchor_idx = torch.tensor(joints, device=DEVICE)
    anchor_tgt = torch.tensor(target, device=DEVICE)

    betas = torch.zeros(smal.B, device=DEVICE, requires_grad=True)
    pose = torch.zeros(smal.K, 3, device=DEVICE)
    pose[0] = matrix_to_axis_angle(torch.tensor(rot0[None], dtype=torch.float32, device=DEVICE))[0]
    pose = pose.detach().requires_grad_()
    scale = torch.tensor(scale0, device=DEVICE, requires_grad=True)
    with torch.no_grad():
        _, joints_world = smal(betas, pose, scale=scale)
        trans = (anchor_tgt.mean(0) - joints_world[anchor_idx].mean(0)).detach()
    trans.requires_grad_()
    huber = torch.nn.SmoothL1Loss(beta=0.05)

    global_opt = torch.optim.Adam([scale, trans, pose], lr=0.02)
    for _ in range(200):
        global_opt.zero_grad()
        _, joints_world = smal(betas, pose, trans=trans, scale=scale)
        huber(joints_world[anchor_idx], anchor_tgt).backward()
        global_opt.step()

    full_opt = torch.optim.Adam([betas, pose, scale, trans], lr=0.01)
    for _ in range(500):
        full_opt.zero_grad()
        verts, joints_world = smal(betas, pose, trans=trans, scale=scale)
        loss = (
            chamfer_distance(verts[None], points[None])[0]
            + 5.0 * huber(joints_world[anchor_idx], anchor_tgt)
            + 1e-3 * (pose[1:] ** 2).sum()
            + 5e-3 * (betas**2).sum()
        )
        loss.backward()
        full_opt.step()

    with torch.no_grad():
        verts, joints_world = smal(betas, pose, trans=trans, scale=scale)
    chamfer = chamfer_distance(verts[None], points[None])[0].item()
    anchor_rms = float(np.sqrt(((joints_world[anchor_idx] - anchor_tgt) ** 2).sum(1).mean().item()))
    diagnostics = FitDiagnostics(
        chamfer=chamfer,
        anchor_rms=anchor_rms,
        lr_residual=float(residual),
        lr_swap=bool(swap),
        n_anchors=len(joints),
        triangulated_count=len(lookup),
        mean_keypoint_confidence=keypoints.mean_confidence,
    )
    return QuadrupedFit(
        betas=betas.detach(),
        pose=pose.detach(),
        scale=float(scale.item()),
        trans=trans.detach(),
        diagnostics=diagnostics,
    )
