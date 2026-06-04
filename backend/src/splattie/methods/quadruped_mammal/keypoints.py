"""Detect 3D anatomy landmarks from a gaussian splat (single-image-derived).

The splat is rendered from 16 internal cameras (upright via a coarse chamfer SMAL fit);
SuperAnimal-Quadruped (DeepLabCut, separate interpreter) detects 2D keypoints per view;
those are triangulated to 3D so the anchors are camera-independent. No extra user input.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import matrix_to_axis_angle

from splattie.methods.object.runtime import run_command
from splattie.methods.quadruped_mammal import runtime
from splattie.methods.quadruped_mammal.gaussians import GaussianSplat, cube_rotations, viewmat
from splattie.methods.quadruped_mammal.schemas import Keypoints3D
from splattie.methods.quadruped_mammal.smal import SMAL, rodrigues

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_AZIMUTHS = tuple(range(0, 360, 45))
_ELEVATIONS = (10.0, -25.0)
_VIEW_PX = 512
_FOV_DEG = 50.0
_CONF_FLOOR = 0.30
_REPROJ_PX = 8.0
_RUNNER = Path(__file__).with_name("superanimal_runner.py")


def sample_points(splat: GaussianSplat, count: int = 8000) -> torch.Tensor:
    """Sample up to ``count`` opaque splat points (device tensor) for chamfer terms."""
    keep = splat.opacity > 0.2
    xyz = splat.xyz[keep] if keep.any() else splat.xyz
    rng = np.random.default_rng(0)
    if len(xyz) > count:
        xyz = xyz[rng.choice(len(xyz), count, replace=False)]
    return torch.tensor(xyz, device=DEVICE)


def _chamfer_up_axis(smal: SMAL, points: torch.Tensor) -> np.ndarray:
    """Coarse chamfer SMAL fit; return the world up-axis so renders come out upright."""
    centroid = points.mean(0)
    extent = points.max(0).values - points.min(0).values
    with torch.no_grad():
        verts, _ = smal(torch.zeros(smal.B, device=DEVICE), torch.zeros(smal.K, 3, device=DEVICE))
        s_extent = verts.max(0).values - verts.min(0).values
        s_center = verts.mean(0)
    scale0 = float((extent.max() / s_extent.max()).item())
    best = (1e9, None, None)
    for matrix in cube_rotations():
        rot = torch.tensor(matrix, device=DEVICE)
        pose = torch.zeros(smal.K, 3, device=DEVICE)
        pose[0] = matrix_to_axis_angle(rot[None])[0]
        trans = centroid - scale0 * (rot @ s_center)
        with torch.no_grad():
            verts, _ = smal(torch.zeros(smal.B, device=DEVICE), pose, trans=trans, scale=scale0)
            dist = chamfer_distance(verts[None], points[None])[0].item()
        if dist < best[0]:
            best = (dist, pose[0].clone(), trans.clone())
    betas = torch.zeros(smal.B, device=DEVICE, requires_grad=True)
    pose = torch.zeros(smal.K, 3, device=DEVICE)
    pose[0] = best[1]
    pose = pose.detach().requires_grad_()
    scale = torch.tensor(scale0, device=DEVICE, requires_grad=True)
    trans = best[2].detach().requires_grad_()
    # Stage 1 locks the global orientation; stage 2 (shape + articulated pose) is what actually
    # settles the up-axis — skipping it leaves the render tilted and SuperAnimal detects nothing.
    global_opt = torch.optim.Adam([scale, trans, pose], lr=0.02)
    for _ in range(250):
        global_opt.zero_grad()
        verts, _ = smal(betas, pose, trans=trans, scale=scale)
        chamfer_distance(verts[None], points[None])[0].backward()
        global_opt.step()
    full_opt = torch.optim.Adam([betas, pose, scale, trans], lr=0.02)
    for _ in range(350):
        full_opt.zero_grad()
        verts, _ = smal(betas, pose, trans=trans, scale=scale)
        loss = chamfer_distance(verts[None], points[None])[0] + 1e-3 * (pose[1:] ** 2).sum() + 5e-3 * (betas**2).sum()
        loss.backward()
        full_opt.step()
    up = (rodrigues(pose.detach()[0:1])[0] @ torch.tensor([0.0, 0.0, 1.0], device=DEVICE)).cpu().numpy()
    return up / np.linalg.norm(up)


def render_views(smal: SMAL, splat: GaussianSplat, views_dir: Path) -> None:
    """Render 16 upright views of the splat + save camera matrices (cameras.npz)."""
    views_dir.mkdir(parents=True, exist_ok=True)
    up = _chamfer_up_axis(smal, sample_points(splat))
    center = splat.xyz.mean(0)
    radius = float(np.abs(splat.xyz - center).max())
    seed = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(float(seed @ up)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], np.float32)
    e0 = np.cross(up, seed)
    e0 = e0 / np.linalg.norm(e0)
    e1 = np.cross(up, e0)

    focal = 0.5 * _VIEW_PX / math.tan(math.radians(_FOV_DEG) / 2.0)
    intrinsics = np.array([[focal, 0, _VIEW_PX / 2], [0, focal, _VIEW_PX / 2], [0, 0, 1]], np.float32)

    def to_t(array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=DEVICE)

    means, quats = to_t(splat.xyz), to_t(splat.quat)
    scales, opac, colors = to_t(splat.scale), to_t(splat.opacity), to_t(splat.color)

    intr_list, view_list, names = [], [], []
    for elevation in _ELEVATIONS:
        for azimuth in _AZIMUTHS:
            az, el = math.radians(azimuth), math.radians(elevation)
            horizontal = math.cos(az) * e0 + math.sin(az) * e1
            eye = center + radius * 2.7 * (math.cos(el) * horizontal + math.sin(el) * up)
            view = viewmat(eye.astype(np.float32), center.astype(np.float32), up.astype(np.float32))
            image, _, _ = rasterization(
                means,
                quats,
                scales,
                opac,
                colors,
                to_t(view[None]),
                to_t(intrinsics[None]),
                _VIEW_PX,
                _VIEW_PX,
                sh_degree=None,
            )
            pixels = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            name = f"view_e{int(elevation):+d}_a{int(azimuth):03d}.png"
            Image.fromarray(pixels).save(views_dir / name)
            intr_list.append(intrinsics)
            view_list.append(view)
            names.append(name)
    np.savez(
        views_dir / "cameras.npz",
        K=np.stack(intr_list),
        V=np.stack(view_list),
        names=names,
        up=up,
        center=center,
        radius=radius,
    )


def _triangulate_one(rows: list[tuple[np.ndarray, float, float, float]]) -> np.ndarray:
    matrix = []
    for proj, x, y, weight in rows:
        matrix.append(weight * (x * proj[2] - proj[0]))
        matrix.append(weight * (y * proj[2] - proj[1]))
    _, _, vh = np.linalg.svd(np.stack(matrix))
    point = vh[-1]
    return point[:3] / point[3]


def _reproj_error(point: np.ndarray, proj: np.ndarray, x: float, y: float) -> float:
    projected = proj @ np.append(point, 1.0)
    return float(np.hypot(projected[0] / projected[2] - x, projected[1] / projected[2] - y))


def _triangulate(views_dir: Path) -> Keypoints3D:
    cameras = np.load(views_dir / "cameras.npz", allow_pickle=True)
    detections = np.load(views_dir / "keypoints2d.npz", allow_pickle=True)
    intr, views, cam_names = cameras["K"], cameras["V"], list(cameras["names"])
    kp2d, bodyparts, view_names = detections["kp"], list(detections["bodyparts"]), list(detections["views"])
    order = [cam_names.index(name) for name in view_names]
    proj = np.stack([intr[o] @ views[o][:3, :4] for o in order])

    positions = np.full((len(bodyparts), 3), np.nan, np.float32)
    support = np.zeros(len(bodyparts), np.int32)
    for part in range(len(bodyparts)):
        obs = [
            (proj[v], float(kp2d[v, part, 0]), float(kp2d[v, part, 1]), float(kp2d[v, part, 2]))
            for v in range(len(view_names))
            if kp2d[v, part, 2] > _CONF_FLOOR
        ]
        if len(obs) < 2:
            continue
        best_count, best_point = -1, None
        for a in range(len(obs)):
            for b in range(a + 1, len(obs)):
                guess = _triangulate_one(
                    [(obs[a][0], obs[a][1], obs[a][2], 1.0), (obs[b][0], obs[b][1], obs[b][2], 1.0)]
                )
                inliers = [o for o in obs if _reproj_error(guess, o[0], o[1], o[2]) < _REPROJ_PX]
                if len(inliers) < 2:
                    continue
                refined = _triangulate_one([(o[0], o[1], o[2], o[3]) for o in inliers])
                if len(inliers) > best_count:
                    best_count, best_point = len(inliers), refined
        if best_point is not None:
            positions[part], support[part] = best_point, best_count
    # Average only over actual detections — SuperAnimal writes -1 for undetected views/keypoints,
    # so a raw mean is dominated by the placeholders and not a meaningful detection-quality signal.
    confidences = kp2d[..., 2]
    detected = confidences[confidences > 0]
    return Keypoints3D(
        bodyparts=[str(b) for b in bodyparts],
        positions=positions,
        support=support,
        mean_confidence=float(detected.mean()) if detected.size else 0.0,
    )


def detect_keypoints_3d(smal: SMAL, splat: GaussianSplat, work_dir: Path) -> Keypoints3D:
    """Render the splat, run SuperAnimal-Quadruped, triangulate -> 3D anatomy landmarks."""
    views_dir = work_dir / "views"
    render_views(smal, splat, views_dir)
    run_command(
        [str(runtime.DLC_PYTHON), str(_RUNNER), str(views_dir)],
        cwd=_RUNNER.parent,
        label="SuperAnimal-Quadruped keypoints",
    )
    return _triangulate(views_dir)
