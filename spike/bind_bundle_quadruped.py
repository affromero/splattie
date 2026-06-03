"""End-to-end quadruped: fit SMAL to the TRELLIS dog splat, bind its LBS onto the
gaussians, emit an animatable .splattie (reusing the object bundle), and prove it
animates by posing a leg and skinning the gaussians."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/ubuntu/Code/splattie/backend/src")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import matrix_to_axis_angle
from scipy.spatial import cKDTree

from smal_torch import SMAL
from splattie.methods.object.bundle import (
    IDENTITY_TRANSFORM,
    RigSkeleton,
    SparseLbsWeights,
    build_object_splattie,
    read_binary_ply,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPIKE = Path("/home/ubuntu/Code/splattie-wt-quadruped-spike/spike")
SMAL_PKL = SPIKE / "backend/vendor/SMAL/smal_online_V1.0/smal_CVPR2017.pkl"
SMAL_PKL = Path("/home/ubuntu/Code/splattie-wt-quadruped-spike/backend/vendor/SMAL/smal_online_V1.0/smal_CVPR2017.pkl")
PLY = SPIKE / "trellis/dog_gaussian.ply"
C0 = 0.28209479177387814


def cube_rotations():
    out = []
    for perm in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3), np.float32)
            for i, p in enumerate(perm):
                m[i, p] = s[i]
            if abs(np.linalg.det(m) - 1) < 1e-3:
                out.append(m)
    return out


def fit(smal, pts):
    pc = pts.mean(0)
    p_ext = pts.max(0).values - pts.min(0).values
    with torch.no_grad():
        v_t, _ = smal(torch.zeros(smal.B, device=DEV), torch.zeros(smal.K, 3, device=DEV))
        s_ext = v_t.max(0).values - v_t.min(0).values
        s_cen = v_t.mean(0)
    scale0 = float((p_ext.max() / s_ext.max()).item())
    best = (1e9, None, None)
    for m in cube_rotations():
        R = torch.tensor(m, device=DEV)
        pose = torch.zeros(smal.K, 3, device=DEV)
        pose[0] = matrix_to_axis_angle(R[None])[0]
        trans = pc - scale0 * (R @ s_cen)
        with torch.no_grad():
            v, _ = smal(torch.zeros(smal.B, device=DEV), pose, trans=trans, scale=scale0)
            d = chamfer_distance(v[None], pts[None])[0].item()
        if d < best[0]:
            best = (d, pose[0].clone(), trans.clone())
    betas = torch.zeros(smal.B, device=DEV, requires_grad=True)
    pose = torch.zeros(smal.K, 3, device=DEV)
    pose[0] = best[1]
    pose = pose.detach().requires_grad_(True)
    scale = torch.tensor(scale0, device=DEV, requires_grad=True)
    trans = best[2].detach().requires_grad_(True)
    for params, iters, full in ([[scale, trans, pose], 250, False], [[betas, pose, scale, trans], 350, True]):
        opt = torch.optim.Adam(params, lr=0.02)
        for _ in range(iters):
            opt.zero_grad()
            v, _ = smal(betas, pose, trans=trans, scale=scale)
            loss = chamfer_distance(v[None], pts[None])[0]
            if full:
                loss = loss + 1e-3 * (pose[1:] ** 2).sum() + 5e-3 * (betas ** 2).sum()
            loss.backward()
            opt.step()
    return betas.detach(), pose.detach(), scale.detach(), trans.detach()


def main():
    smal = SMAL(str(SMAL_PKL), device=DEV)
    rows = read_binary_ply(PLY).vertices
    xyz = np.column_stack([rows["x"], rows["y"], rows["z"]]).astype(np.float32)  # ALL gaussians
    rgb = np.clip(np.column_stack([rows[f"f_dc_{i}"] for i in range(3)]) * C0 + 0.5, 0, 1)
    print(f"splat: {len(xyz)} gaussians")

    rng = np.random.default_rng(0)
    fit_pts = torch.tensor(xyz[rng.choice(len(xyz), 8000, replace=False)], device=DEV)
    betas, pose, scale, trans = fit(smal, fit_pts)
    with torch.no_grad():
        verts, joints, G_rest = smal(betas, pose, trans=trans, scale=scale, return_G=True)
    fch = chamfer_distance(verts[None], fit_pts[None])[0].item()
    print(f"fit chamfer={fch:.5f}")

    # bind: per-gaussian LBS from nearest SMAL vertex -> top-4
    V = verts.cpu().numpy()
    J = joints.cpu().numpy()
    W = smal.weights.cpu().numpy()  # (Vsmal, 33)
    _, nn = cKDTree(V).query(xyz, k=1)
    gw = W[nn]  # (Ngauss, 33)
    k = 4
    idx = np.argpartition(-gw, k - 1, axis=1)[:, :k]
    w = np.take_along_axis(gw, idx, axis=1)
    order = np.argsort(-w, axis=1)
    idx = np.take_along_axis(idx, order, axis=1).astype(np.int64)
    w = np.take_along_axis(w, order, axis=1)
    w = w / np.clip(w.sum(1, keepdims=True), 1e-8, None)
    print(f"bind: top-{k} LBS weights per gaussian; mean dominant weight={w[:, 0].mean():.3f}")

    # bundle (reuse object .splattie writer)
    parents = [int(p) if p >= 0 else -1 for p in smal.parents]
    skel = RigSkeleton(
        names=[f"joint{i}" for i in range(33)], parents=parents,
        rest_positions=[tuple(float(c) for c in J[i]) for i in range(33)],
        rig="smal-quadruped", joint_count=33,
    )
    lbs = SparseLbsWeights(num_gaussians=len(xyz), joint_count=33, k=k,
                           indices=idx.reshape(-1).tolist(), weights=w.reshape(-1).astype(float).tolist())
    out_dir = SPIKE / "bundle"
    bundle_path, n = build_object_splattie(
        ply_path=PLY, output_dir=out_dir, model_id="dog", skeleton=skel, lbs_weights=lbs,
        source_image_path=SPIKE / "animal.png", transform=IDENTITY_TRANSFORM,
    )
    print(f"WROTE .splattie: {bundle_path}  ({n} gaussians, {bundle_path.stat().st_size//1024} KB)")

    # animation proof: rotate a front-leg joint, skin the gaussians via the bound weights
    pose_new = pose.clone()
    pose_new[8] = pose_new[8] + torch.tensor([0.0, 1.1, 0.0], device=DEV)  # swing front leg
    with torch.no_grad():
        _, _, G_new = smal(betas, pose_new, trans=trans, scale=scale, return_G=True)
    Gr = G_rest.cpu().numpy()
    Gn = G_new.cpu().numpy()
    M = np.einsum("jik,jkl->jil", Gn, np.linalg.inv(Gr))  # per-joint delta (SMAL space)
    x_smal = (xyz - trans.cpu().numpy()) / scale.item()
    xh = np.concatenate([x_smal, np.ones((len(xyz), 1), np.float32)], 1)
    acc = np.zeros((len(xyz), 3), np.float32)
    for kk in range(k):
        Mi = M[idx[:, kk]]
        acc += w[:, kk:kk + 1] * np.einsum("nij,nj->ni", Mi, xh)[:, :3]
    posed = (acc * scale.item() + trans.cpu().numpy()).astype(np.float32)
    moved = np.linalg.norm(posed - xyz, axis=1)
    print(f"animation: {(moved > 0.02).sum()} gaussians moved >2cm when the front leg is posed")

    # render rest vs posed (profile = y-z), colored by splat
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (P, ttl) in zip(axs, [(xyz, "bound rest pose"), (posed, "front leg posed (LBS-skinned gaussians)")]):
        o = np.argsort(P[:, 0])
        ax.scatter(P[o, 1], P[o, 2], s=2, c=rgb[o], marker=".", edgecolors="none")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl)
    fig.suptitle("Quadruped .splattie — SMAL-bound gaussians animate (profile y-z)")
    fig.tight_layout()
    fig.savefig("/home/ubuntu/quadruped_animation.png", dpi=120, bbox_inches="tight")
    print("wrote /home/ubuntu/quadruped_animation.png")


if __name__ == "__main__":
    main()
