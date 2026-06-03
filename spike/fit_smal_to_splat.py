"""Fit SMAL (shape+pose+global) to a TRELLIS gaussian splat's 3D points via chamfer.

This is the spike's core: prove a SMAL anatomical skeleton can be registered onto a
TRELLIS reconstruction of a quadruped, so the rig lands inside the splat.
"""

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

from smal_torch import SMAL

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMAL_PKL = "/home/ubuntu/Code/splattie-wt-quadruped-spike/backend/vendor/SMAL/smal_online_V1.0/smal_CVPR2017.pkl"
PLY = "/home/ubuntu/Code/splattie-wt-quadruped-spike/spike/trellis/dog_gaussian.ply"
OUT = "/home/ubuntu/smal_fit_to_splat.png"


def load_splat(ply, n=8000):
    from splattie.methods.object.bundle import read_binary_ply

    r = read_binary_ply(Path(ply)).vertices
    xyz = np.column_stack([r["x"], r["y"], r["z"]]).astype(np.float32)
    if "opacity" in (r.dtype.names or ()):
        keep = 1 / (1 + np.exp(-r["opacity"].astype(np.float32))) > 0.2
        xyz = xyz[keep]
    rng = np.random.default_rng(0)
    if len(xyz) > n:
        xyz = xyz[rng.choice(len(xyz), n, replace=False)]
    return torch.tensor(xyz, device=DEV)


def cube_rotations():
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3), np.float32)
            for i, p in enumerate(perm):
                m[i, p] = signs[i]
            if abs(np.linalg.det(m) - 1) < 1e-3:
                rots.append(m)
    return rots  # 24 proper rotations


def main():
    smal = SMAL(SMAL_PKL, device=DEV)
    pts = load_splat(PLY)
    pc = pts.mean(0)
    p_ext = (pts.max(0).values - pts.min(0).values)

    with torch.no_grad():
        v_t, _ = smal(torch.zeros(smal.B, device=DEV), torch.zeros(smal.K, 3, device=DEV))
        s_ext = v_t.max(0).values - v_t.min(0).values
        s_cen = v_t.mean(0)
    scale0 = float((p_ext.max() / s_ext.max()).item())

    # 24-rotation init search: which global orientation best aligns SMAL to the splat?
    best = (1e9, None, None)
    for m in cube_rotations():
        R = torch.tensor(m, device=DEV)
        po = matrix_to_axis_angle(R[None])[0]
        pose = torch.zeros(smal.K, 3, device=DEV)
        pose[0] = po
        trans = pc - scale0 * (R @ s_cen)
        with torch.no_grad():
            v, _ = smal(torch.zeros(smal.B, device=DEV), pose, trans=trans, scale=scale0)
            d = chamfer_distance(v[None], pts[None])[0].item()
        if d < best[0]:
            best = (d, po.clone(), trans.clone())
    print(f"init search: best chamfer={best[0]:.4f}")

    betas = torch.zeros(smal.B, device=DEV, requires_grad=True)
    pose = torch.zeros(smal.K, 3, device=DEV)
    pose[0] = best[1]
    pose = pose.detach().requires_grad_(True)
    scale = torch.tensor(scale0, device=DEV, requires_grad=True)
    trans = best[2].detach().requires_grad_(True)

    def step(params, iters, full):
        opt = torch.optim.Adam(params, lr=0.02)
        for it in range(iters):
            opt.zero_grad()
            v, j = smal(betas, pose, trans=trans, scale=scale)
            cham = chamfer_distance(v[None], pts[None])[0]
            reg = 0.0
            if full:
                reg = 1e-3 * (pose[1:] ** 2).sum() + 5e-3 * (betas ** 2).sum()
            loss = cham + reg
            loss.backward()
            opt.step()
            if it % 100 == 0 or it == iters - 1:
                print(f"  [{'full' if full else 'glob'}] it{it:3d} chamfer={cham.item():.5f}")

    print("stage 1: global (scale/trans/orient)")
    step([scale, trans, pose], 250, full=False)  # pose grad flows mostly through pose[0] early
    print("stage 2: full (shape + articulated pose)")
    step([betas, pose, scale, trans], 350, full=True)

    with torch.no_grad():
        verts, joints = smal(betas, pose, trans=trans, scale=scale)
    V = verts.cpu().numpy()
    J = joints.cpu().numpy()
    P = pts.cpu().numpy()
    fch = chamfer_distance(verts[None], pts[None])[0].item()
    print(f"final chamfer={fch:.5f}")

    # render: splat points (grey) + fitted SMAL skeleton (red bones/yellow joints), 3 views
    views = [(0, 2, "x-z"), (1, 2, "y-z"), (0, 1, "x-y (top)")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, b, name) in zip(axs, views):
        ax.scatter(P[:, a], P[:, b], s=2, c="#9aa", marker=".", edgecolors="none", alpha=0.5)
        ax.scatter(V[:, a], V[:, b], s=1, c="#7EB8F0", marker=".", edgecolors="none", alpha=0.3)
        for i, p in enumerate(smal.parents):
            if p >= 0:
                ax.plot([J[i, a], J[p, a]], [J[i, b], J[p, b]], "-", c="red", lw=1.8, zorder=5)
        ax.scatter(J[:, a], J[:, b], c="yellow", edgecolors="k", s=18, zorder=6, linewidths=0.5)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"SMAL skeleton fitted to TRELLIS dog splat (chamfer={fch:.4f}) — grey=splat, blue=SMAL mesh, red=rig")
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
