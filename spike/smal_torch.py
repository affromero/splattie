"""Minimal differentiable SMAL (SMPL-style) forward pass in PyTorch.

forward(betas, pose, trans, scale) -> (posed_vertices, posed_joints), all differentiable,
so SMAL can be optimised (pose+shape+global) to match a 3D point cloud (a TRELLIS splat).
"""

from __future__ import annotations

# chumpy / numpy<2 compat shim for py3.11 (SMAL pkl unpickles via chumpy)
import inspect
import pickle
from pathlib import Path

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
import numpy as np

for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("str", str), ("unicode", str), ("bool8", bool)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
import chumpy  # noqa: F401  (must import after shim, before unpickle)
import torch


def _arr(x) -> np.ndarray:
    return np.asarray(x.r if hasattr(x, "r") else x, dtype=np.float32)


def rodrigues(axis_angle: torch.Tensor) -> torch.Tensor:
    """(K,3) axis-angle -> (K,3,3) rotation matrices."""
    theta = torch.norm(axis_angle + 1e-12, dim=1, keepdim=True)  # (K,1)
    r = (axis_angle / theta).unsqueeze(-1)  # (K,3,1)
    cos = torch.cos(theta).unsqueeze(-1)
    sin = torch.sin(theta).unsqueeze(-1)
    z = torch.zeros_like(r[:, 0, 0])
    rx, ry, rz = r[:, 0, 0], r[:, 1, 0], r[:, 2, 0]
    K = torch.stack([z, -rz, ry, rz, z, -rx, -ry, rx, z], dim=1).view(-1, 3, 3)
    eye = torch.eye(3, device=axis_angle.device).unsqueeze(0)
    rrt = torch.matmul(r, r.transpose(1, 2))
    return cos * eye + (1 - cos) * rrt + sin * K


class SMAL:
    def __init__(self, pkl_path: str | Path, device: str = "cpu"):
        with open(pkl_path, "rb") as f:
            m = pickle.load(f, encoding="latin1")
        self.device = device
        t = lambda a: torch.tensor(_arr(a), device=device)
        self.v_template = t(m["v_template"])          # (V,3)
        self.shapedirs = t(m["shapedirs"])            # (V,3,B)
        self.posedirs = t(m["posedirs"])              # (V,3,9*(K-1))
        jr = m["J_regressor"]
        jr = jr.todense() if hasattr(jr, "todense") else _arr(jr)
        self.J_regressor = torch.tensor(np.asarray(jr, dtype=np.float32), device=device)  # (K,V)
        self.weights = t(m["weights"])                # (V,K)
        self.faces = np.asarray(m["f"]).astype(np.int64)
        kt = np.asarray(m["kintree_table"]).astype(np.int64)
        parents = kt[0].copy()
        parents[parents > 10**8] = -1                 # root
        self.parents = parents
        self.K = int(self.weights.shape[1])
        self.V = int(self.v_template.shape[0])
        self.B = int(self.shapedirs.shape[2])

    def __call__(self, betas, pose, trans=None, scale=None, return_G=False):
        dev = self.device
        v_shaped = self.v_template + torch.einsum("vck,k->vc", self.shapedirs, betas)
        J = torch.einsum("kv,vc->kc", self.J_regressor, v_shaped)         # (K,3)
        R = rodrigues(pose)                                              # (K,3,3)
        eye = torch.eye(3, device=dev)
        pose_feat = (R[1:] - eye).reshape(-1)                            # (9*(K-1),)
        v_posed = v_shaped + torch.einsum("vck,k->vc", self.posedirs, pose_feat)

        def rt(Ri, ti):
            G = torch.eye(4, device=dev)
            G = torch.cat([torch.cat([Ri, ti.view(3, 1)], 1), G[3:4]], 0)
            return G

        results = []
        for i in range(self.K):
            p = self.parents[i]
            local = rt(R[i], J[i] if p == -1 else J[i] - J[p])
            results.append(local if p == -1 else results[p] @ local)
        G = torch.stack(results)                                         # (K,4,4) world transforms
        joints = G[:, :3, 3]                                            # posed joint positions

        # rest-pose removal for skinning
        Jh = torch.cat([J, torch.zeros(self.K, 1, device=dev)], 1).unsqueeze(-1)  # (K,4,1)
        GJ = torch.matmul(G, Jh)                                         # (K,4,1)
        pack = torch.cat([torch.zeros(self.K, 4, 3, device=dev), GJ], 2)  # (K,4,4) in last col
        G_skin = G - pack
        T = torch.einsum("vk,kij->vij", self.weights, G_skin)           # (V,4,4)
        vh = torch.cat([v_posed, torch.ones(self.V, 1, device=dev)], 1).unsqueeze(-1)
        verts = torch.matmul(T, vh)[:, :3, 0]

        if scale is not None:
            verts, joints = verts * scale, joints * scale
        if trans is not None:
            verts, joints = verts + trans, joints + trans
        if return_G:
            return verts, joints, G  # G = SMAL-space world joint transforms (pre scale/trans)
        return verts, joints
