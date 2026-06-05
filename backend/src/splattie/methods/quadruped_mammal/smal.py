"""Minimal differentiable SMAL (Skinned Multi-Animal Linear) forward pass in PyTorch.

``SMAL(betas, pose, trans, scale)`` returns ``(vertices, joints)`` (optionally the SMAL-space
world joint transforms ``G``), all differentiable — so the model can be optimised
(pose + shape + global) to register onto a 3D point cloud (a TRELLIS gaussian splat).

The SMAL pkl unpickles via chumpy, which needs numpy<2 / py3.11 names restored first; the
shim below mirrors the one in ``methods/lam/method.py`` for FLAME.
"""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
import numpy as np
import numpy.typing as npt

for _name, _type in [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode", str),
    ("bool8", bool),
]:
    if not hasattr(np, _name):
        setattr(np, _name, _type)
import chumpy  # noqa: E402, F401  (must import after the shim, before unpickling the pkl)
import torch  # noqa: E402
from beartype import beartype  # noqa: E402
from jaxtyping import Float, jaxtyped  # noqa: E402


@jaxtyped(typechecker=beartype)
def _arr(value: object) -> Float[npt.NDArray[np.float32], "..."]:
    return np.asarray(value.r if hasattr(value, "r") else value, dtype=np.float32)


@jaxtyped(typechecker=beartype)
def rodrigues(axis_angle: Float[torch.Tensor, "k 3"]) -> Float[torch.Tensor, "k 3 3"]:
    """Convert ``(K, 3)`` axis-angle vectors to ``(K, 3, 3)`` rotation matrices."""
    theta = torch.norm(axis_angle + 1e-12, dim=1, keepdim=True)
    r = (axis_angle / theta).unsqueeze(-1)
    cos = torch.cos(theta).unsqueeze(-1)
    sin = torch.sin(theta).unsqueeze(-1)
    zero = torch.zeros_like(r[:, 0, 0])
    rx, ry, rz = r[:, 0, 0], r[:, 1, 0], r[:, 2, 0]
    skew = torch.stack([zero, -rz, ry, rz, zero, -rx, -ry, rx, zero], dim=1).view(-1, 3, 3)
    eye = torch.eye(3, device=axis_angle.device).unsqueeze(0)
    rrt = torch.matmul(r, r.transpose(1, 2))
    return cos * eye + (1 - cos) * rrt + sin * skew


class SMAL:
    """Differentiable SMAL forward. ``__call__`` is batch-free (one mesh per call)."""

    def __init__(self, pkl_path: str | Path, device: str = "cpu") -> None:
        with Path(pkl_path).open("rb") as handle:
            model = pickle.load(handle, encoding="latin1")  # noqa: S301  (trusted vendored SMAL model)
        self.device = device

        @jaxtyped(typechecker=beartype)
        def tensor(value: object) -> Float[torch.Tensor, "..."]:
            return torch.tensor(_arr(value), device=device)

        self.v_template = tensor(model["v_template"])
        self.shapedirs = tensor(model["shapedirs"])
        self.posedirs = tensor(model["posedirs"])
        regressor = model["J_regressor"]
        regressor = regressor.todense() if hasattr(regressor, "todense") else _arr(regressor)
        self.J_regressor = torch.tensor(np.asarray(regressor, dtype=np.float32), device=device)
        self.weights = tensor(model["weights"])
        self.faces = np.asarray(model["f"]).astype(np.int64)
        kintree = np.asarray(model["kintree_table"]).astype(np.int64)
        parents = kintree[0].copy()
        parents[parents > 10**8] = -1
        self.parents = parents
        self.K = int(self.weights.shape[1])
        self.V = int(self.v_template.shape[0])
        self.B = int(self.shapedirs.shape[2])

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        betas: Float[torch.Tensor, "b"],
        pose: Float[torch.Tensor, "k 3"],
        trans: Float[torch.Tensor, "3"] | None = None,
        scale: Float[torch.Tensor, ""] | float | None = None,
        *,
        return_world: bool = False,
    ) -> tuple[Float[torch.Tensor, "..."], ...]:
        device = self.device
        v_shaped = self.v_template + torch.einsum("vck,k->vc", self.shapedirs, betas)
        joints_rest = torch.einsum("kv,vc->kc", self.J_regressor, v_shaped)
        rotations = rodrigues(pose)
        eye = torch.eye(3, device=device)
        pose_feature = (rotations[1:] - eye).reshape(-1)
        v_posed = v_shaped + torch.einsum("vck,k->vc", self.posedirs, pose_feature)

        @jaxtyped(typechecker=beartype)
        def rigid(
            rotation: Float[torch.Tensor, "3 3"], translation: Float[torch.Tensor, "3"]
        ) -> Float[torch.Tensor, "4 4"]:
            bottom = torch.eye(4, device=device)[3:4]
            return torch.cat([torch.cat([rotation, translation.view(3, 1)], 1), bottom], 0)

        transforms: list[torch.Tensor] = []
        for joint in range(self.K):
            parent = self.parents[joint]
            offset = joints_rest[joint] if parent == -1 else joints_rest[joint] - joints_rest[parent]
            local = rigid(rotations[joint], offset)
            transforms.append(local if parent == -1 else transforms[parent] @ local)
        world = torch.stack(transforms)
        joints = world[:, :3, 3]

        joints_h = torch.cat([joints_rest, torch.zeros(self.K, 1, device=device)], 1).unsqueeze(-1)
        world_joints = torch.matmul(world, joints_h)
        pack = torch.cat([torch.zeros(self.K, 4, 3, device=device), world_joints], 2)
        skinning = world - pack
        per_vertex = torch.einsum("vk,kij->vij", self.weights, skinning)
        v_homogeneous = torch.cat([v_posed, torch.ones(self.V, 1, device=device)], 1).unsqueeze(-1)
        vertices = torch.matmul(per_vertex, v_homogeneous)[:, :3, 0]

        if scale is not None:
            vertices, joints = vertices * scale, joints * scale
        if trans is not None:
            vertices, joints = vertices + trans, joints + trans
        if return_world:
            return vertices, joints, world  # world = SMAL-space joint transforms (pre scale/trans)
        return vertices, joints
