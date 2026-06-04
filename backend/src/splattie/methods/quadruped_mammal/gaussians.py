"""Shared gaussian-splat geometry helpers (load, camera matrices, quaternion math)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from splattie.methods.object.bundle import read_binary_ply

SH_C0 = 0.28209479177387814


class GaussianSplat:
    """Decoded gaussian attributes in the PLY's native frame (numpy, CPU)."""

    def __init__(self, ply_path: Path) -> None:
        rows = read_binary_ply(ply_path).vertices
        self.xyz = np.column_stack([rows["x"], rows["y"], rows["z"]]).astype(np.float32)
        quat = np.column_stack([rows[f"rot_{i}"] for i in range(4)]).astype(np.float32)
        self.quat = quat / np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-9, None)
        self.scale = np.exp(np.column_stack([rows[f"scale_{i}"] for i in range(3)]).astype(np.float32))
        self.opacity = 1.0 / (1.0 + np.exp(-rows["opacity"].astype(np.float32)))
        self.color = np.clip(np.column_stack([rows[f"f_dc_{i}"] for i in range(3)]) * SH_C0 + 0.5, 0.0, 1.0).astype(
            np.float32
        )

    def __len__(self) -> int:
        """Return the number of gaussians."""
        return len(self.xyz)


def viewmat(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV-style world->camera view matrix that keeps ``up`` pointing up in the image."""
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], 0)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    return view.astype(np.float32)


def rotation_about(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix about ``axis`` (need not be unit) by ``theta`` radians."""
    unit = axis / np.linalg.norm(axis)
    x, y, z = unit
    cos, sin, t = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return np.array(
        [
            [t * x * x + cos, t * x * y - sin * z, t * x * z + sin * y],
            [t * x * y + sin * z, t * y * y + cos, t * y * z - sin * x],
            [t * x * z - sin * y, t * y * z + sin * x, t * z * z + cos],
        ],
        np.float32,
    )


def quat_from_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Return the unit quaternion (w, x, y, z) for a rotation about ``axis`` by ``theta``."""
    unit = axis / np.linalg.norm(axis)
    s = math.sin(theta / 2.0)
    return np.array([math.cos(theta / 2.0), unit[0] * s, unit[1] * s, unit[2] * s], np.float32)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product ``left * right`` for wxyz quats (``left`` a single quat, ``right`` (N,4))."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        1,
    ).astype(np.float32)


def cube_rotations() -> list[np.ndarray]:
    """Return the 24 proper (det=+1) axis-permutation rotation matrices, for orientation search."""
    out: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            mat = np.zeros((3, 3), np.float32)
            for axis, target in enumerate(perm):
                mat[axis, target] = signs[axis]
            if abs(np.linalg.det(mat) - 1.0) < 1e-3:
                out.append(mat)
    return out
