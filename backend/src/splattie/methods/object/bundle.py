"""Object `.splattie` bundle adapter.

This is the production boundary from the feasibility spike: it accepts rigged
object data (gaussian PLY + arbitrary skeleton + sparse LBS weights), applies
the viewer coordinate transform explicitly, and writes the same bundle shape
that body assets use. Upstream reconstruction/rigging can evolve separately as
long as it hands this module aligned splats, joints, and per-splat weights.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Protocol, Self

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Float, jaxtyped
from pydantic import ConfigDict, Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from splattie.methods.bundle_common import (
    RigSpec,
    build_manifest,
    bundle_splattie,
    count_ply_vertices,
    read_widget_version,
)
from splattie.types import AssetType, SplatFormat

FloatArray = Float[npt.NDArray[np.float32], "..."]

_TOP_K_LIMIT = 4
_LBSW_MAGIC = b"LBSW"
_LBSW_VERSION = 1
_PYDANTIC_CONFIG = ConfigDict(arbitrary_types_allowed=True, extra="forbid", populate_by_name=True)


class HasDType(Protocol):
    """Small protocol for structured NumPy records."""

    dtype: np.dtype


class ObjectTransformName(str, Enum):
    """Supported object coordinate transforms."""

    IDENTITY = "identity"
    VIEWER_UPRIGHT_180X = "viewer-upright-180x"


class QuaternionAxis(str, Enum):
    """Widget quaternion helper axes."""

    X = "x"
    Y = "y"
    Z = "z"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True), kw_only=True)
class ObjectViewerTransform:
    """Rigid transform applied to make object splats render upright in the widget."""

    name: ObjectTransformName
    matrix: FloatArray
    quaternion_wxyz: tuple[float, float, float, float]


IDENTITY_TRANSFORM = ObjectViewerTransform(
    name=ObjectTransformName.IDENTITY,
    matrix=np.eye(3, dtype=np.float32),
    quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
)

OBJECT_VIEWER_TRANSFORM = ObjectViewerTransform(
    name=ObjectTransformName.VIEWER_UPRIGHT_180X,
    matrix=np.diag([1.0, -1.0, -1.0]).astype(np.float32),
    quaternion_wxyz=(0.0, 1.0, 0.0, 0.0),
)

OBJECT_RIG = RigSpec(
    rig="puppeteer-object",
    topology="object-auto",
    splat_format=SplatFormat.PLY,
    skeleton_file="skeleton.json",
    weights_file="lbs_weights.bin",
    expression=None,
)


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class CameraConfig:
    """Widget camera config serialized into states.json."""

    theta: float
    phi: float
    radius: float
    look_at: str = Field(default="auto", alias="lookAt")
    fov: float = 45


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class SaccadeConfig:
    """Inert object saccade settings."""

    enabled: bool
    amplitude: float
    interval_ms: tuple[int, int] = Field(..., alias="intervalMs")
    move_ms: int = Field(..., alias="moveMs")


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class GazeConfig:
    """Inert object gaze config retained for widget schema compatibility."""

    intensity: float
    smoothing_tau: float = Field(..., alias="smoothingTau")
    deadzone: float
    max_eye_yaw: float = Field(..., alias="maxEyeYaw")
    max_eye_pitch: float = Field(..., alias="maxEyePitch")
    max_neck_yaw: float = Field(..., alias="maxNeckYaw")
    max_neck_pitch: float = Field(..., alias="maxNeckPitch")
    saccade: SaccadeConfig


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class GhostConfig:
    """Subtle idle motion config."""

    amplitude: float
    frequency: float
    wobble: float


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class ObjectExpressionConfig:
    """Per-state object expression coefficients."""


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class ObjectTrackingConfig:
    """Object widget pointer tracking gains."""

    head: float
    torso: float


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class ObjectPoseConfig:
    """Per-state object pose overrides."""


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class StateDefinition:
    """One object interaction state."""

    ghost: GhostConfig
    expression: ObjectExpressionConfig
    camera: CameraConfig
    rotation: tuple[float, float, float]
    tracking: ObjectTrackingConfig
    pose: ObjectPoseConfig


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class ObjectStateSet:
    """Object widget states."""

    idle: StateDefinition
    hover: StateDefinition
    click: StateDefinition


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class TransitionConfig:
    """Widget transition config."""

    duration: float
    easing: str


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class ObjectTransitions:
    """Object state transition configs."""

    idle_hover: TransitionConfig
    hover_idle: TransitionConfig
    any_click: TransitionConfig

    def jsonable(self) -> Mapping[str, object]:
        """Return the widget's transition-key shape."""
        return {
            "idle->hover": TypeAdapter(TransitionConfig).dump_python(self.idle_hover, mode="json", by_alias=True),
            "hover->idle": TypeAdapter(TransitionConfig).dump_python(self.hover_idle, mode="json", by_alias=True),
            "*->click": TypeAdapter(TransitionConfig).dump_python(self.any_click, mode="json", by_alias=True),
        }


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class WidgetDefaults:
    """Object widget default config."""

    camera: CameraConfig
    gaze: GazeConfig


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class ObjectWidgetConfig:
    """Full object states.json payload."""

    defaults: WidgetDefaults
    states: ObjectStateSet
    transitions: ObjectTransitions

    def jsonable(self) -> Mapping[str, object]:
        """Return the widget states.json shape."""
        adapter = TypeAdapter(ObjectWidgetConfig)
        payload = adapter.dump_python(self, mode="json", by_alias=True)
        payload["transitions"] = self.transitions.jsonable()
        return payload


OBJECT_CAMERA = CameraConfig(theta=0, phi=82, radius=1.35, look_at="auto", fov=45)
DEFAULT_STATES_OBJECT = ObjectWidgetConfig(
    defaults=WidgetDefaults(
        camera=OBJECT_CAMERA,
        gaze=GazeConfig(
            intensity=0,
            smoothing_tau=0.18,
            deadzone=0.06,
            max_eye_yaw=0,
            max_eye_pitch=0,
            max_neck_yaw=0,
            max_neck_pitch=0,
            saccade=SaccadeConfig(enabled=False, amplitude=0, interval_ms=(3000, 6500), move_ms=90),
        ),
    ),
    states=ObjectStateSet(
        idle=StateDefinition(
            ghost=GhostConfig(amplitude=0.001, frequency=0.25, wobble=0.08),
            expression=ObjectExpressionConfig(),
            camera=OBJECT_CAMERA,
            rotation=(0, 0, 0),
            tracking=ObjectTrackingConfig(head=0, torso=0),
            pose=ObjectPoseConfig(),
        ),
        hover=StateDefinition(
            ghost=GhostConfig(amplitude=0.002, frequency=0.4, wobble=0.12),
            expression=ObjectExpressionConfig(),
            camera=CameraConfig(theta=0, phi=82, radius=1.25, look_at="auto", fov=45),
            rotation=(0, 0, 0),
            tracking=ObjectTrackingConfig(head=0, torso=0),
            pose=ObjectPoseConfig(),
        ),
        click=StateDefinition(
            ghost=GhostConfig(amplitude=0.001, frequency=0.6, wobble=0.05),
            expression=ObjectExpressionConfig(),
            camera=CameraConfig(theta=0, phi=82, radius=1.18, look_at="auto", fov=45),
            rotation=(0, 0, 0),
            tracking=ObjectTrackingConfig(head=0, torso=0),
            pose=ObjectPoseConfig(),
        ),
    ),
    transitions=ObjectTransitions(
        idle_hover=TransitionConfig(duration=0.25, easing="ease-out"),
        hover_idle=TransitionConfig(duration=0.35, easing="ease-in"),
        any_click=TransitionConfig(duration=0.1, easing="snap"),
    ),
)

_PLY_TYPES: Mapping[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True), kw_only=True)
class BinaryPly:
    """Binary little-endian PLY vertex data with scalar properties."""

    vertices: np.ndarray
    properties: list[tuple[str, str]]


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class RigSkeleton:
    """Rig-agnostic skeleton schema consumed by the widget."""

    names: list[str]
    parents: list[int]
    rest_positions: list[tuple[float, float, float]] = Field(..., alias="restPositions")
    rig: str = OBJECT_RIG.rig
    joint_count: int | None = Field(default=None, alias="jointCount")

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Validate skeleton lengths and parent ordering."""
        joint_count = self.joint_count or len(self.names)
        if joint_count != len(self.names):
            msg = "skeleton.jointCount must match skeleton.names length"
            raise ValueError(msg)
        if len(self.parents) != len(self.names):
            msg = "skeleton.parents must match skeleton.names length"
            raise ValueError(msg)
        if len(self.rest_positions) != len(self.names):
            msg = "skeleton.restPositions must match skeleton.names length"
            raise ValueError(msg)
        root_count = 0
        for idx, parent in enumerate(self.parents):
            if parent == -1:
                root_count += 1
                continue
            if parent < 0 or parent >= len(self.names) or parent == idx:
                msg = f"skeleton parent for joint {idx} is invalid"
                raise ValueError(msg)
        if root_count == 0:
            msg = "skeleton must contain at least one root joint"
            raise ValueError(msg)
        self.joint_count = joint_count
        return self

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Self:
        """Parse a JSON-like payload into a validated skeleton."""
        return cls(**payload)

    def transformed(self, transform: ObjectViewerTransform) -> Self:
        """Return a skeleton whose rest positions use the viewer coordinate frame."""
        rest = np.asarray(self.rest_positions, dtype=np.float32)
        transformed = _transform_points(rest, transform)
        return RigSkeleton(
            rig=self.rig,
            joint_count=self.joint_count,
            names=list(self.names),
            parents=list(self.parents),
            rest_positions=[tuple(float(v) for v in row) for row in transformed],
        )

    def topologically_sorted(self, weights: SparseLbsWeights) -> tuple[Self, SparseLbsWeights]:
        """Return parent-before-child skeleton order plus remapped sparse weights."""
        children: list[list[int]] = [[] for _ in self.names]
        roots: list[int] = []
        for idx, parent in enumerate(self.parents):
            if parent == -1:
                roots.append(idx)
            else:
                children[parent].append(idx)

        order: list[int] = []
        cursor = 0
        queue = list(roots)
        while cursor < len(queue):
            old_idx = queue[cursor]
            cursor += 1
            order.append(old_idx)
            queue.extend(children[old_idx])
        if len(order) != len(self.names):
            msg = "skeleton hierarchy contains a cycle or unreachable joint"
            raise ValueError(msg)

        old_to_new = [0 for _ in self.names]
        for new_idx, old_idx in enumerate(order):
            old_to_new[old_idx] = new_idx

        sorted_skeleton = RigSkeleton(
            rig=self.rig,
            joint_count=self.joint_count,
            names=[self.names[old_idx] for old_idx in order],
            parents=[-1 if self.parents[old_idx] == -1 else old_to_new[self.parents[old_idx]] for old_idx in order],
            rest_positions=[self.rest_positions[old_idx] for old_idx in order],
        )
        remapped_weights = SparseLbsWeights(
            num_gaussians=weights.num_gaussians,
            joint_count=weights.joint_count,
            k=weights.k,
            indices=[old_to_new[idx] for idx in weights.indices],
            weights=list(weights.weights),
        )
        return sorted_skeleton, remapped_weights

    def jsonable(self) -> Mapping[str, object]:
        """Return the skeleton JSON payload."""
        return TypeAdapter(RigSkeleton).dump_python(self, mode="json", by_alias=True)


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class SparseLbsWeights:
    """Sparse top-K per-gaussian LBS weights."""

    num_gaussians: int = Field(..., alias="numGaussians")
    joint_count: int = Field(..., alias="jointCount")
    k: int
    indices: list[int]
    weights: list[float]

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Validate lengths and influence count."""
        if self.k < 1 or self.k > _TOP_K_LIMIT:
            msg = f"object LBS weights require 1 <= k <= {_TOP_K_LIMIT}, got {self.k}"
            raise ValueError(msg)
        expected = self.num_gaussians * self.k
        if len(self.indices) != expected or len(self.weights) != expected:
            msg = f"weights length must be numGaussians*k ({expected})"
            raise ValueError(msg)
        return self

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Self:
        """Parse a JSON-like payload into validated sparse weights."""
        return cls(**payload)

    def normalized(self, *, joint_count: int) -> Self:
        """Return validated, row-normalized weights for binary bundle storage."""
        indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        values = np.asarray(self.weights, dtype=np.float32).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= joint_count) or np.any(indices > np.iinfo(np.uint16).max):
            msg = "weights.indices contain invalid joint indices"
            raise ValueError(msg)
        if not np.isfinite(values).all() or np.any(values < 0):
            msg = "weights.weights must be finite non-negative values"
            raise ValueError(msg)

        indices_2d = indices.reshape(self.num_gaussians, self.k)
        values_2d = values.reshape(self.num_gaussians, self.k)
        sums = values_2d.sum(axis=1, keepdims=True)
        missing = sums[:, 0] <= 0
        if np.any(missing):
            values_2d[missing] = 0
            values_2d[missing, 0] = 1
            indices_2d[missing, 0] = 0
            sums = values_2d.sum(axis=1, keepdims=True)
        values_2d = values_2d / np.clip(sums, 1e-8, None)
        return SparseLbsWeights(
            num_gaussians=self.num_gaussians,
            joint_count=joint_count,
            k=self.k,
            indices=indices_2d.astype(np.uint16, copy=False).reshape(-1).astype(int).tolist(),
            weights=values_2d.astype(np.float32, copy=False).reshape(-1).astype(float).tolist(),
        )


def _read_ply_header_lines(path: Path, file_obj: BinaryIO) -> list[str]:
    header_lines: list[str] = []
    while True:
        line = file_obj.readline()
        if not line:
            msg = f"{path} ended before end_header"
            raise ValueError(msg)
        text = line.decode("ascii", errors="strict").strip()
        header_lines.append(text)
        if text == "end_header":
            return header_lines


def _parse_vertex_layout(path: Path, header_lines: list[str]) -> tuple[int, list[tuple[str, str]]]:
    if header_lines[:2] != ["ply", "format binary_little_endian 1.0"]:
        msg = f"{path} must be a binary_little_endian PLY"
        raise ValueError(msg)

    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts or parts[0] in {"ply", "format", "comment", "obj_info", "end_header"}:
            continue
        if parts[0] == "element":
            if parts[1] != "vertex":
                msg = f"{path} has unsupported non-vertex PLY element {parts[1]!r}"
                raise ValueError(msg)
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if in_vertex and parts[0] == "property":
            _append_ply_property(path, properties, parts)

    if vertex_count is None:
        msg = f"{path} has no vertex element"
        raise ValueError(msg)
    if not properties:
        msg = f"{path} vertex element has no scalar properties"
        raise ValueError(msg)
    return vertex_count, properties


def _append_ply_property(path: Path, properties: list[tuple[str, str]], parts: list[str]) -> None:
    if parts[1] == "list":
        msg = f"{path} has unsupported list property in vertex element"
        raise ValueError(msg)
    if parts[1] not in _PLY_TYPES:
        msg = f"{path} has unsupported PLY property type {parts[1]!r}"
        raise ValueError(msg)
    properties.append((parts[2], parts[1]))


def read_binary_ply(path: Path) -> BinaryPly:
    """Read a gaussian binary little-endian PLY while preserving vertex properties."""
    with path.open("rb") as f:
        vertex_count, properties = _parse_vertex_layout(path, _read_ply_header_lines(path, f))
        dtype = np.dtype([(name, _PLY_TYPES[ply_type]) for name, ply_type in properties])
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
        if len(vertices) != vertex_count:
            msg = f"{path} expected {vertex_count} vertices, read {len(vertices)}"
            raise ValueError(msg)

    return BinaryPly(vertices=vertices, properties=properties)


def write_binary_ply(path: Path, ply: BinaryPly) -> None:
    """Write a binary little-endian PLY with the given scalar vertex properties."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(ply.vertices)}\n"
        + "".join(f"property {ply_type} {name}\n" for name, ply_type in ply.properties)
        + "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        ply.vertices.tofile(f)


def _require_fields(vertices: HasDType, fields: tuple[str, ...]) -> None:
    names = set(vertices.dtype.names or ())
    missing = sorted(set(fields) - names)
    if missing:
        msg = f"PLY is missing required fields: {missing}"
        raise ValueError(msg)


@jaxtyped(typechecker=beartype)
def _transform_points(points: FloatArray, transform: ObjectViewerTransform) -> FloatArray:
    return (np.asarray(points, dtype=np.float32) @ transform.matrix.T).astype(np.float32)


@jaxtyped(typechecker=beartype)
def _quat_multiply_wxyz(left: tuple[float, float, float, float], right: FloatArray) -> FloatArray:
    lw, lx, ly, lz = left
    rw = right[:, 0]
    rx = right[:, 1]
    ry = right[:, 2]
    rz = right[:, 3]
    out = np.empty_like(right, dtype=np.float32)
    out[:, 0] = lw * rw - lx * rx - ly * ry - lz * rz
    out[:, 1] = lw * rx + lx * rw + ly * rz - lz * ry
    out[:, 2] = lw * ry - lx * rz + ly * rw + lz * rx
    out[:, 3] = lw * rz + lx * ry - ly * rx + lz * rw
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norm, out=np.zeros_like(out), where=norm > 0)


def transform_gaussian_ply(ply: BinaryPly, transform: ObjectViewerTransform = OBJECT_VIEWER_TRANSFORM) -> BinaryPly:
    """Apply a rigid viewer transform to positions, normals, and wxyz gaussian rotations."""
    out = ply.vertices.copy()
    _require_fields(out, ("x", "y", "z"))
    xyz = np.column_stack([out["x"], out["y"], out["z"]]).astype(np.float32)
    transformed = _transform_points(xyz, transform)
    out["x"], out["y"], out["z"] = transformed[:, 0], transformed[:, 1], transformed[:, 2]

    if {"nx", "ny", "nz"}.issubset(out.dtype.names or ()):
        normals = np.column_stack([out["nx"], out["ny"], out["nz"]]).astype(np.float32)
        transformed_normals = _transform_points(normals, transform)
        out["nx"], out["ny"], out["nz"] = (
            transformed_normals[:, 0],
            transformed_normals[:, 1],
            transformed_normals[:, 2],
        )

    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(out.dtype.names or ()):
        rotations = np.column_stack([out[f"rot_{idx}"] for idx in range(4)]).astype(np.float32)
        transformed_rots = _quat_multiply_wxyz(transform.quaternion_wxyz, rotations)
        for idx in range(4):
            out[f"rot_{idx}"] = transformed_rots[:, idx]

    return BinaryPly(vertices=out, properties=ply.properties)


def write_lbs_weights_binary(path: Path, weights: SparseLbsWeights) -> None:
    """Write sparse LBS weights in the compact `LBSW` binary format."""
    indices = np.asarray(weights.indices, dtype="<u2").reshape(-1)
    values = np.asarray(weights.weights, dtype=np.float32).reshape(-1)
    expected = weights.num_gaussians * weights.k
    if len(indices) != expected or len(values) != expected:
        msg = f"LBSW expected {expected} indices/weights"
        raise ValueError(msg)

    with path.open("wb") as f:
        f.write(_LBSW_MAGIC)
        f.write(struct.pack("<IIII", _LBSW_VERSION, weights.num_gaussians, weights.joint_count, weights.k))
        f.write(indices.tobytes())
        f.write(values.astype("<f2").tobytes())


def read_lbs_weights_binary(path: Path) -> SparseLbsWeights:
    """Read `LBSW` files for tests and offline inspection."""
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != _LBSW_MAGIC:
        msg = f"{path} is not an LBSW weights file"
        raise ValueError(msg)
    version, num_gaussians, joint_count, k = struct.unpack("<IIII", data[4:20])
    if version != _LBSW_VERSION:
        msg = f"unsupported LBSW version {version}"
        raise ValueError(msg)
    count = num_gaussians * k
    indices_offset = 20
    weights_offset = indices_offset + count * 2
    expected_size = weights_offset + count * 2
    if len(data) != expected_size:
        msg = f"{path} has {len(data)} bytes, expected {expected_size}"
        raise ValueError(msg)
    indices = np.frombuffer(data, dtype="<u2", count=count, offset=indices_offset).astype(int).tolist()
    values = (
        np.frombuffer(data, dtype="<f2", count=count, offset=weights_offset).astype(np.float32).astype(float).tolist()
    )
    return SparseLbsWeights(
        num_gaussians=num_gaussians,
        joint_count=joint_count,
        k=k,
        indices=indices,
        weights=values,
    )


def build_object_splattie(
    *,
    ply_path: Path,
    output_dir: Path,
    model_id: str,
    skeleton: RigSkeleton,
    lbs_weights: SparseLbsWeights,
    source_image_path: Path | None = None,
    transform: ObjectViewerTransform = OBJECT_VIEWER_TRANSFORM,
) -> tuple[Path, int]:
    """Write a widget-loadable object `.splattie` from a rigged gaussian PLY."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_skeleton, sorted_weights = skeleton.topologically_sorted(lbs_weights)
    transformed_skeleton = sorted_skeleton.transformed(transform)
    sparse_weights = sorted_weights.normalized(joint_count=len(transformed_skeleton.names))

    transformed_ply_path = output_dir / f"{model_id}.ply"
    skeleton_path = output_dir / OBJECT_RIG.skeleton_file
    weights_path = output_dir / OBJECT_RIG.weights_file

    write_binary_ply(transformed_ply_path, transform_gaussian_ply(read_binary_ply(ply_path), transform))
    skeleton_path.write_text(json.dumps(transformed_skeleton.jsonable(), separators=(",", ":")) + "\n")
    write_lbs_weights_binary(weights_path, sparse_weights)

    num_gaussians = count_ply_vertices(transformed_ply_path)
    if num_gaussians != sparse_weights.num_gaussians:
        msg = f"PLY has {num_gaussians} gaussians but weights declare {sparse_weights.num_gaussians}"
        raise ValueError(msg)

    manifest = build_manifest(
        splat_filename=transformed_ply_path.name,
        num_gaussians=num_gaussians,
        widget_version=read_widget_version(),
        asset_type=AssetType.object,
        rig=OBJECT_RIG,
        generator_method="trellis-puppeteer",
        generator_method_version="object-rig-v1",
        generator_tool="splattie-backend",
        source_image_path=source_image_path,
    )
    animation = manifest["animation"]
    if isinstance(animation, dict) and isinstance(animation.get("weights"), dict):
        animation["weights"]["format"] = "lbsw-v1"
    manifest.setdefault("metadata", {})
    manifest["metadata"]["viewerTransform"] = transform.name.value

    bundle_path = output_dir / f"{model_id}.splattie"
    bundle_splattie(
        output_path=bundle_path,
        splat_path=transformed_ply_path,
        manifest=manifest,
        states=DEFAULT_STATES_OBJECT.jsonable(),
        rig_files={
            OBJECT_RIG.skeleton_file: skeleton_path,
            OBJECT_RIG.weights_file: weights_path,
        },
    )
    return bundle_path, num_gaussians


def object_transform_from_name(name: ObjectTransformName) -> ObjectViewerTransform:
    """Resolve a CLI-safe transform name."""
    if name is ObjectTransformName.IDENTITY:
        return IDENTITY_TRANSFORM
    if name is ObjectTransformName.VIEWER_UPRIGHT_180X:
        return OBJECT_VIEWER_TRANSFORM
    msg = f"unknown object transform {name!r}"
    raise ValueError(msg)


def quaternion_xyzw(axis: QuaternionAxis, degrees: float) -> tuple[float, float, float, float]:
    """Small helper for authoring object state poses in widget xyzw convention."""
    half = math.radians(degrees) / 2.0
    s = math.sin(half)
    c = math.cos(half)
    if axis is QuaternionAxis.X:
        return (s, 0.0, 0.0, c)
    if axis is QuaternionAxis.Y:
        return (0.0, s, 0.0, c)
    if axis is QuaternionAxis.Z:
        return (0.0, 0.0, s, c)
    msg = f"unknown axis {axis!r}"
    raise ValueError(msg)
