"""Bind Puppeteer mesh rigs onto TRELLIS gaussian splats.

Puppeteer predicts a RigNet-style skeleton plus per-mesh-vertex skin weights.
TRELLIS emits a gaussian PLY and a mesh from the same latent object, but their
axis conventions may differ. This module picks the best signed-axis alignment,
transfers mesh weights to nearest gaussians, and returns the typed bundle inputs.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from pydantic import ConfigDict, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from splattie.methods.object.bundle import BinaryPly, RigSkeleton, SparseLbsWeights, read_binary_ply, write_binary_ply

FloatPoints = Float[npt.NDArray[np.float32], "points 3"]
FloatWeights = Float[npt.NDArray[np.float32], "vertices joints"]
FloatWeightRows = Float[npt.NDArray[np.float32], "rows joints"]
FloatTopWeights = Float[npt.NDArray[np.float32], "rows k"]
IntTopMatrix = Int[npt.NDArray[np.integer], "rows k"]

_PYDANTIC_CONFIG = ConfigDict(arbitrary_types_allowed=True, extra="forbid", populate_by_name=True)
_NEAREST_FALLBACK_CHUNK_SIZE = 512


@dataclass(config=_PYDANTIC_CONFIG, frozen=True, kw_only=True)
class JointInfluence:
    """One mesh-vertex joint influence parsed from Puppeteer's skin file."""

    joint_name: str
    weight: float


@dataclass(config=_PYDANTIC_CONFIG, frozen=True, kw_only=True)
class VertexSkin:
    """All joint influences for one mesh vertex."""

    vertex_index: int
    influences: tuple[JointInfluence, ...]


@dataclass(config=_PYDANTIC_CONFIG, frozen=True, kw_only=True)
class RigSkin:
    """Puppeteer/RigNet skeleton and mesh-vertex skin weights."""

    joint_names: list[str]
    parent_indices: list[int]
    joint_positions: list[tuple[float, float, float]]
    root_index: int
    skins: list[VertexSkin]

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Validate skeleton and skinning references."""
        joint_count = len(self.joint_names)
        if joint_count == 0:
            msg = "rig skin contains no joints"
            raise ValueError(msg)
        if len(self.parent_indices) != joint_count or len(self.joint_positions) != joint_count:
            msg = "rig skin joint arrays must have matching lengths"
            raise ValueError(msg)
        if self.root_index < 0 or self.root_index >= joint_count:
            msg = "rig skin root index is invalid"
            raise ValueError(msg)
        if not self.skins:
            msg = "rig skin contains no vertex weights"
            raise ValueError(msg)
        names = set(self.joint_names)
        for skin in self.skins:
            if skin.vertex_index < 0:
                msg = "skin vertex index must be non-negative"
                raise ValueError(msg)
            for influence in skin.influences:
                if influence.joint_name not in names:
                    msg = f"skin influence references unknown joint {influence.joint_name!r}"
                    raise ValueError(msg)
        return self

    def joint_to_index(self) -> Mapping[str, int]:
        """Return joint-name to index mapping."""
        return {name: idx for idx, name in enumerate(self.joint_names)}


@dataclass(config=_PYDANTIC_CONFIG, frozen=True, kw_only=True)
class AxisTransform:
    """Signed axis mapping from gaussian coordinates into mesh coordinates."""

    axes: tuple[int, int, int]
    signs: tuple[float, float, float]

    def jsonable(self) -> Mapping[str, object]:
        """Return a compact JSON payload."""
        return {
            "axes": list(self.axes),
            "signs": list(self.signs),
            "formula": "mesh_xyz = gaussian_xyz[:, axes] * signs",
        }


@dataclass(config=_PYDANTIC_CONFIG, frozen=True, kw_only=True)
class DistanceStats:
    """Nearest-mesh transfer quality stats."""

    mean: float
    median: float
    p95: float
    max: float


@dataclass(config=_PYDANTIC_CONFIG, frozen=True, kw_only=True)
class BindingSummary:
    """Human-readable binding metadata for inspection and smoke tests."""

    gaussian_count: int
    mesh_vertex_count: int
    mesh_face_count: int
    joint_count: int
    root_joint: str
    top_k: int
    gaussian_to_mesh_transform: AxisTransform
    nearest_mesh_distance: DistanceStats
    gaussian_bbox_min: tuple[float, float, float]
    gaussian_bbox_max: tuple[float, float, float]
    mesh_bbox_min: tuple[float, float, float]
    mesh_bbox_max: tuple[float, float, float]
    rigged_splat_npz: str
    dominant_joint_preview_ply: str | None

    def jsonable(self) -> Mapping[str, object]:
        """Return the JSON payload written next to the binding output."""
        return TypeAdapter(BindingSummary).dump_python(self, mode="json")


@dataclass(config=_PYDANTIC_CONFIG, frozen=True, kw_only=True)
class RiggedSplatBinding:
    """Typed object bundle inputs produced by gaussian-to-rig binding."""

    skeleton: RigSkeleton
    lbs_weights: SparseLbsWeights
    summary: BindingSummary
    rigged_splat_npz: Path
    summary_json: Path
    dominant_joint_preview_ply: Path | None


def parse_rignet_skin(path: Path) -> RigSkin:
    """Parse Puppeteer's RigNet-style skeleton plus skinning text file."""
    joints: MutableMapping[str, tuple[float, float, float]] = {}
    hierarchy: list[tuple[str, str]] = []
    skins: list[VertexSkin] = []
    root_name: str | None = None

    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "joints":
            name, position = _parse_joint_line(path, line_no, parts)
            joints[name] = position
        elif tag == "root":
            root_name = _parse_root_line(path, line_no, parts)
        elif tag == "hier":
            hierarchy.append(_parse_hier_line(path, line_no, parts))
        elif tag == "skin":
            skins.append(_parse_skin_line(path, line_no, parts))

    return _rig_skin_from_parts(path, joints, root_name, hierarchy, skins)


def _parse_joint_line(path: Path, line_no: int, parts: Sequence[str]) -> tuple[str, tuple[float, float, float]]:
    if len(parts) != 5:
        msg = f"{path}:{line_no}: joints line must have 5 fields"
        raise ValueError(msg)
    return parts[1], (float(parts[2]), float(parts[3]), float(parts[4]))


def _parse_root_line(path: Path, line_no: int, parts: Sequence[str]) -> str:
    if len(parts) != 2:
        msg = f"{path}:{line_no}: root line must have 2 fields"
        raise ValueError(msg)
    return parts[1]


def _parse_hier_line(path: Path, line_no: int, parts: Sequence[str]) -> tuple[str, str]:
    if len(parts) != 3:
        msg = f"{path}:{line_no}: hier line must have 3 fields"
        raise ValueError(msg)
    return parts[1], parts[2]


def _rig_skin_from_parts(
    path: Path,
    joints: Mapping[str, tuple[float, float, float]],
    root_name: str | None,
    hierarchy: Sequence[tuple[str, str]],
    skins: Sequence[VertexSkin],
) -> RigSkin:
    if not joints:
        msg = f"{path} contains no joints"
        raise ValueError(msg)
    if root_name is None:
        msg = f"{path} contains no root"
        raise ValueError(msg)
    if root_name not in joints:
        msg = f"{path} root joint {root_name!r} is not declared"
        raise ValueError(msg)

    joint_names = list(joints.keys())
    joint_to_index = {name: idx for idx, name in enumerate(joint_names)}
    parent_indices = [-1 for _ in joint_names]
    for parent, child in hierarchy:
        if parent not in joint_to_index or child not in joint_to_index:
            msg = f"{path} hierarchy references undeclared joint {parent!r}->{child!r}"
            raise ValueError(msg)
        parent_indices[joint_to_index[child]] = joint_to_index[parent]

    return RigSkin(
        joint_names=joint_names,
        parent_indices=parent_indices,
        joint_positions=[joints[name] for name in joint_names],
        root_index=joint_to_index[root_name],
        skins=list(skins),
    )


def _parse_skin_line(path: Path, line_no: int, parts: Sequence[str]) -> VertexSkin:
    if len(parts) < 4 or len(parts) % 2 != 0:
        msg = f"{path}:{line_no}: skin line must contain vertex plus joint/weight pairs"
        raise ValueError(msg)
    influences = tuple(
        JointInfluence(joint_name=parts[idx], weight=float(parts[idx + 1])) for idx in range(2, len(parts), 2)
    )
    return VertexSkin(vertex_index=int(parts[1]), influences=influences)


@jaxtyped(typechecker=beartype)
def build_vertex_weight_matrix(vertex_count: int, rig: RigSkin) -> FloatWeights:
    """Build dense mesh-vertex weights from parsed RigNet skin data."""
    joint_to_index = rig.joint_to_index()
    weights = np.zeros((vertex_count, len(rig.joint_names)), dtype=np.float32)
    for skin in rig.skins:
        if skin.vertex_index >= vertex_count:
            msg = f"skin vertex index {skin.vertex_index} exceeds mesh vertex count {vertex_count}"
            raise ValueError(msg)
        for influence in skin.influences:
            weights[skin.vertex_index, joint_to_index[influence.joint_name]] = influence.weight

    sums = weights.sum(axis=1, keepdims=True)
    missing = np.flatnonzero(sums[:, 0] <= 0)
    if len(missing) > 0:
        msg = f"{len(missing)} mesh vertices have no skin weights"
        raise ValueError(msg)
    return np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0).astype(np.float32)


@jaxtyped(typechecker=beartype)
def top_k_weights(weights: FloatWeightRows, k: int) -> tuple[IntTopMatrix, FloatTopWeights]:
    """Return row-normalized top-k joint weights."""
    if k < 1:
        msg = "top_k must be >= 1"
        raise ValueError(msg)
    top_k = min(k, weights.shape[1])
    top_indices = np.argpartition(-weights, kth=top_k - 1, axis=1)[:, :top_k]
    top_values = np.take_along_axis(weights, top_indices, axis=1)
    order = np.argsort(-top_values, axis=1)
    top_indices = np.take_along_axis(top_indices, order, axis=1).astype(np.int64)
    top_values = np.take_along_axis(top_values, order, axis=1).astype(np.float32)
    sums = top_values.sum(axis=1, keepdims=True)
    return top_indices, np.divide(top_values, sums, out=np.zeros_like(top_values), where=sums > 0).astype(np.float32)


@jaxtyped(typechecker=beartype)
def choose_gaussian_to_mesh_transform(
    gaussian_xyz: Float[npt.NDArray[np.float32], "gaussians 3"],
    mesh_vertices: Float[npt.NDArray[np.float32], "mesh_vertices 3"],
) -> AxisTransform:
    """Choose the signed-axis transform with the lowest nearest-mesh distance."""
    sample_count = min(50_000, len(gaussian_xyz))
    if sample_count < len(gaussian_xyz):
        sample_indices = np.linspace(0, len(gaussian_xyz) - 1, sample_count, dtype=np.int64)
        sample = gaussian_xyz[sample_indices]
    else:
        sample = gaussian_xyz

    best_score = float("inf")
    best_axes = (0, 1, 2)
    best_signs = (1.0, 1.0, 1.0)
    for axes in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            transformed = transform_gaussian_to_mesh(sample, AxisTransform(axes=axes, signs=signs))
            nearest_distance, _ = nearest_vertices(transformed, mesh_vertices)
            score = float(np.mean(nearest_distance))
            if score < best_score:
                best_score = score
                best_axes = axes
                best_signs = signs
    return AxisTransform(axes=best_axes, signs=best_signs)


@jaxtyped(typechecker=beartype)
def transform_gaussian_to_mesh(xyz: FloatPoints, transform: AxisTransform) -> FloatPoints:
    """Apply the selected gaussian-to-mesh axis transform."""
    axes = np.asarray(transform.axes, dtype=np.int64)
    signs = np.asarray(transform.signs, dtype=np.float32)
    return (xyz[:, axes] * signs).astype(np.float32)


@jaxtyped(typechecker=beartype)
def transform_mesh_to_gaussian(xyz: FloatPoints, transform: AxisTransform) -> FloatPoints:
    """Invert the selected gaussian-to-mesh axis transform."""
    out = np.empty_like(xyz, dtype=np.float32)
    for mesh_axis, gaussian_axis in enumerate(transform.axes):
        out[:, gaussian_axis] = xyz[:, mesh_axis] * transform.signs[mesh_axis]
    return out


@jaxtyped(typechecker=beartype)
def nearest_vertices(
    query_xyz: Float[npt.NDArray[np.float32], "queries 3"],
    reference_xyz: Float[npt.NDArray[np.float32], "references 3"],
) -> tuple[Float[npt.NDArray[np.float32], "queries"], Int[npt.NDArray[np.integer], "queries"]]:
    """Find nearest reference vertex for each query point."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return _nearest_vertices_numpy(query_xyz, reference_xyz)

    tree = cKDTree(reference_xyz)
    try:
        distance, index = tree.query(query_xyz, k=1, workers=-1)
    except TypeError:
        distance, index = tree.query(query_xyz, k=1)
    return distance.astype(np.float32), index.astype(np.int64)


@jaxtyped(typechecker=beartype)
def _nearest_vertices_numpy(
    query_xyz: Float[npt.NDArray[np.float32], "queries 3"],
    reference_xyz: Float[npt.NDArray[np.float32], "references 3"],
) -> tuple[Float[npt.NDArray[np.float32], "queries"], Int[npt.NDArray[np.integer], "queries"]]:
    distances = np.empty(len(query_xyz), dtype=np.float32)
    indices = np.empty(len(query_xyz), dtype=np.int64)
    for start in range(0, len(query_xyz), _NEAREST_FALLBACK_CHUNK_SIZE):
        stop = min(start + _NEAREST_FALLBACK_CHUNK_SIZE, len(query_xyz))
        diff = query_xyz[start:stop, None, :] - reference_xyz[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        local_index = np.argmin(sq_dist, axis=1)
        indices[start:stop] = local_index
        distances[start:stop] = np.sqrt(sq_dist[np.arange(stop - start), local_index])
    return distances, indices


def bind_rigged_splat(
    *,
    gaussian_ply: Path,
    mesh_obj: Path,
    rig_skin: Path,
    output_dir: Path,
    model_id: str,
    top_k: int = 4,
    write_preview: bool = True,
) -> RiggedSplatBinding:
    """Transfer Puppeteer mesh skin weights onto TRELLIS gaussians."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gaussian_rows = read_binary_ply(gaussian_ply).vertices
    _require_gaussian_xyz(gaussian_rows)
    gaussian_xyz = np.column_stack([gaussian_rows["x"], gaussian_rows["y"], gaussian_rows["z"]]).astype(np.float32)

    mesh_vertices, mesh_face_count = _load_mesh_vertices(mesh_obj)
    rig = parse_rignet_skin(rig_skin)
    vertex_weights = build_vertex_weight_matrix(len(mesh_vertices), rig)

    axis_transform = choose_gaussian_to_mesh_transform(gaussian_xyz, mesh_vertices)
    binding_xyz = transform_gaussian_to_mesh(gaussian_xyz, axis_transform)
    nearest_distance, nearest_vertex = nearest_vertices(binding_xyz, mesh_vertices)
    splat_weights = vertex_weights[nearest_vertex]
    splat_joint_indices, splat_joint_weights = top_k_weights(splat_weights, top_k)
    joint_positions = transform_mesh_to_gaussian(np.asarray(rig.joint_positions, dtype=np.float32), axis_transform)

    skeleton = RigSkeleton(
        rig="puppeteer-object",
        joint_count=len(rig.joint_names),
        names=list(rig.joint_names),
        parents=list(rig.parent_indices),
        rest_positions=[tuple(float(value) for value in row) for row in joint_positions],
    )
    lbs_weights = SparseLbsWeights(
        num_gaussians=len(gaussian_xyz),
        joint_count=len(rig.joint_names),
        k=int(splat_joint_indices.shape[1]),
        indices=splat_joint_indices.reshape(-1).astype(int).tolist(),
        weights=splat_joint_weights.reshape(-1).astype(float).tolist(),
    )

    npz_path = output_dir / f"{model_id}_rigged_splat.npz"
    preview_path = output_dir / f"{model_id}_dominant_joint_preview.ply" if write_preview else None
    if preview_path is not None:
        write_dominant_joint_preview(preview_path, gaussian_xyz, splat_joint_indices[:, 0].astype(np.int64))
    _write_binding_npz(
        npz_path=npz_path,
        gaussian_rows=gaussian_rows,
        gaussian_ply=gaussian_ply,
        transform=axis_transform,
        rig=rig,
        joint_positions_gaussian=joint_positions,
        splat_joint_indices=splat_joint_indices,
        splat_joint_weights=splat_joint_weights,
        nearest_vertex=nearest_vertex,
        nearest_distance=nearest_distance,
    )

    summary = _binding_summary(
        gaussian_xyz=gaussian_xyz,
        mesh_vertices=mesh_vertices,
        mesh_face_count=mesh_face_count,
        rig=rig,
        top_k=top_k,
        transform=axis_transform,
        nearest_distance=nearest_distance,
        npz_path=npz_path,
        preview_path=preview_path,
    )
    summary_json = output_dir / f"{model_id}_rigged_splat_summary.json"
    summary_json.write_text(json.dumps(summary.jsonable(), indent=2) + "\n")

    return RiggedSplatBinding(
        skeleton=skeleton,
        lbs_weights=lbs_weights,
        summary=summary,
        rigged_splat_npz=npz_path,
        summary_json=summary_json,
        dominant_joint_preview_ply=preview_path,
    )


def _require_gaussian_xyz(rows: object) -> None:
    dtype = getattr(rows, "dtype", None)
    names = dtype.names if isinstance(dtype, np.dtype) else ()
    missing = sorted({"x", "y", "z"} - set(names or ()))
    if missing:
        msg = f"gaussian PLY is missing required fields: {missing}"
        raise ValueError(msg)


@jaxtyped(typechecker=beartype)
def _load_mesh_vertices(mesh_obj: Path) -> tuple[Float[npt.NDArray[np.float32], "mesh_vertices 3"], int]:
    import trimesh

    mesh = trimesh.load(mesh_obj, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = tuple(mesh.geometry.values())
        if not geometries:
            msg = f"{mesh_obj} contains no mesh geometry"
            raise ValueError(msg)
        mesh = trimesh.util.concatenate(geometries)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        msg = f"{mesh_obj} did not load as a vertex mesh"
        raise ValueError(msg)
    return vertices, len(faces)


@jaxtyped(typechecker=beartype)
def write_dominant_joint_preview(
    path: Path,
    xyz: Float[npt.NDArray[np.float32], "gaussians 3"],
    dominant_joint: Int[npt.NDArray[np.integer], "gaussians"],
) -> None:
    """Write a colored preview PLY for inspecting transferred dominant joints."""
    max_joint = int(dominant_joint.max()) if len(dominant_joint) else 0
    palette = np.asarray(
        [[(idx * 73 + 31) % 256, (idx * 151 + 67) % 256, (idx * 199 + 109) % 256] for idx in range(max_joint + 1)],
        dtype=np.uint8,
    )
    colors = palette[dominant_joint]
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rows = np.empty(len(xyz), dtype=dtype)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rows["red"], rows["green"], rows["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    write_binary_ply(
        path,
        BinaryPly(
            vertices=rows,
            properties=[
                ("x", "float"),
                ("y", "float"),
                ("z", "float"),
                ("red", "uchar"),
                ("green", "uchar"),
                ("blue", "uchar"),
            ],
        ),
    )


@jaxtyped(typechecker=beartype)
def _write_binding_npz(
    *,
    npz_path: Path,
    gaussian_rows: object,
    gaussian_ply: Path,
    transform: AxisTransform,
    rig: RigSkin,
    joint_positions_gaussian: Float[npt.NDArray[np.float32], "joints 3"],
    splat_joint_indices: Int[npt.NDArray[np.integer], "gaussians k"],
    splat_joint_weights: Float[npt.NDArray[np.float32], "gaussians k"],
    nearest_vertex: Int[npt.NDArray[np.integer], "gaussians"],
    nearest_distance: Float[npt.NDArray[np.float32], "gaussians"],
) -> None:
    np.savez_compressed(
        npz_path,
        gaussians=gaussian_rows,
        gaussian_properties=np.asarray(gaussian_rows.dtype.names or ()),
        gaussian_ply=np.asarray(str(gaussian_ply)),
        gaussian_to_mesh_axes=np.asarray(transform.axes, dtype=np.int16),
        gaussian_to_mesh_signs=np.asarray(transform.signs, dtype=np.float32),
        joint_names=np.asarray(rig.joint_names),
        joint_positions_mesh=np.asarray(rig.joint_positions, dtype=np.float32),
        joint_positions_gaussian=joint_positions_gaussian,
        root_index=np.asarray(rig.root_index, dtype=np.int16),
        parent_indices=np.asarray(rig.parent_indices, dtype=np.int16),
        splat_joint_indices=splat_joint_indices.astype(np.int16),
        splat_joint_weights=splat_joint_weights.astype(np.float32),
        nearest_mesh_vertex=nearest_vertex.astype(np.int32),
        nearest_mesh_distance=nearest_distance.astype(np.float32),
    )


@jaxtyped(typechecker=beartype)
def _binding_summary(
    *,
    gaussian_xyz: Float[npt.NDArray[np.float32], "gaussians 3"],
    mesh_vertices: Float[npt.NDArray[np.float32], "mesh_vertices 3"],
    mesh_face_count: int,
    rig: RigSkin,
    top_k: int,
    transform: AxisTransform,
    nearest_distance: Float[npt.NDArray[np.float32], "gaussians"],
    npz_path: Path,
    preview_path: Path | None,
) -> BindingSummary:
    return BindingSummary(
        gaussian_count=len(gaussian_xyz),
        mesh_vertex_count=len(mesh_vertices),
        mesh_face_count=mesh_face_count,
        joint_count=len(rig.joint_names),
        root_joint=rig.joint_names[rig.root_index],
        top_k=top_k,
        gaussian_to_mesh_transform=transform,
        nearest_mesh_distance=DistanceStats(
            mean=float(np.mean(nearest_distance)),
            median=float(np.median(nearest_distance)),
            p95=float(np.percentile(nearest_distance, 95)),
            max=float(np.max(nearest_distance)),
        ),
        gaussian_bbox_min=_tuple3(gaussian_xyz.min(axis=0)),
        gaussian_bbox_max=_tuple3(gaussian_xyz.max(axis=0)),
        mesh_bbox_min=_tuple3(mesh_vertices.min(axis=0)),
        mesh_bbox_max=_tuple3(mesh_vertices.max(axis=0)),
        rigged_splat_npz=str(npz_path),
        dominant_joint_preview_ply=str(preview_path) if preview_path is not None else None,
    )


@jaxtyped(typechecker=beartype)
def _tuple3(values: Float[npt.NDArray[np.float32], "3"]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))
