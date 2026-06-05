"""Rig-aware `.splattie` bundle recompression.

PlayCanvas compressed PLY reorders gaussians into spatial chunks. Rigged bundles
store LBS weights in splat-index order, so recompression must recover the new
file order and re-permute the per-splat payload before repacking the archive.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from klogr import get_logger
from pydantic import BaseModel, ConfigDict, TypeAdapter

from splattie.compression.compressed_ply import compress_ply, decode_ply
from splattie.methods.object.bundle import (
    SparseLbsWeights,
    read_binary_ply,
    read_lbs_weights_binary,
    write_lbs_weights_binary,
)

logger = get_logger()

FloatPositions = Float[npt.NDArray[np.float32], "splats 3"]
IntPermutation = Int[npt.NDArray[np.integer], "splats"]

_COMPRESSED_MARKER = b"packed_position"
_MAX_REPAIR_COUNT = 2048
_MAX_REPAIR_RATIO = 0.005
_SPARSE_WEIGHTS_ADAPTER = TypeAdapter(SparseLbsWeights)


class CompressBundleResult(BaseModel):
    """Outcome of rig-aware bundle compression."""

    model_config = ConfigDict(frozen=True)

    splat_entry: str
    weights_entry: str
    already_compressed: bool
    size_before: int
    size_after: int


def is_compressed_ply_bytes(data: bytes) -> bool:
    """Return whether a PLY payload appears to be PlayCanvas compressed PLY."""
    return _COMPRESSED_MARKER in data[:2048]


@jaxtyped(typechecker=beartype)
def read_ply_positions(path: Path) -> FloatPositions:
    """Read gaussian centers from a plain binary PLY."""
    vertices = read_binary_ply(path).vertices
    names = set(vertices.dtype.names or ())
    missing = {"x", "y", "z"} - names
    if missing:
        msg = f"{path} is missing gaussian center fields: {sorted(missing)}"
        raise ValueError(msg)
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32, copy=False)


@jaxtyped(typechecker=beartype)
def recover_permutation(original_positions: FloatPositions, new_positions: FloatPositions) -> IntPermutation:
    """Return original indices for each splat in the new compressed-file order."""
    if original_positions.shape != new_positions.shape:
        msg = f"position arrays must have the same shape, got {original_positions.shape} and {new_positions.shape}"
        raise ValueError(msg)
    if original_positions.shape[0] == 0:
        msg = "cannot recover permutation for an empty PLY"
        raise ValueError(msg)

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        msg = "scipy is required to recover compressed-Ply splat permutations"
        raise RuntimeError(msg) from exc

    distances, nearest = cKDTree(original_positions).query(new_positions)
    permutation = np.asarray(nearest, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    repaired = _repair_duplicate_matches(original_positions, new_positions, permutation, distances)
    _assert_permutation(repaired, original_positions.shape[0])
    return repaired


@jaxtyped(typechecker=beartype)
def reorder_sparse_weights(weights: SparseLbsWeights, permutation: IntPermutation) -> SparseLbsWeights:
    """Reorder gaussian-major sparse LBS weights into the new splat order."""
    if len(permutation) != weights.num_gaussians:
        msg = f"permutation length {len(permutation)} does not match {weights.num_gaussians} gaussians"
        raise ValueError(msg)
    _assert_permutation(permutation, weights.num_gaussians)

    row_indices = np.asarray(weights.indices, dtype=np.int64).reshape(weights.num_gaussians, weights.k)
    row_values = np.asarray(weights.weights, dtype=np.float32).reshape(weights.num_gaussians, weights.k)
    order = np.asarray(permutation, dtype=np.int64)
    return SparseLbsWeights(
        num_gaussians=weights.num_gaussians,
        joint_count=weights.joint_count,
        k=weights.k,
        indices=row_indices[order].reshape(-1).astype(int).tolist(),
        weights=row_values[order].reshape(-1).astype(float).tolist(),
    )


def compress_bundle_rigged(src: Path, dst: Path) -> CompressBundleResult:
    """Rewrite a rigged `.splattie` with compressed PLY and aligned LBS weights."""
    src = src.resolve()
    dst = dst.resolve()
    size_before = src.stat().st_size

    with zipfile.ZipFile(str(src), "r") as zf:
        entries = [(info, zf.read(info.filename)) for info in zf.infolist() if not info.is_dir()]

    names = [info.filename for info, _data in entries]
    payload = {info.filename: data for info, data in entries}
    manifest = _manifest_payload(payload.get("manifest.json"))
    splat_entry = _splat_entry_name(names, manifest)
    weights_entry = _weights_entry_name(names, manifest)
    _reject_flame_topology(manifest)

    if is_compressed_ply_bytes(payload[splat_entry]):
        if src != dst:
            shutil.copyfile(src, dst)
        return CompressBundleResult(
            splat_entry=splat_entry,
            weights_entry=weights_entry,
            already_compressed=True,
            size_before=size_before,
            size_after=size_before,
        )

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        ply_path = work / "splat.ply"
        compressed_path = work / "splat.compressed.ply"
        decoded_path = work / "splat.decoded.ply"
        ply_path.write_bytes(payload[splat_entry])

        original_positions = read_ply_positions(ply_path)
        compress_ply(ply_path, compressed_path)
        decode_ply(compressed_path, decoded_path)
        new_positions = read_ply_positions(decoded_path)
        permutation = recover_permutation(original_positions, new_positions)

        weights = _read_sparse_weights_bytes(work, weights_entry, payload[weights_entry])
        if weights.num_gaussians != len(permutation):
            msg = f"{weights_entry} declares {weights.num_gaussians} gaussians but {splat_entry} has {len(permutation)}"
            raise ValueError(msg)
        payload[splat_entry] = compressed_path.read_bytes()
        payload[weights_entry] = _write_sparse_weights_bytes(
            work, weights_entry, reorder_sparse_weights(weights, permutation)
        )

    _write_bundle(entries, payload, dst)
    size_after = dst.stat().st_size
    logger.info(f"Compressed rigged bundle {src.name}: {size_before // 1024} KB -> {size_after // 1024} KB")
    return CompressBundleResult(
        splat_entry=splat_entry,
        weights_entry=weights_entry,
        already_compressed=False,
        size_before=size_before,
        size_after=size_after,
    )


@jaxtyped(typechecker=beartype)
def _repair_duplicate_matches(
    original_positions: FloatPositions,
    new_positions: FloatPositions,
    permutation: IntPermutation,
    distances: Float[npt.NDArray[np.float32], "splats"],
) -> IntPermutation:
    counts = np.bincount(permutation, minlength=original_positions.shape[0])
    duplicate_slots = np.flatnonzero(counts > 1)
    if len(duplicate_slots) == 0:
        return permutation

    repaired = np.asarray(permutation, dtype=np.int64).copy()
    missing = np.flatnonzero(counts == 0).astype(np.int64)
    displaced: list[int] = []
    for original_idx in duplicate_slots:
        new_indices = np.flatnonzero(repaired == original_idx)
        keep_offset = int(np.argmin(distances[new_indices]))
        displaced.extend(int(new_idx) for new_idx in np.delete(new_indices, keep_offset))

    if len(displaced) != len(missing):
        msg = f"cannot repair permutation: {len(displaced)} duplicates but {len(missing)} missing originals"
        raise ValueError(msg)

    repair_limit = min(_MAX_REPAIR_COUNT, max(1, int(original_positions.shape[0] * _MAX_REPAIR_RATIO)))
    if len(displaced) > repair_limit:
        msg = f"compressed PLY produced {len(displaced)} duplicate nearest-neighbor matches; refusing to guess"
        raise ValueError(msg)

    distance_matrix = np.linalg.norm(
        new_positions[np.asarray(displaced, dtype=np.int64), None, :] - original_positions[missing][None, :, :],
        axis=2,
    )
    assigned_rows: set[int] = set()
    assigned_cols: set[int] = set()
    while len(assigned_rows) < len(displaced):
        masked = distance_matrix.copy()
        if assigned_rows:
            masked[list(assigned_rows), :] = np.inf
        if assigned_cols:
            masked[:, list(assigned_cols)] = np.inf
        flat_idx = int(np.argmin(masked))
        row, col = np.unravel_index(flat_idx, masked.shape)
        if not np.isfinite(masked[row, col]):
            msg = "could not complete duplicate nearest-neighbor repair"
            raise ValueError(msg)
        repaired[displaced[row]] = int(missing[col])
        assigned_rows.add(int(row))
        assigned_cols.add(int(col))

    logger.info(f"Repaired {len(displaced)} duplicate compressed-PLY nearest-neighbor matches")
    return repaired


@jaxtyped(typechecker=beartype)
def _assert_permutation(permutation: IntPermutation, expected: int) -> None:
    order = np.asarray(permutation, dtype=np.int64)
    if len(order) != expected:
        msg = f"expected permutation of length {expected}, got {len(order)}"
        raise ValueError(msg)
    if np.any(order < 0) or np.any(order >= expected):
        msg = "permutation contains out-of-range indices"
        raise ValueError(msg)
    counts = np.bincount(order, minlength=expected)
    if not np.all(counts == 1):
        duplicates = int(np.count_nonzero(counts > 1))
        missing = int(np.count_nonzero(counts == 0))
        msg = f"permutation is not bijective: {duplicates} duplicates, {missing} missing"
        raise ValueError(msg)


def _manifest_payload(data: bytes | None):
    if data is None:
        msg = "bundle has no manifest.json"
        raise ValueError(msg)
    manifest = json.loads(data.decode("utf-8"))
    if not isinstance(manifest, dict):
        msg = "manifest.json must be a JSON object"
        raise TypeError(msg)
    return manifest


def _splat_entry_name(names: list[str], manifest: object) -> str:
    try:
        entry = manifest["avatar"]["splat"]["file"]
    except (KeyError, TypeError):
        entry = None
    if isinstance(entry, str) and entry in names:
        return entry

    plys = [name for name in names if name.endswith(".ply")]
    if len(plys) != 1:
        msg = f"cannot locate splat PLY: expected exactly one .ply entry, found {plys}"
        raise ValueError(msg)
    return plys[0]


def _weights_entry_name(names: list[str], manifest: object) -> str:
    try:
        entry = manifest["animation"]["weights"]["file"]
    except (KeyError, TypeError):
        entry = None
    if isinstance(entry, str) and entry in names:
        return entry

    weight_entries = [name for name in names if name in {"lbs_weights.bin", "lbs_weights.json", "lbs_weight_20k.json"}]
    if len(weight_entries) != 1:
        msg = f"cannot locate per-splat weights entry: found {weight_entries}"
        raise ValueError(msg)
    return weight_entries[0]


def _reject_flame_topology(manifest: object) -> None:
    try:
        topology = manifest["avatar"]["splat"]["topology"]
    except (KeyError, TypeError):
        topology = None
    if topology == "flame-20k":
        msg = "refusing to compress flame-20k head bundle because splat order is FLAME topology"
        raise ValueError(msg)


def _read_sparse_weights_bytes(work: Path, entry: str, data: bytes) -> SparseLbsWeights:
    if entry.endswith(".bin"):
        path = work / "weights.bin"
        path.write_bytes(data)
        return read_lbs_weights_binary(path)
    if entry.endswith(".json"):
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            msg = f"{entry} must be a JSON object"
            raise ValueError(msg)
        return SparseLbsWeights.from_payload(payload)
    msg = f"unsupported weights entry format: {entry}"
    raise ValueError(msg)


def _write_sparse_weights_bytes(work: Path, entry: str, weights: SparseLbsWeights) -> bytes:
    if entry.endswith(".bin"):
        path = work / "weights.reordered.bin"
        write_lbs_weights_binary(path, weights)
        return path.read_bytes()
    if entry.endswith(".json"):
        payload = _SPARSE_WEIGHTS_ADAPTER.dump_python(weights, mode="json", by_alias=True)
        return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    msg = f"unsupported weights entry format: {entry}"
    raise ValueError(msg)


def _write_bundle(entries: list[tuple[zipfile.ZipInfo, bytes]], payload: object, dst: Path) -> None:
    tmp_out = dst.with_suffix(dst.suffix + ".tmp")
    with zipfile.ZipFile(str(tmp_out), "w", zipfile.ZIP_DEFLATED) as zf:
        for info, _data in entries:
            cloned = zipfile.ZipInfo(info.filename, info.date_time)
            cloned.comment = info.comment
            cloned.create_system = info.create_system
            cloned.external_attr = info.external_attr
            cloned.extra = info.extra
            cloned.internal_attr = info.internal_attr
            cloned.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(cloned, payload[info.filename])
    tmp_out.replace(dst)
