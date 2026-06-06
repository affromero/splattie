"""Rig-aware `.splattie` bundle recompression.

PlayCanvas compressed PLY reorders gaussians into recursive Morton order before
chunking. Rigged bundles store LBS weights in splat-index order, so
recompression must mirror that exact order and re-permute the per-splat payload
before repacking the archive.
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
from jaxtyping import Float, Int, UInt32, jaxtyped
from klogr import get_logger
from pydantic import BaseModel, ConfigDict, TypeAdapter

from splattie.compression.compressed_ply import compress_ply
from splattie.methods.object.bundle import (
    SparseLbsWeights,
    read_binary_ply,
    read_lbs_weights_binary,
    write_lbs_weights_binary,
)

logger = get_logger()

FloatPositions = Float[npt.NDArray[np.float32], "splats 3"]
IntPermutation = Int[npt.NDArray[np.integer], "splats"]
UIntMorton = UInt32[npt.NDArray[np.uint32], "splats"]

_COMPRESSED_MARKER = b"packed_position"
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
def compressed_ply_permutation(original_positions: FloatPositions) -> IntPermutation:
    """Return original indices in PlayCanvas compressed-PLY vertex order."""
    if original_positions.shape[0] == 0:
        msg = "cannot compute compressed-PLY permutation for an empty PLY"
        raise ValueError(msg)
    if not np.all(np.isfinite(original_positions)):
        msg = "cannot compute compressed-PLY permutation for non-finite gaussian centers"
        raise ValueError(msg)

    order = np.arange(original_positions.shape[0], dtype=np.int64)
    _sort_morton_order(original_positions, order, 0, len(order))
    _assert_permutation(order, original_positions.shape[0])
    return order


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
        ply_path.write_bytes(payload[splat_entry])

        original_positions = read_ply_positions(ply_path)
        permutation = compressed_ply_permutation(original_positions)
        compress_ply(ply_path, compressed_path)

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
def _sort_morton_order(positions: FloatPositions, order: IntPermutation, start: int, end: int) -> None:
    """Mirror @playcanvas/splat-transform sortMortonOrder in-place.

    PlayCanvas sorts equal Morton-code buckets recursively only when the bucket
    is larger than one compressed-PLY chunk (256 splats). Stable ordering inside
    smaller equal-code buckets is part of the emitted vertex order.
    """
    if end - start == 0:
        return

    subset = order[start:end]
    points = positions[subset]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    lengths = maxs - mins
    if not np.all(np.isfinite(lengths)):
        msg = "invalid gaussian center extents for compressed-PLY Morton sort"
        raise ValueError(msg)
    if np.all(lengths == 0):
        return

    multipliers = np.divide(
        1024.0,
        lengths,
        out=np.zeros(3, dtype=np.float32),
        where=lengths != 0,
    )
    coords = np.minimum(1023, (points - mins) * multipliers).astype(np.uint32, copy=False)
    morton = _encode_morton3(coords[:, 0], coords[:, 1], coords[:, 2])
    stable = np.argsort(morton, kind="stable")
    order[start:end] = subset[stable]
    sorted_morton = morton[stable]

    bucket_start = start
    local_start = 0
    while local_start < len(sorted_morton):
        local_end = local_start + 1
        while local_end < len(sorted_morton) and sorted_morton[local_end] == sorted_morton[local_start]:
            local_end += 1
        if local_end - local_start > 256:
            _sort_morton_order(positions, order, bucket_start, bucket_start + local_end - local_start)
        bucket_start += local_end - local_start
        local_start = local_end


@jaxtyped(typechecker=beartype)
def _part_1_by_2(values: UIntMorton) -> UIntMorton:
    out = values & np.uint32(0x000003FF)
    out = (out ^ (out << np.uint32(16))) & np.uint32(0xFF0000FF)
    out = (out ^ (out << np.uint32(8))) & np.uint32(0x0300F00F)
    out = (out ^ (out << np.uint32(4))) & np.uint32(0x030C30C3)
    return (out ^ (out << np.uint32(2))) & np.uint32(0x09249249)


@jaxtyped(typechecker=beartype)
def _encode_morton3(
    x: UIntMorton,
    y: UIntMorton,
    z: UIntMorton,
) -> UIntMorton:
    return (_part_1_by_2(z) << np.uint32(2)) + (_part_1_by_2(y) << np.uint32(1)) + _part_1_by_2(x)


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
