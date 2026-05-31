"""Object `.splattie` bundle commands."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from klogr import get_logger

from splattie.methods.object.bundle import (
    ObjectTransformName,
    RigSkeleton,
    SparseLbsWeights,
    build_object_splattie,
    object_transform_from_name,
)

logger = get_logger()


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        msg = f"{path} must contain a JSON object"
        raise TypeError(msg)
    return payload


def _load_rigged_npz(path: Path) -> tuple[RigSkeleton, SparseLbsWeights]:
    payload = np.load(path)
    required = {
        "joint_names",
        "parent_indices",
        "joint_positions_gaussian",
        "splat_joint_indices",
        "splat_joint_weights",
    }
    missing = sorted(required - set(payload.files))
    if missing:
        msg = f"{path} is missing required arrays: {missing}"
        raise ValueError(msg)

    names = payload["joint_names"].astype(str).tolist()
    parents = payload["parent_indices"].astype(int).tolist()
    rest_positions = payload["joint_positions_gaussian"].astype(np.float32).tolist()
    indices = payload["splat_joint_indices"].astype(np.int64)
    weights = payload["splat_joint_weights"].astype(np.float32)

    if indices.shape != weights.shape or indices.ndim != 2:
        msg = "splat_joint_indices and splat_joint_weights must both be [numGaussians, k]"
        raise ValueError(msg)

    skeleton = RigSkeleton(
        rig="puppeteer-object",
        joint_count=len(names),
        names=names,
        parents=parents,
        rest_positions=rest_positions,
    )
    lbs_weights = SparseLbsWeights(
        num_gaussians=int(indices.shape[0]),
        joint_count=len(names),
        k=int(indices.shape[1]),
        indices=indices.reshape(-1).astype(int).tolist(),
        weights=weights.reshape(-1).astype(float).tolist(),
    )
    return skeleton, lbs_weights


def bundle_object(
    ply_path: Path,
    output_dir: Path,
    model_id: str | None = None,
    *,
    skeleton_json: Path | None = None,
    weights_json: Path | None = None,
    rigged_splat_npz: Path | None = None,
    source_image_path: Path | None = None,
    viewer_transform: ObjectTransformName = ObjectTransformName.VIEWER_UPRIGHT_180X,
) -> None:
    """Bundle a rigged arbitrary object into a widget-loadable `.splattie`.

    Args:
        ply_path: Binary little-endian gaussian PLY to include in the bundle.
        output_dir: Directory where `<model_id>.splattie` and intermediates are written.
        model_id: Bundle stem. Defaults to `ply_path.stem`.
        skeleton_json: Skeleton JSON with `names`, `parents`, and `restPositions`.
        weights_json: Sparse LBS weights JSON with `numGaussians`, `k`, `indices`, and `weights`.
        rigged_splat_npz: Spike/bind output containing joint arrays and splat weights.
        source_image_path: Optional source image path hashed into manifest metadata.
        viewer_transform: Coordinate transform to apply before packaging.

    Provide either (`skeleton_json` and `weights_json`) or `rigged_splat_npz`.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_id = model_id or ply_path.stem

    if rigged_splat_npz is not None:
        if skeleton_json is not None or weights_json is not None:
            msg = "Use either rigged_splat_npz or skeleton_json+weights_json, not both"
            raise ValueError(msg)
        skeleton, weights = _load_rigged_npz(rigged_splat_npz)
    else:
        if skeleton_json is None or weights_json is None:
            msg = "skeleton_json and weights_json are required when rigged_splat_npz is not provided"
            raise ValueError(msg)
        skeleton = RigSkeleton.from_payload(_load_json_mapping(skeleton_json))
        weights = SparseLbsWeights.from_payload(_load_json_mapping(weights_json))

    bundle_path, num_gaussians = build_object_splattie(
        ply_path=ply_path,
        output_dir=output_dir,
        model_id=resolved_id,
        skeleton=skeleton,
        lbs_weights=weights,
        source_image_path=source_image_path,
        transform=object_transform_from_name(viewer_transform),
    )
    logger.info(f"Object .splattie: {bundle_path} ({num_gaussians} gaussians)")
