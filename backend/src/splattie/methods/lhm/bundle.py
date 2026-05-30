"""LHM body `.splattie` bundle adapter (Phase 1.C).

Turns LHM's exported gaussian PLY into a widget-loadable body `.splattie` by
extracting the SMPL-X rig the widget needs to skin + pose it:

- **skeleton.json** — the 55 SMPL-X joints (names + parents + rest positions). The
  gaussians are baked into the *photographed* body pose (arms at rest), so the rest
  positions are the matching **posed** joints emitted alongside the PLY by
  `infer_mesh` (``<stem>_joints.json``). The widget's rest pose is therefore the
  identity — no shoulder-down rotation, hence no skinning stretch. Falls back to the
  canonical ``J_regressor @ (v_template + betas·shapedirs)`` joints only when pose
  estimation was unavailable (the legacy A-pose export).
- **lbs_weights.json** — the per-gaussian linear-blend-skinning weights
  (``smplx_model.skinning_weight`` [N, 55]) stored sparse top-K (each gaussian binds
  to a few joints), renormalised. The lower-arm gaussians are then re-bound rigidly to
  the elbow (reweight_lower_arm_rigid) so that *rotating* the arm (editor IK /
  animation) swings it as a solid piece instead of stretching — the rest position is
  set by the bake, this only affects how the limb deforms when moved.

The widget (Phase 1.D) reads these to skin the gaussians via SMPL-X LBS and apply
the head/torso look-at + arm IK. Reuses `bundle_common.build_manifest` /
`bundle_splattie` so body bundles share the head bundle's shape (manifest + splat +
rig + states.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Float, jaxtyped

from splattie.methods.bundle_common import (
    RigSpec,
    build_manifest,
    bundle_splattie,
    count_ply_vertices,
    read_widget_version,
)
from splattie.types import AssetType, SplatFormat

# SMPL-X kinematic joints, in order (mirrors smpl_x.py:148 joints_name). Pelvis is
# the root; the body widget uses Spine/Neck/Head for the look-at.
JOINTS_NAME: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine_1",
    "L_Knee",
    "R_Knee",
    "Spine_2",
    "L_Ankle",
    "R_Ankle",
    "Spine_3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "Jaw",
    "L_Eye",
    "R_Eye",
    "L_Index_1",
    "L_Index_2",
    "L_Index_3",
    "L_Middle_1",
    "L_Middle_2",
    "L_Middle_3",
    "L_Pinky_1",
    "L_Pinky_2",
    "L_Pinky_3",
    "L_Ring_1",
    "L_Ring_2",
    "L_Ring_3",
    "L_Thumb_1",
    "L_Thumb_2",
    "L_Thumb_3",
    "R_Index_1",
    "R_Index_2",
    "R_Index_3",
    "R_Middle_1",
    "R_Middle_2",
    "R_Middle_3",
    "R_Pinky_1",
    "R_Pinky_2",
    "R_Pinky_3",
    "R_Ring_1",
    "R_Ring_2",
    "R_Ring_3",
    "R_Thumb_1",
    "R_Thumb_2",
    "R_Thumb_3",
)

# How many joints each gaussian binds to in the bundle (SMPL-X LBS is sparse).
_TOP_K = 4

# Lower-arm region: a gaussian is "forearm/hand" when its nearest joint is the elbow,
# wrist, or a finger. Only these are re-bound — rigidly, to the elbow (_LOWER_ARM_TARGETS)
# — so that rotating the limb (editor IK / animation) swings the forearm as one solid
# piece instead of the approximate per-gaussian LBS stretching it into thin tendrils.
# The collar/shoulder/upper-arm are deliberately left with their original soft weights
# so the shoulder + chest still deform naturally (re-binding them dragged the torso).
_ARM_JOINTS = frozenset({"L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"})
_FINGER_TOKENS = ("Index", "Middle", "Pinky", "Ring", "Thumb")
_LOWER_ARM_TARGETS = frozenset({"L_Elbow", "R_Elbow"})

# The SMPL-X body rig: skeleton + per-gaussian weights are generated per body (the
# skeleton is betas-specific), so the bundler writes them via rig_files.
BODY_RIG = RigSpec(
    rig="smplx",
    topology="smplx-voxel",
    splat_format=SplatFormat.PLY,
    skeleton_file="skeleton.json",
    weights_file="lbs_weights.json",
    expression=None,
)

# Default widget states for a body. Parallels DEFAULT_STATES_HEAD but tracks
# head + torso look-at (the chosen body interaction) instead of eyes, and frames
# the full standing figure. The editor (1.E) tunes camera/pose per body.
DEFAULT_STATES_BODY: dict = {
    "defaults": {"camera": {"theta": 0, "phi": 90, "radius": 2.4, "fov": 45}},
    "states": {
        "idle": {
            "ghost": {"amplitude": 0.004, "frequency": 0.3, "wobble": 0.2},
            "expression": {},
            "camera": {"theta": 0, "phi": 90, "radius": 2.4, "fov": 45},
            "rotation": [0, 0, 0],
            "tracking": {"head": 1.0, "torso": 0.3},
        },
        "hover": {
            "ghost": {"amplitude": 0.006, "frequency": 0.5, "wobble": 0.3},
            "expression": {},
            "camera": {"theta": 0, "phi": 90, "radius": 2.2, "fov": 45},
            "rotation": [0, 0, 0],
            "tracking": {"head": 1.0, "torso": 0.5},
        },
        "click": {
            "ghost": {"amplitude": 0.002, "frequency": 0.8, "wobble": 0.1},
            "expression": {},
            "camera": {"theta": 0, "phi": 88, "radius": 2.0, "fov": 48},
            "rotation": [0, 0, 0],
            "tracking": {"head": 0.6, "torso": 0.2},
        },
    },
    "transitions": {
        "idle->hover": {"duration": 0.3, "easing": "ease-out"},
        "hover->idle": {"duration": 0.5, "easing": "ease-in"},
        "*->click": {"duration": 0.1, "easing": "snap"},
    },
}


@jaxtyped(typechecker=beartype)
def _rest_joints(
    smplx_model: object, betas: Float[npt.NDArray[np.float32], "n_betas"]
) -> Float[npt.NDArray[np.float32], "55 3"]:
    """Rest-pose SMPL-X joints [55, 3] for the person's betas, in gaussian space.

    ``J_regressor @ (v_template + betas·shapedirs)`` — the same canonical shaping the
    gaussians use, so the skeleton overlays the body (validated on h100).
    """
    import torch

    layer = getattr(smplx_model, "smplx_layer", None) or smplx_model.layer["neutral"]
    n_betas = int(getattr(layer, "num_betas", 10))
    device = layer.v_template.device
    bt = torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(-1)[:n_betas]
    v_shaped = layer.v_template + torch.einsum("l,vcl->vc", bt, layer.shapedirs[:, :, :n_betas])
    j_regressor = layer.J_regressor
    j_regressor = j_regressor.to_dense() if j_regressor.is_sparse else j_regressor
    joints = (j_regressor.to(device) @ v_shaped)[:55]
    return joints.detach().cpu().numpy().astype(np.float32)


@jaxtyped(typechecker=beartype)
def extract_body_rig(
    smplx_model: object,
    betas: Float[npt.NDArray[np.float32], "n_betas"],
    posed_joints: Float[npt.NDArray[np.float32], "55 3"] | None = None,
) -> tuple[dict, dict]:
    """Extract (skeleton, sparse-weights) dicts from the loaded LHM SMPL-X model.

    ``posed_joints`` (from ``infer_mesh``'s ``<stem>_joints.json``) are the joints in
    the baked-pose frame; when given they become the skeleton's rest positions so the
    widget's identity rest pose matches the gaussians. Without them the canonical
    (A-pose) joints are used (legacy export path).
    """
    weights = smplx_model.skinning_weight.detach().cpu().numpy().astype(np.float32)  # [N, 55]
    n_gaussians = int(weights.shape[0])

    # Sparse top-K per gaussian, renormalised so the kept weights sum to 1.
    top = np.argsort(-weights, axis=1)[:, :_TOP_K].astype(np.int32)  # [N, K]
    top_w = np.take_along_axis(weights, top, axis=1)  # [N, K]
    top_w = top_w / np.clip(top_w.sum(axis=1, keepdims=True), 1e-8, None)

    joints = posed_joints if posed_joints is not None else _rest_joints(smplx_model, betas)  # [55, 3]
    parents = _smplx_parents(smplx_model)

    skeleton = {
        "rig": "smplx",
        "jointCount": len(JOINTS_NAME),
        "names": list(JOINTS_NAME),
        "parents": parents,
        "restPositions": joints.tolist(),
    }
    lbs_weights = {
        "numGaussians": n_gaussians,
        "jointCount": len(JOINTS_NAME),
        "k": _TOP_K,
        "indices": top.reshape(-1).tolist(),
        "weights": top_w.reshape(-1).astype(np.float32).round(6).tolist(),
    }
    return skeleton, lbs_weights


def _smplx_parents(smplx_model: object) -> list[int]:
    """Parent index per joint (root = -1), from the SMPL-X kinematic tree."""
    layer = getattr(smplx_model, "smplx_layer", None) or smplx_model.layer["neutral"]
    import torch

    parents = layer.parents
    parents = parents.detach().cpu().numpy() if torch.is_tensor(parents) else np.asarray(parents)
    return [int(p) for p in parents[:55]]


@jaxtyped(typechecker=beartype)
def parse_ply_xyz(data: bytes) -> Float[npt.NDArray[np.float32], "n 3"]:
    """Read gaussian XYZ positions ``[N, 3]`` from a binary little-endian PLY blob.

    The skinning re-weight needs per-gaussian positions in the same (baked-pose) space
    as the skeleton. Bodies are always binary little-endian PLY (gaussian splats); ascii
    is unsupported. Properties are read by name, so extra channels (color, opacity,
    scale, rotation) are tolerated.
    """
    marker = b"end_header\n"
    header_end = data.index(marker) + len(marker)
    header = data[:header_end].decode("ascii", errors="ignore")
    if "format binary_little_endian" not in header:
        msg = "parse_ply_xyz expects a binary little-endian PLY"
        raise ValueError(msg)
    np_of = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
        "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
    }  # fmt: skip
    n_verts = 0
    fields: list[tuple[str, str]] = []
    for line in header.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property" and parts[1] != "list":
            fields.append((parts[2], np_of[parts[1]]))
    arr = np.frombuffer(data, dtype=np.dtype(fields), count=n_verts, offset=header_end)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)


@jaxtyped(typechecker=beartype)
def reweight_lower_arm_rigid(
    positions: Float[npt.NDArray[np.float32], "n 3"],
    skeleton: dict,
    lbs_weights: dict,
    *,
    sigma: float = 0.10,
) -> None:
    """Re-bind arm-region gaussians to the upper-arm chain (down to the elbow).

    The bodies are baked into their resting pose, but the per-gaussian LBS weights are
    LHM's approximate skinning. When the limb is *rotated* (editor IK / animation) that
    approximate skinning stretches the lower arm into thin tendrils. Re-binding every
    arm-region gaussian (classified by its nearest joint) to the nearest of
    {collar, shoulder, elbow} with a Gaussian proximity falloff makes the lower arm
    swing as one rigid piece. Leg/torso gaussians (nearest joint outside the arm) are
    left untouched. Mutates ``lbs_weights`` in place (its ``indices`` / ``weights``).
    """
    names = skeleton["names"]
    rest = np.asarray(skeleton["restPositions"], dtype=np.float32)  # [J, 3]
    arm_class = np.array(
        [i for i, n in enumerate(names) if n in _ARM_JOINTS or any(t in n for t in _FINGER_TOKENS)],
        dtype=np.int64,
    )
    targets = np.array(sorted(i for i, n in enumerate(names) if n in _LOWER_ARM_TARGETS), dtype=np.int64)
    k = int(lbs_weights["k"])
    n = positions.shape[0]
    idx = np.asarray(lbs_weights["indices"], dtype=np.int64).reshape(n, k)
    wt = np.asarray(lbs_weights["weights"], dtype=np.float64).reshape(n, k)

    dist2 = ((positions[:, None, :] - rest[None, :, :]) ** 2).sum(-1)  # [N, J]
    is_arm = np.isin(dist2.argmin(axis=1), arm_class)  # nearest joint is arm-region

    take = min(k, targets.size)
    fall = np.exp(-((np.sqrt(dist2[:, targets]) / sigma) ** 2))  # [N, T] proximity falloff
    order = np.argsort(-fall, axis=1)[:, :take]  # closest targets per gaussian
    top = np.take_along_axis(fall, order, axis=1)
    underflow = top.sum(axis=1) <= 1e-12  # too far for any falloff -> nearest target only
    top[underflow] = 0.0
    top[underflow, 0] = 1.0
    top /= top.sum(axis=1, keepdims=True)
    glob = targets[order]  # [N, take] global joint indices

    idx[is_arm, :take] = glob[is_arm]
    wt[is_arm, :take] = top[is_arm]
    if take < k:
        wt[is_arm, take:] = 0.0
    lbs_weights["indices"] = idx.reshape(-1).tolist()
    lbs_weights["weights"] = np.round(wt, 6).reshape(-1).tolist()


@jaxtyped(typechecker=beartype)
def build_body_splattie(
    *,
    ply_path: Path,
    output_dir: Path,
    model_id: str,
    smplx_model: object,
    betas: Float[npt.NDArray[np.float32], "n_betas"],
    posed_joints: Float[npt.NDArray[np.float32], "55 3"] | None = None,
    source_image_path: Path | None = None,
) -> tuple[Path, int]:
    """Write a widget-loadable body `.splattie` from the PLY + the SMPL-X rig.

    ``posed_joints`` are the baked-pose skeleton joints from ``infer_mesh``'s
    ``<stem>_joints.json``; when given, the rig's rest pose matches the photographed
    body so the widget needs no rest rotation (and no arm stretch). Returns
    (bundle_path, num_gaussians).
    """
    skeleton, lbs_weights = extract_body_rig(smplx_model, np.asarray(betas, dtype=np.float32), posed_joints)
    # Re-bind the lower arms rigidly to the elbow so rotating the limb (editor IK /
    # animation) swings it as one solid piece instead of stretching the photo-avatar's
    # gaussians into tendrils. Rest position is unaffected — only the LBS weights change.
    reweight_lower_arm_rigid(parse_ply_xyz(ply_path.read_bytes()), skeleton, lbs_weights)

    skeleton_path = output_dir / BODY_RIG.skeleton_file
    weights_path = output_dir / BODY_RIG.weights_file
    skeleton_path.write_text(json.dumps(skeleton))
    weights_path.write_text(json.dumps(lbs_weights))

    num_gaussians = count_ply_vertices(ply_path)
    manifest = build_manifest(
        splat_filename=ply_path.name,
        num_gaussians=num_gaussians,
        widget_version=read_widget_version(),
        asset_type=AssetType.BODY,
        rig=BODY_RIG,
        generator_method="lhm",
        generator_method_version="500m-siggraph2025",
        generator_tool="splattie-backend",
        source_image_path=source_image_path,
    )
    bundle_path = output_dir / f"{model_id}.splattie"
    bundle_splattie(
        output_path=bundle_path,
        splat_path=ply_path,
        manifest=manifest,
        states=DEFAULT_STATES_BODY,
        rig_files={
            BODY_RIG.skeleton_file: skeleton_path,
            BODY_RIG.weights_file: weights_path,
        },
    )
    return bundle_path, num_gaussians
