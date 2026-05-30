#!/usr/bin/env python3
"""Export FLAME expression blendshape basis for the splattie-widget.

Extracts the per-vertex expression displacement basis from LAM's upsampled
FLAME model (20K vertices × 3 × N_expr). The output is a compact binary
that the web client loads to compute per-frame vertex displacements:

    position[i] += sum(weight_j * basis[i][j]) for each expression j

Usage:
    # From the backend directory, with uv:
    uv run python scripts/export_expression_basis.py \
        --shape-param data/generations/<id>/shape_param.pt \
        --output ../packages/splattie-widget/public/expression_basis.bin \
        --num-expressions 20

    # Without a specific shape (uses zero/mean shape):
    uv run python scripts/export_expression_basis.py \
        --output ../packages/splattie-widget/public/expression_basis.bin

Binary format:
    bytes 0-3:   magic "EXPR" (4 bytes)
    bytes 4-7:   num_vertices (uint32 LE)
    bytes 8-11:  num_expressions (uint32 LE)
    bytes 12+:   float32 LE array, shape (num_vertices, num_expressions, 3)
                 row-major: [v0_e0_x, v0_e0_y, v0_e0_z, v0_e1_x, ...]

Sidecar JSON (same path, .json extension):
    { "num_vertices": N, "num_expressions": M, "bytes": B,
      "labels": ["expr_0", "expr_1", ...] }
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import tyro
from klogr import get_logger

logger = get_logger()

VENDOR_LAM = Path(__file__).resolve().parents[1] / "vendor" / "LAM"


def patch_torch_load() -> None:
    _original = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original(*args, **kwargs)

    torch.load = _patched


def load_flame_head(device: str = "cuda", num_expressions: int = 50) -> object:
    lam_path = str(VENDOR_LAM)
    if lam_path not in sys.path:
        sys.path.insert(0, lam_path)

    patch_torch_load()

    from lam.models.rendering.flame_model.flame import FlameHeadSubdivided
    from omegaconf import OmegaConf

    config_path = VENDOR_LAM / "configs" / "inference" / "lam-20k-8gpu.yaml"
    cfg = OmegaConf.load(str(config_path))

    human_model_path = str(VENDOR_LAM / "model_zoo" / "human_parametric_models")
    flame = FlameHeadSubdivided(
        flame_model_path=f"{human_model_path}/flame_assets/flame/flame2023.pkl",
        flame_lmk_embedding_path=f"{human_model_path}/flame_assets/flame/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path=f"{human_model_path}/flame_assets/flame/head_template_mesh.obj",
        flame_parts_path=f"{human_model_path}/flame_assets/flame/FLAME_masks.pkl",
        shape_params=cfg.model.get("shape_param_dim", 10),
        expr_params=num_expressions,
        subdivide_num=1,
    ).to(device)
    flame.eval()
    return flame


REGION_NAMES = ["jaw", "lips", "brow", "nose", "cheek", "eyes", "forehead", "neck"]


def _label_expressions(
    flame: object,
    basis_np: np.ndarray,
    num_expr: int,
    num_verts: int,
) -> list[str]:
    """Assign semantic names to PCA expression components using FLAME vertex masks."""
    import pickle

    masks_path = Path(flame.flame_model_dir) / "FLAME_masks.pkl"
    if not masks_path.exists():
        return [f"expr_{i}" for i in range(num_expr)]

    with open(masks_path, "rb") as f:
        masks = pickle.load(f, encoding="latin1")

    region_map: dict[str, set[int]] = {}
    for name in REGION_NAMES:
        key = name if name in masks else next((k for k in masks if name in k.lower()), None)
        if key and hasattr(masks[key], "__iter__"):
            region_map[name] = set(int(v) for v in masks[key] if int(v) < num_verts)

    if not region_map:
        return [f"expr_{i}" for i in range(num_expr)]

    labels: list[str] = []
    used_names: dict[str, int] = {}
    for i in range(num_expr):
        disp = basis_np[:, i, :]
        per_vert_mag = np.linalg.norm(disp, axis=1)

        best_region = "face"
        best_score = 0.0
        for rname, vids in region_map.items():
            vids_arr = np.array(list(vids))
            vids_arr = vids_arr[vids_arr < num_verts]
            if len(vids_arr) == 0:
                continue
            score = float(per_vert_mag[vids_arr].mean())
            if score > best_score:
                best_score = score
                best_region = rname

        # Direction hint from dominant axis of top-displaced vertices
        top_verts = np.argsort(per_vert_mag)[-100:]
        mean_disp = disp[top_verts].mean(axis=0)
        dominant_axis = int(np.argmax(np.abs(mean_disp)))
        direction = ""
        if dominant_axis == 0:
            direction = "L" if mean_disp[0] > 0 else "R"
        elif dominant_axis == 1:
            direction = "Up" if mean_disp[1] > 0 else "Down"
        elif dominant_axis == 2:
            direction = "Fwd" if mean_disp[2] > 0 else "Back"

        base = f"{best_region}{direction}"
        count = used_names.get(base, 0)
        used_names[base] = count + 1
        label = base if count == 0 else f"{base}{count + 1}"
        labels.append(label)

    return labels


def export_basis(
    flame: object,
    shape_param: torch.Tensor | None,
    num_expressions: int,
    output_path: Path,
) -> None:
    device = flame.v_template.device

    n_shape = flame.n_shape_params
    n_expr_available = flame.n_expr_params
    num_expr = min(num_expressions, n_expr_available)

    if shape_param is None:
        shape_param = torch.zeros(1, n_shape, device=device)
    else:
        shape_param = shape_param.to(device)
        if shape_param.dim() == 1:
            shape_param = shape_param.unsqueeze(0)

    # shapedirs_up: (num_verts_up, 3, n_shape + n_expr [+ teeth])
    # Expression basis starts at index n_shape
    expr_basis = flame.shapedirs_up[:, :, n_shape : n_shape + num_expr]
    # expr_basis shape: (num_verts, 3, num_expr)

    num_verts = expr_basis.shape[0]
    logger.info(f"Vertices: {num_verts}, Expressions: {num_expr}")
    logger.info(f"Basis shape: {expr_basis.shape}")

    # Transpose to (num_verts, num_expr, 3) for easier web consumption
    basis_np = expr_basis.permute(0, 2, 1).contiguous().cpu().numpy().astype(np.float32)

    # Analyze which facial region each expression affects most
    labels = _label_expressions(flame, basis_np, num_expr, num_verts)

    for i in range(num_expr):
        mag = np.linalg.norm(basis_np[:, i, :], axis=1).max()
        logger.info(f"  {i:02d} {labels[i]:20s} max_disp={mag:.6f}")

    # Write binary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"EXPR")
        f.write(struct.pack("<II", num_verts, num_expr))
        f.write(basis_np.tobytes())

    file_size = output_path.stat().st_size
    logger.info(f"Binary: {output_path} ({file_size / 1024:.0f} KB)")

    # Write sidecar JSON
    json_path = output_path.with_suffix(".json")
    meta = {
        "num_vertices": int(num_verts),
        "num_expressions": int(num_expr),
        "bytes": int(file_size),
        "format": "float32_le, shape (num_vertices, num_expressions, 3)",
        "labels": labels,
        "indices": {label: i for i, label in enumerate(labels)},
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata: {json_path}")


def main(
    output: Path,
    shape_param: Path | None = None,
    num_expressions: int = 20,
    device: str = "cuda",
) -> None:
    """Export the FLAME expression basis to a .bin file.

    Args:
        output: Output .bin path.
        shape_param: Path to shape_param.pt (person-specific). Uses the mean shape if omitted.
        num_expressions: Number of expression coefficients to export.
        device: Device to run FLAME on (cuda or cpu).

    """
    shape = None
    if shape_param:
        shape = torch.load(shape_param, map_location="cpu")
        logger.info(f"Loaded shape param: {shape.shape}")

    logger.info("Loading FLAME model...")
    flame = load_flame_head(device=device, num_expressions=num_expressions)
    logger.info(f"FLAME loaded: {flame.n_shape_params} shape, {flame.n_expr_params} expr params")

    export_basis(flame, shape, num_expressions, output)
    logger.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)
