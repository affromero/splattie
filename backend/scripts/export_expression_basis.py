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

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

VENDOR_LAM = Path(__file__).resolve().parents[1] / "vendor" / "LAM"


def patch_torch_load() -> None:
    _original = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original(*args, **kwargs)

    torch.load = _patched


def load_flame_head(device: str = "cuda") -> object:
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
        expr_params=cfg.model.get("expr_param_dim", 10),
        subdivide_num=1,
    ).to(device)
    flame.eval()
    return flame


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
    print(f"Vertices: {num_verts}, Expressions: {num_expr}")
    print(f"Basis shape: {expr_basis.shape}")

    # Transpose to (num_verts, num_expr, 3) for easier web consumption
    basis_np = expr_basis.permute(0, 2, 1).contiguous().cpu().numpy().astype(np.float32)

    # Show magnitude of each expression for debugging
    for i in range(num_expr):
        mag = np.linalg.norm(basis_np[:, i, :], axis=1).max()
        print(f"  expr_{i:02d}: max displacement = {mag:.6f}")

    # Write binary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"EXPR")
        f.write(struct.pack("<II", num_verts, num_expr))
        f.write(basis_np.tobytes())

    file_size = output_path.stat().st_size
    print(f"Binary: {output_path} ({file_size / 1024:.0f} KB)")

    # Write sidecar JSON
    json_path = output_path.with_suffix(".json")
    meta = {
        "num_vertices": int(num_verts),
        "num_expressions": int(num_expr),
        "bytes": int(file_size),
        "format": "float32_le, shape (num_vertices, num_expressions, 3)",
        "labels": [f"expr_{i}" for i in range(num_expr)],
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FLAME expression basis")
    parser.add_argument(
        "--shape-param",
        type=str,
        default=None,
        help="Path to shape_param.pt (person-specific). Uses mean shape if omitted.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output .bin path")
    parser.add_argument(
        "--num-expressions", type=int, default=20, help="Number of expression coefficients to export (default: 20)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    shape_param = None
    if args.shape_param:
        shape_param = torch.load(args.shape_param, map_location="cpu")
        print(f"Loaded shape param: {shape_param.shape}")

    print("Loading FLAME model...")
    flame = load_flame_head(device=args.device)
    print(f"FLAME loaded: {flame.n_shape_params} shape, {flame.n_expr_params} expr params")

    export_basis(flame, shape_param, args.num_expressions, Path(args.output))
    print("Done.")


if __name__ == "__main__":
    main()
