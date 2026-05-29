#!/usr/bin/env python3
"""Export the 52 ARKit blendshapes as a per-vertex basis for the splattie-widget.

Unlike export_expression_basis.py (which exports FLAME's PCA expression directions —
not semantic), this exports LAM's true ARKit blendshape basis (`flame_arkit_bs.npy`,
52 named blendshapes). The widget then drives real, named expressions: setting the
`mouthSmileLeft` weight produces an actual smile.

The base-res (5023-vertex) blendshapes are upsampled to the 20K gaussian topology by
the same teeth-aware subdivider LAM uses for its avatars, so vertex order matches.

Usage (on the GPU box):
    uv run python scripts/export_arkit_basis.py \
        --output ../apps/web/public/demos/expression_basis.bin

Binary format (identical to export_expression_basis.py, so the widget loads it
unchanged): "EXPR" magic, num_vertices u32, num_expressions u32 (=52), then
float32 LE (num_vertices, num_expressions, 3). Sidecar .json carries the ARKit
labels + name→index map.
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

# ARKit ARFaceAnchor.BlendShapeLocation canonical order (52). The .npy column order
# is undocumented in LAM; this is the de-facto standard. The export prints a per-
# channel region/axis analysis so the order can be verified (e.g. index 24 jawOpen
# must light up the jaw moving down).
ARKIT_52_NAMES = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]
REGION_NAMES = ["jaw", "lips", "nose", "eyes", "forehead", "neck", "cheek"]


def patch_torch_load() -> None:
    _original = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original(*args, **kwargs)

    torch.load = _patched


def load_flame_arkit(device: str, arkit_bs_path: Path) -> object:
    """FLAME subdivided model whose expression channels ARE the 52 ARKit blendshapes."""
    lam_path = str(VENDOR_LAM)
    if lam_path not in sys.path:
        sys.path.insert(0, lam_path)
    patch_torch_load()
    # FLAME's flame2023.pkl unpickles chumpy objects — restore the py3.11/numpy shims.
    from splattie.methods.lam.method import _patch_chumpy_compat

    _patch_chumpy_compat()

    from lam.models.rendering.flame_model.flame_arkit import FlameHeadSubdivided
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(VENDOR_LAM / "configs" / "inference" / "lam-20k-8gpu.yaml"))
    human_model_path = str(VENDOR_LAM / "model_zoo" / "human_parametric_models")
    flame = FlameHeadSubdivided(
        flame_model_path=f"{human_model_path}/flame_assets/flame/flame2023.pkl",
        flame_lmk_embedding_path=f"{human_model_path}/flame_assets/flame/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path=f"{human_model_path}/flame_assets/flame/head_template_mesh.obj",
        flame_parts_path=f"{human_model_path}/flame_assets/flame/FLAME_masks.pkl",
        shape_params=cfg.model.get("shape_param_dim", 10),
        expr_params=52,
        subdivide_num=1,
        flame_arkit_bs_path=str(arkit_bs_path),
    ).to(device)
    flame.eval()
    return flame


def verify_order(basis_np: np.ndarray, masks_path: Path) -> None:
    """Print each channel's dominant region + motion axis to sanity-check the names."""
    import pickle

    region_map: dict[str, np.ndarray] = {}
    num_verts = basis_np.shape[0]
    if masks_path.exists():
        with open(masks_path, "rb") as f:
            masks = pickle.load(f, encoding="latin1")
        for name in REGION_NAMES:
            key = name if name in masks else next((k for k in masks if name in k.lower()), None)
            if key and hasattr(masks[key], "__iter__"):
                ids = np.array([int(v) for v in masks[key] if int(v) < num_verts])
                if len(ids):
                    region_map[name] = ids

    print(f"{'idx':>3} {'name':22} {'region':9} {'axis':5} {'max_disp':>9}")
    for i, name in enumerate(ARKIT_52_NAMES):
        disp = basis_np[:, i, :]
        mag = np.linalg.norm(disp, axis=1)
        region = "?"
        best = 0.0
        for rname, ids in region_map.items():
            s = float(mag[ids].mean())
            if s > best:
                best, region = s, rname
        top = np.argsort(mag)[-100:]
        md = disp[top].mean(axis=0)
        axis = ["L/R", "Up/Dn", "F/B"][int(np.argmax(np.abs(md)))]
        print(f"{i:>3} {name:22} {region:9} {axis:5} {mag.max():9.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the 52 ARKit blendshape basis")
    parser.add_argument("--output", type=str, required=True, help="Output .bin path")
    parser.add_argument(
        "--flame-arkit-bs",
        type=str,
        default=str(VENDOR_LAM / "model_zoo/human_parametric_models/flame_assets/flame/flame_arkit_bs.npy"),
        help="Path to flame_arkit_bs.npy (52, V, 3)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading FLAME (ARKit blendshapes)...")
    flame = load_flame_arkit(args.device, Path(args.flame_arkit_bs))
    n_shape = flame.n_shape_params

    # shapedirs_up: (V_up, 3, n_shape + 52). The expression slice is the 52 ARKit
    # blendshapes upsampled to the gaussian topology.
    arkit = flame.shapedirs_up[:, :, n_shape : n_shape + 52]  # (V_up, 3, 52)
    num_verts = arkit.shape[0]
    basis_np = arkit.permute(0, 2, 1).contiguous().cpu().numpy().astype(np.float32)  # (V_up, 52, 3)
    print(f"Vertices: {num_verts}, Blendshapes: 52, basis {basis_np.shape}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"EXPR")
        f.write(struct.pack("<II", num_verts, 52))
        f.write(basis_np.tobytes())
    size = out.stat().st_size
    print(f"Binary: {out} ({size / 1024:.0f} KB)")

    meta = {
        "num_vertices": int(num_verts),
        "num_expressions": 52,
        "bytes": int(size),
        "format": "float32_le, shape (num_vertices, num_expressions, 3)",
        "system": "arkit",
        "labels": ARKIT_52_NAMES,
        "indices": {name: i for i, name in enumerate(ARKIT_52_NAMES)},
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {out.with_suffix('.json')}")

    # Best-effort: confirm the assumed ARKit name order against per-channel motion.
    masks_path = VENDOR_LAM / "model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"
    try:
        verify_order(basis_np, masks_path)
    except Exception as exc:
        print(f"(order verification skipped: {exc})")
    print("Done.")


if __name__ == "__main__":
    main()
