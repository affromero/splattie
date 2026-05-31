"""FLAME expression-basis export commands."""

from __future__ import annotations

import importlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from klogr import get_logger

logger = get_logger()

VENDOR_LAM = Path(__file__).resolve().parents[3] / "vendor" / "LAM"

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
REGION_NAMES = ["jaw", "lips", "brow", "nose", "cheek", "eyes", "forehead", "neck"]


def patch_torch_load() -> None:
    """Patch torch.load to default weights_only=False so trusted FLAME pickles load."""
    original = torch.load

    def patched(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched


def _ensure_lam_importable() -> None:
    lam_path = str(VENDOR_LAM)
    if lam_path not in sys.path:
        sys.path.insert(0, lam_path)
    patch_torch_load()
    from splattie.methods.lam.method import _patch_chumpy_compat

    _patch_chumpy_compat()


def _human_model_path() -> str:
    return str(VENDOR_LAM / "model_zoo" / "human_parametric_models")


def load_flame_head(device: str = "cuda", num_expressions: int = 50) -> object:
    """Load the FLAME head model from the vendored LAM submodule."""
    _ensure_lam_importable()
    flame_module = importlib.import_module("lam.models.rendering.flame_model.flame")
    omega_conf = importlib.import_module("omegaconf").OmegaConf
    config_path = VENDOR_LAM / "configs" / "inference" / "lam-20k-8gpu.yaml"
    cfg = omega_conf.load(str(config_path))
    human_model_path = _human_model_path()
    flame = flame_module.FlameHeadSubdivided(
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


def load_flame_arkit(device: str, arkit_bs_path: Path) -> object:
    """Load the FLAME subdivided model whose expression channels are ARKit blendshapes."""
    _ensure_lam_importable()
    from lam.models.rendering.flame_model.flame_arkit import FlameHeadSubdivided
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(VENDOR_LAM / "configs" / "inference" / "lam-20k-8gpu.yaml"))
    human_model_path = _human_model_path()
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


def _load_region_map(flame: object, num_verts: int) -> dict[str, set[int]]:
    masks_path = Path(flame.flame_model_dir) / "FLAME_masks.pkl"
    if not masks_path.exists():
        return {}
    masks = torch.load(masks_path, map_location="cpu", encoding="latin1")
    region_map: dict[str, set[int]] = {}
    for name in REGION_NAMES:
        key = name if name in masks else next((k for k in masks if name in k.lower()), None)
        if key and hasattr(masks[key], "__iter__"):
            region_map[name] = {int(v) for v in masks[key] if int(v) < num_verts}
    return region_map


def _dominant_region(per_vert_mag: npt.NDArray[np.float32], region_map: dict[str, set[int]]) -> str:
    best_region = "face"
    best_score = 0.0
    for rname, vids in region_map.items():
        vids_arr = np.array(list(vids))
        if vids_arr.size == 0:
            continue
        score = float(per_vert_mag[vids_arr].mean())
        if score > best_score:
            best_score = score
            best_region = rname
    return best_region


def _direction_hint(disp: npt.NDArray[np.float32], per_vert_mag: npt.NDArray[np.float32]) -> str:
    top_verts = np.argsort(per_vert_mag)[-100:]
    mean_disp = disp[top_verts].mean(axis=0)
    dominant_axis = int(np.argmax(np.abs(mean_disp)))
    if dominant_axis == 0:
        return "L" if mean_disp[0] > 0 else "R"
    if dominant_axis == 1:
        return "Up" if mean_disp[1] > 0 else "Down"
    return "Fwd" if mean_disp[2] > 0 else "Back"


def _label_expressions(
    flame: object,
    basis_np: npt.NDArray[np.float32],
    num_expr: int,
    num_verts: int,
) -> list[str]:
    region_map = _load_region_map(flame, num_verts)
    if not region_map:
        return [f"expr_{i}" for i in range(num_expr)]

    labels: list[str] = []
    used_names: dict[str, int] = {}
    for i in range(num_expr):
        disp = basis_np[:, i, :]
        per_vert_mag = np.linalg.norm(disp, axis=1)
        base = f"{_dominant_region(per_vert_mag, region_map)}{_direction_hint(disp, per_vert_mag)}"
        count = used_names.get(base, 0)
        used_names[base] = count + 1
        labels.append(base if count == 0 else f"{base}{count + 1}")
    return labels


def _write_basis(output_path: Path, basis_np: npt.NDArray[np.float32], labels: list[str], system: str) -> None:
    num_verts, num_expr, _ = basis_np.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(b"EXPR")
        f.write(struct.pack("<II", num_verts, num_expr))
        f.write(basis_np.tobytes())

    size = output_path.stat().st_size
    meta = {
        "num_vertices": int(num_verts),
        "num_expressions": int(num_expr),
        "bytes": int(size),
        "format": "float32_le, shape (num_vertices, num_expressions, 3)",
        "system": system,
        "labels": labels,
        "indices": {label: i for i, label in enumerate(labels)},
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Binary: {output_path} ({size / 1024:.0f} KB)")
    logger.info(f"Metadata: {output_path.with_suffix('.json')}")


def export_expression_basis(
    output: Path,
    shape_param: Path | None = None,
    num_expressions: int = 20,
    device: str = "cuda",
) -> None:
    """Export FLAME PCA expression basis to a widget-loadable .bin file.

    Args:
        output: Output .bin path.
        shape_param: Path to shape_param.pt. Uses the mean shape if omitted.
        num_expressions: Number of expression coefficients to export.
        device: Device to run FLAME on (cuda or cpu).

    """
    shape = None
    if shape_param:
        shape = torch.load(shape_param, map_location="cpu")
        logger.info(f"Loaded shape param: {shape.shape}")

    flame = load_flame_head(device=device, num_expressions=num_expressions)
    device_obj = flame.v_template.device
    n_shape = flame.n_shape_params
    num_expr = min(num_expressions, flame.n_expr_params)

    if shape is None:
        shape = torch.zeros(1, n_shape, device=device_obj)
    else:
        shape = shape.to(device_obj)
        if shape.dim() == 1:
            shape = shape.unsqueeze(0)

    expr_basis = flame.shapedirs_up[:, :, n_shape : n_shape + num_expr]
    num_verts = int(expr_basis.shape[0])
    basis_np = expr_basis.permute(0, 2, 1).contiguous().cpu().numpy().astype(np.float32)
    labels = _label_expressions(flame, basis_np, num_expr, num_verts)

    for i, label in enumerate(labels):
        mag = np.linalg.norm(basis_np[:, i, :], axis=1).max()
        logger.info(f"  {i:02d} {label:20s} max_disp={mag:.6f}")

    _write_basis(output, basis_np, labels, "flame-pca")


def _verify_arkit_order(basis_np: npt.NDArray[np.float32], masks_path: Path) -> None:
    if not masks_path.exists():
        return
    masks = torch.load(masks_path, map_location="cpu", encoding="latin1")
    region_map: dict[str, npt.NDArray[np.int64]] = {}
    num_verts = basis_np.shape[0]
    for name in REGION_NAMES:
        key = name if name in masks else next((k for k in masks if name in k.lower()), None)
        if key and hasattr(masks[key], "__iter__"):
            ids = np.array([int(v) for v in masks[key] if int(v) < num_verts])
            if len(ids):
                region_map[name] = ids

    logger.info(f"{'idx':>3} {'name':22} {'region':9} {'axis':5} {'max_disp':>9}")
    for i, name in enumerate(ARKIT_52_NAMES):
        disp = basis_np[:, i, :]
        mag = np.linalg.norm(disp, axis=1)
        region = "?"
        best = 0.0
        for rname, ids in region_map.items():
            score = float(mag[ids].mean())
            if score > best:
                best, region = score, rname
        top = np.argsort(mag)[-100:]
        md = disp[top].mean(axis=0)
        axis = ["L/R", "Up/Dn", "F/B"][int(np.argmax(np.abs(md)))]
        logger.info(f"{i:>3} {name:22} {region:9} {axis:5} {mag.max():9.5f}")


def export_arkit_basis(
    output: Path,
    flame_arkit_bs: Path = VENDOR_LAM / "model_zoo/human_parametric_models/flame_assets/flame/flame_arkit_bs.npy",
    device: str = "cuda",
) -> None:
    """Export LAM's 52 named ARKit blendshape basis to a widget-loadable .bin file.

    Args:
        output: Output .bin path.
        flame_arkit_bs: Path to flame_arkit_bs.npy.
        device: Device to run FLAME on (cuda or cpu).

    """
    flame = load_flame_arkit(device, flame_arkit_bs)
    n_shape = flame.n_shape_params
    arkit = flame.shapedirs_up[:, :, n_shape : n_shape + 52]
    basis_np = arkit.permute(0, 2, 1).contiguous().cpu().numpy().astype(np.float32)
    logger.info(f"Vertices: {basis_np.shape[0]}, Blendshapes: 52, basis {basis_np.shape}")
    _write_basis(output, basis_np, ARKIT_52_NAMES, "arkit")

    masks_path = VENDOR_LAM / "model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"
    try:
        _verify_arkit_order(basis_np, masks_path)
    except Exception as exc:
        logger.info(f"order verification skipped: {exc}")
