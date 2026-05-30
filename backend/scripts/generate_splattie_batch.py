#!/usr/bin/env python3
"""Batch-generate `.splattie` demos from portrait/full-body images on GPU.

One generator, two asset types (``--asset-type``):

- ``head`` (LAM): FLAME tracking + 3DGS head reconstruction via LAM's CLI, then
  bundle the PLY with the shared FLAME rig. Run from ``backend/vendor/LAM/``.
- ``body`` (LHM): SMPL-X body reconstruction via the in-process ``LHMMethod`` (Multi-HMR
  betas → infer_mesh → skeleton + per-gaussian LBS weights bundle). Run from ``backend/``.

Both emit ``<name>.splattie`` into ``--output-dir``.

Usage:
    # heads (from backend/vendor/LAM/)
    python ../../scripts/generate_splattie_batch.py --asset-type head \
        --images-dir /tmp/pexels_heads --output-dir /tmp/out
    # bodies (from backend/)
    python scripts/generate_splattie_batch.py --asset-type body \
        --images-dir /tmp/pexels_bodies --output-dir /tmp/out
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import tyro
from klogr import get_logger

from splattie.methods.bundle_common import (
    DEFAULT_STATES_HEAD,
    HEAD_RIG,
    build_manifest,
    bundle_splattie,
    count_ply_vertices,
    read_widget_version,
)
from splattie.types import AssetType

logger = get_logger()


def run_lam_inference(image_path: Path, name: str) -> Path:
    """Run LAM inference via CLI. Returns path to the generated offset PLY."""
    Path("tracking_output/preprocess").mkdir(parents=True, exist_ok=True)
    Path(f"tracking_output/export/{name}").mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "lam.launch",
        "infer.lam",
        "--config",
        "configs/inference/lam-20k-8gpu.yaml",
        "model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
        f"image_input={image_path}",
        "save_ply=true",
        "save_img=true",
        "export_video=true",
        "export_mesh=false",
        "vis_motion=false",
        "render_fps=30",
        f"motion_seqs_dir=tracking_output/export/{name}/",
        "motion_img_dir=null",
        "rank=0",
        "nodes=0",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONPATH"] = "." + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300, check=False)
    if result.returncode != 0:
        logger.info(f"  STDERR: {result.stderr[-500:]}")
        msg = f"LAM inference failed for {name}"
        raise RuntimeError(msg)

    return Path(f"exps/cano_gs/{name}_gs_offset.ply")


def generate_head(image_path: Path, name: str, output_dir: Path, widget_version: str) -> Path:
    """LAM head: CLI inference → bundle PLY + shared FLAME rig → `<name>.splattie`."""
    run_lam_inference(image_path, name)
    abs_ply = Path(f"exps/cano_gs/{name}.ply")
    if not abs_ply.exists():
        msg = f"Absolute PLY not found: {abs_ply}"
        raise FileNotFoundError(msg)

    splattie_path = output_dir / f"{name}.splattie"
    manifest = build_manifest(
        splat_filename=f"{name}.ply",
        num_gaussians=count_ply_vertices(abs_ply),
        widget_version=widget_version,
        asset_type=AssetType.HEAD,
        rig=HEAD_RIG,
        generator_method="lam",
        generator_tool="generate_splattie_batch",
        source_image_path=image_path,
    )
    bundle_splattie(output_path=splattie_path, splat_path=abs_ply, manifest=manifest, states=DEFAULT_STATES_HEAD)
    return splattie_path


def generate_body(method: object, image_path: Path, name: str, output_dir: Path) -> Path:
    """LHM body: in-process generate → copy the produced `.splattie` to `<name>.splattie`."""
    import numpy as np
    from PIL import Image

    img = np.array(Image.open(image_path).convert("RGB"))
    mask = np.ones(img.shape[:2], dtype=np.bool_)
    result = method.generate(img, mask)  # type: ignore[attr-defined]

    produced = Path("data/generations") / result.model_id / f"{result.model_id}.splattie"
    splattie_path = output_dir / f"{name}.splattie"
    shutil.copy(produced, splattie_path)
    return splattie_path


def main(
    images_dir: Path,
    output_dir: Path,
    asset_type: Literal["head", "body"] = "head",
) -> None:
    """Batch-generate `.splattie` demos (head or body) from a directory of images.

    Args:
        images_dir: Directory of input images (.jpg/.jpeg/.png).
        output_dir: Directory to write `.splattie` bundles into.
        asset_type: ``head`` (LAM/FLAME) or ``body`` (LHM/SMPL-X).

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    widget_version = read_widget_version()
    logger.info(f"Asset type: {asset_type} | widget version: {widget_version}\n")

    image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not image_files:
        logger.info(f"No images found in {images_dir}")
        return
    logger.info(f"Found {len(image_files)} images\n")

    body_method = None
    if asset_type == "body":
        from splattie.methods.lhm.method import LHMMethod

        body_method = LHMMethod()
        body_method.load()

    for i, img_path in enumerate(image_files, 1):
        name = img_path.stem
        logger.info(f"[{i}/{len(image_files)}] {img_path.name}")
        try:
            if asset_type == "body":
                out = generate_body(body_method, img_path, name, output_dir)
            else:
                out = generate_head(img_path, name, output_dir, widget_version)
            logger.info(f"  OK -> {out} ({out.stat().st_size // 1024} KB)\n")
        except Exception as e:
            logger.info(f"  FAILED: {e}\n")
            continue

    splattie_files = list(output_dir.glob("*.splattie"))
    logger.info(f"Done! {len(splattie_files)} .splattie files in {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
