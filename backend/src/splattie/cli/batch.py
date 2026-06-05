"""Batch `.splattie` generation commands."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from klogr import get_logger
from PIL import Image

from splattie.methods.base import AssetGenerationMethod
from splattie.methods.lam.method import LAMMethod
from splattie.methods.lhm.method import LHMMethod
from splattie.methods.object.method import ObjectRigMethod
from splattie.methods.quadruped_mammal.method import QuadrupedMammalMethod
from splattie.types import AssetType

logger = get_logger()

_STORAGE = Path("data/generations")
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png"})


def _method_for(asset_type: AssetType) -> AssetGenerationMethod:
    if asset_type is AssetType.body:
        return LHMMethod()
    if asset_type is AssetType.object:
        return ObjectRigMethod()
    if asset_type is AssetType.head:
        return LAMMethod()
    if asset_type is AssetType.quadruped_mammal:
        return QuadrupedMammalMethod()
    msg = f"no batch method wired for asset type {asset_type.value!r}"
    raise NotImplementedError(msg)


def _generate_one(method: AssetGenerationMethod, image_path: Path, output_path: Path) -> None:
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.ones(image.shape[:2], dtype=np.bool_)
    result = method.generate(image, mask)
    produced = _STORAGE / result.model_id / f"{result.model_id}.splattie"
    if not produced.exists():
        msg = f"generation returned {result.model_id}, but {produced} is missing"
        raise FileNotFoundError(msg)
    shutil.copy(produced, output_path)


def generate_splattie_batch(
    images_dir: Path,
    output_dir: Path,
    asset_type: AssetType = AssetType.head,
) -> None:
    """Batch-generate `.splattie` files from a directory of source images.

    Args:
        images_dir: Directory of input images (.jpg/.jpeg/.png).
        output_dir: Directory to write `<image-stem>.splattie` bundles into.
        asset_type: `head` uses LAM, `body` uses LHM, `object` uses TRELLIS + Puppeteer.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not image_files:
        logger.info(f"No images found in {images_dir}")
        return

    method = _method_for(asset_type)
    logger.info(f"{asset_type.value}: loading method for {len(image_files)} images")
    method.load()
    try:
        for index, image_path in enumerate(image_files, 1):
            output_path = output_dir / f"{image_path.stem}.splattie"
            logger.info(f"[{index}/{len(image_files)}] {image_path.name}")
            try:
                _generate_one(method, image_path, output_path)
                logger.info(f"  OK -> {output_path} ({output_path.stat().st_size // 1024} KB)")
            except Exception as exc:
                logger.error(f"  FAILED {image_path.stem}: {exc}")  # noqa: TRY400 - keep batch going
    finally:
        method.unload()

    logger.info(f"Done: {len(list(output_dir.glob('*.splattie')))} .splattie files in {output_dir}")
