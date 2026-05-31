"""TRELLIS + Puppeteer arbitrary object generation method."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Bool, UInt8, jaxtyped
from klogr import get_logger

from splattie.methods.object import runtime
from splattie.methods.object.bind import bind_rigged_splat
from splattie.methods.object.bundle import build_object_splattie
from splattie.methods.object.puppeteer import rig_object_mesh_with_puppeteer
from splattie.methods.object.reconstruct import reconstruct_object_with_trellis
from splattie.methods.registry import registry
from splattie.types import AssetType, GenerationResult, MethodCapabilities, MethodInfo

logger = get_logger()

STORAGE_DIR = Path("data/generations")
METHOD_ID = "trellis-puppeteer"


@registry.register
class ObjectRigMethod:
    """Single image to rigged arbitrary object gaussian splat."""

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            id=METHOD_ID,
            name="TRELLIS + Puppeteer",
            description="Single image -> rigged object 3D Gaussian splat with arbitrary LBS skeleton",
            paper_url="https://arxiv.org/abs/2412.01506",
            repo_url="https://github.com/affromero/TRELLIS",
            asset_type=AssetType.object,
        )

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            supports_single_image=True,
            supports_expression=False,
            max_output_gaussians=500_000,
            typical_inference_seconds=120.0,
        )

    def load(self) -> None:
        # No fallback: if vendor code or weights are missing, generation raises.
        runtime.require_object_runtime()

    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
    ) -> GenerationResult:
        self.load()
        model_id = uuid.uuid4().hex[:12]
        output_dir = (STORAGE_DIR / model_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._generate_object(image, mask, model_id, output_dir)

    @jaxtyped(typechecker=beartype)
    def _generate_object(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
        model_id: str,
        output_dir: Path,
    ) -> GenerationResult:
        from PIL import Image as PILImage

        img_path = output_dir / f"{model_id}.png"
        PILImage.fromarray(_mask_to_white_background(image, mask)).save(str(img_path))
        pipeline_dir = output_dir / "object_pipeline"
        pipeline_dir.mkdir(exist_ok=True)

        with runtime.inference_lock:
            reconstruction = reconstruct_object_with_trellis(
                image_path=img_path,
                output_dir=pipeline_dir / "trellis",
                model_id=model_id,
            )
            puppeteer = rig_object_mesh_with_puppeteer(
                mesh_obj=reconstruction.mesh_obj,
                output_dir=pipeline_dir / "puppeteer",
                model_id=model_id,
            )
            binding = bind_rigged_splat(
                gaussian_ply=reconstruction.gaussian_ply,
                mesh_obj=puppeteer.input_mesh,
                rig_skin=puppeteer.skin_txt,
                output_dir=pipeline_dir / "binding",
                model_id=model_id,
            )
            bundle_path, num_gaussians = build_object_splattie(
                ply_path=reconstruction.gaussian_ply,
                output_dir=output_dir,
                model_id=model_id,
                skeleton=binding.skeleton,
                lbs_weights=binding.lbs_weights,
                source_image_path=img_path,
            )

        logger.info(f"Object .splattie: {bundle_path.name} ({num_gaussians} gaussians)")
        splattie_url = f"/storage/{model_id}/{bundle_path.name}"
        return GenerationResult(
            model_id=model_id,
            splattie_url=splattie_url,
            splattie_size_bytes=bundle_path.stat().st_size,
            num_gaussians=num_gaussians,
            method_id=METHOD_ID,
        )

    def unload(self) -> None:
        # TRELLIS and Puppeteer run in subprocesses, so process exit releases GPU memory.
        return None


@jaxtyped(typechecker=beartype)
def _mask_to_white_background(
    image: UInt8[npt.NDArray[np.uint8], "h w 3"],
    mask: Bool[npt.NDArray[np.bool_], "h w"],
) -> UInt8[npt.NDArray[np.uint8], "h w 3"]:
    """Apply the segmentation mask as a white background for object reconstruction."""
    if mask.shape != image.shape[:2]:
        msg = f"mask shape {mask.shape} does not match image shape {image.shape[:2]}"
        raise ValueError(msg)
    out = image.copy()
    out[~mask] = 255
    return out
