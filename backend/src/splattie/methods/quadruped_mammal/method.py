"""TripoSplat/TRELLIS + SuperAnimal-anchored SMAL quadruped-mammal generation method.

Single image -> gaussian splat (TripoSplat by default, TRELLIS optional) -> 3D SuperAnimal
keypoints -> keypoint-anchored SMAL fit -> LBS bind -> gaze-enabled `.splattie`. Non-mammals are
rejected by the fit's detection gate (NotAQuadrupedMammalError) — there is no fallback rig.

The reconstruction backend defaults to TripoSplat (cleaner animal faces) and is overridable via the
``SPLATTIE_QUADRUPED_BACKEND`` env var (``trellis`` / ``triposplat``).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Bool, UInt8, jaxtyped
from klogr import get_logger

from splattie.methods.quadruped_mammal import runtime
from splattie.methods.quadruped_mammal.bind import bind_and_bundle
from splattie.methods.quadruped_mammal.fit import fit_smal
from splattie.methods.quadruped_mammal.gaussians import GaussianSplat
from splattie.methods.quadruped_mammal.keypoints import DEVICE, detect_keypoints_3d
from splattie.methods.quadruped_mammal.reconstruct import ReconstructBackend, reconstruct_gaussian_splat
from splattie.methods.quadruped_mammal.smal import SMAL
from splattie.methods.registry import registry
from splattie.types import AssetType, GenerationResult, MethodCapabilities, MethodInfo

logger = get_logger()

STORAGE_DIR = Path("data/generations")
METHOD_ID = "trellis-smal-quadruped"
# Reconstruction backend default; override per-deploy with SPLATTIE_QUADRUPED_BACKEND=trellis.
_BACKEND_ENV = "SPLATTIE_QUADRUPED_BACKEND"


def _resolve_backend() -> ReconstructBackend:
    raw = os.environ.get(_BACKEND_ENV, ReconstructBackend.triposplat)
    try:
        return ReconstructBackend(raw)
    except ValueError as exc:
        allowed = ", ".join(b.value for b in ReconstructBackend)
        msg = f"{_BACKEND_ENV}={raw!r} is not a valid reconstruction backend; allowed: {allowed}"
        raise ValueError(msg) from exc


RECONSTRUCT_BACKEND = _resolve_backend()
# Gate is detection-only: the fit raises NotAQuadrupedMammalError when SuperAnimal can't find
# a quadruped (non-mammals). Calibration showed |betas| does NOT separate out-of-family
# megafauna (deer 1.00 vs elephant 1.04), so there is no reliable shape gate — such inputs are
# in-scope-but-degraded. shape_norm is still recorded in diagnostics for observability.


@registry.register
class QuadrupedMammalMethod:
    """Single image to a rigged quadruped-mammal gaussian splat with a SMAL skeleton."""

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            id=METHOD_ID,
            name="TRELLIS + SMAL (SuperAnimal-anchored)",
            description="Single image -> rigged quadruped-mammal 3D Gaussian splat with a SMAL skeleton + head-follow",
            paper_url="https://smal.is.tue.mpg.de/",
            repo_url="https://github.com/affromero/TRELLIS",
            asset_type=AssetType.quadruped_mammal,
        )

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            supports_single_image=True,
            supports_expression=False,
            max_output_gaussians=500_000,
            typical_inference_seconds=180.0,
        )

    def load(self) -> None:
        # No fallback: missing SMAL / DeepLabCut / the selected reconstruction backend raises.
        runtime.require_quadruped_runtime(RECONSTRUCT_BACKEND)

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
        return self._generate(image, mask, model_id, output_dir)

    @jaxtyped(typechecker=beartype)
    def _generate(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
        model_id: str,
        output_dir: Path,
    ) -> GenerationResult:
        from PIL import Image as PILImage

        img_path = output_dir / f"{model_id}.png"
        PILImage.fromarray(_mask_to_white_background(image, mask)).save(str(img_path))
        pipeline_dir = output_dir / "quadruped_pipeline"
        pipeline_dir.mkdir(exist_ok=True)

        with runtime.inference_lock:
            gaussian_ply = reconstruct_gaussian_splat(
                image_path=img_path,
                output_dir=pipeline_dir / "reconstruct",
                model_id=model_id,
                backend=RECONSTRUCT_BACKEND,
            )
            splat = GaussianSplat(gaussian_ply)
            smal = SMAL(str(runtime.SMAL_PKL), device=DEVICE)
            keypoints = detect_keypoints_3d(smal, splat, pipeline_dir / "keypoints")
            fit = fit_smal(smal, splat, keypoints)  # raises NotAQuadrupedMammalError for non-mammals
            logger.info(f"Quadruped SMAL fit {model_id}: {fit.diagnostics.model_dump(by_alias=True)}")
            bundle_path, num_gaussians = bind_and_bundle(
                smal,
                splat,
                fit,
                ply_path=gaussian_ply,
                output_dir=output_dir,
                model_id=model_id,
                source_image_path=img_path,
            )

        logger.info(f"Quadruped .splattie: {bundle_path.name} ({num_gaussians} gaussians)")
        return GenerationResult(
            model_id=model_id,
            splattie_url=f"/storage/{model_id}/{bundle_path.name}",
            splattie_size_bytes=bundle_path.stat().st_size,
            num_gaussians=num_gaussians,
            method_id=METHOD_ID,
        )

    def unload(self) -> None:
        # TRELLIS + SuperAnimal run in subprocesses; process exit releases GPU memory.
        return None


@jaxtyped(typechecker=beartype)
def _mask_to_white_background(
    image: UInt8[npt.NDArray[np.uint8], "h w 3"],
    mask: Bool[npt.NDArray[np.bool_], "h w"],
) -> UInt8[npt.NDArray[np.uint8], "h w 3"]:
    """Composite the subject over a white background for reconstruction."""
    if mask.shape != image.shape[:2]:
        msg = f"mask shape {mask.shape} does not match image shape {image.shape[:2]}"
        raise ValueError(msg)
    out = image.copy()
    out[~mask] = 255
    return out
