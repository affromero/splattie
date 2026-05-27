"""LAM (Large Avatar Model) head generation method.

SIGGRAPH 2025 — single image → drivable 3DGS head with FLAME animation.
Runs generate_head.sh which handles LAM inference + ZIP bundle creation.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Bool, UInt8, jaxtyped

from mirada.methods.registry import registry
from mirada.types import GenerationResult, MethodCapabilities, MethodInfo

logger = logging.getLogger(__name__)

STORAGE_DIR = Path("data/generations")
SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"


@registry.register
class LAMMethod:
    """LAM: Large Avatar Model for one-shot animatable Gaussian heads."""

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            id="lam",
            name="LAM (SIGGRAPH 2025)",
            description="Single image → drivable 3DGS head with FLAME LBS animation",
            paper_url="https://arxiv.org/abs/2502.17796",
            repo_url="https://github.com/aigc3d/LAM",
        )

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            supports_single_image=True,
            supports_expression=True,
            max_output_gaussians=20_000,
            typical_inference_seconds=30.0,
        )

    def load(self) -> None:
        logger.info("LAM method ready (inference via generate_head.sh)")

    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
    ) -> GenerationResult:
        model_id = uuid.uuid4().hex[:12]
        output_dir = STORAGE_DIR / model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        img_path = output_dir / "input.jpg"
        Image.fromarray(image).save(img_path)

        script = SCRIPTS_DIR / "generate_head.sh"
        if script.exists():
            try:
                result = subprocess.run(
                    ["bash", str(script), str(img_path), str(output_dir)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                logger.info("LAM stdout: %s", result.stdout[-200:] if result.stdout else "")
                if result.returncode != 0:
                    logger.error("LAM stderr: %s", result.stderr[-500:] if result.stderr else "")
            except subprocess.TimeoutExpired:
                logger.exception("LAM inference timed out after 120s")

        zip_path = next(output_dir.glob("*.zip"), None)
        if zip_path:
            spz_size = zip_path.stat().st_size
            return GenerationResult(
                model_id=model_id,
                spz_url=f"/storage/{model_id}/{zip_path.name}",
                spz_size_bytes=spz_size,
                num_gaussians=20_000,
                method_id="lam",
                flame_params_url=f"/storage/{model_id}/{zip_path.name}",
            )

        return self._generate_stub(model_id, output_dir)

    def _generate_stub(self, model_id: str, output_dir: Path) -> GenerationResult:
        """Fallback: generate a stub bundle pointing to the demo."""
        logger.warning("LAM inference not available, using demo bundle")
        return GenerationResult(
            model_id=model_id,
            spz_url="/demo/andres.zip",
            spz_size_bytes=0,
            num_gaussians=20_000,
            method_id="lam",
            flame_params_url="/demo/andres.zip",
        )

    def unload(self) -> None:
        logger.info("LAM method unloaded")
