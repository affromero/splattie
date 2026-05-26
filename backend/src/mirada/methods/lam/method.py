"""LAM (Large Avatar Model) head generation method.

SIGGRAPH 2025 — single image → drivable 3DGS head with FLAME animation.
"""

from __future__ import annotations

import json
import logging
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
            max_output_gaussians=100_000,
            typical_inference_seconds=1.4,
        )

    def load(self) -> None:
        logger.info("LAM model loading (stub — GPU required for real inference)")

    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
    ) -> GenerationResult:
        model_id = uuid.uuid4().hex[:12]
        output_dir = STORAGE_DIR / model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        num_gaussians = 50_000
        spz_data = self._generate_stub_spz(num_gaussians)
        spz_path = output_dir / "head.spz"
        spz_path.write_bytes(spz_data)

        flame_params = self._generate_stub_flame_params(num_gaussians)
        flame_path = output_dir / "flame.json"
        flame_path.write_text(json.dumps(flame_params))

        return GenerationResult(
            model_id=model_id,
            spz_url=f"/storage/{model_id}/head.spz",
            spz_size_bytes=len(spz_data),
            num_gaussians=num_gaussians,
            method_id="lam",
            flame_params_url=f"/storage/{model_id}/flame.json",
        )

    def unload(self) -> None:
        logger.info("LAM model unloaded")

    def _generate_stub_spz(self, num_gaussians: int) -> bytes:
        """Generate a stub SPZ file for development without GPU."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((num_gaussians, 3)).astype(np.float32) * 0.1
        return positions.tobytes()

    def _generate_stub_flame_params(self, num_gaussians: int) -> dict:
        """Generate stub FLAME parameters for development."""
        return {
            "num_gaussians": num_gaussians,
            "num_bones": 5,
            "left_eye_bone_index": 3,
            "right_eye_bone_index": 4,
            "bone_names": ["root", "neck", "jaw", "left_eye", "right_eye"],
        }
