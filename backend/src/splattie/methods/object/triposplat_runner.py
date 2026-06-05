"""Subprocess entry point for TripoSplat image-to-3D-gaussian reconstruction.

Run with ``cwd`` set to the vendored TripoSplat checkout so the ``ckpts/`` paths
and the ``triposplat`` / ``model`` imports resolve. Emits a gaussian PLY; the
parent process meshes it on CPU (see ``gaussian_to_mesh``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

_PYDANTIC_CONFIG = ConfigDict(extra="forbid", populate_by_name=True)

# Checkpoint paths are relative to the TripoSplat checkout (the subprocess cwd),
# matching the layout setup-gpu.sh writes into vendor/TripoSplat/ckpts.
_CKPT = "ckpts/diffusion_models/triposplat_fp16.safetensors"
_DECODER = "ckpts/vae/triposplat_vae_decoder_fp16.safetensors"
_DINOV3 = "ckpts/clip_vision/dino_v3_vit_h.safetensors"
_FLUX2_VAE = "ckpts/vae/flux2-vae.safetensors"
_RMBG = "ckpts/background_removal/birefnet.safetensors"


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class TriposplatRunMetadata:
    """Metadata written next to the TripoSplat gaussian output."""

    gaussian_ply: str
    num_gaussians: int
    seed: int
    steps: int
    guidance_scale: float
    shift: float

    def jsonable(self) -> object:
        """Return JSON-serializable metadata."""
        return TypeAdapter(TriposplatRunMetadata).dump_python(self, mode="json")


def main() -> None:
    """Run TripoSplat and write a gaussian PLY (default Z-up->Y-up transform)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--num-gaussians", default=262144, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--guidance-scale", default=3.0, type=float)
    parser.add_argument("--shift", default=3.0, type=float)
    parser.add_argument("--erode-radius", default=1, type=int)
    args = parser.parse_args()

    from triposplat import TripoSplatPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = TripoSplatPipeline(
        ckpt_path=_CKPT,
        decoder_path=_DECODER,
        dinov3_path=_DINOV3,
        flux2_vae_encoder_path=_FLUX2_VAE,
        rmbg_path=_RMBG,
        device="cuda",
    )
    gaussian, _prepared = pipeline.run(
        str(args.image_path),
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        num_gaussians=args.num_gaussians,
        erode_radius=args.erode_radius,
        show_progress=False,
    )

    gaussian_ply = args.output_dir / f"{args.model_id}_gaussian.ply"
    gaussian.save_ply(str(gaussian_ply))

    metadata = TriposplatRunMetadata(
        gaussian_ply=str(gaussian_ply),
        num_gaussians=args.num_gaussians,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
    )
    (args.output_dir / f"{args.model_id}_triposplat.json").write_text(json.dumps(metadata.jsonable(), indent=2) + "\n")


if __name__ == "__main__":
    main()
