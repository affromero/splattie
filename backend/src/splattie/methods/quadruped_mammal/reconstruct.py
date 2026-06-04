"""TRELLIS reconstruction for the quadruped pipeline — gaussian PLY only.

The SMAL fit registers against the gaussian splat directly, so (unlike the object method)
the TRELLIS mesh is not needed; this returns just the gaussian PLY path.
"""

from __future__ import annotations

from pathlib import Path

from splattie.methods.object.reconstruct import reconstruct_object_with_trellis


def reconstruct_gaussian_splat(*, image_path: Path, output_dir: Path, model_id: str) -> Path:
    """Run TRELLIS and return the gaussian PLY path."""
    reconstruction = reconstruct_object_with_trellis(
        image_path=image_path,
        output_dir=output_dir,
        model_id=model_id,
    )
    return reconstruction.gaussian_ply
