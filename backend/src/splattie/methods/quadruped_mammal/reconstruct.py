"""Image-to-3D-gaussian reconstruction for the quadruped pipeline — gaussian PLY only.

The SMAL fit registers against the gaussian splat directly, so (unlike the object method) no mesh is
needed; this returns just the gaussian PLY path. The backend is selectable via the ``ReconstructBackend``
enum: TripoSplat (VAST-AI) is the default because it reconstructs animal muzzles/faces far more cleanly
than TRELLIS (which melts flat/foreshortened animal faces); TRELLIS stays available as an option.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from splattie.methods.object.reconstruct import reconstruct_object_with_trellis
from splattie.methods.object.runtime import (
    VENDOR_TRIPOSPLAT,
    python_module_args,
    run_command,
    vendor_python_env,
)


class ReconstructBackend(StrEnum):
    """Image->3D-gaussian reconstruction backend.

    `str`-valued so it serializes to its plain value and compares equal to that string, while staying a
    single typed source of truth (matching `AssetType`/`SplatFormat`).
    """

    trellis = "trellis"
    triposplat = "triposplat"


def reconstruct_gaussian_splat(
    *, image_path: Path, output_dir: Path, model_id: str, backend: ReconstructBackend = ReconstructBackend.triposplat
) -> Path:
    """Run the chosen image->3D-gaussian backend and return the gaussian PLY path.

    TripoSplat is the default for animals: it reconstructs muzzles/faces far more cleanly than TRELLIS
    (which melts flat/foreshortened animal faces). The object method keeps TRELLIS; pass
    ``backend=ReconstructBackend.trellis`` (or the string ``"trellis"``) to fall back here.
    """
    backend = ReconstructBackend(backend)  # coerce env-var / plain-string callers to the enum
    if backend is ReconstructBackend.triposplat:
        return _reconstruct_with_triposplat(image_path=image_path, output_dir=output_dir, model_id=model_id)
    reconstruction = reconstruct_object_with_trellis(
        image_path=image_path,
        output_dir=output_dir,
        model_id=model_id,
    )
    return reconstruction.gaussian_ply


def _reconstruct_with_triposplat(*, image_path: Path, output_dir: Path, model_id: str) -> Path:
    """Run TripoSplat (default backend) in its vendored subprocess; return only the gaussian PLY.

    The object method also meshes the splat for Puppeteer; the SMAL fit skins the gaussians directly,
    so the mesh step is skipped here.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    args = ["--image-path", str(image_path), "--output-dir", str(output_dir), "--model-id", model_id]
    run_command(
        python_module_args("splattie.methods.object.triposplat_runner", args),
        cwd=VENDOR_TRIPOSPLAT,
        env=vendor_python_env(pythonpath_roots=(VENDOR_TRIPOSPLAT,)),
        label="TripoSplat quadruped reconstruction",
    )
    gaussian_ply = output_dir / f"{model_id}_gaussian.ply"
    if not gaussian_ply.exists():
        msg = f"TripoSplat did not produce expected gaussian PLY {gaussian_ply}"
        raise FileNotFoundError(msg)
    return gaussian_ply
