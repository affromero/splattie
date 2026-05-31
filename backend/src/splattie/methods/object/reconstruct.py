"""TRELLIS reconstruction orchestration for object generation."""

from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from splattie.methods.object.runtime import VENDOR_TRELLIS, python_module_args, run_command, vendor_python_env

_PYDANTIC_CONFIG = ConfigDict(frozen=True, extra="forbid")


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class TrellisSamplingConfig:
    """TRELLIS sampling parameters used by the production object method."""

    seed: int = 7
    sparse_steps: int = 12
    sparse_cfg: float = 7.5
    slat_steps: int = 12
    slat_cfg: float = 3.0


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class ObjectReconstruction:
    """TRELLIS gaussian and mesh outputs."""

    gaussian_ply: Path
    mesh_obj: Path
    mesh_arrays_npz: Path


_DEFAULT_TRELLIS_SAMPLING = TrellisSamplingConfig()


def reconstruct_object_with_trellis(
    *,
    image_path: Path,
    output_dir: Path,
    model_id: str,
    config: TrellisSamplingConfig = _DEFAULT_TRELLIS_SAMPLING,
) -> ObjectReconstruction:
    """Run TRELLIS in a subprocess and return generated splat/mesh paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--image-path",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--model-id",
        model_id,
        "--seed",
        str(config.seed),
        "--sparse-steps",
        str(config.sparse_steps),
        "--sparse-cfg",
        str(config.sparse_cfg),
        "--slat-steps",
        str(config.slat_steps),
        "--slat-cfg",
        str(config.slat_cfg),
    ]
    run_command(
        python_module_args("splattie.methods.object.trellis_runner", args),
        cwd=VENDOR_TRELLIS,
        env=vendor_python_env(pythonpath_roots=(VENDOR_TRELLIS,)),
        label="TRELLIS object reconstruction",
    )

    result = ObjectReconstruction(
        gaussian_ply=output_dir / f"{model_id}_gaussian.ply",
        mesh_obj=output_dir / f"{model_id}_mesh.obj",
        mesh_arrays_npz=output_dir / f"{model_id}_mesh_arrays.npz",
    )
    for path in (result.gaussian_ply, result.mesh_obj, result.mesh_arrays_npz):
        if not path.exists():
            msg = f"TRELLIS did not produce expected output {path}"
            raise FileNotFoundError(msg)
    return result
