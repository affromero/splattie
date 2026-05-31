"""Puppeteer rigging orchestration for reconstructed object meshes."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from klogr import get_logger
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from splattie.methods.object.runtime import (
    PUPPETEER_SKELETON_WEIGHTS,
    PUPPETEER_SKINNING_WEIGHTS,
    VENDOR_PUPPETEER,
    run_command,
    vendor_python_env,
)

_PYDANTIC_CONFIG = ConfigDict(frozen=True, extra="forbid")
logger = get_logger()


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class PuppeteerRiggingConfig:
    """Puppeteer inference parameters."""

    input_pc_num: int = 8192
    max_input_faces: int = 25_000
    simplification_aggression: int = 7
    torchrun_port: int = 10009
    skinning_depth: int = 1


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class PuppeteerRiggingOutput:
    """Puppeteer rigging outputs used by the binding step."""

    input_mesh: Path
    skeleton_txt: Path
    skin_txt: Path
    skin_npy: Path
    results_dir: Path
    input_vertex_count: int
    input_face_count: int


_DEFAULT_PUPPETEER_RIGGING = PuppeteerRiggingConfig()
_PRIME_CUDA_AND_RUN = (
    "import runpy, sys, torch; "
    "torch.cuda.init(); "
    "sys.argv = [sys.argv[1], *sys.argv[2:]]; "
    "runpy.run_path(sys.argv[0], run_name='__main__')"
)


def rig_object_mesh_with_puppeteer(
    *,
    mesh_obj: Path,
    output_dir: Path,
    model_id: str,
    config: PuppeteerRiggingConfig = _DEFAULT_PUPPETEER_RIGGING,
) -> PuppeteerRiggingOutput:
    """Run Puppeteer skeleton + skinning stages for one mesh."""
    results_dir = output_dir / "puppeteer_results"
    input_dir = output_dir / "puppeteer_input"
    skeletons_dir = results_dir / "skeletons"
    input_dir.mkdir(parents=True, exist_ok=True)
    skeletons_dir.mkdir(parents=True, exist_ok=True)

    input_mesh = input_dir / f"{model_id}.obj"
    prepared_mesh = prepare_puppeteer_input_mesh(source_mesh=mesh_obj, output_mesh=input_mesh, config=config)

    _run_skeleton_stage(input_dir=input_dir, results_dir=results_dir, config=config)
    predicted_skeleton = results_dir / "skel_results" / f"{model_id}_pred.txt"
    if not predicted_skeleton.exists():
        msg = f"Puppeteer skeleton stage did not produce {predicted_skeleton}"
        raise FileNotFoundError(msg)
    skeleton_txt = skeletons_dir / f"{model_id}.txt"
    shutil.copyfile(predicted_skeleton, skeleton_txt)

    _run_skinning_stage(input_dir=input_dir, skeletons_dir=skeletons_dir, results_dir=results_dir, config=config)
    skin_txt = results_dir / "skin_results" / "generate" / f"{model_id}_skin.txt"
    skin_npy = results_dir / "skin_results" / "generate" / f"{model_id}_skin.npy"
    if not skin_txt.exists():
        msg = f"Puppeteer skinning stage did not produce {skin_txt}"
        raise FileNotFoundError(msg)
    if not skin_npy.exists():
        msg = f"Puppeteer skinning stage did not produce {skin_npy}"
        raise FileNotFoundError(msg)

    final_dir = results_dir / "final_rigging"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_skin = final_dir / f"{model_id}.txt"
    shutil.copyfile(skin_txt, final_skin)
    return PuppeteerRiggingOutput(
        input_mesh=input_mesh,
        skeleton_txt=skeleton_txt,
        skin_txt=final_skin,
        skin_npy=skin_npy,
        results_dir=results_dir,
        input_vertex_count=prepared_mesh.vertex_count,
        input_face_count=prepared_mesh.face_count,
    )


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class PreparedPuppeteerMesh:
    """Mesh written to Puppeteer's input directory."""

    path: Path
    vertex_count: int
    face_count: int
    source_vertex_count: int
    source_face_count: int


def prepare_puppeteer_input_mesh(
    *,
    source_mesh: Path,
    output_mesh: Path,
    config: PuppeteerRiggingConfig = _DEFAULT_PUPPETEER_RIGGING,
) -> PreparedPuppeteerMesh:
    """Write the exact mesh that Puppeteer will skin."""
    if config.max_input_faces < 1:
        msg = "max_input_faces must be >= 1"
        raise ValueError(msg)

    mesh = _load_mesh(source_mesh)
    source_vertex_count = len(mesh.vertices)
    source_face_count = len(mesh.faces)
    if source_face_count > config.max_input_faces:
        mesh = _simplify_mesh(mesh, face_count=config.max_input_faces, aggression=config.simplification_aggression)
    mesh.remove_unreferenced_vertices()

    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)
    return PreparedPuppeteerMesh(
        path=output_mesh,
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        source_vertex_count=source_vertex_count,
        source_face_count=source_face_count,
    )


def _load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = tuple(mesh.geometry.values())
        if not geometries:
            msg = f"{path} contains no mesh geometry"
            raise ValueError(msg)
        mesh = trimesh.util.concatenate(geometries)
    _validate_mesh_arrays(path, np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64))
    return mesh


@jaxtyped(typechecker=beartype)
def _validate_mesh_arrays(
    path: Path,
    vertices: Float[npt.NDArray[np.float32], "vertices 3"],
    faces: Int[npt.NDArray[np.integer], "faces 3"],
) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        msg = f"{path} did not load as a vertex mesh"
        raise ValueError(msg)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        msg = f"{path} did not load as a triangular mesh"
        raise ValueError(msg)


def _simplify_mesh(mesh: Any, *, face_count: int, aggression: int) -> Any:
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=face_count, aggression=aggression)
    except ModuleNotFoundError as exc:
        msg = "Mesh simplification requires the fast-simplification package; run backend GPU setup again."
        raise RuntimeError(msg) from exc
    if len(simplified.faces) == 0:
        msg = "mesh simplification produced an empty mesh"
        raise ValueError(msg)
    return simplified


def _try_run_skeleton_stage(args: list[str]) -> subprocess.CalledProcessError | None:
    try:
        run_command(
            args,
            cwd=VENDOR_PUPPETEER / "skeleton",
            env=vendor_python_env(pythonpath_roots=(VENDOR_PUPPETEER,)),
            label="Puppeteer skeleton generation",
        )
    except subprocess.CalledProcessError as exc:
        return exc
    return None


def _run_skeleton_stage(*, input_dir: Path, results_dir: Path, config: PuppeteerRiggingConfig) -> None:
    args = [
        sys.executable,
        "-c",
        _PRIME_CUDA_AND_RUN,
        "demo.py",
        "--input_dir",
        str(input_dir),
        "--pretrained_weights",
        str(PUPPETEER_SKELETON_WEIGHTS),
        "--output_dir",
        str(results_dir),
        "--save_name",
        "skel_results",
        "--input_pc_num",
        str(config.input_pc_num),
        "--apply_marching_cubes",
        "--joint_token",
        "--seq_shuffle",
    ]
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, 4):
        last_error = _try_run_skeleton_stage(args)
        if last_error is None:
            return
        if attempt == 3:
            raise last_error
        delay_seconds = attempt * 10
        logger.warning(f"Puppeteer skeleton generation failed on attempt {attempt}/3; retrying in {delay_seconds}s")
        time.sleep(delay_seconds)


def _run_skinning_stage(
    *,
    input_dir: Path,
    skeletons_dir: Path,
    results_dir: Path,
    config: PuppeteerRiggingConfig,
) -> None:
    args = [
        sys.executable,
        "main.py",
        "--num_workers",
        "1",
        "--batch_size",
        "1",
        "--generate",
        "--save_skin_npy",
        "--pretrained_weights",
        str(PUPPETEER_SKINNING_WEIGHTS),
        "--input_skel_folder",
        str(skeletons_dir),
        "--mesh_folder",
        str(input_dir),
        "--post_filter",
        "--depth",
        str(config.skinning_depth),
        "--save_folder",
        str(results_dir / "skin_results"),
    ]
    run_command(
        args,
        cwd=VENDOR_PUPPETEER / "skinning",
        env=vendor_python_env(pythonpath_roots=(VENDOR_PUPPETEER,)),
        label="Puppeteer skinning",
    )
