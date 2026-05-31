"""Subprocess entry point for TRELLIS image-to-3D reconstruction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

_PYDANTIC_CONFIG = ConfigDict(extra="forbid", populate_by_name=True)


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class TrellisRunMetadata:
    """Metadata written next to TRELLIS reconstruction outputs."""

    gaussian_ply: str
    mesh_obj: str
    mesh_arrays_npz: str
    mesh_vertices: int
    mesh_faces: int
    seed: int

    def jsonable(self) -> object:
        """Return JSON-serializable metadata."""
        return TypeAdapter(TrellisRunMetadata).dump_python(self, mode="json")


def main() -> None:
    """Run TRELLIS and write gaussian PLY + mesh OBJ."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--sparse-steps", default=12, type=int)
    parser.add_argument("--sparse-cfg", default=7.5, type=float)
    parser.add_argument("--slat-steps", default=12, type=int)
    parser.add_argument("--slat-cfg", default=3.0, type=float)
    args = parser.parse_args()

    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")

    from trellis.pipelines import TrellisImageTo3DPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    image = Image.open(args.image_path)
    outputs = pipeline.run(
        image,
        seed=args.seed,
        formats=["mesh", "gaussian"],
        sparse_structure_sampler_params={"steps": args.sparse_steps, "cfg_strength": args.sparse_cfg},
        slat_sampler_params={"steps": args.slat_steps, "cfg_strength": args.slat_cfg},
    )

    gaussian = outputs["gaussian"][0]
    gaussian_ply = args.output_dir / f"{args.model_id}_gaussian.ply"
    gaussian.save_ply(str(gaussian_ply))

    mesh_result = outputs["mesh"][0]
    vertices = mesh_result.vertices.detach().cpu().numpy().astype(np.float32)
    faces = mesh_result.faces.detach().cpu().numpy().astype(np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh_obj = args.output_dir / f"{args.model_id}_mesh.obj"
    mesh.export(mesh_obj)

    mesh_arrays = args.output_dir / f"{args.model_id}_mesh_arrays.npz"
    np.savez(mesh_arrays, vertices=vertices, faces=faces)

    metadata = TrellisRunMetadata(
        gaussian_ply=str(gaussian_ply),
        mesh_obj=str(mesh_obj),
        mesh_arrays_npz=str(mesh_arrays),
        mesh_vertices=len(vertices),
        mesh_faces=len(faces),
        seed=args.seed,
    )
    (args.output_dir / f"{args.model_id}_trellis.json").write_text(json.dumps(metadata.jsonable(), indent=2) + "\n")


if __name__ == "__main__":
    main()
