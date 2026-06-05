"""CPU tests for Puppeteer mesh-rig to gaussian binding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from splattie.methods.object.bind import bind_rigged_splat, parse_rignet_skin
from splattie.methods.object.bundle import BinaryPly, write_binary_ply


def _fixture_gaussian_ply(path: Path) -> Path:
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4")])
    rows = np.zeros(3, dtype=dtype)
    rows["x"] = [0.0, 1.0, 0.0]
    rows["y"] = [0.0, 0.0, 1.0]
    rows["z"] = [0.0, 0.0, 0.0]
    rows["opacity"] = [1.0, 1.0, 1.0]
    write_binary_ply(
        path,
        BinaryPly(
            vertices=rows,
            properties=[("x", "float"), ("y", "float"), ("z", "float"), ("opacity", "float")],
        ),
    )
    return path


def _fixture_mesh(path: Path) -> Path:
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return path


def _fixture_skin(path: Path) -> Path:
    path.write_text(
        "joints root 0 0 0\n"
        "joints tip 1 0 0\n"
        "root root\n"
        "hier root tip\n"
        "skin 0 root 1.0\n"
        "skin 1 tip 1.0\n"
        "skin 2 root 0.5 tip 0.5\n"
    )
    return path


def test_parse_rignet_skin(tmp_path: Path) -> None:
    rig = parse_rignet_skin(_fixture_skin(tmp_path / "skin.txt"))
    assert rig.joint_names == ["root", "tip"]
    assert rig.parent_indices == [-1, 0]
    assert rig.root_index == 0
    assert len(rig.skins) == 3


def test_bind_rigged_splat_transfers_mesh_weights(tmp_path: Path) -> None:
    binding = bind_rigged_splat(
        gaussian_ply=_fixture_gaussian_ply(tmp_path / "gaussian.ply"),
        mesh_obj=_fixture_mesh(tmp_path / "mesh.obj"),
        rig_skin=_fixture_skin(tmp_path / "skin.txt"),
        output_dir=tmp_path / "binding",
        model_id="toy",
        top_k=2,
    )

    assert binding.skeleton.names == ["root", "tip"]
    assert binding.skeleton.parents == [-1, 0]
    assert binding.lbs_weights.num_gaussians == 3
    assert binding.lbs_weights.k == 2
    assert binding.rigged_splat_npz.exists()
    assert binding.dominant_joint_preview_ply is not None
    assert binding.dominant_joint_preview_ply.exists()

    summary = json.loads(binding.summary_json.read_text())
    assert summary["gaussian_count"] == 3
    assert summary["mesh_vertex_count"] == 3
    assert summary["nearest_mesh_distance"]["max"] == 0.0
