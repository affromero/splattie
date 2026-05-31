"""CPU tests for object `.splattie` bundle production."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from splattie.methods.object.bundle import (
    DEFAULT_STATES_OBJECT,
    OBJECT_RIG,
    BinaryPly,
    RigSkeleton,
    SparseLbsWeights,
    build_object_splattie,
    read_binary_ply,
    read_lbs_weights_binary,
    transform_gaussian_ply,
    write_binary_ply,
    write_lbs_weights_binary,
)


def _fixture_ply(path: Path) -> Path:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
            ("opacity", "<f4"),
        ]
    )
    rows = np.zeros(2, dtype=dtype)
    rows["x"] = [1.0, -1.0]
    rows["y"] = [2.0, -2.0]
    rows["z"] = [3.0, -3.0]
    rows["nx"] = [0.0, 0.0]
    rows["ny"] = [1.0, 0.0]
    rows["nz"] = [0.0, 1.0]
    rows["rot_0"] = 1.0
    rows["opacity"] = [0.5, 0.75]
    write_binary_ply(
        path,
        BinaryPly(
            vertices=rows,
            properties=[
                ("x", "float"),
                ("y", "float"),
                ("z", "float"),
                ("nx", "float"),
                ("ny", "float"),
                ("nz", "float"),
                ("rot_0", "float"),
                ("rot_1", "float"),
                ("rot_2", "float"),
                ("rot_3", "float"),
                ("opacity", "float"),
            ],
        ),
    )
    return path


def _skeleton() -> RigSkeleton:
    return RigSkeleton(
        rig="puppeteer-object",
        joint_count=2,
        names=["root", "arm"],
        parents=[-1, 0],
        rest_positions=[(0.0, 1.0, 2.0), (0.5, -1.0, -2.0)],
    )


def _weights() -> SparseLbsWeights:
    return SparseLbsWeights(
        num_gaussians=2,
        joint_count=2,
        k=2,
        indices=[0, 1, 1, 0],
        weights=[0.75, 0.25, 0.4, 0.6],
    )


def test_transform_gaussian_ply_flips_viewer_axis_and_rotations(tmp_path: Path) -> None:
    ply_path = _fixture_ply(tmp_path / "object.ply")
    transformed = transform_gaussian_ply(read_binary_ply(ply_path))
    rows = transformed.vertices

    assert np.allclose(np.column_stack([rows["x"], rows["y"], rows["z"]]), [[1, -2, -3], [-1, 2, 3]])
    assert np.allclose(np.column_stack([rows["nx"], rows["ny"], rows["nz"]]), [[0, -1, 0], [0, 0, -1]])
    # TRELLIS gaussian rotations are stored wxyz. Left-multiplying identity by
    # the viewer 180-degree X rotation gives [w, x, y, z] = [0, 1, 0, 0].
    assert np.allclose(np.column_stack([rows[f"rot_{idx}"] for idx in range(4)]), [[0, 1, 0, 0], [0, 1, 0, 0]])


def test_lbsw_binary_round_trip(tmp_path: Path) -> None:
    path = tmp_path / OBJECT_RIG.weights_file
    write_lbs_weights_binary(path, _weights())
    loaded = read_lbs_weights_binary(path)

    assert loaded.num_gaussians == 2
    assert loaded.joint_count == 2
    assert loaded.k == 2
    assert loaded.indices == [0, 1, 1, 0]
    assert np.allclose(loaded.weights, [0.75, 0.25, 0.4, 0.6], atol=5e-4)


def test_skeleton_sort_remaps_weight_indices() -> None:
    skeleton = RigSkeleton(
        rig="puppeteer-object",
        joint_count=3,
        names=["leaf", "root", "branch"],
        parents=[2, -1, 1],
        rest_positions=[(2, 0, 0), (0, 0, 0), (1, 0, 0)],
    )
    weights = SparseLbsWeights(
        num_gaussians=1,
        joint_count=3,
        k=2,
        indices=[0, 2],
        weights=[0.7, 0.3],
    )

    sorted_skeleton, sorted_weights = skeleton.topologically_sorted(weights)

    assert sorted_skeleton.names == ["root", "branch", "leaf"]
    assert sorted_skeleton.parents == [-1, 0, 1]
    assert sorted_weights.indices == [2, 1]


def test_object_bundle_is_widget_loadable(tmp_path: Path) -> None:
    ply_path = _fixture_ply(tmp_path / "source.ply")
    out_dir = tmp_path / "bundle"
    bundle_path, num_gaussians = build_object_splattie(
        ply_path=ply_path,
        output_dir=out_dir,
        model_id="toy",
        skeleton=_skeleton(),
        lbs_weights=_weights(),
    )

    assert num_gaussians == 2
    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
        assert names == {"manifest.json", "toy.ply", "skeleton.json", "lbs_weights.bin", "states.json"}
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["assetType"] == "object"
        assert manifest["animation"]["skeleton"] == {"file": "skeleton.json", "rig": "puppeteer-object"}
        assert manifest["animation"]["weights"] == {"file": "lbs_weights.bin", "format": "lbsw-v1"}
        assert manifest["avatar"]["splat"]["topology"] == "object-auto"
        assert manifest["metadata"]["viewerTransform"] == "viewer-upright-180x"
        states = json.loads(zf.read("states.json"))
        assert states == DEFAULT_STATES_OBJECT.jsonable()
        skeleton = json.loads(zf.read("skeleton.json"))
        assert skeleton["restPositions"] == [[0.0, -1.0, -2.0], [0.5, 1.0, 2.0]]
