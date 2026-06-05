"""Asset generation method tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from splattie.methods.base import AssetGenerationMethod
from splattie.methods.lam.method import LAMMethod
from splattie.methods.lhm.method import LHMMethod
from splattie.methods.object.bundle import RigSkeleton, SparseLbsWeights
from splattie.methods.object.method import ObjectRigMethod
from splattie.methods.object.puppeteer import PuppeteerRiggingOutput
from splattie.methods.object.reconstruct import ObjectReconstruction
from splattie.methods.quadruped_mammal.method import QuadrupedMammalMethod
from splattie.methods.quadruped_mammal.schemas import FitDiagnostics
from splattie.methods.registry import registry
from splattie.types import AssetType, ReconstructBackend
from tests.gpu import GPU_TEST_SKIP_REASON, gpu_tests_enabled

_DEMOS = Path(__file__).resolve().parents[2] / "apps" / "web" / "public" / "demos"
# A committed demo portrait — LAM's FLAME tracking needs a real face.
_FACE_IMAGE = _DEMOS / "heads" / "h1.jpg"
# A committed full-body demo — LHM segments the person (rembg) and requires a portrait
# (taller-than-wide) person bbox, so the body pipeline needs a standing-figure image.
_BODY_IMAGE = _DEMOS / "bodies" / "b1.jpg"


def test_lam_implements_protocol() -> None:
    method = LAMMethod()
    assert isinstance(method, AssetGenerationMethod)


def test_lam_info() -> None:
    method = LAMMethod()
    assert method.info.id == "lam"
    assert "SIGGRAPH" in method.info.name


def test_lam_is_a_head_method() -> None:
    assert LAMMethod().info.asset_type is AssetType.head
    assert LAMMethod().info.asset_type == "head"


def test_lam_capabilities() -> None:
    method = LAMMethod()
    caps = method.capabilities
    assert caps.supports_single_image is True
    assert caps.max_output_gaussians > 0


def test_lam_generate_propagates_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """No silent fallback: if the model can't load, generation raises (no demo bundle).

    Forces the GPU/weights boundary to fail so it runs on any machine (GPU or not).
    """
    import splattie.methods.lam.method as lam_method

    def _boom() -> object:
        msg = "simulated model-load failure (no GPU / weights)"
        raise RuntimeError(msg)

    monkeypatch.setattr(lam_method, "_load_model", _boom)
    method = LAMMethod()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.bool_)
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure is acceptable; the point is it does NOT fall back
        method.generate(image, mask)


@pytest.mark.skipif(not gpu_tests_enabled(), reason=GPU_TEST_SKIP_REASON)
@pytest.mark.skipif(not _FACE_IMAGE.exists(), reason="demo portrait not present")
def test_lam_generate_produces_bundle() -> None:
    """On a real GPU, generation produces a widget-loadable `.splattie` bundle."""
    from PIL import Image

    method = LAMMethod()
    method.load()

    image = np.array(Image.open(_FACE_IMAGE).convert("RGB"))
    mask = np.ones(image.shape[:2], dtype=np.bool_)

    result = method.generate(image, mask)
    assert result.method_id == "lam"
    assert result.num_gaussians > 0
    assert result.splattie_url.endswith(".splattie")

    method.unload()


def test_registry_has_lam() -> None:
    methods = registry.list_available()
    ids = [m.id for m in methods]
    assert "lam" in ids


def test_registry_get() -> None:
    method = registry.get("lam")
    assert method.info.id == "lam"


def test_registry_default() -> None:
    assert registry.default_method_id == "lam"


def test_registry_for_asset_type() -> None:
    """Endpoints select by category; the registry resolves the method behind it."""
    assert registry.for_asset_type(AssetType.head).info.id == "lam"
    assert registry.for_asset_type(AssetType.body).info.id == "lhm"
    assert registry.for_asset_type(AssetType.object).info.id == "trellis-puppeteer"
    assert registry.for_asset_type(AssetType.quadruped_mammal).info.id == "trellis-smal-quadruped"


def test_lhm_implements_protocol() -> None:
    assert isinstance(LHMMethod(), AssetGenerationMethod)


def test_lhm_is_a_body_method() -> None:
    assert LHMMethod().info.asset_type is AssetType.body
    assert LHMMethod().info.asset_type == "body"


def test_lhm_registered() -> None:
    assert registry.get("lhm").info.id == "lhm"
    assert registry.get("lhm").info.asset_type == "body"


def test_lhm_generate_propagates_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """No silent fallback: body generation raises if the model can't load.

    Forces the GPU/weights boundary to fail so it runs on any machine (GPU or not).
    """
    import splattie.methods.lhm.method as lhm_method

    def _boom() -> object:
        msg = "simulated model-load failure (no GPU / weights)"
        raise RuntimeError(msg)

    monkeypatch.setattr(lhm_method, "load_inferrer", _boom)
    method = LHMMethod()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.bool_)
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure is fine; no silent fallback
        method.generate(image, mask)


@pytest.mark.skipif(not gpu_tests_enabled(), reason=GPU_TEST_SKIP_REASON)
@pytest.mark.skipif(not _BODY_IMAGE.exists(), reason="demo body image not present")
def test_lhm_generate_produces_body() -> None:
    """On a real GPU, LHM produces a canonical-pose body gaussian asset from one body image."""
    from PIL import Image

    method = LHMMethod()
    method.load()

    image = np.array(Image.open(_BODY_IMAGE).convert("RGB"))
    mask = np.ones(image.shape[:2], dtype=np.bool_)

    result = method.generate(image, mask)
    assert result.method_id == "lhm"
    assert result.num_gaussians > 0
    # 1.C: the body method emits a widget-loadable .splattie (SMPL-X skeleton + weights).
    assert result.splattie_url.endswith(".splattie")

    method.unload()


def test_object_implements_protocol() -> None:
    assert isinstance(ObjectRigMethod(), AssetGenerationMethod)


def test_object_is_an_object_method() -> None:
    assert ObjectRigMethod().info.asset_type is AssetType.object
    assert ObjectRigMethod().info.asset_type == "object"


def test_object_registered() -> None:
    assert registry.get("trellis-puppeteer").info.id == "trellis-puppeteer"
    assert registry.get("trellis-puppeteer").info.asset_type == "object"


def test_object_generate_propagates_runtime_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """No silent fallback: object generation raises if vendor runtime is incomplete."""
    import splattie.methods.object.runtime as object_runtime

    def _boom() -> object:
        msg = "simulated object-runtime failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(object_runtime, "require_object_runtime", _boom)
    method = ObjectRigMethod()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.bool_)
    with pytest.raises(Exception):  # noqa: B017, PT011 - any failure is fine; no silent fallback
        method.generate(image, mask)


def test_object_generate_binds_against_puppeteer_input_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Puppeteer skins its prepared input mesh, so binding must use that same mesh."""
    import splattie.methods.object.method as object_method

    model_id = "objecttest01"
    trellis_mesh = tmp_path / "trellis.obj"
    puppeteer_mesh = tmp_path / "puppeteer.obj"
    observed_meshes: list[Path] = []
    skeleton = RigSkeleton(
        rig="puppeteer-object",
        joint_count=1,
        names=["root"],
        parents=[-1],
        rest_positions=[(0.0, 0.0, 0.0)],
    )
    weights = SparseLbsWeights(
        num_gaussians=1,
        joint_count=1,
        k=1,
        indices=[0],
        weights=[1.0],
    )

    def _fake_reconstruct_object_with_trellis(
        *,
        image_path: Path,
        output_dir: Path,
        model_id: str,
    ) -> ObjectReconstruction:
        assert image_path.exists()
        assert output_dir.name == "trellis"
        assert model_id == "objecttest01"
        return ObjectReconstruction(
            gaussian_ply=tmp_path / "gaussian.ply",
            mesh_obj=trellis_mesh,
            mesh_arrays_npz=tmp_path / "mesh_arrays.npz",
        )

    def _fake_rig_object_mesh_with_puppeteer(
        *,
        mesh_obj: Path,
        output_dir: Path,
        model_id: str,
    ) -> PuppeteerRiggingOutput:
        assert mesh_obj == trellis_mesh
        assert output_dir.name == "puppeteer"
        assert model_id == "objecttest01"
        return PuppeteerRiggingOutput(
            input_mesh=puppeteer_mesh,
            skeleton_txt=tmp_path / "skeleton.txt",
            skin_txt=tmp_path / "skin.txt",
            skin_npy=tmp_path / "skin.npy",
            results_dir=tmp_path / "puppeteer_results",
            input_vertex_count=3,
            input_face_count=1,
        )

    def _fake_bind_rigged_splat(
        *,
        gaussian_ply: Path,
        mesh_obj: Path,
        rig_skin: Path,
        output_dir: Path,
        model_id: str,
    ) -> object:
        assert gaussian_ply == tmp_path / "gaussian.ply"
        assert rig_skin == tmp_path / "skin.txt"
        assert output_dir.name == "binding"
        assert model_id == "objecttest01"
        observed_meshes.append(mesh_obj)
        return SimpleNamespace(skeleton=skeleton, lbs_weights=weights)

    def _fake_build_object_splattie(
        *,
        ply_path: Path,
        output_dir: Path,
        model_id: str,
        skeleton: RigSkeleton,
        lbs_weights: SparseLbsWeights,
        source_image_path: Path,
    ) -> tuple[Path, int]:
        assert ply_path == tmp_path / "gaussian.ply"
        assert skeleton.rig == "puppeteer-object"
        assert lbs_weights.num_gaussians == 1
        assert source_image_path.exists()
        bundle = output_dir / f"{model_id}.splattie"
        bundle.write_bytes(b"bundle")
        return bundle, 1

    def _fake_load(self: ObjectRigMethod) -> None:
        assert isinstance(self, ObjectRigMethod)

    monkeypatch.setattr(object_method, "reconstruct_object_with_trellis", _fake_reconstruct_object_with_trellis)
    monkeypatch.setattr(object_method, "rig_object_mesh_with_puppeteer", _fake_rig_object_mesh_with_puppeteer)
    monkeypatch.setattr(object_method, "bind_rigged_splat", _fake_bind_rigged_splat)
    monkeypatch.setattr(object_method, "build_object_splattie", _fake_build_object_splattie)
    monkeypatch.setattr(object_method, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(ObjectRigMethod, "load", _fake_load)
    monkeypatch.setattr(object_method.uuid, "uuid4", lambda: SimpleNamespace(hex=model_id))

    result = ObjectRigMethod().generate(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4), dtype=np.bool_),
    )

    assert observed_meshes == [puppeteer_mesh]
    assert result.method_id == "trellis-puppeteer"
    assert result.splattie_url == f"/storage/{model_id}/{model_id}.splattie"


def test_quadruped_implements_protocol() -> None:
    assert isinstance(QuadrupedMammalMethod(), AssetGenerationMethod)


def test_quadruped_is_a_quadruped_method() -> None:
    info = QuadrupedMammalMethod().info
    assert info.id == "trellis-smal-quadruped"
    assert info.asset_type is AssetType.quadruped_mammal
    assert info.asset_type == "quadruped_mammal"


def test_quadruped_registered() -> None:
    assert registry.get("trellis-smal-quadruped").info.id == "trellis-smal-quadruped"


def test_quadruped_generate_propagates_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import splattie.methods.quadruped_mammal.method as quadruped_method

    def _boom(_backend: object = None) -> None:
        msg = "simulated SMAL weights-missing failure"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(quadruped_method.runtime, "require_quadruped_runtime", _boom)
    with pytest.raises(FileNotFoundError, match="SMAL"):
        QuadrupedMammalMethod().generate(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4), dtype=np.bool_),
        )


@pytest.mark.parametrize("backend_choice", [None, ReconstructBackend.trellis])
def test_quadruped_generate_produces_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_choice: ReconstructBackend | None
) -> None:
    """End-to-end orchestration (reconstruct -> keypoints -> fit -> bind), GPU stages mocked.

    Parametrized over the backend so the per-request choice (default TripoSplat, or TRELLIS) is
    proven to reach reconstruction.
    """
    import splattie.methods.quadruped_mammal.method as quadruped_method

    expected_backend = backend_choice or ReconstructBackend.triposplat
    model_id = "critter00001"
    gaussian_ply = tmp_path / "gaussian.ply"
    diagnostics = FitDiagnostics(
        chamfer=0.004,
        anchor_rms=0.05,
        lr_residual=0.09,
        lr_swap=False,
        n_anchors=16,
        triangulated_count=39,
        mean_keypoint_confidence=0.53,
        shape_norm=3.2,
    )

    def _fake_reconstruct(*, image_path: Path, output_dir: Path, model_id: str, backend: ReconstructBackend) -> Path:
        assert image_path.exists()
        assert output_dir.name == "reconstruct"
        assert model_id == "critter00001"
        assert backend == expected_backend  # the method threads its per-request backend to reconstruction
        return gaussian_ply

    def _fake_bind_and_bundle(
        _smal: object,
        _splat: object,
        _fit: object,
        *,
        ply_path: Path,
        output_dir: Path,
        model_id: str,
        source_image_path: Path,
    ) -> tuple[Path, int]:
        assert ply_path == gaussian_ply
        assert source_image_path.exists()
        bundle = output_dir / f"{model_id}.splattie"
        bundle.write_bytes(b"bundle")
        return bundle, 169120

    monkeypatch.setattr(quadruped_method, "reconstruct_gaussian_splat", _fake_reconstruct)
    monkeypatch.setattr(quadruped_method, "GaussianSplat", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(quadruped_method, "SMAL", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(quadruped_method, "detect_keypoints_3d", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(quadruped_method, "fit_smal", lambda *_a, **_k: SimpleNamespace(diagnostics=diagnostics))
    monkeypatch.setattr(quadruped_method, "bind_and_bundle", _fake_bind_and_bundle)
    monkeypatch.setattr(quadruped_method, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(QuadrupedMammalMethod, "load", lambda *_a, **_k: None)
    monkeypatch.setattr(quadruped_method.uuid, "uuid4", lambda: SimpleNamespace(hex=model_id))

    method = QuadrupedMammalMethod(backend=backend_choice) if backend_choice else QuadrupedMammalMethod()
    result = method.generate(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4), dtype=np.bool_),
    )

    assert result.method_id == "trellis-smal-quadruped"
    assert result.num_gaussians == 169120
    assert result.splattie_url == f"/storage/{model_id}/{model_id}.splattie"


def test_batch_method_for_wires_every_asset_type() -> None:
    """Every AssetType must resolve to a batch generation method.

    Guards `generate-splattie-batch` against a missing import/wiring (a NameError that only
    surfaces at CLI call time, not at module import — ruff's F821 does not catch it).
    """
    from splattie.cli.batch import _method_for

    expected = {
        AssetType.head: "LAMMethod",
        AssetType.body: "LHMMethod",
        AssetType.object: "ObjectRigMethod",
        AssetType.quadruped_mammal: "QuadrupedMammalMethod",
    }
    for asset_type in AssetType:
        assert type(_method_for(asset_type)).__name__ == expected[asset_type]


def test_method_resolution_threads_quadruped_backend() -> None:
    """The API + CLI method resolvers pass the chosen reconstruction backend to the quadruped method."""
    from splattie.api.routes.generate import _method_for as api_method_for
    from splattie.cli.batch import _method_for as cli_method_for

    for resolve in (api_method_for, cli_method_for):
        chosen = resolve(AssetType.quadruped_mammal, ReconstructBackend.trellis)
        assert isinstance(chosen, QuadrupedMammalMethod)
        assert chosen.backend is ReconstructBackend.trellis
        assert resolve(AssetType.quadruped_mammal, None).backend is ReconstructBackend.triposplat  # default
        # Other categories ignore the backend (no quadruped method returned).
        assert not isinstance(resolve(AssetType.object, ReconstructBackend.trellis), QuadrupedMammalMethod)
