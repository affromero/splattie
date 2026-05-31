"""LHM body generation method (SIGGRAPH 2025 — arxiv 2503.10625).

Single image → SMPL-X-anchored 3DGS body. We drive LHM's canonical pipeline:
Multi-HMR pose estimation for the person's body shape (betas), then `infer_mesh` with a
neutral SMPL-X pose + the real face crop → a canonical-pose gaussian PLY. Background
removal uses rembg. The widget animates the SMPL-X rig client-side. Outputs a raw
gaussian PLY; the `.splattie` bundle + per-gaussian LBS weight extraction land in the
LHM bundle adapter (Phase 1.C).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Bool, UInt8, jaxtyped
from klogr import get_logger

from splattie.methods.lhm.bundle import build_body_splattie
from splattie.methods.lhm.runtime import VENDOR_LHM, chdir, inference_lock, load_inferrer
from splattie.methods.registry import registry
from splattie.types import AssetType, GenerationResult, MethodCapabilities, MethodInfo

logger = get_logger()

STORAGE_DIR = Path("data/generations")
# 500M model: shape_param_dim=10 (configs/inference/human-lrm-500M.yaml).
_SHAPE_PARAM_DIM = 10


@registry.register
class LHMMethod:
    """LHM: Large Animatable Human Reconstruction Model — one-shot SMPL-X body splat."""

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            id="lhm",
            name="LHM (SIGGRAPH 2025)",
            description="Single image → SMPL-X-anchored 3DGS body with LBS animation",
            paper_url="https://arxiv.org/abs/2503.10625",
            repo_url="https://github.com/aigc3d/LHM",
            asset_type=AssetType.body,
        )

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            supports_single_image=True,
            supports_expression=False,
            max_output_gaussians=40_000,
            typical_inference_seconds=20.0,
        )

    def load(self) -> None:
        # No fallback: missing GPU/weights raises (caller returns 500).
        load_inferrer()

    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
    ) -> GenerationResult:
        # No fallback: inference failure propagates as a 500. LHM re-derives its own
        # person mask (rembg) inside infer_mesh, so the upstream `mask` is unused here.
        model_id = uuid.uuid4().hex[:12]
        output_dir = (STORAGE_DIR / model_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._generate_with_lhm(image, model_id, output_dir)

    @jaxtyped(typechecker=beartype)
    def _generate_with_lhm(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        model_id: str,
        output_dir: Path,
    ) -> GenerationResult:
        from PIL import Image as PILImage

        img_path = output_dir / f"{model_id}.jpg"
        PILImage.fromarray(image).save(str(img_path))
        tmp_dir = output_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        inferrer = load_inferrer()

        # LHM uses cwd-relative asset paths; serialize and run from vendor/LHM.
        with inference_lock, chdir(VENDOR_LHM):
            # Person-specific body shape + pose from Multi-HMR. The pose (axis-angle
            # [53, 3]) bakes the arms into their photographed rest position; neutral
            # betas + canonical A-pose only if pose estimation is unavailable.
            if inferrer.pose_estimator is not None:
                est = inferrer.pose_estimator(str(img_path))
                betas = est.beta if est.beta is not None else np.zeros(_SHAPE_PARAM_DIM, dtype=np.float32)
                pose_rotvec = est.full_pose  # [53, 3], or None when no full body detected
            else:
                logger.warning("LHM pose estimator unavailable → neutral betas + canonical A-pose")
                betas = np.zeros(_SHAPE_PARAM_DIM, dtype=np.float32)
                pose_rotvec = None

            logger.info("Running LHM infer_mesh (body gaussians baked to the photographed pose)...")
            inferrer.infer_mesh(
                str(img_path),
                dump_tmp_dir=str(tmp_dir),
                dump_mesh_dir=str(output_dir),
                shape_param=betas,
                pose_rotvec=pose_rotvec,
            )

            # infer_mesh names the output after the image stem ("<id>.jpg" -> "<id>.ply").
            ply_path = output_dir / f"{model_id}.ply"
            if not ply_path.exists():
                msg = f"LHM infer_mesh did not produce {ply_path}"
                raise FileNotFoundError(msg)

            # When a pose was baked, infer_mesh drops the matching posed skeleton next to
            # the PLY; the rig uses it so the widget's rest pose is the identity.
            joints_path = output_dir / f"{model_id}_joints.json"
            posed_joints = (
                np.asarray(json.loads(joints_path.read_text()), dtype=np.float32) if joints_path.exists() else None
            )

            # Bundle the PLY + SMPL-X rig (posed skeleton + per-gaussian LBS weights) into
            # a widget-loadable body .splattie (assetType=body).
            bundle_path, num_gaussians = build_body_splattie(
                ply_path=ply_path,
                output_dir=output_dir,
                model_id=model_id,
                smplx_model=inferrer.model.renderer.smplx_model,
                betas=np.asarray(betas, dtype=np.float32),
                posed_joints=posed_joints,
                source_image_path=img_path,
            )

        logger.info(f"LHM body .splattie: {bundle_path.name} ({num_gaussians} gaussians)")
        splattie_url = f"/storage/{model_id}/{bundle_path.name}"
        return GenerationResult(
            model_id=model_id,
            splattie_url=splattie_url,
            splattie_size_bytes=bundle_path.stat().st_size,
            num_gaussians=num_gaussians,
            method_id="lhm",
        )

    def unload(self) -> None:
        from splattie.methods.lhm.runtime import unload_model

        unload_model()
