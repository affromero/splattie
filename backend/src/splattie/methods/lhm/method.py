"""LHM body generation method (SIGGRAPH 2025 — arxiv 2503.10625).

Single image → canonical SMPL-X-anchored 3DGS body. We run LHM's canonical
`infer_mesh` forward with a neutral SMPL-X pose (no mmpose pose estimation needed
for the canonical avatar — the widget animates the rig client-side) and rembg
masking. Outputs a raw gaussian PLY; the `.splattie` bundle + per-gaussian LBS
weight extraction happen in the LHM bundle adapter (Phase 1.C).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Bool, UInt8, jaxtyped

from splattie.methods.lhm.runtime import VENDOR_LHM, chdir, inference_lock, load_model
from splattie.methods.registry import registry
from splattie.types import AssetType, GenerationResult, MethodCapabilities, MethodInfo

logger = logging.getLogger(__name__)

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
            asset_type=AssetType.BODY,
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
        load_model()

    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
    ) -> GenerationResult:
        # No fallback: inference failure propagates as a 500.
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
        import torch
        from PIL import Image as PILImage

        img_path = output_dir / "input.jpg"
        PILImage.fromarray(image).save(str(img_path))

        model, cfg = load_model()
        source_size = cfg.dataset.source_image_res
        src_head_size = cfg.dataset.get("src_head_size", 112)

        ply_path = output_dir / f"{model_id}.ply"
        device, dtype = "cuda", torch.float32

        # LHM uses cwd-relative asset paths; serialize and run from vendor/LHM.
        with inference_lock, chdir(VENDOR_LHM):
            from LHM.runners.infer.human_lrm import infer_preprocess_image  # patched: mmpose-free
            from rembg import remove

            # Person mask via rembg (the canonical SAM2-free parsing fallback).
            parsing_mask = np.array(remove(PILImage.open(img_path).convert("RGB")).convert("RGBA"))[:, :, 3]

            ref_image, _, _ = infer_preprocess_image(
                str(img_path),
                mask=parsing_mask,
                intr=None,
                pad_ratio=0,
                bg_color=1.0,
                max_tgt_size=896,
                aspect_standard=5.0 / 3,
                enlarge_ratio=[1.0, 1.0],
                render_tgt_size=source_size,
                multiply=14,
                need_mask=True,
            )

            # Canonical avatar: no head crop, neutral SMPL-X pose. The widget poses
            # the rig client-side; person-specific betas (via pose estimation) are a
            # future fidelity upgrade.
            src_head_rgb = torch.zeros(1, 3, src_head_size, src_head_size, dtype=dtype)

            smplx_params = {
                "betas": torch.zeros(1, _SHAPE_PARAM_DIM, device=device),
                "root_pose": torch.zeros(1, 1, 3, device=device),
                "body_pose": torch.zeros(1, 1, 21, 3, device=device),
                "jaw_pose": torch.zeros(1, 1, 3, device=device),
                "leye_pose": torch.zeros(1, 1, 3, device=device),
                "reye_pose": torch.zeros(1, 1, 3, device=device),
                "lhand_pose": torch.zeros(1, 1, 15, 3, device=device),
                "rhand_pose": torch.zeros(1, 1, 15, 3, device=device),
                "expr": torch.zeros(1, 1, 100, device=device),
                "trans": torch.zeros(1, 1, 3, device=device),
            }

            model.to(dtype)
            logger.info("Running LHM forward pass (canonical body gaussians)...")
            with torch.no_grad():
                gs_app_model_list, query_points, transform_mat_neutral_pose = model.infer_single_view(
                    ref_image.unsqueeze(0).to(device, dtype),
                    src_head_rgb.unsqueeze(0).to(device, dtype),
                    None,
                    None,
                    None,
                    None,
                    None,
                    smplx_params={k: v.to(device) for k, v in smplx_params.items()},
                )
                smplx_params["transform_mat_neutral_pose"] = transform_mat_neutral_pose
                output_gs = model.animation_infer_gs(gs_app_model_list, query_points, smplx_params)
                output_gs.save_ply(str(ply_path))

        num_gaussians = _count_ply_vertices(ply_path)
        logger.info("LHM body PLY saved: %s (%d gaussians)", ply_path.name, num_gaussians)

        # Raw output for now; the .splattie bundle + per-gaussian LBS weights land in 1.C.
        ply_url = f"/storage/{model_id}/{model_id}.ply"
        return GenerationResult(
            model_id=model_id,
            spz_url=ply_url,
            spz_size_bytes=ply_path.stat().st_size,
            num_gaussians=num_gaussians,
            method_id="lhm",
            rig_params_url=ply_url,
        )

    def unload(self) -> None:
        from splattie.methods.lhm.runtime import unload_model

        unload_model()


def _count_ply_vertices(ply_path: Path) -> int:
    with ply_path.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    msg = f"No vertex count in PLY header: {ply_path}"
    raise ValueError(msg)
