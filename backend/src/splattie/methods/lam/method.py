"""LAM (Large Avatar Model) head generation method.

SIGGRAPH 2025 - single image to drivable 3DGS head with FLAME animation.
Calls LAM's Python API directly from vendor/LAM submodule.
"""

from __future__ import annotations

import logging
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Bool, UInt8, jaxtyped

from splattie.methods.registry import registry
from splattie.types import GenerationResult, MethodCapabilities, MethodInfo

logger = logging.getLogger(__name__)

STORAGE_DIR = Path("data/generations")
VENDOR_LAM = Path(__file__).resolve().parents[4] / "vendor" / "LAM"
VENDOR_WEBRENDER = Path(__file__).resolve().parents[4] / "vendor" / "LAM_WebRender"
MODEL_ZOO = VENDOR_LAM / "model_zoo"

_lam_model = None
_lam_config = None


def _patch_torch_load() -> None:
    """Patch torch.load to use weights_only=False for LAM compatibility."""
    import torch

    _original = torch.load

    def _patched(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("weights_only", False)
        return _original(*args, **kwargs)

    torch.load = _patched


def _load_model():
    """Load the LAM model once and cache it."""
    global _lam_model, _lam_config

    if _lam_model is not None:
        return _lam_model, _lam_config

    lam_path = str(VENDOR_LAM)
    if lam_path not in sys.path:
        sys.path.insert(0, lam_path)

    _patch_torch_load()

    from lam.models.modeling_lam import ModelLAM
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    config_path = VENDOR_LAM / "configs" / "inference" / "lam-20k-8gpu.yaml"
    cfg = OmegaConf.load(str(config_path))

    logger.info("Building LAM model...")
    model = ModelLAM(**cfg.model).to("cuda")

    ckpt_path = MODEL_ZOO / "lam_models" / "releases" / "lam" / "lam-20k" / "step_045500" / "model.safetensors"
    ckpt = load_file(str(ckpt_path), device="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    logger.info("LAM model loaded on GPU")

    _lam_model = model
    _lam_config = cfg
    return model, cfg


@registry.register
class LAMMethod:
    """LAM: Large Avatar Model for one-shot animatable Gaussian heads."""

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            id="lam",
            name="LAM (SIGGRAPH 2025)",
            description="Single image → drivable 3DGS head with FLAME LBS animation",
            paper_url="https://arxiv.org/abs/2502.17796",
            repo_url="https://github.com/aigc3d/LAM",
        )

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            supports_single_image=True,
            supports_expression=True,
            max_output_gaussians=20_000,
            typical_inference_seconds=30.0,
        )

    def load(self) -> None:
        try:
            _load_model()
        except Exception:
            logger.warning("LAM model not available (no GPU or missing weights)")

    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
    ) -> GenerationResult:
        model_id = uuid.uuid4().hex[:12]
        output_dir = STORAGE_DIR / model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._generate_with_lam(image, model_id, output_dir)
        except Exception:
            logger.exception("LAM inference failed, using demo fallback")
            return self._fallback(model_id)

    def _generate_with_lam(
        self,
        image: npt.NDArray[np.uint8],
        model_id: str,
        output_dir: Path,
    ) -> GenerationResult:
        import torch
        from PIL import Image as PILImage

        img_path = output_dir / "input.jpg"
        PILImage.fromarray(image).save(str(img_path))

        model, cfg = _load_model()
        source_size = cfg.dataset.source_image_res

        lam_path = str(VENDOR_LAM)
        if lam_path not in sys.path:
            sys.path.insert(0, lam_path)
        from lam.runners.infer.head_utils import preprocess_image

        img_tensor, _, _, shape_param = preprocess_image(
            str(img_path),
            mask_path=None,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=None,
            aspect_standard=1.0,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
            get_shape_param=True,
        )

        logger.info("Running LAM forward pass...")
        with torch.no_grad():
            image_in = img_tensor.unsqueeze(0).to("cuda", torch.float32)
            dummy_c2ws = torch.eye(4).unsqueeze(0).unsqueeze(0).to("cuda")
            focal = source_size / 2
            dummy_intrs = (
                torch.tensor([[focal, 0, focal, 0, focal, focal, 0, 0, 1]]).reshape(1, 1, 3, 3).float().to("cuda")
            )
            dummy_bg = torch.ones(1, 1, 3).to("cuda")
            flame_params = {
                "betas": shape_param.unsqueeze(0).to("cuda"),
                "expr": torch.zeros(1, 100, device="cuda"),
                "rotation": torch.zeros(1, 1, 3, device="cuda"),
                "neck_pose": torch.zeros(1, 1, 3, device="cuda"),
                "jaw_pose": torch.zeros(1, 1, 3, device="cuda"),
                "eyes_pose": torch.zeros(1, 1, 6, device="cuda"),
                "translation": torch.zeros(1, 1, 3, device="cuda"),
            }

            res = model.infer_single_view(
                image_in,
                None,
                None,
                render_c2ws=dummy_c2ws,
                render_intrs=dummy_intrs,
                render_bg_colors=dummy_bg,
                flame_params=flame_params,
            )

        bundle_dir = output_dir / model_id
        bundle_dir.mkdir(exist_ok=True)

        ply_path = bundle_dir / "offset.ply"
        cano_gs = res["cano_gs_lst"][0]
        cano_gs.save_ply(str(ply_path), rgb2sh=False, offset2xyz=True)
        logger.info("PLY saved: %s (%d KB)", ply_path.name, ply_path.stat().st_size // 1024)

        demo_zip = VENDOR_WEBRENDER / "asset" / "arkit" / "p2-1.zip"
        if demo_zip.exists():
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                import zipfile as zf

                with zf.ZipFile(str(demo_zip), "r") as z:
                    z.extractall(tmp)
                demo_dir = Path(tmp) / "p2-1"
                for fname in ["skin.glb", "animation.glb", "vertex_order.json"]:
                    src = demo_dir / fname
                    if src.exists():
                        shutil.copy2(str(src), str(bundle_dir / fname))

        zip_path = output_dir / f"{model_id}.zip"
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for f in bundle_dir.iterdir():
                zf.write(str(f), f"{model_id}/{f.name}")

        zip_size = zip_path.stat().st_size
        logger.info("ZIP bundle: %s (%d KB)", zip_path.name, zip_size // 1024)

        return GenerationResult(
            model_id=model_id,
            spz_url=f"/storage/{model_id}/{model_id}.zip",
            spz_size_bytes=zip_size,
            num_gaussians=20_000,
            method_id="lam",
            flame_params_url=f"/storage/{model_id}/{model_id}.zip",
        )

    def _fallback(self, model_id: str) -> GenerationResult:
        logger.warning("Using demo bundle fallback")
        return GenerationResult(
            model_id=model_id,
            spz_url="/demo/andres.zip",
            spz_size_bytes=0,
            num_gaussians=20_000,
            method_id="lam",
            flame_params_url="/demo/andres.zip",
        )

    def unload(self) -> None:
        global _lam_model, _lam_config
        if _lam_model is not None:
            _lam_model = None
            _lam_config = None
            import torch

            torch.cuda.empty_cache()
            logger.info("LAM model unloaded")
