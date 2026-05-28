"""LAM (Large Avatar Model) head generation method.

SIGGRAPH 2025 - single image to drivable 3DGS head with FLAME animation.
Calls LAM's Python API directly from vendor/LAM submodule.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Bool, UInt8, jaxtyped

from splattie.methods.bundle_common import (
    DEFAULT_STATES_HEAD,
    HEAD_RIG,
    build_manifest,
    bundle_splattie,
    count_ply_vertices,
    read_widget_version,
)
from splattie.methods.registry import registry
from splattie.types import AssetType, GenerationResult, MethodCapabilities, MethodInfo

logger = logging.getLogger(__name__)

STORAGE_DIR = Path("data/generations")
VENDOR_LAM = Path(__file__).resolve().parents[4] / "vendor" / "LAM"
MODEL_ZOO = VENDOR_LAM / "model_zoo"

_lam_model = None
_lam_config = None
_flame_tracker = None
# LAM + its FLAME tracker read/write cwd-relative paths and share scratch dirs,
# so inference is serialized and run from vendor/LAM.
_inference_lock = threading.Lock()
_TRACKING_DIR = (STORAGE_DIR / "_flame_tracking").resolve()


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_flame_tracker():
    """Load the FLAME single-image tracker once and cache it."""
    global _flame_tracker
    if _flame_tracker is not None:
        return _flame_tracker

    lam_path = str(VENDOR_LAM)
    if lam_path not in sys.path:
        sys.path.insert(0, lam_path)
    _patch_torch_load()
    _patch_chumpy_compat()

    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    tracking_models = VENDOR_LAM / "model_zoo" / "flame_tracking_models"
    _TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    # The tracker's internal vhap/FLAME assets are cwd-relative to vendor/LAM.
    with _chdir(VENDOR_LAM):
        _flame_tracker = FlameTrackingSingleImage(
            output_dir=str(_TRACKING_DIR),
            alignment_model_path=str(tracking_models / "68_keypoints_model.pkl"),
            vgghead_model_path=str(tracking_models / "vgghead" / "vgg_heads_l.trcd"),
            human_matting_path=str(tracking_models / "matting" / "stylematte_synth.pt"),
            facebox_model_path=str(tracking_models / "FaceBoxesV2.pth"),
            detect_iris_landmarks=False,
        )
    logger.info("FLAME tracker loaded")
    return _flame_tracker


def _patch_torch_load() -> None:
    """Patch torch.load to use weights_only=False for LAM compatibility."""
    import torch

    _original = torch.load

    def _patched(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("weights_only", False)
        return _original(*args, **kwargs)

    torch.load = _patched


def _patch_chumpy_compat() -> None:
    """Restore stdlib/numpy names that chumpy 0.70 needs.

    FLAME's ``flame2023.pkl`` unpickles chumpy objects, but chumpy 0.70 predates
    Python 3.11 (which removed ``inspect.getargspec``) and numpy >= 1.24 (which
    removed ``np.bool`` / ``np.int`` / …). Restore the shims so the pickle loads.
    """
    import inspect

    import numpy as np

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    # chumpy/__init__.py does `from numpy import bool, int, float, complex,
    # object, unicode, str, ...` — all removed/renamed in modern numpy.
    for name, typ in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, typ)


def _load_model():
    """Load the LAM model once and cache it."""
    global _lam_model, _lam_config

    if _lam_model is not None:
        return _lam_model, _lam_config

    lam_path = str(VENDOR_LAM)
    if lam_path not in sys.path:
        sys.path.insert(0, lam_path)

    _patch_torch_load()
    _patch_chumpy_compat()

    from lam.models.modeling_lam import ModelLAM
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    config_path = VENDOR_LAM / "configs" / "inference" / "lam-20k-8gpu.yaml"
    cfg = OmegaConf.load(str(config_path))
    # The config points at FLAME/SMPL-X assets via a cwd-relative
    # "./model_zoo/human_parametric_models" path, which only resolves when the
    # process runs from vendor/LAM. Absolutize it so the model loads no matter
    # the working dir (the API server runs from backend/).
    cfg.model.human_model_path = str(VENDOR_LAM / "model_zoo" / "human_parametric_models")

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
            asset_type="head",
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
        # No fallback: if the model can't load (no GPU / missing weights), let it
        # raise so the caller returns a 500 instead of silently serving a demo.
        _load_model()

    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
    ) -> GenerationResult:
        # No fallback: inference failure propagates as a 500 (no-fallback rule).
        model_id = uuid.uuid4().hex[:12]
        output_dir = STORAGE_DIR / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._generate_with_lam(image, model_id, output_dir)

    @jaxtyped(typechecker=beartype)
    def _generate_with_lam(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        model_id: str,
        output_dir: Path,
    ) -> GenerationResult:
        import torch
        from PIL import Image as PILImage

        output_dir = output_dir.resolve()
        img_path = output_dir / "input.jpg"
        PILImage.fromarray(image).save(str(img_path))

        model, cfg = _load_model()
        tracker = _load_flame_tracker()
        source_size = cfg.dataset.source_image_res

        lam_path = str(VENDOR_LAM)
        if lam_path not in sys.path:
            sys.path.insert(0, lam_path)
        from lam.runners.infer.head_utils import preprocess_image

        ply_path = output_dir / f"{model_id}.ply"

        # LAM and its FLAME tracker use cwd-relative paths and a shared scratch
        # dir, so serialize and run them from vendor/LAM.
        with _inference_lock, _chdir(VENDOR_LAM):
            # FLAME single-image fitting → cropped image + mask + shape params.
            if tracker.preprocess(str(img_path)) != 0:
                msg = "FLAME tracking preprocess failed (no face detected?)"
                raise RuntimeError(msg)
            if tracker.optimize() != 0:
                msg = "FLAME tracking optimize failed"
                raise RuntimeError(msg)
            code, tracked_dir = tracker.export()
            if code != 0:
                msg = "FLAME tracking export failed"
                raise RuntimeError(msg)
            tracked_dir = Path(tracked_dir)
            tracked_img = tracked_dir / "images" / "00000_00.png"
            tracked_mask = tracked_dir / "fg_masks" / "00000_00.png"

            img_tensor, _, _, shape_param = preprocess_image(
                str(tracked_img),
                mask_path=str(tracked_mask),
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

            # We only need the canonical gaussian model (cano_gs), not LAM's driving
            # animation — the widget animates the FLAME rig client-side. So replicate
            # just the pre-animation half of infer_single_view (forward_gs) and skip
            # forward_animate_gs (which needs per-frame driving params we don't have).
            logger.info("Running LAM forward pass (canonical gaussians only)...")
            import math

            from einops import rearrange

            with torch.no_grad():
                image_in = img_tensor.unsqueeze(0).to("cuda", torch.float32)
                flame_params = {
                    "betas": shape_param.unsqueeze(0).to("cuda"),
                    "expr": torch.zeros(1, 100, device="cuda"),
                    "rotation": torch.zeros(1, 1, 3, device="cuda"),
                    "neck_pose": torch.zeros(1, 1, 3, device="cuda"),
                    "jaw_pose": torch.zeros(1, 1, 3, device="cuda"),
                    "eyes_pose": torch.zeros(1, 1, 6, device="cuda"),
                    "translation": torch.zeros(1, 1, 3, device="cuda"),
                }

                query_points = None
                if model.latent_query_points_type.startswith("e2e_flame"):
                    query_points, flame_params = model.renderer.get_query_points(flame_params, device=image_in.device)
                latent_points, image_feats = model.forward_latent_points(
                    image_in[:, 0], camera=None, query_points=query_points
                )
                image_feats_bchw = rearrange(
                    image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1]))
                )
                gs_model_list, _, _, _ = model.renderer.forward_gs(
                    gs_hidden_features=latent_points,
                    query_points=query_points,
                    flame_data=flame_params,
                    additional_features={
                        "image_feats": image_feats,
                        "image": image_in[:, 0],
                        "image_feats_bchw": image_feats_bchw,
                    },
                )
                cano_gs = gs_model_list[0]

            # Save the absolute-position PLY (rgb2sh=True, offset2xyz=False) — the
            # same flavor the demo batch builder ships and the widget renders. The
            # shared bundler then wraps it with the canonical FLAME rig + manifest
            # so the served .splattie matches a batch-built head demo in shape.
            cano_gs.save_ply(str(ply_path), rgb2sh=True, offset2xyz=False)
        logger.info("PLY saved: %s (%d KB)", ply_path.name, ply_path.stat().st_size // 1024)

        num_gaussians = count_ply_vertices(ply_path)
        manifest = build_manifest(
            splat_filename=f"{model_id}.ply",
            num_gaussians=num_gaussians,
            widget_version=read_widget_version(),
            asset_type=AssetType.HEAD,
            rig=HEAD_RIG,
            generator_tool="lam/method.py",
            source_image_path=img_path,
        )
        splattie_path = output_dir / f"{model_id}.splattie"
        bundle_splattie(
            output_path=splattie_path,
            splat_path=ply_path,
            manifest=manifest,
            states=DEFAULT_STATES_HEAD,
        )
        bundle_size = splattie_path.stat().st_size
        logger.info("Bundle: %s (%d KB)", splattie_path.name, bundle_size // 1024)

        bundle_url = f"/storage/{model_id}/{model_id}.splattie"
        return GenerationResult(
            model_id=model_id,
            spz_url=bundle_url,
            spz_size_bytes=bundle_size,
            num_gaussians=num_gaussians,
            method_id="lam",
            rig_params_url=bundle_url,
        )

    def unload(self) -> None:
        global _lam_model, _lam_config
        if _lam_model is not None:
            _lam_model = None
            _lam_config = None
            import torch

            torch.cuda.empty_cache()
            logger.info("LAM model unloaded")
