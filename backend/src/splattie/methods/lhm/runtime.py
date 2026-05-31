"""LHM model loading + shared runtime helpers (cached, GPU).

LHM is vendored at backend/vendor/LHM (fork affromero/LHM @ splattie). We construct
LHM's canonical `HumanLRMInferrer` and drive its *real* pipeline:

    betas = pose_estimator(image)          # Multi-HMR -> person's body shape
    infer_mesh(image, shape_param=betas)   # canonical pose + real face crop -> save_ply

The avatar is reconstructed in a neutral SMPL-X pose (the widget animates the rig
client-side) but with the person's real body shape + face crop. Background removal uses
rembg (`parsingnet` stays None -> our fork's rembg fallback; no sam2). The render-only
`diff_gaussian_rasterization` import is optional in the fork's gs_renderer — the export
path (forward -> animation_infer_gs -> save_ply) never rasterizes.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

from klogr import get_logger

logger = get_logger()

VENDOR_LHM = Path(__file__).resolve().parents[4] / "vendor" / "LHM"
# AutoModelQuery resolves this name to the downloaded snapshot (download_weights.sh).
_MODEL_NAME = "LHM-500M"

_inferrer = None
# LHM uses cwd-relative asset paths + shared scratch dirs -> serialize + run from
# vendor/LHM (mirrors the LAM method).
inference_lock = threading.Lock()


@contextlib.contextmanager
def chdir(path: Path) -> Iterator[None]:
    """Temporarily change the working directory (LHM uses cwd-relative paths)."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def patch_chumpy_compat() -> None:
    """Restore stdlib/numpy names chumpy 0.70 needs (FLAME pkl in SMPL-X).

    Same shim as the LAM method: chumpy predates py3.11 (`inspect.getargspec`) and
    numpy >= 1.24 (`np.bool` / `np.unicode` / …).
    """
    import inspect

    import numpy as np

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
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


def load_inferrer():  # noqa: ANN201 - LHM's HumanLRMInferrer is untyped (vendored).
    """Build + cache LHM's HumanLRMInferrer on GPU (real pose + face pipeline).

    Raises if weights/GPU are missing (no fallback). `parse_configs()` reads the model
    name + flags from `sys.argv` (LHM's CLI), so we mirror inference_mesh.sh's invocation
    in-process instead of spawning a subprocess.
    """
    global _inferrer
    if _inferrer is not None:
        return _inferrer

    if str(VENDOR_LHM) not in sys.path:
        sys.path.insert(0, str(VENDOR_LHM))
    patch_chumpy_compat()

    saved_argv = sys.argv
    sys.argv = [
        "splattie-lhm",
        f"model_name={_MODEL_NAME}",
        "image_input=./train_data/example_imgs/",
        "export_mesh=True",
        "motion_seqs_dir=None",
        "motion_img_dir=None",
        "vis_motion=false",
        "motion_img_need_mask=true",
    ]
    try:
        with chdir(VENDOR_LHM):
            from LHM.runners.infer.human_lrm import HumanLRMInferrer

            logger.info("Building LHM-500M inferrer (model + Multi-HMR pose + face detector)...")
            inferrer = HumanLRMInferrer()
    finally:
        sys.argv = saved_argv

    logger.info(
        "LHM inferrer ready "
        f"(pose_estimator={inferrer.pose_estimator is not None}, "
        f"facedetect={inferrer.facedetect is not None}, "
        f"parsingnet={inferrer.parsingnet is not None}; mask via rembg)"
    )
    _inferrer = inferrer
    return inferrer


def unload_model() -> None:
    """Release the cached LHM inferrer + free GPU memory."""
    global _inferrer
    if _inferrer is None:
        return
    _inferrer = None
    import torch

    torch.cuda.empty_cache()
    logger.info("LHM inferrer unloaded")
