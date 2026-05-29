"""LHM model loading + shared runtime helpers (cached, GPU).

LHM is vendored at backend/vendor/LHM (fork affromero/LHM @ splattie, which makes
mmpose/sam2 optional). We construct ModelHumanLRM directly and run the canonical
`infer_mesh` forward for the *canonical* body avatar — neutral SMPL-X pose, so no
pose estimation (mmpose) is needed. The widget animates the SMPL-X rig client-side.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

VENDOR_LHM = Path(__file__).resolve().parents[4] / "vendor" / "LHM"
# model_dict key for the 500M (SapDino body+head SD3.5) checkpoint.
_EXP_TYPE = "human_lrm_sapdino_bh_sd3_5"
_HF_SNAPSHOT_GLOB = "pretrained_models/huggingface/models--3DAIGC--LHM-500M/snapshots/*"

_lhm_model = None
_lhm_cfg = None
# LHM uses cwd-relative asset paths + shared scratch dirs → serialize + run from
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


def _hf_snapshot() -> Path:
    base = VENDOR_LHM / "pretrained_models" / "huggingface" / "models--3DAIGC--LHM-500M" / "snapshots"
    snaps = sorted(base.glob("*")) if base.exists() else []
    if not snaps:
        msg = (
            f"LHM-500M weights not found under {base}. Run backend/scripts/setup-gpu.sh "
            "(or LHM's download_weights.sh) to populate the vendored weights."
        )
        raise FileNotFoundError(msg)
    return snaps[-1]


def load_model() -> tuple:
    """Build + cache the LHM-500M model on GPU. Raises if weights/GPU missing (no fallback)."""
    global _lhm_model, _lhm_cfg
    if _lhm_model is not None:
        return _lhm_model, _lhm_cfg

    lhm_path = str(VENDOR_LHM)
    if lhm_path not in sys.path:
        sys.path.insert(0, lhm_path)
    patch_chumpy_compat()

    from LHM.models import model_dict
    from LHM.utils.hf_hub import wrap_model_hub
    from omegaconf import OmegaConf

    with chdir(VENDOR_LHM):
        cfg = OmegaConf.load(str(VENDOR_LHM / "configs" / "inference" / "human-lrm-500M.yaml"))
        logger.info("Building LHM-500M model...")
        model_cls = wrap_model_hub(model_dict[_EXP_TYPE])
        model = model_cls.from_pretrained(str(_hf_snapshot())).to("cuda")
        model.eval()
    logger.info("LHM model loaded on GPU")

    _lhm_model = model
    _lhm_cfg = cfg
    return model, cfg


def unload_model() -> None:
    """Release the cached LHM model + free GPU memory."""
    global _lhm_model, _lhm_cfg
    if _lhm_model is None:
        return
    _lhm_model = None
    _lhm_cfg = None
    import torch

    torch.cuda.empty_cache()
    logger.info("LHM model unloaded")
