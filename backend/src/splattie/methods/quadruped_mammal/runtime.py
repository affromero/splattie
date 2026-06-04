"""Runtime paths + readiness checks for the quadruped_mammal method.

The SMAL model (MPI noncommercial weights) and the SuperAnimal/DeepLabCut interpreter live
outside the backend's own venv; their locations are env-configurable so Docker/deploy can
relocate them without code changes. Defaults match the local GPU-host layout.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from splattie.methods.object.runtime import VENDOR_TRELLIS

VENDOR_ROOT = Path(__file__).resolve().parents[4] / "vendor"
__all__ = ["DLC_PYTHON", "SMAL_PKL", "VENDOR_TRELLIS", "inference_lock", "require_quadruped_runtime"]
_DEFAULT_SMAL_PKL = VENDOR_ROOT / "SMAL" / "smal_online_V1.0" / "smal_CVPR2017.pkl"
_DEFAULT_DLC_PYTHON = Path("/home/ubuntu/dlc-venv/bin/python")


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else default


# SPLATTIE_SMAL_PKL / SPLATTIE_DLC_PYTHON override these (e.g. inside Docker).
SMAL_PKL = _env_path("SPLATTIE_SMAL_PKL", _DEFAULT_SMAL_PKL)
DLC_PYTHON = _env_path("SPLATTIE_DLC_PYTHON", _DEFAULT_DLC_PYTHON)

# SMAL fitting (gsplat render) + SuperAnimal both hold large GPU state; serialize like object.
inference_lock = threading.Lock()


def require_quadruped_runtime() -> None:
    """Fail early when SMAL, the DeepLabCut interpreter, or TRELLIS is missing.

    TRELLIS supplies the gaussian splat; Puppeteer is NOT required (the quadruped rig comes
    from SMAL), so this deliberately does not call the object method's Puppeteer-inclusive check.
    """
    required = (SMAL_PKL, DLC_PYTHON, VENDOR_TRELLIS / "trellis")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        msg = (
            "Quadruped generation runtime is incomplete. Run `bash backend/scripts/setup-gpu.sh` "
            "(SMAL weights + DeepLabCut venv + TRELLIS), or set SPLATTIE_SMAL_PKL / SPLATTIE_DLC_PYTHON. "
            f"Missing:\n{formatted}"
        )
        raise FileNotFoundError(msg)
