"""Shared runtime helpers for the TRELLIS + Puppeteer object method."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

from klogr import get_logger

logger = get_logger()

VENDOR_ROOT = Path(__file__).resolve().parents[4] / "vendor"
VENDOR_TRELLIS = VENDOR_ROOT / "TRELLIS"
VENDOR_TRIPOSPLAT = VENDOR_ROOT / "TripoSplat"  # optional image->3D-gaussian backend (alongside TRELLIS)
VENDOR_PUPPETEER = VENDOR_ROOT / "Puppeteer"

# TripoSplat checkpoints, downloaded by setup-gpu.sh under vendor/TripoSplat/ckpts.
TRIPOSPLAT_CKPTS = VENDOR_TRIPOSPLAT / "ckpts"
TRIPOSPLAT_FLOW_MODEL = TRIPOSPLAT_CKPTS / "diffusion_models" / "triposplat_fp16.safetensors"

PUPPETEER_SKELETON_WEIGHTS = VENDOR_PUPPETEER / "skeleton" / "skeleton_ckpts" / "puppeteer_skeleton_w_diverse_pose.pth"
PUPPETEER_SKINNING_WEIGHTS = (
    VENDOR_PUPPETEER / "skinning" / "skinning_ckpts" / "puppeteer_skin_w_diverse_pose_depth1.pth"
)
PUPPETEER_MICHELANGELO = (
    VENDOR_PUPPETEER
    / "skeleton"
    / "third_partys"
    / "Michelangelo"
    / "checkpoints"
    / "aligned_shape_latents"
    / "shapevae-256.ckpt"
)
PUPPETEER_SKINNING_MICHELANGELO = (
    VENDOR_PUPPETEER
    / "skinning"
    / "third_partys"
    / "Michelangelo"
    / "checkpoints"
    / "aligned_shape_latents"
    / "shapevae-256.ckpt"
)
PUPPETEER_PARTFIELD = VENDOR_PUPPETEER / "skinning" / "third_partys" / "PartField" / "ckpt" / "model_objaverse.ckpt"

# TRELLIS and Puppeteer both use large shared GPU state and cwd-relative caches.
inference_lock = threading.Lock()


def require_object_runtime() -> None:
    """Fail early when vendored code or required Puppeteer weights are missing."""
    missing = [path for path in _required_paths() if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        msg = (
            "Object generation runtime is incomplete. Initialize submodules and run "
            "`bash backend/scripts/setup-gpu.sh` on the GPU host. Missing:\n"
            f"{formatted}"
        )
        raise FileNotFoundError(msg)


def _required_paths() -> tuple[Path, ...]:
    return (
        VENDOR_TRELLIS / "trellis",
        VENDOR_TRELLIS / "trellis" / "representations" / "mesh" / "flexicubes" / "flexicubes.py",
        VENDOR_PUPPETEER / "skeleton" / "demo.py",
        VENDOR_PUPPETEER / "skeleton" / "third_partys" / "Michelangelo" / "encode.py",
        VENDOR_PUPPETEER / "skinning" / "main.py",
        VENDOR_PUPPETEER / "skinning" / "third_partys" / "PartField" / "encode.py",
        PUPPETEER_SKELETON_WEIGHTS,
        PUPPETEER_SKINNING_WEIGHTS,
        PUPPETEER_MICHELANGELO,
        PUPPETEER_SKINNING_MICHELANGELO,
        PUPPETEER_PARTFIELD,
    )


def vendor_python_env(*, pythonpath_roots: Sequence[Path] = ()) -> Mapping[str, str]:
    """Build subprocess environment with explicit vendored import roots."""
    env = os.environ.copy()
    roots = [str(path) for path in pythonpath_roots]
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    if roots:
        env["PYTHONPATH"] = os.pathsep.join(roots)
    env.setdefault("ATTN_BACKEND", "xformers")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_SOCKET_IFNAME", "lo")
    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    env.setdefault("SPCONV_ALGO", "native")
    return env


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    label: str,
) -> None:
    """Run a vendored command and raise with context on failure."""
    logger.info(f"Running {label}: {' '.join(args)}")
    subprocess.run(
        list(args),
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        check=True,
    )


def python_module_args(module: str, args: Sequence[str]) -> list[str]:
    """Return a command list for running a first-party module in this interpreter."""
    return [sys.executable, "-m", module, *args]
