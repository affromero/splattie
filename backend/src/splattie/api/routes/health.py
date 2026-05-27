"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from splattie.methods.registry import registry

router = APIRouter()

_NO_GPU: dict = {"available": False, "device": None, "model_loaded": False}


def _gpu_status() -> dict:
    """Check GPU availability and LAM model status."""
    try:
        import torch
    except ImportError:
        return _NO_GPU

    if not torch.cuda.is_available():
        return _NO_GPU

    device_name = torch.cuda.get_device_name(0)
    vram_total_mb = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    vram_used_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)

    from splattie.methods.lam.method import _lam_model

    return {
        "available": True,
        "device": device_name,
        "vram_total_mb": vram_total_mb,
        "vram_used_mb": vram_used_mb,
        "model_loaded": _lam_model is not None,
    }


@router.get("/health")
def health() -> dict:
    """Return service health status."""
    methods = registry.list_available()
    gpu = _gpu_status()
    return {
        "status": "ok",
        "gpu": gpu,
        "methods_loaded": [m.id for m in methods],
    }
