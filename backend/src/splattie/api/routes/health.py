"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

from splattie.methods.registry import registry

router = APIRouter()
_PYDANTIC_CONFIG = ConfigDict(extra="forbid", populate_by_name=True)


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class GpuStatus:
    """GPU status payload returned by the health route."""

    available: bool
    device: str | None
    model_loaded: bool = Field(alias="modelLoaded")
    vram_total_mb: int | None = Field(default=None, alias="vramTotalMb")
    vram_used_mb: int | None = Field(default=None, alias="vramUsedMb")


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HealthResponse:
    """Service health payload."""

    status: str
    gpu: GpuStatus
    methods_loaded: list[str] = Field(alias="methodsLoaded")


_HEALTH_RESPONSE = TypeAdapter(HealthResponse)
_NO_GPU = GpuStatus(available=False, device=None, model_loaded=False)


def _gpu_status() -> GpuStatus:
    """Check GPU availability and LAM model status."""
    try:
        import torch
    except ImportError:
        return _NO_GPU

    if not torch.cuda.is_available():
        return _NO_GPU

    device_name = torch.cuda.get_device_name(0)
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    vram_used_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)

    from splattie.methods.lam.method import _lam_model

    return GpuStatus(
        available=True,
        device=device_name,
        vram_total_mb=vram_total_mb,
        vram_used_mb=vram_used_mb,
        model_loaded=_lam_model is not None,
    )


@router.get("/health")
def health() -> JSONResponse:
    """Return service health status."""
    methods = registry.list_available()
    payload = HealthResponse(
        status="ok",
        gpu=_gpu_status(),
        methods_loaded=[m.id for m in methods],
    )
    return JSONResponse(_HEALTH_RESPONSE.dump_python(payload, mode="json", by_alias=True, exclude_none=True))
