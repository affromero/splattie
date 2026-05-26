"""Available methods endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from mirada.methods.registry import registry

router = APIRouter()


@router.get("/models")
def list_models() -> dict:
    """List available head generation methods."""
    methods = registry.list_available()
    return {
        "methods": [m.model_dump() for m in methods],
        "default": registry.default_method_id,
    }
