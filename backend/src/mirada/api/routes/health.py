"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from mirada.methods.registry import registry

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Return service health status."""
    methods = registry.list_available()
    return {
        "status": "ok",
        "gpu": "none (dev mode)",
        "methods_loaded": [m.id for m in methods],
    }
