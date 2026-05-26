"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Force LAM method registration on import
import mirada.methods.lam.method  # noqa: F401
from mirada.api.routes import generate, health, models, segment


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Mirada API",
        description="3DGS head generation from a single photo",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(segment.router)
    app.include_router(generate.router)
    app.include_router(models.router)

    @app.get("/storage/{model_id}/{filename}")
    async def serve_storage(model_id: str, filename: str) -> FileResponse:
        """Serve generated files (SPZ, FLAME params, masks)."""
        for base in [Path("data/generations"), Path("data/uploads")]:
            path = base / model_id / filename
            if path.exists():
                return FileResponse(path)
        return FileResponse(status_code=404, path="")

    return app
