"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from splattie.api.routes import generate, health, models, segment, stats
from splattie.methods.lam.method import LAMMethod
from splattie.methods.lhm.method import LHMMethod
from splattie.methods.object.method import ObjectRigMethod
from splattie.methods.quadruped_mammal.method import QuadrupedMammalMethod

# Keep explicit references so method modules are imported and registry decorators run.
_REGISTERED_METHOD_CLASSES = (LAMMethod, LHMMethod, ObjectRigMethod, QuadrupedMammalMethod)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Splattie API",
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
    app.include_router(stats.router)

    @app.get("/storage/{model_id}/{filename}")
    async def serve_storage(model_id: str, filename: str) -> FileResponse:
        """Serve generated files (SPZ, FLAME params, masks)."""
        for base in [Path("data/generations"), Path("data/uploads")]:
            path = base / model_id / filename
            if path.exists():
                return FileResponse(path)
        return FileResponse(status_code=404, path="")

    return app
