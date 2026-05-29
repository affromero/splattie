"""Visitor analytics endpoints.

- ``POST /track`` is public: the browser beacon posts page views here.
- ``GET /admin/stats`` is protected by a shared bearer token (ADMIN_API_TOKEN)
  and is called server-to-server by the Next.js admin dashboard.
"""

from __future__ import annotations

import os
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from splattie.stats.store import StatsStore

router = APIRouter()

_store: StatsStore | None = None


def get_store() -> StatsStore:
    """Lazily build the process-wide stats store from STATS_DB_PATH."""
    global _store
    if _store is None:
        _store = StatsStore(os.environ.get("STATS_DB_PATH", "data/stats.db"))
    return _store


def _client_ip(request: Request) -> str:
    """Real client IP, honoring the reverse proxy's X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def require_admin(request: Request) -> None:
    """Reject requests without a valid ADMIN_API_TOKEN bearer token."""
    token = os.environ.get("ADMIN_API_TOKEN", "")
    provided = request.headers.get("authorization", "")
    if not token or not secrets.compare_digest(provided, f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="unauthorized")


Store = Annotated[StatsStore, Depends(get_store)]


class TrackPayload(BaseModel):
    """Beacon payload posted by the frontend on page views and avatar events."""

    type: str = Field(default="pageview", max_length=32)
    path: str = Field(default="/", max_length=512)
    referrer: str | None = Field(default=None, max_length=2048)
    meta: dict[str, Any] | None = None


@router.post("/track", status_code=204)
async def track(request: Request, store: Store) -> Response:
    """Record a page view / event from the browser beacon.

    The body is sent as text/plain JSON so the beacon stays a CORS-simple
    request (no preflight) and works with navigator.sendBeacon.
    """
    raw = await request.body()
    try:
        payload = TrackPayload.model_validate_json(raw) if raw else TrackPayload()
    except ValidationError:
        return Response(status_code=204)  # never fail a beacon on bad input

    store.record(
        event_type=payload.type,
        path=payload.path,
        referrer=payload.referrer,
        ip=_client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
        meta=payload.meta,
    )
    return Response(status_code=204)


@router.get("/admin/stats", dependencies=[Depends(require_admin)])
async def admin_stats(store: Store, days: int = 30) -> JSONResponse:
    """Return aggregate analytics for the trailing window (auth required)."""
    return JSONResponse(store.stats(days=days))
