"""Tests for visitor analytics storage and endpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from splattie.api.app import create_app
from splattie.api.routes import stats as stats_routes
from splattie.stats.store import StatsStore, classify_device, referrer_host, visitor_hash

DESKTOP_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120 Safari/537.36"
MOBILE_UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) Safari/604.1"
BOT_UA = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"


@pytest.fixture
def store(tmp_path: Path) -> StatsStore:
    return StatsStore(str(tmp_path / "stats.db"))


def test_device_classification() -> None:
    assert classify_device(DESKTOP_UA) == "desktop"
    assert classify_device(MOBILE_UA) == "mobile"
    assert classify_device(BOT_UA) == "bot"


def test_referrer_reduced_to_host() -> None:
    assert referrer_host("https://news.ycombinator.com/item?id=1") == "news.ycombinator.com"
    assert referrer_host(None) == "direct"
    assert referrer_host("") == "direct"


def test_visitor_hash_is_stable_per_day_and_anonymous() -> None:
    ts = int(time.time())
    first = visitor_hash("1.2.3.4", DESKTOP_UA, ts)
    again = visitor_hash("1.2.3.4", DESKTOP_UA, ts)
    other_ip = visitor_hash("9.9.9.9", DESKTOP_UA, ts)
    next_day = visitor_hash("1.2.3.4", DESKTOP_UA, ts + 86400)

    assert first == again  # same person, same day -> one visitor
    assert first != other_ip  # different person
    assert first != next_day  # cannot be tracked across days
    assert "1.2.3.4" not in first  # no raw PII leaks into the id


def test_counts_pageviews_and_unique_visitors(store: StatsStore) -> None:
    now = int(time.time())
    store.record(event_type="pageview", path="/", referrer=None, ip="1.1.1.1", user_agent=DESKTOP_UA, ts=now)
    store.record(
        event_type="pageview",
        path="/view/demo",
        referrer=None,
        ip="1.1.1.1",
        user_agent=DESKTOP_UA,
        ts=now,
    )
    store.record(
        event_type="pageview",
        path="/",
        referrer="https://t.co/x",
        ip="2.2.2.2",
        user_agent=MOBILE_UA,
        ts=now,
    )

    result = store.stats(days=30, now_ts=now)
    assert result["summary"]["pageviews"] == 3
    assert result["summary"]["visitors"] == 2


def test_bots_excluded_from_human_metrics(store: StatsStore) -> None:
    now = int(time.time())
    store.record(event_type="pageview", path="/", referrer=None, ip="1.1.1.1", user_agent=DESKTOP_UA, ts=now)
    store.record(event_type="pageview", path="/", referrer=None, ip="3.3.3.3", user_agent=BOT_UA, ts=now)

    result = store.stats(days=30, now_ts=now)
    assert result["summary"]["pageviews"] == 1
    assert result["summary"]["visitors"] == 1
    assert result["summary"]["bots"] == 1


def test_avatar_events_and_breakdowns(store: StatsStore) -> None:
    now = int(time.time())
    store.record(
        event_type="avatar_create",
        path="/create",
        referrer=None,
        ip="1.1.1.1",
        user_agent=DESKTOP_UA,
        ts=now,
    )
    store.record(
        event_type="avatar_view",
        path="/view/abc",
        referrer="https://x.com/p",
        ip="2.2.2.2",
        user_agent=MOBILE_UA,
        ts=now,
    )
    store.record(
        event_type="pageview",
        path="/",
        referrer="https://x.com/p",
        ip="2.2.2.2",
        user_agent=MOBILE_UA,
        ts=now,
    )

    result = store.stats(days=30, now_ts=now)
    assert result["summary"]["avatar_creates"] == 1
    assert result["summary"]["avatar_views"] == 1
    assert {"referrer": "x.com", "count": 1} in result["top_referrers"]
    assert {"path": "/", "views": 1} in result["top_paths"]
    assert {"device": "mobile", "count": 1} in result["devices"]


def test_old_events_outside_window_are_excluded(store: StatsStore) -> None:
    now = int(time.time())
    store.record(
        event_type="pageview",
        path="/",
        referrer=None,
        ip="1.1.1.1",
        user_agent=DESKTOP_UA,
        ts=now - 40 * 86400,
    )
    store.record(event_type="pageview", path="/", referrer=None, ip="1.1.1.1", user_agent=DESKTOP_UA, ts=now)

    result = store.stats(days=30, now_ts=now)
    assert result["summary"]["pageviews"] == 1


# --- endpoint tests -------------------------------------------------------


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, StatsStore]:
    """App + isolated store + a known admin token."""
    monkeypatch.setenv("ADMIN_API_TOKEN", "test-token")
    test_store = StatsStore(str(tmp_path / "endpoint.db"))
    app = create_app()
    app.dependency_overrides[stats_routes.get_store] = lambda: test_store
    return app, test_store


async def test_track_endpoint_persists_event(app_client: tuple[FastAPI, StatsStore]) -> None:
    app, test_store = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Mirror the browser beacon: text/plain body, not application/json.
        resp = await ac.post(
            "/track",
            content=json.dumps({"type": "pageview", "path": "/", "referrer": "https://hn.com/x"}),
            headers={
                "content-type": "text/plain",
                "user-agent": DESKTOP_UA,
                "x-forwarded-for": "5.5.5.5",
            },
        )
    assert resp.status_code == 204
    assert test_store.stats(days=30)["summary"]["pageviews"] == 1


async def test_admin_stats_requires_token(app_client: tuple[FastAPI, StatsStore]) -> None:
    app, _ = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        unauth = await ac.get("/admin/stats")
        wrong = await ac.get("/admin/stats", headers={"authorization": "Bearer nope"})
        ok = await ac.get("/admin/stats", headers={"authorization": "Bearer test-token"})

    assert unauth.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 200
    assert "summary" in ok.json()
