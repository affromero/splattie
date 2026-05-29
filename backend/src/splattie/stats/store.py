"""SQLite-backed event store for visitor analytics.

Privacy notes:
- No raw IP addresses or full user agents that identify a person are kept.
- Visitors are counted via a daily-rotating salted hash of (ip + user agent),
  so the same person is one "visitor" within a UTC day but cannot be tracked
  across days or reversed back to an IP.
- Country is a coarse, client-supplied hint (derived from the browser's
  timezone/locale), never from the IP.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_BOT_MARKERS = (
    "bot",
    "crawl",
    "spider",
    "slurp",
    "bingpreview",
    "facebookexternalhit",
    "embedly",
    "preview",
    "headless",
    "lighthouse",
    "pingdom",
    "uptime",
)
_MOBILE_MARKERS = ("mobi", "android", "iphone", "ipad", "ipod")


def classify_device(user_agent: str) -> str:
    """Return 'bot', 'mobile', or 'desktop' from a user-agent string."""
    ua = user_agent.lower()
    if any(marker in ua for marker in _BOT_MARKERS):
        return "bot"
    if any(marker in ua for marker in _MOBILE_MARKERS):
        return "mobile"
    return "desktop"


def referrer_host(referrer: str | None) -> str:
    """Reduce a referrer URL to its host, or 'direct' when absent."""
    if not referrer:
        return "direct"
    try:
        host = urlparse(referrer).netloc
    except ValueError:
        host = ""
    return (host or "direct")[:128]


def visitor_hash(ip: str, user_agent: str, ts: int) -> str:
    """Daily-rotating, salted, one-way visitor identifier."""
    day = time.strftime("%Y-%m-%d", time.gmtime(ts))
    salt = os.environ.get("STATS_SALT", "")
    raw = f"{salt}|{day}|{ip}|{user_agent}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class StatsStore:
    """A small append-mostly event log with aggregate queries."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=5.0)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        con = self._connect()
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    path TEXT NOT NULL DEFAULT '',
                    referrer TEXT NOT NULL DEFAULT 'direct',
                    visitor TEXT NOT NULL DEFAULT '',
                    device TEXT NOT NULL DEFAULT 'desktop',
                    country TEXT NOT NULL DEFAULT '',
                    meta TEXT
                )
                """
            )
            # Migration: add `country` to event logs created before it existed.
            columns = {row[1] for row in con.execute("PRAGMA table_info(events)").fetchall()}
            if "country" not in columns:
                con.execute("ALTER TABLE events ADD COLUMN country TEXT NOT NULL DEFAULT ''")
            con.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(type, ts)")
            con.commit()
        finally:
            con.close()

    def record(
        self,
        *,
        event_type: str,
        path: str,
        referrer: str | None,
        ip: str,
        user_agent: str,
        country: str | None = None,
        meta: dict[str, Any] | None = None,
        ts: int | None = None,
    ) -> None:
        """Persist a single event."""
        ts = ts if ts is not None else int(time.time())
        con = self._connect()
        try:
            con.execute(
                "INSERT INTO events (ts, type, path, referrer, visitor, device, country, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    event_type[:32],
                    path[:512],
                    referrer_host(referrer),
                    visitor_hash(ip, user_agent, ts),
                    classify_device(user_agent),
                    (country or "")[:32],
                    json.dumps(meta) if meta else None,
                ),
            )
            con.commit()
        finally:
            con.close()

    def stats(self, *, days: int = 30, now_ts: int | None = None) -> dict[str, Any]:
        """Return aggregate metrics for the trailing ``days`` window.

        Bots are excluded from human-facing counts but reported separately.
        """
        days = max(1, min(days, 365))
        now = now_ts if now_ts is not None else int(time.time())
        since = now - days * 86400
        con = self._connect()
        try:
            cur = con.cursor()

            def scalar(query: str, args: tuple[Any, ...]) -> int:
                return int(cur.execute(query, args).fetchone()[0])

            pageviews = scalar(
                "SELECT COUNT(*) FROM events WHERE type='pageview' AND device!='bot' AND ts>=?",
                (since,),
            )
            visitors = scalar(
                "SELECT COUNT(DISTINCT visitor) FROM events WHERE device!='bot' AND ts>=?",
                (since,),
            )
            creates = scalar(
                "SELECT COUNT(*) FROM events WHERE type='avatar_create' AND ts>=?",
                (since,),
            )
            avatar_views = scalar(
                "SELECT COUNT(*) FROM events WHERE type='avatar_view' AND device!='bot' AND ts>=?",
                (since,),
            )
            editor_opens = scalar(
                "SELECT COUNT(*) FROM events WHERE type='editor_open' AND device!='bot' AND ts>=?",
                (since,),
            )
            bots = scalar(
                "SELECT COUNT(*) FROM events WHERE device='bot' AND ts>=?",
                (since,),
            )
            pageviews_today = scalar(
                "SELECT COUNT(*) FROM events WHERE type='pageview' AND device!='bot' AND ts>=?",
                (now - 86400,),
            )
            pageviews_7d = scalar(
                "SELECT COUNT(*) FROM events WHERE type='pageview' AND device!='bot' AND ts>=?",
                (now - 7 * 86400,),
            )

            timeseries = [
                {"day": row["d"], "pageviews": int(row["pv"]), "visitors": int(row["uv"])}
                for row in cur.execute(
                    "SELECT date(ts,'unixepoch') d, "
                    "SUM(CASE WHEN type='pageview' THEN 1 ELSE 0 END) pv, "
                    "COUNT(DISTINCT visitor) uv "
                    "FROM events WHERE device!='bot' AND ts>=? GROUP BY d ORDER BY d",
                    (since,),
                ).fetchall()
            ]
            top_paths = [
                {"path": row["path"], "views": int(row["c"])}
                for row in cur.execute(
                    "SELECT path, COUNT(*) c FROM events WHERE type='pageview' AND device!='bot' "
                    "AND ts>=? GROUP BY path ORDER BY c DESC LIMIT 10",
                    (since,),
                ).fetchall()
            ]
            top_referrers = [
                {"referrer": row["referrer"], "count": int(row["c"])}
                for row in cur.execute(
                    "SELECT referrer, COUNT(*) c FROM events WHERE type='pageview' AND device!='bot' "
                    "AND ts>=? GROUP BY referrer ORDER BY c DESC LIMIT 10",
                    (since,),
                ).fetchall()
            ]
            devices = [
                {"device": row["device"], "count": int(row["c"])}
                for row in cur.execute(
                    "SELECT device, COUNT(*) c FROM events WHERE type='pageview' "
                    "AND ts>=? GROUP BY device ORDER BY c DESC",
                    (since,),
                ).fetchall()
            ]
            top_countries = [
                {"country": row["country"], "visitors": int(row["c"])}
                for row in cur.execute(
                    "SELECT country, COUNT(DISTINCT visitor) c FROM events "
                    "WHERE device!='bot' AND country!='' AND ts>=? "
                    "GROUP BY country ORDER BY c DESC LIMIT 12",
                    (since,),
                ).fetchall()
            ]

            return {
                "range_days": days,
                "summary": {
                    "pageviews": pageviews,
                    "visitors": visitors,
                    "avatar_creates": creates,
                    "avatar_views": avatar_views,
                    "editor_opens": editor_opens,
                    "bots": bots,
                    "pageviews_today": pageviews_today,
                    "pageviews_7d": pageviews_7d,
                },
                "timeseries": timeseries,
                "top_paths": top_paths,
                "top_referrers": top_referrers,
                "devices": devices,
                "top_countries": top_countries,
                "demo_clicks": self._demo_clicks(cur, since),
            }
        finally:
            con.close()

    @staticmethod
    def _demo_clicks(cur: sqlite3.Cursor, since: int) -> list[dict[str, Any]]:
        """Count demo-portrait clicks by demo id (parsed from the event meta)."""
        counts: dict[str, int] = {}
        for row in cur.execute(
            "SELECT meta FROM events WHERE type='demo_click' AND device!='bot' AND ts>=?",
            (since,),
        ).fetchall():
            if not row["meta"]:
                continue
            try:
                demo_id = json.loads(row["meta"]).get("id")
            except (ValueError, TypeError):
                demo_id = None
            if isinstance(demo_id, str) and demo_id:
                counts[demo_id] = counts.get(demo_id, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
        return [{"id": demo_id, "clicks": clicks} for demo_id, clicks in ranked]
