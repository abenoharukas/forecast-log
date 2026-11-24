"""Tests for live news & 5-day prediction reference API (`news_feed.py` + `server.py`)."""

from __future__ import annotations

import io
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from forecast_log import news_feed, server
from forecast_log.bars import Bar, Series
from forecast_log.db import latest_news_status, save_news_fetch, upsert_series
from forecast_log.group_chat import build_market_context


def _seed_reference_env(root: Path) -> None:
    today = datetime.now(news_feed.SHANGHAI).date()
    # Seed events.json with upcoming CPI (in 2 days -> trigger window), NFP (5 days -> trigger window), FOMC (10 days)
    events_payload = {
        "generated_at": "2026-11-20T00:00:00-05:00",
        "source": "test",
        "calendar": {
            "cpi": [(today + timedelta(days=2)).isoformat()],
            "nfp": [(today + timedelta(days=5)).isoformat()],
            "fomc": [(today + timedelta(days=10)).isoformat()],
        },
    }
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "events.json").write_text(json.dumps(events_payload), encoding="utf-8")

    # Seed cross-asset bars for reference snapshot
    start = today - timedelta(days=40)
    for sym, base, step in (
        ("^NDX", 20000.0, 15.0),
        ("^TNX", 4.5, -0.01),
        ("DX-Y.NYB", 106.0, -0.05),
        ("HYG", 78.0, 0.05),
        ("TLT", 92.0, -0.02),
        ("^VIX", 18.0, 0.1),
        ("^VXN", 22.0, 0.1),
        ("GC=F", 2600.0, 2.0),
    ):
        bars = tuple(
            Bar(start + timedelta(days=i), base + i * step, is_closed=True)
            for i in range(30)
        )
        upsert_series(root, Series(sym, "America/New_York", "2026-11-23T21:00:00-05:00", bars))


def test_needs_news_refresh_and_cooldown(tmp_path: Path):
    now = datetime(2026, 11, 24, 3, 0, 0, tzinfo=timezone.utc)  # 11:00 Beijing on 2026-11-24
    assert news_feed.needs_news_refresh(tmp_path, cooldown_sec=900, now=now) is True

    # Stale date (10 days old) -> needs refresh
    save_news_fetch(
        tmp_path,
        {
            "as_of": "2026-11-14",
            "source": "wsj-markets",
            "fetched_at": "2026-11-14T03:00:00+00:00",
            "documents": [
                {
                    "id": "old1",
                    "published_at": "2026-11-14",
                    "title": "Old market headline",
                    "url": "https://example.com/old1",
                    "domain": "example.com",
                }
            ],
        },
    )
    assert news_feed.needs_news_refresh(tmp_path, cooldown_sec=900, now=now) is True

    # Today's news fetched 2 minutes ago -> within cooldown, no refresh needed
    save_news_fetch(
        tmp_path,
        {
            "as_of": "2026-11-24",
            "source": "wsj-markets",
            "fetched_at": "2026-11-24T02:58:00+00:00",
            "documents": [
                {
                    "id": "fresh1",
                    "published_at": "2026-11-24T02:50:00+00:00",
                    "title": "Fed signals rate cut as stocks rally",
                    "url": "https://example.com/fresh1",
                    "domain": "wsj.com",
                }
            ],
        },
    )
    assert news_feed.needs_news_refresh(tmp_path, cooldown_sec=900, now=now) is False

    # 20 minutes later -> cooldown expired
    later = now + timedelta(minutes=20)
    assert news_feed.needs_news_refresh(tmp_path, cooldown_sec=900, now=later) is True


def test_refresh_news_persists_and_builds_references(tmp_path: Path, monkeypatch):
    _seed_reference_env(tmp_path)
    today = datetime.now(news_feed.SHANGHAI).date()
    calls = {"rss": 0, "wiki": 0}

    def fake_rss(name: str, url: str, timeout: float = 8.0):
        calls["rss"] += 1
        if name == "wsj-markets":
            return (
                [
                    {
                        "id": "wsj1",
                        "published_at": f"{today.isoformat()}T02:30:00+00:00",
                        "title": "Stocks rally as Fed hints at rate cut amid easing inflation",
                        "text": "Stocks rally as Fed hints at rate cut amid easing inflation",
                        "url": "https://www.wsj.com/articles/rally-1",
                        "domain": "www.wsj.com",
                        "source": "wsj-markets",
                    },
                    {
                        "id": "wsj2",
                        "published_at": f"{today.isoformat()}T01:15:00+00:00",
                        "title": "Tariff dispute raises recession fears across global markets",
                        "text": "Tariff dispute raises recession fears across global markets",
                        "url": "https://www.wsj.com/articles/tariff-2",
                        "domain": "www.wsj.com",
                        "source": "wsj-markets",
                    },
                ],
                "",
            )
        if name == "yahoo-ssec":
            return (
                [
                    {
                        "id": "cn1",
                        "published_at": f"{today.isoformat()}T02:00:00+00:00",
                        "title": "China PBOC announces new liquidity support for Shanghai equities",
                        "text": "China PBOC announces new liquidity support for Shanghai equities",
                        "url": "https://finance.yahoo.com/news/china-pboc-1",
                        "domain": "finance.yahoo.com",
                        "source": "yahoo-ssec",
                    }
                ],
                "",
            )
        return [], ""

    def fake_backfill(root: Path, day: date, max_days: int = 14):
        calls["wiki"] += 1
        yday = (day - timedelta(days=1)).isoformat()
        save_news_fetch(
            root,
            {
                "as_of": yday,
                "source": "wikipedia-current-events",
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "documents": [
                    {
                        "id": "wk1",
                        "published_at": yday,
                        "title": "Global central banks hold emergency talks on oil price shock",
                        "url": "https://en.wikipedia.org/wiki/Portal:Current_events/test",
                        "domain": "en.wikipedia.org",
                    }
                ],
            },
        )
        return [yday], ""

    monkeypatch.setattr(news_feed, "_fetch_rss", fake_rss)
    monkeypatch.setattr(news_feed, "_backfill_recent_wiki_gap", fake_backfill)

    # First call (non-force) refreshes because DB has no news yet
    first = news_feed.refresh_news(tmp_path, force=False, market="ndx", score_llm=False)
    assert first["ok"] is True
    assert first["cached"] is False
    assert first["news"]["latest_as_of"] == today.isoformat()
    assert len(first["news"]["items"]) >= 3
    assert first["news"]["backfilled_days"] == [(today - timedelta(days=1)).isoformat()]
    assert calls["wiki"] == 1
    assert calls["rss"] == len(news_feed.LIVE_FEEDS)

    # Keywords & veto watch computed
    assert "tariff" in first["keywords"]["risk_off_hits"]
    assert "recession" in first["keywords"]["risk_off_hits"]
    assert first["keywords"]["veto_watch"] is True
    assert "rate cut" in first["keywords"]["risk_on_hits"]

    # Macro calendar & cross assets reference included
    assert first["macro_calendar"]["available"] is True
    cal_by_key = {ev["key"]: ev for ev in first["macro_calendar"]["events"]}
    assert cal_by_key["cpi"]["in_window"] is True
    assert cal_by_key["nfp"]["in_window"] is True
    assert cal_by_key["fomc"]["in_window"] is False
    symbols = [a["symbol"] for a in first["cross_assets"]]
    assert "^TNX" in symbols and "HYG/TLT" in symbols and "^VIX" in symbols

    # Second call (non-force) hits cooldown cache -> zero upstream fetches
    prev_rss = calls["rss"]
    second = news_feed.refresh_news(tmp_path, force=False, market="ndx", score_llm=False)
    assert second["cached"] is True
    assert calls["rss"] == prev_rss

    # Force call bypasses cooldown
    third = news_feed.refresh_news(tmp_path, force=True, market="ndx", score_llm=False)
    assert third["cached"] is False
    assert calls["rss"] > prev_rss

    # Group chat context now includes macro reference & latest news
    ctx = build_market_context(tmp_path, "ndx")
    assert "5. 宏观与跨资产参考" in ctx["text"]
    assert "6. 最新新闻参考" in ctx["text"]
    assert latest_news_status(tmp_path)["total_rows"] >= 4


def _dispatch_handler(method: str, path: str) -> tuple[int, dict]:
    """Invoke `server.Handler` in-memory without opening sockets (sandbox & proxy safe)."""
    h = server.Handler.__new__(server.Handler)
    h.path = path
    h.headers = {}
    h.rfile = io.BytesIO()
    h.wfile = io.BytesIO()
    status_holder = {"code": 0}

    def fake_send_response(code, message=None):
        status_holder["code"] = code

    h.send_response = fake_send_response
    h.send_header = lambda k, v: None
    h.end_headers = lambda: None
    if method == "POST":
        h.do_POST()
    else:
        h.do_GET()
    return status_holder["code"], json.loads(h.wfile.getvalue().decode("utf-8"))


def test_http_news_endpoints(tmp_path: Path, monkeypatch):
    _seed_reference_env(tmp_path)
    today = datetime.now(news_feed.SHANGHAI).date()

    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(news_feed, "trigger_llm_news_scoring", lambda *a, **kw: {"running": False})
    monkeypatch.setattr(
        news_feed,
        "_fetch_rss",
        lambda name, url, timeout=8.0: (
            [
                {
                    "id": f"{name}-1",
                    "published_at": f"{today.isoformat()}T02:00:00+00:00",
                    "title": f"{name}: Nasdaq futures gain on tech earnings",
                    "text": f"{name}: Nasdaq futures gain on tech earnings",
                    "url": f"https://example.com/{name}/1",
                    "domain": "example.com",
                    "source": name,
                }
            ]
            if name == "wsj-markets"
            else [],
            "",
        ),
    )
    monkeypatch.setattr(news_feed, "_backfill_recent_wiki_gap", lambda root, day, max_days=14: ([], ""))

    status, body = _dispatch_handler("POST", "/api/news/refresh?force=1&market=ndx")
    assert status == 200
    assert body["ok"] is True
    assert body["cached"] is False
    assert len(body["news"]["items"]) == 1
    assert "earnings" in body["keywords"]["hits"]

    status_get, get_body = _dispatch_handler("GET", "/api/news?market=ndx")
    assert status_get == 200
    assert get_body["ok"] is True
    assert get_body["cached"] is True
    assert len(get_body["news"]["items"]) == 1


def test_end_to_end_llm_news_scoring_reads_sqlite_and_feeds_cross(tmp_path: Path, monkeypatch):
    from forecast_log import cross, providers
    from forecast_log.agent.llm import Reply, ScriptedLLM

    today = datetime.now(news_feed.SHANGHAI).date()
    day_iso = today.isoformat()
    save_news_fetch(
        tmp_path,
        {
            "as_of": day_iso,
            "source": "wsj-markets",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "documents": [
                {
                    "id": "n1",
                    "published_at": f"{day_iso}T02:00:00+00:00",
                    "title": "Emergency tariff shock triggers bank sell-off and recession warning",
                    "url": "https://wsj.com/1",
                    "domain": "wsj.com",
                }
            ],
        },
    )

    # 1. cross._keyword_features now reads directly from SQLite news table
    kw_map = cross._keyword_features(tmp_path, [day_iso])
    assert kw_map[day_iso] >= 3

    # 2. DeepSeek & Qwen score straight from SQLite news table
    ds_reply = json.dumps(
        {
            "days": [
                {
                    "as_of": day_iso,
                    "surprise": "strong",
                    "direction": "bearish",
                    "category": "trade",
                    "evidence": "Emergency tariff shock triggers bank sell-off",
                }
            ]
        }
    )
    qw_reply = json.dumps(
        {
            "days": [
                {
                    "as_of": day_iso,
                    "regime": "risk_off",
                    "severity": 0.78,
                    "novelty": 0.85,
                    "events": [
                        {
                            "type": "trade",
                            "direction": "adverse",
                            "severity": 0.8,
                            "evidence": "Emergency tariff shock",
                        }
                    ],
                    "note": "Severe tariff shock and banking stress.",
                }
            ]
        }
    )
    monkeypatch.setattr(
        news_feed,
        "_resolve_llm_models",
        lambda root: ("deepseek-chat", "qwen3.8-flash", {"DEEPSEEK_API_KEY": "k1", "QWEN_API_KEY": "k2"}),
    )
    monkeypatch.setattr(
        providers,
        "build",
        lambda model, root, env=None: (
            ScriptedLLM([Reply(content=qw_reply)], name=model)
            if "qwen" in model
            else ScriptedLLM([Reply(content=ds_reply)], name=model)
        ),
    )

    result = news_feed.score_recent_news_with_llms(tmp_path, days_back=5, force_today=True, recompute_signals=False)
    assert result["deepseek_scored"] == 1
    assert result["qwen_scored"] == 1
    assert result["errors"] == []

    # 3. cross._news_labels and cross._news_providers both pick up the newly scored day
    ds_labels, _ = cross._news_labels(tmp_path)
    assert ds_labels[day_iso]["surprise"] == "strong"
    assert ds_labels[day_iso]["direction"] == "bearish"
    cols, _ = cross._news_providers(tmp_path, [day_iso])
    assert cols["news_qwen3.8-flash"] == [1]

    # 4. load_llm_indicators exposes both models cleanly
    ind = news_feed.load_llm_indicators(tmp_path)
    assert ind["deepseek"]["as_of"] == day_iso
    assert ind["deepseek"]["veto"] is True
    assert ind["qwen"]["as_of"] == day_iso
    assert ind["qwen"]["veto"] is True
    assert ind["qwen"]["severity"] == 0.78
