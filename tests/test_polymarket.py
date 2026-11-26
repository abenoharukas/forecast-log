"""Tests for Polymarket macro/china-tariff/US-index odds aggregation (`polymarket.py`)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecast_log import news_feed, polymarket
from forecast_log.bars import Bar, Series
from forecast_log.db import upsert_series
from forecast_log.group_chat import build_market_context


def _sample_events() -> dict[str, list[dict]]:
    return {
        "Fed": [
            {
                "id": "ev-fed",
                "slug": "fed-decision-in-december",
                "title": "Fed Decision in December?",
                "active": True,
                "closed": False,
                "volume24hr": 5300000.0,
                "liquidity": 1200000.0,
                "endDate": "2026-12-18T00:00:00Z",
                "markets": [
                    {
                        "question": "Will the Fed cut rates by 25 bps in December?",
                        "groupItemTitle": "25 bps decrease",
                        "outcomes": '["Yes", "No"]',
                        "outcomePrices": '["0.625", "0.375"]',
                        "oneDayPriceChange": 0.085,
                        "volume24hr": 3200000.0,
                        "active": True,
                        "closed": False,
                    },
                    {
                        "question": "Will the Fed keep rates unchanged in December?",
                        "groupItemTitle": "No change",
                        "outcomes": '["Yes", "No"]',
                        "outcomePrices": '["0.35", "0.65"]',
                        "oneDayPriceChange": -0.075,
                        "volume24hr": 2100000.0,
                        "active": True,
                        "closed": False,
                    },
                ],
            },
            {
                "id": "ev-nba",
                "slug": "nba-cavaliers-vs-knicks",
                "title": "NBA Cavaliers vs. Knicks",
                "active": True,
                "closed": False,
                "volume24hr": 9999999.0,
                "liquidity": 500000.0,
                "markets": [
                    {
                        "question": "Cavaliers vs. Knicks",
                        "outcomes": '["Cavaliers", "Knicks"]',
                        "outcomePrices": '["0.4", "0.6"]',
                        "volume24hr": 9999999.0,
                        "active": True,
                        "closed": False,
                    }
                ],
            },
        ],
        "China": [
            {
                "id": "ev-tariff",
                "slug": "us-tariff-rate-on-china-dec-2026",
                "title": "U.S. tariff rate on China on December 31, 2026?",
                "active": True,
                "closed": False,
                "volume24hr": 21842.0,
                "liquidity": 45000.0,
                "endDate": "2026-12-31T00:00:00Z",
                "markets": [
                    {
                        "question": "Will U.S. tariff rate on China exceed 50%?",
                        "groupItemTitle": ">50%",
                        "outcomes": '["Yes", "No"]',
                        "outcomePrices": '["0.44", "0.56"]',
                        "oneDayPriceChange": -0.03,
                        "volume24hr": 15000.0,
                        "active": True,
                        "closed": False,
                    }
                ],
            },
            {
                "id": "ev-taiwan-noise",
                "slug": "will-china-invade-taiwan-before-2027",
                "title": "Will China invade Taiwan before 2027?",
                "active": True,
                "closed": False,
                "volume24hr": 90000.0,
                "liquidity": 50000.0,
                "markets": [
                    {
                        "question": "Will China invade Taiwan before 2027?",
                        "outcomes": '["Yes", "No"]',
                        "outcomePrices": '["0.02", "0.98"]',
                        "volume24hr": 90000.0,
                        "active": True,
                        "closed": False,
                    }
                ],
            },
        ],
        "Nasdaq": [
            {
                "id": "ev-ndx",
                "slug": "ndx-close-dec-2026",
                "title": "What will Nasdaq 100 (NDX) close at in December?",
                "active": True,
                "closed": False,
                "volume24hr": 18500.0,
                "liquidity": 32000.0,
                "endDate": "2026-12-31T00:00:00Z",
                "markets": [
                    {
                        "question": "Nasdaq 100 above 28,000?",
                        "groupItemTitle": ">28,000",
                        "outcomes": '["Yes", "No"]',
                        "outcomePrices": '["0.71", "0.29"]',
                        "oneDayPriceChange": 0.04,
                        "volume24hr": 12000.0,
                        "active": True,
                        "closed": False,
                    }
                ],
            }
        ],
    }


def test_refresh_polymarket_filters_caches_and_feeds_chat(tmp_path: Path, monkeypatch):
    sample = _sample_events()
    calls = {"n": 0}

    def fake_search(query: str, timeout: float = 8.0):
        calls["n"] += 1
        return sample.get(query, []), ""

    monkeypatch.setattr(polymarket, "_fetch_search", fake_search)

    res = polymarket.refresh_polymarket(tmp_path, force=False)
    assert res["available"] is True
    ids = {it["id"] for it in res["items"]}
    assert ids == {"ev-fed", "ev-tariff", "ev-ndx"}
    assert "ev-nba" not in ids
    assert "ev-taiwan-noise" not in ids

    by_id = {it["id"]: it for it in res["items"]}
    assert by_id["ev-fed"]["category"] == "macro"
    assert by_id["ev-fed"]["lead_prob"] == 0.625
    assert by_id["ev-fed"]["delta_24h"] == 0.085
    assert by_id["ev-tariff"]["category"] == "china_tariff"
    assert by_id["ev-ndx"]["category"] == "us_index"

    # Cooldown prevents duplicate upstream calls
    prev_calls = calls["n"]
    res_cached = polymarket.refresh_polymarket(tmp_path, force=False, cooldown_sec=900)
    assert res_cached["items"] == res["items"]
    assert calls["n"] == prev_calls

    # News bundle & group chat context pick up cached Polymarket odds
    today = datetime.now(news_feed.SHANGHAI).date()
    bars = tuple(Bar(today - timedelta(days=30 - i), 20000.0 + i * 10.0, is_closed=True) for i in range(30))
    upsert_series(tmp_path, Series("^NDX", "America/New_York", "2026-11-25T21:00:00-05:00", bars))

    bundle = news_feed.load_news_bundle(tmp_path, market="ndx")
    assert bundle["polymarket"]["available"] is True
    assert len(bundle["polymarket"]["items"]) == 3

    ctx = build_market_context(tmp_path, "ndx")
    assert "7. 预测市场赔率参考 (Polymarket)" in ctx["text"]
    assert "Fed Decision in December?" in ctx["text"]
    assert "62.5%" in ctx["text"]
    assert "Δ24h +8.5%" in ctx["text"]
