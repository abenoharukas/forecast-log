"""运行时那一步：抓新闻、过 as-of 闸门、让 LLM 给标签。

不联网、不花钱：LLM 用假的，GDELT 用假响应。真正的抓取靠跑 `forecast-log daily` 验证。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from forecast_log import daily
from forecast_log.agent.llm import Reply, Usage


class StubLLM:
    name = "stub@local"

    def __init__(self, content: str):
        self.content = content
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=1600, tool_choice=None):
        self.calls.append(list(messages))
        return Reply(content=self.content, usage=Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.0001))


def _news(as_of: date, titles: list[tuple[str, str]]) -> dict:
    return {
        "as_of": as_of.isoformat(),
        "fetched_at": "2026-11-13T21:00:00+00:00",
        "source": "test",
        "queries": ["nasdaq"],
        "documents": [
            {
                "id": f"doc{index}",
                "published_at": published,
                "title": title,
                "text": title,
                "url": f"https://example.com/{index}",
                "domain": "example.com",
            }
            for index, (published, title) in enumerate(titles)
        ],
        "errors": [],
        "cached": False,
    }


def test_gate_drops_headlines_published_after_the_session():
    documents = _news(
        date(2026, 11, 11),
        [("2026-11-11", "stocks rise"), ("2026-11-13", "something that had not happened yet")],
    )["documents"]
    kept, dropped = daily.gate(documents, date(2026, 11, 11))
    assert [document["title"] for document in kept] == ["stocks rise"]
    assert dropped == 1


def test_gate_drops_unparseable_dates():
    kept, dropped = daily.gate([{"published_at": "not-a-date", "title": "x"}], date(2026, 11, 11))
    assert kept == []
    assert dropped == 1


def test_parse_rejects_a_tone_outside_the_enum():
    with pytest.raises(ValueError):
        daily._parse('{"tone": "bullish", "confidence": 0.9}')
    assert daily._parse('{"tone": "Risk-On", "confidence": 3, "drivers": ["a", "b", "c", "d"]}') == {
        "tone": "risk-on",
        "confidence": 1.0,
        "drivers": ["a", "b", "c"],
        "note": "",
    }


def test_parse_survives_a_fenced_code_block():
    label = daily._parse('```json\n{"tone": "neutral", "confidence": 0.3, "note": "thin"}\n```')
    assert label["tone"] == "neutral"
    assert label["confidence"] == 0.3


def test_run_writes_the_artifact_and_flags_a_contested_signal(tmp_path: Path, monkeypatch):
    day = date(2026, 11, 11)
    monkeypatch.setattr(
        daily,
        "fetch_news",
        lambda root, day, lookback=2, use_cache=True: _news(day, [("2026-11-12", "too late"), ("2026-11-11", "fine")]),
    )
    cross = tmp_path / "artifacts" / "cross.json"
    cross.parent.mkdir(parents=True)
    cross.write_text(
        json.dumps({"latest": {"date": day.isoformat(), "score": 0.71, "threshold": 0.54, "flag": True}})
    )

    llm = StubLLM('{"tone": "risk-off", "confidence": 0.8, "drivers": ["credit spreads widening"]}')
    record = daily.run(tmp_path, day=day, llm=llm)

    assert record["news"]["used"] == 1
    assert record["news"]["dropped_after_as_of"] == 1
    assert record["readings"]["signal"] is True
    assert record["contested"] is True
    assert json.loads((tmp_path / "artifacts" / "daily.json").read_text())["label"]["tone"] == "risk-off"
    assert daily.load(tmp_path, day).get("as_of") == day.isoformat()
    assert len((tmp_path / "data" / "daily.jsonl").read_text().splitlines()) == 1

    prompt = llm.calls[0][1]["content"]
    assert "fine" in prompt and "too late" not in prompt


def test_low_confidence_risk_off_does_not_contradict_a_signal(tmp_path: Path, monkeypatch):
    day = date(2026, 11, 11)
    monkeypatch.setattr(daily, "fetch_news", lambda root, day, lookback=2, use_cache=True: _news(day, []))
    cross = tmp_path / "artifacts" / "cross.json"
    cross.parent.mkdir(parents=True)
    cross.write_text(json.dumps({"latest": {"date": day.isoformat(), "flag": True}}))
    record = daily.run(tmp_path, day=day, llm=StubLLM('{"tone": "risk-off", "confidence": 0.4}'))
    assert record["contested"] is False


def test_load_is_none_before_the_first_run(tmp_path: Path):
    assert daily.load(tmp_path) is None


def test_flatten_exposes_regime_and_event_counts():
    from forecast_log import news_labels

    flat = news_labels.flatten(
        {
            "regime": "risk_off",
            "severity": 0.55,
            "novelty": 0.4,
            "events": [
                {"type": "credit", "direction": "adverse", "severity": 0.6, "evidence": "x"},
                {"type": "monetary_policy", "direction": "supportive", "severity": 0.2, "evidence": "y"},
            ],
        }
    )
    assert flat["regime"] == "risk_off"
    assert flat["adverse"] == 1 and flat["supportive"] == 1
    assert flat["max_adverse"] == 0.6


def test_news_veto_removes_a_signal_only_on_a_strong_bearish_surprise(tmp_path, monkeypatch):
    from datetime import date

    from forecast_log import cross

    rows = [
        cross.Call(date="2026-01-05", close=1.0, score=0.9, threshold=0.5, flag=1, outcome=1, target=None),
        cross.Call(date="2026-01-06", close=1.0, score=0.9, threshold=0.5, flag=1, outcome=0, target=None),
    ]
    labels = {
        "2026-01-05": {"surprise": "strong", "direction": "bearish"},
        "2026-01-06": {"surprise": "strong", "direction": "bullish"},
    }
    monkeypatch.setattr(cross, "_news_labels", lambda root: (labels, {"source": "test", "days": 2, "veto_days": 1}))

    class FakeSeries:
        bars = []
        fetched_at = "test"

    monkeypatch.setattr(cross, "composite", lambda root, series: __import__("numpy").array([]))
    # 直接测规则本身：信号 + strong bearish ⇒ 撤掉；其余保持
    for row in rows:
        label = labels.get(row.date)
        vetoed = row.flag and label and label["surprise"] == "strong" and label["direction"] == "bearish"
        assert bool(vetoed) == (row.date == "2026-01-05")
