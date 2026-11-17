import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from forecast_log.bars import Bar, Series, last_completed_bar, save_series
from forecast_log.yahoo import _assert_daily
from forecast_log.commands import lock_command, resolve_command, score_command
from forecast_log.horizon import sessions_after, sessions_from_as_of_through_resolve
from forecast_log.paths import series_path
from forecast_log.questions import load_questions

NY = ZoneInfo("America/New_York")


def test_monthly_history_is_rejected():
    monthly = tuple(Bar(date(2000, 1, 1) + timedelta(days=31 * i), 18.0) for i in range(12))
    with pytest.raises(RuntimeError):
        _assert_daily("^VXN", monthly)


def test_live_horizons_match_the_question_file():
    assert sessions_from_as_of_through_resolve(date(2026, 11, 11), date(2026, 11, 20)) == 7
    assert sessions_from_as_of_through_resolve(date(2026, 11, 11), date(2026, 11, 30)) == 12
    after = sessions_after(date(2026, 11, 11), 12)
    assert after[0] == date(2026, 11, 12)
    assert after[6] == date(2026, 11, 20)
    assert after[11] == date(2026, 11, 30)
    assert date(2026, 11, 26) not in after


def test_questions_are_strictly_greater_than():
    from forecast_log.paths import repo_root

    questions = load_questions(repo_root() / "questions.json")
    assert [question.id for question in questions] == [
        "vxn-2026-11-20-gt-30",
        "vxn-2026-11-30-gt-30",
        "ndx-2026-11-30-gt-baseline",
    ]


def test_same_day_bar_is_ignored_before_the_close():
    bars = (
        Bar(date(2026, 11, 11), 17.49),
        Bar(date(2026, 11, 12), 18.0),
    )
    series = Series("^VXN", "America/New_York", "t", bars)
    morning = datetime(2026, 11, 12, 10, 0, tzinfo=NY)
    evening = datetime(2026, 11, 12, 16, 15, tzinfo=NY)
    assert last_completed_bar(series, morning).date == date(2026, 11, 11)
    assert last_completed_bar(series, evening).date == date(2026, 11, 12)


def _write_cache(root, symbol, closes, start):
    day = start
    bars = []
    for close in closes:
        bars.append(Bar(day, float(close)))
        day += timedelta(days=1)
    save_series(
        series_path(root, symbol),
        Series(symbol, "America/New_York", "2026-11-12T12:00:00-05:00", tuple(bars)),
    )
    return bars


def test_lock_writes_once_and_score_waits_for_settlement(tmp_path):
    questions = {
        "questions": [
            {
                "id": "vxn-demo",
                "resolve_session": "2026-11-20",
                "symbol": "^VXN",
                "comparison": "gt",
                "threshold": 30,
                "yes_when": "^VXN daily Close > 20",
            }
        ]
    }
    (tmp_path / "questions.json").write_text(json.dumps(questions))
    # Long enough for the analogue model: a year of state history plus 30 finished analogues.
    _write_cache(tmp_path, "^VXN", [18.0] * 400, date(2025, 9, 6))
    _write_cache(tmp_path, "^NDX", [100.0 + (i % 5) for i in range(400)], date(2025, 9, 6))
    # Last cached session is 2026-10-10. Freeze "now" after that close and before resolution.
    now = datetime(2026, 10, 11, 12, 0, tzinfo=NY)
    written = lock_command(tmp_path, now=now)
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["evidence"][-1]["date"] <= payload["as_of_session"]
    with pytest.raises(SystemExit):
        lock_command(tmp_path, now=now)
    assert score_command(tmp_path) is None


def test_resolution_keeps_the_first_close(tmp_path, monkeypatch):
    questions = {
        "questions": [
            {
                "id": "vxn-demo",
                "resolve_session": "2026-11-11",
                "symbol": "^VXN",
                "comparison": "gt",
                "threshold": 30,
                "yes_when": "^VXN daily Close > 20",
            }
        ]
    }
    (tmp_path / "questions.json").write_text(json.dumps(questions))
    calls = {"n": 0}

    def fake_fetch(symbol):
        calls["n"] += 1
        close = 17.49 if calls["n"] == 1 else 30.0
        return Series(
            symbol,
            "America/New_York",
            f"fetch-{calls['n']}",
            (
                Bar(date(2026, 11, 10), 18.0),
                Bar(date(2026, 11, 11), close),
                Bar(date(2026, 11, 12), 18.2),
            ),
        )

    monkeypatch.setattr("forecast_log.commands.fetch_daily", fake_fetch)
    now = datetime(2026, 11, 12, 12, 0, tzinfo=NY)
    resolve_command(tmp_path, now=now)
    resolve_command(tmp_path, now=now)
    document = json.loads((tmp_path / "resolutions.json").read_text())
    assert document["resolutions"] == [
        {
            "question_id": "vxn-demo",
            "resolve_session": "2026-11-11",
            "symbol": "^VXN",
            "close": 17.49,
            "threshold": 30.0,
            "outcome": 0,
            "fetched_at": "fetch-1",
        }
    ]
    assert calls["n"] == 1


def test_missing_session_is_not_replaced_with_another_day(tmp_path, monkeypatch):
    questions = {
        "questions": [
            {
                "id": "vxn-demo",
                "resolve_session": "2026-11-11",
                "symbol": "^VXN",
                "comparison": "gt",
                "threshold": 30,
                "yes_when": "x",
            }
        ]
    }
    (tmp_path / "questions.json").write_text(json.dumps(questions))

    def fake_fetch(symbol):
        return Series(
            symbol,
            "America/New_York",
            "fetch",
            (Bar(date(2026, 11, 10), 18.0), Bar(date(2026, 11, 12), 19.0)),
        )

    monkeypatch.setattr("forecast_log.commands.fetch_daily", fake_fetch)
    resolve_command(tmp_path, now=datetime(2026, 11, 13, 12, 0, tzinfo=NY))
    document = json.loads((tmp_path / "resolutions.json").read_text())
    assert document["resolutions"] == []
    assert document["failures"][0]["reason"] == "no daily close for the resolve session"


def test_unclosed_intraday_bar_is_refetched_and_locked_after_close(tmp_path, monkeypatch):
    from forecast_log.bars import is_session_closed
    from forecast_log.db import get_series, unclosed_bars_summary, upsert_series
    from forecast_log.yahoo import fetch_incremental

    intraday_at = "2026-11-16T21:53:45-05:00"
    assert not is_session_closed(date(2026, 11, 17), "Asia/Shanghai", intraday_at)

    base_bars = [
        Bar(date(2026, 11, 10) + timedelta(days=i), 3600.0 + i, open=3595.0, high=3610.0, low=3590.0, is_closed=True)
        for i in range(6)
    ]
    intraday_bar = Bar(
        date(2026, 11, 17),
        3574.29,
        open=3585.67,
        high=3586.25,
        low=3565.59,
        volume=1000,
        is_closed=False,
    )
    upsert_series(
        tmp_path,
        Series("000001.SS", "Asia/Shanghai", intraday_at, tuple([*base_bars, intraday_bar])),
    )
    assert unclosed_bars_summary(tmp_path) == {"000001.SS": ["2026-11-17"]}

    post_close_at = "2026-11-17T02:30:00-05:00"
    assert is_session_closed(date(2026, 11, 17), "Asia/Shanghai", post_close_at)

    captured_starts = []

    def fake_fetch_daily(symbol, timeout=60, start=None):
        captured_starts.append(start)
        closed_bar = Bar(
            date(2026, 11, 17),
            3592.88,
            open=3585.67,
            high=3598.00,
            low=3565.59,
            volume=5000,
            is_closed=True,
        )
        return Series(symbol, "Asia/Shanghai", post_close_at, tuple([*base_bars[-3:], closed_bar]))

    monkeypatch.setattr("forecast_log.yahoo.fetch_daily", fake_fetch_daily)

    existing = get_series(tmp_path, "000001.SS")
    assert existing is not None
    assert not existing.bars[-1].is_closed

    refreshed = fetch_incremental("000001.SS", existing)
    upsert_series(tmp_path, refreshed)

    assert captured_starts[0] <= date(2026, 11, 12)
    final_series = get_series(tmp_path, "000001.SS")
    assert final_series.bars[-1].close == 3592.88
    assert final_series.bars[-1].volume == 5000
    assert final_series.bars[-1].is_closed is True
    assert unclosed_bars_summary(tmp_path) == {}

