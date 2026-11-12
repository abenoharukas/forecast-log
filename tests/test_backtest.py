from datetime import date, timedelta

import numpy as np

from forecast_log.backtest import (
    MIN_HISTORY,
    _climatology,
    _rows_vix,
    brier,
    forward_levels,
    timeline_document,
)
from forecast_log.bars import Bar, Series


def _series(closes):
    day = date(2010, 1, 4)
    bars = []
    for close in closes:
        bars.append(Bar(day, float(close)))
        day += timedelta(days=1)
    return Series("^VIX", "America/New_York", "t", tuple(bars))


def test_brier_of_one_half_is_a_quarter():
    assert brier([0.5, 0.5, 0.5], [0, 1, 1]) == 0.25


def test_backtest_probability_does_not_see_the_later_close():
    rng = np.random.default_rng(0)
    closes = np.clip(18 + rng.normal(0, 0.4, size=MIN_HISTORY + 40), 12, 28)
    series = _series(closes)
    rows = _rows_vix(series, horizon=5, threshold=20)
    spiked = Series(
        series.symbol,
        series.timezone,
        series.fetched_at,
        series.bars + (Bar(series.bars[-1].date + timedelta(days=1), 90.0),),
    )
    again = _rows_vix(spiked, horizon=5, threshold=20)
    by_date = {row["as_of"]: row["probability"] for row in again}
    assert rows
    for row in rows:
        assert by_date[row["as_of"]] == row["probability"]
    assert {row["outcome"] for row in rows}.issubset({0, 1})


def test_timeline_columns_stay_aligned():
    series = _series([18.0, 19.0, 17.0])
    sample = [
        {
            "as_of": "2010-01-04",
            "resolve": "2010-01-08",
            "probability": 0.15678,
            "outcome": 0,
            "median_close": 18.129,
            "actual_close": 17.5,
        }
    ]
    rows = {key: sample for key in ("vix-h7-gt-20", "vix-h12-gt-20", "ndx-h12-gt-asof")}
    document = timeline_document(series, series, rows)
    forecast = document["questions"]["vix-h7-gt-20"]["forecast"]
    assert document["vix"]["time"] == ["2010-01-04", "2010-01-05", "2010-01-06"]
    assert len(forecast["time"]) == len(forecast["probability"]) == len(forecast["median"])
    assert forecast["probability"] == [0.1568]
    assert forecast["median"] == [18.13]
    assert document["questions"]["ndx-h12-gt-asof"]["threshold"] is None
    assert document["questions"]["vix-h7-gt-20"]["forward"] == []


def test_forward_path_is_only_after_the_last_close():
    day = date(2025, 1, 2)
    bars = []
    level = 17.2
    while day <= date(2026, 11, 11):
        if day.weekday() < 5:
            bars.append(Bar(day, level))
            level = 16.4 if level > 18.4 else level + 0.07
        day += timedelta(days=1)
    series = Series("^VIX", "America/New_York", "t", tuple(bars))
    forward = forward_levels(series, 12, "vix", 20.0)
    assert [point["time"] for point in forward] == [
        "2026-11-12",
        "2026-11-13",
        "2026-11-16",
        "2026-11-17",
        "2026-11-18",
        "2026-11-19",
        "2026-11-20",
        "2026-11-23",
        "2026-11-24",
        "2026-11-25",
        "2026-11-27",
        "2026-11-30",
    ]
    assert all(point["low"] <= point["median"] <= point["high"] for point in forward)
    later = Series(
        series.symbol,
        series.timezone,
        series.fetched_at,
        series.bars + (Bar(date(2026, 11, 12), 40.0),),
    )
    assert forward_levels(series, 7, "vix", 20.0)[0]["median"] != forward_levels(
        later, 7, "vix", 20.0
    )[0]["median"]


def test_climatology_waits_until_the_outcome_session():
    rows = [
        {"as_of": "2020-01-02", "resolve": "2020-01-10", "outcome": 1},
        {"as_of": "2020-01-03", "resolve": "2020-01-13", "outcome": 0},
    ]
    assert _climatology(rows) == [None, None]
    matured = rows + [{"as_of": "2020-01-10", "resolve": "2020-01-20", "outcome": 0}]
    # One resolved outcome is below the minimum of 30, so the baseline stays unset.
    assert _climatology(matured) == [None, None, None]
