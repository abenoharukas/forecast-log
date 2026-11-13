from datetime import date, timedelta

import numpy as np

from forecast_log.backtest import (
    _climatology,
    _rows_vxn,
    brier,
    forward_levels,
    timeline_document,
)
from forecast_log.bars import Bar, Series
from forecast_log.estimate import min_index


def _series(closes):
    day = date(2010, 1, 4)
    bars = []
    for close in closes:
        bars.append(Bar(day, float(close)))
        day += timedelta(days=1)
    return Series("^VXN", "America/New_York", "t", tuple(bars))


def test_brier_of_one_half_is_a_quarter():
    assert brier([0.5, 0.5, 0.5], [0, 1, 1]) == 0.25


def test_backtest_probability_does_not_see_the_later_close():
    rng = np.random.default_rng(0)
    closes = np.clip(18 + rng.normal(0, 0.4, size=min_index(5) + 240), 12, 28)
    series = _series(closes)
    rows = _rows_vxn(series, horizon=5, threshold=20)
    spiked = Series(
        series.symbol,
        series.timezone,
        series.fetched_at,
        series.bars + (Bar(series.bars[-1].date + timedelta(days=1), 90.0),),
    )
    again = _rows_vxn(spiked, horizon=5, threshold=20)
    by_date = {row["as_of"]: row["probability"] for row in again}
    assert rows
    for row in rows:
        assert by_date[row["as_of"]] == row["probability"]
    assert {row["outcome"] for row in rows}.issubset({0, 1})


def test_daily_forecast_predicts_each_session_from_the_one_before():
    from forecast_log.backtest import daily_forecasts

    series = _series(np.clip(18 + np.sin(np.arange(600) / 9.0) * 3, 10, 30))
    daily = daily_forecasts(series)
    rows = daily["rows"]
    assert rows
    first = rows[0]
    position = [bar.date.isoformat() for bar in series.bars].index(first["time"])
    # Each row predicts its own session from the session before it.
    assert first["previous"] == series.bars[position - 1].close
    assert first["actual"] == series.bars[position].close
    assert rows[-1]["time"] == series.bars[-1].date.isoformat()
    assert first["low"] <= first["median"] <= first["high"]
    assert 0 <= first["risk"] <= 100
    skill = daily["skill"]
    assert 0.5 <= skill["band_coverage"] <= 1.0
    assert skill["baseline_mae"] > 0


def test_timeline_columns_stay_aligned():
    series = _series([18.0, 19.0, 17.0])
    sample = [
        {
            "as_of": "2010-01-04",
            "resolve": "2010-01-08",
            "as_of_close": 18.0,
            "probability": 0.15678,
            "outcome": 0,
            "median_close": 18.129,
            "actual_close": 17.5,
        }
    ]
    rows = {key: sample for key in ("vxn-h7-gt-30", "vxn-h12-gt-30", "ndx-h12-gt-asof")}
    document = timeline_document(series, series, rows)
    forecast = document["questions"]["vxn-h7-gt-30"]["forecast"]
    assert document["vxn"]["time"] == ["2010-01-04", "2010-01-05", "2010-01-06"]
    assert len(forecast["time"]) == len(forecast["probability"]) == len(forecast["median"])
    assert forecast["probability"] == [0.1568]
    assert forecast["median"] == [18.13]
    assert document["questions"]["ndx-h12-gt-asof"]["threshold"] is None
    assert document["questions"]["vxn-h7-gt-30"]["forward"] == []


def test_forward_path_is_only_after_the_last_close():
    day = date(2025, 1, 2)
    bars = []
    level = 17.2
    while day <= date(2026, 11, 11):
        if day.weekday() < 5:
            bars.append(Bar(day, level))
            level = 16.4 if level > 18.4 else level + 0.07
        day += timedelta(days=1)
    series = Series("^VXN", "America/New_York", "t", tuple(bars))
    forward = forward_levels(series, 12, 30.0)
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
    assert forward_levels(series, 7, 30.0)[0]["median"] != forward_levels(
        later, 7, 30.0
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


def test_rolling_path_never_reads_the_sessions_it_is_scored_on():
    from forecast_log.backtest import PATH_HORIZON, rolling_paths

    rng = np.random.default_rng(3)
    closes = np.clip(19 + rng.normal(0, 0.5, size=min_index(PATH_HORIZON) + 300), 12, 30)
    series = _series(closes)
    rows = rolling_paths(series)
    assert rows
    # Overwrite every close after the middle day: forecasts made before it cannot move.
    cut = len(series.bars) // 2
    shattered = tuple(
        Bar(bar.date, 90.0 if position > cut else bar.close)
        for position, bar in enumerate(series.bars)
    )
    again = {row["as_of"]: row for row in rolling_paths(Series("^VXN", "America/New_York", "t", shattered))}
    for row in rows:
        if row["resolve"] is not None and row["resolve"] <= series.bars[cut].date.isoformat():
            assert again[row["as_of"]]["path"] == row["path"]


def test_rolling_path_scores_against_the_close_that_printed():
    from forecast_log.backtest import PATH_HORIZON, path_document, path_summary, rolling_paths

    series = _series(np.clip(18 + np.sin(np.arange(min_index(7) + 300) / 9.0) * 3, 10, 30))
    rows = rolling_paths(series)
    dates = [bar.date.isoformat() for bar in series.bars]
    first = rows[0]
    assert first["as_of"] == dates[dates.index(first["as_of"])]
    assert first["resolve"] == dates[dates.index(first["as_of"]) + PATH_HORIZON]
    assert first["actual"][-1] == series.bars[dates.index(first["resolve"])].close
    assert len(first["path"]) == PATH_HORIZON
    assert all(point["low"] <= point["median"] <= point["high"] for point in first["path"])
    summary = path_summary(rows)
    # The newest rows run past the last close, so the scored sample is shorter than the paths.
    assert summary["n"] == sum(1 for row in rows if row["actual"][-1] is not None)
    assert 0 < summary["n"] < len(rows)
    assert [step["step"] for step in summary["steps"]] == list(range(1, PATH_HORIZON + 1))
    document = path_document(rows)
    assert (
        len(document["time"])
        == len(document["median"])
        == len(document["low"])
        == summary["n"]
    )
    assert document["summary"]["terminal"]["step"] == PATH_HORIZON
    assert 0.0 <= document["summary"]["terminal"]["coverage"] <= 1.0


def test_direction_probability_never_reads_ahead():
    from forecast_log.direction import walk_forward

    rng = np.random.default_rng(7)
    closes = np.clip(3000 * np.cumprod(1 + rng.normal(0.0004, 0.012, size=900)), 500, 12000)
    series = _series(closes)
    matrix = np.column_stack(
        [np.log(np.clip(14 + rng.normal(0, 1, size=len(closes)), 8, 30)),
         np.log(np.clip(90 + rng.normal(0, 6, size=len(closes)), 60, 140))]
    )
    rows = walk_forward(series, matrix)
    assert rows
    cut = len(series.bars) // 2
    shattered = Series(
        series.symbol,
        series.timezone,
        series.fetched_at,
        tuple(
            Bar(bar.date, 9000.0 if position > cut else bar.close)
            for position, bar in enumerate(series.bars)
        ),
    )
    again = {row.date: row.probability for row in walk_forward(shattered, matrix)}
    for row in rows:
        if row.date <= series.bars[cut].date.isoformat():
            assert again[row.date] == row.probability


def test_direction_outcome_is_five_sessions_later():
    from forecast_log.direction import HORIZON, walk_forward

    closes = [100.0] * 400 + [100.0 + position for position in range(1, 400)]
    series = _series(closes)
    rows = walk_forward(series, None)
    assert rows
    first = rows[0]
    dates = [bar.date.isoformat() for bar in series.bars]
    position = dates.index(first.date)
    assert first.target == dates[position + HORIZON]
    for row in rows:
        if row.outcome is not None:
            spot = dates.index(row.date)
            expected = 1 if series.bars[spot + HORIZON].close > series.bars[spot].close else 0
            assert row.outcome == expected
