from datetime import date, timedelta

import numpy as np
import pytest

from forecast_log.bars import Bar
from forecast_log.estimate import (
    LONG_LOOKBACK,
    MIN_CANDIDATES,
    AnalogModel,
    estimate,
    min_index,
)


def bars_from_closes(closes, start=date(2000, 1, 3)):
    day = start
    out = []
    for close in closes:
        out.append(Bar(day, float(close)))
        day += timedelta(days=1)
    return out


def wave(count=1200, period=120.0, low=18.0, high=26.0):
    """Mean-reverting closes: a smooth cycle in log space."""
    t = np.arange(count)
    shape = (1 - np.cos(2 * np.pi * t / period)) / 2
    return low * np.exp(np.log(high / low) * shape)


def test_state_vector_only_uses_the_past():
    closes = wave()
    model = AnalogModel(closes)
    spike = closes.copy()
    spike[-1] *= 2.0
    bumped = AnalogModel(spike)
    assert np.allclose(model.state[:-1], bumped.state[:-1])
    assert not np.allclose(model.state[-1], bumped.state[-1])


def test_forecast_is_anchored_at_the_as_of_close():
    closes = wave()
    index = len(closes) - 1
    result = AnalogModel(closes).forecast(index, 7, 30.0, with_path=True)
    assert result.as_of_close == closes[index]
    assert result.path[0].low <= result.path[0].median <= result.path[0].high
    assert len(result.path) == 7
    assert [point.horizon for point in result.path] == list(range(1, 8))
    # The last path point is the headline level, so the path and the number cannot drift apart.
    assert result.path[-1].median == pytest.approx(result.median_close)
    assert 0.0 <= result.probability <= 1.0


def test_forecast_leads_instead_of_restating_the_past_value():
    """On a cycle a peak must project down, and the projected change must follow the real one.

    This is the regression test for the old estimator: it reported the median *level* of past
    analogues, so the "forecast" was the actual series delayed by the horizon.
    """
    horizon = 20
    closes = wave(count=1200, period=120.0)
    model = AnalogModel(closes)

    peak = int(np.argmax(closes[-120:])) + len(closes) - 120
    result = model.forecast(peak, horizon, 1e9, with_path=True)
    assert closes[peak] == closes.max()
    assert result.path[-1].median < closes[peak] * 0.97

    predicted = []
    realized = []
    for index in range(min_index(horizon), len(closes) - horizon, 3):
        forecast = model.forecast(index, horizon, 1e9)
        predicted.append(np.log(forecast.median_close / closes[index]))
        realized.append(np.log(closes[index + horizon] / closes[index]))
    predicted = np.array(predicted)
    realized = np.array(realized)
    assert predicted.size > 100
    assert np.mean(np.sign(predicted) == np.sign(realized)) > 0.6
    assert np.corrcoef(predicted, realized)[0, 1] > 0.4


def test_probability_uses_a_strict_greater_than():
    closes = np.full(500, 18.0)
    result = AnalogModel(closes).forecast(len(closes) - 1, 7, 18.0)
    assert result.rule == "all_analogues" or result.rule == "analog"
    assert result.probability == 0.0


def test_threshold_below_the_current_level_is_almost_certain():
    closes = wave()
    result = AnalogModel(closes).forecast(len(closes) - 1, 7, 1.0)
    assert result.probability == 1.0


def test_estimate_drops_bars_after_as_of():
    bars = bars_from_closes(wave(count=600))
    as_of = bars[-1].date
    later = bars + [Bar(as_of + timedelta(days=1), 90.0)]
    assert estimate(bars, as_of, 5, 30.0) == estimate(later, as_of, 5, 30.0)


def test_too_little_history_is_refused():
    bars = bars_from_closes(wave(count=LONG_LOOKBACK + MIN_CANDIDATES))
    with pytest.raises(ValueError):
        estimate(bars, bars[-1].date, 7, 30.0)
    assert min_index(7) > len(bars) - 1


def test_unknown_as_of_is_refused():
    bars = bars_from_closes(wave(count=600))
    with pytest.raises(ValueError):
        estimate(bars, date(1999, 1, 4), 5, 30.0)
