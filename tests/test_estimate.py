from datetime import date, timedelta

import numpy as np
import pytest

from forecast_log.bars import Bar
from forecast_log.estimate import estimate_ndx, estimate_vix, rolling_sample_std


def bars_from_closes(closes, start=date(2000, 1, 3)):
    day = start
    out = []
    for close in closes:
        out.append(Bar(day, float(close)))
        day += timedelta(days=1)
    return out


def test_vix_band_uses_only_nearby_starts_and_strict_greater_than():
    bars = bars_from_closes([10, 11, 10, 12, 30])
    estimate = estimate_vix(bars, bars[3].date, horizon=2, threshold=20, min_n=1)
    assert estimate.rule == "band"
    assert estimate.sample_size == 1
    assert estimate.probability == 0.0
    assert estimate.median_close == 12.0


def test_vix_equal_to_threshold_is_not_yes():
    bars = bars_from_closes([18, 18, 20, 18])
    estimate = estimate_vix(bars, bars[2].date, horizon=1, threshold=20, band=0.10, min_n=1)
    assert estimate.probability == 0.0


def test_vix_estimate_drops_bars_after_as_of():
    bars = bars_from_closes([18.0] * 40 + [17.5])
    as_of = bars[-1].date
    later = bars + [Bar(as_of + timedelta(days=1), 80.0)]
    before = estimate_vix(bars, as_of, horizon=5, threshold=20, min_n=5)
    after = estimate_vix(later, as_of, horizon=5, threshold=20, min_n=5)
    assert before == after
    assert before.probability == 0.0


def test_ndx_flat_prices_need_a_strict_gain():
    bars = bars_from_closes([10.0] * 40)
    as_of = bars[-1].date
    at_the_line = estimate_ndx(bars, as_of, horizon=5, baseline=10, vol_window=5, min_n=1)
    below_the_line = estimate_ndx(bars, as_of, horizon=5, baseline=9, vol_window=5, min_n=1)
    assert at_the_line.probability == 0.0
    assert below_the_line.probability == 1.0


def test_ndx_estimate_drops_bars_after_as_of():
    bars = bars_from_closes([100.0 + (i % 3) for i in range(40)])
    as_of = bars[-1].date
    later = bars + [Bar(as_of + timedelta(days=1), 250.0)]
    assert estimate_ndx(bars, as_of, horizon=5, vol_window=5, min_n=5) == estimate_ndx(
        later, as_of, horizon=5, vol_window=5, min_n=5
    )


def test_rolling_sample_std_matches_numpy():
    returns = np.array([0.01, -0.02, 0.03, 0.0, 0.02, -0.01])
    got = rolling_sample_std(returns, window=3)
    for end in range(3, len(returns) + 1):
        expected = float(np.std(returns[end - 3 : end], ddof=1))
        assert got[end] == pytest.approx(expected)
