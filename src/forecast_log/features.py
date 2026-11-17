"""Forward-looking inputs: the CBOE volatility complex around the fear index.

The settlement price stays `^VXN` (see questions.json). These series are inputs only.
They are the only data in the repo that looks ahead of the fear index:

- the short end of the S&P variance curve, `^VIX9D / ^VIX3M`. Backwardation is the
  documented leading state, and it correlates -0.29 with the next 7 sessions of VXN
  before anything else is conditioned on.
- `^VVIX`, the volatility of VIX, relative to its own one-year midpoint.

^VIX9D starts 2011-01-03, so a model with these inputs starts around 2012.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from forecast_log.bars import Series, has_series, load_series

FEATURE_SYMBOLS = ("^VIX9D", "^VIX3M", "^VVIX")
LONG_LOOKBACK = 252


def missing(root: Path) -> list[str]:
    return [symbol for symbol in FEATURE_SYMBOLS if not has_series(root, symbol)]


def _aligned(series: Series, dates: list[date]) -> np.ndarray:
    """Last close at or before each date. Never reads ahead."""
    out = np.full(len(dates), np.nan)
    closes = [(bar.date, bar.close) for bar in series.bars]
    pointer = 0
    last = np.nan
    for position, day in enumerate(dates):
        while pointer < len(closes) and closes[pointer][0] <= day:
            last = closes[pointer][1]
            pointer += 1
        out[position] = last
    return out


def _rolling_median(values: np.ndarray, window: int = LONG_LOOKBACK) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    out[window - 1 :] = np.median(windows, axis=1)
    return out


def load_matrix(root: Path, dates: list[date]) -> np.ndarray | None:
    """Two features per date, or None when the files have not been fetched."""
    if missing(root):
        return None
    short = _aligned(load_series(root, "^VIX9D"), dates)
    long = _aligned(load_series(root, "^VIX3M"), dates)
    vol_of_vol = _aligned(load_series(root, "^VVIX"), dates)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.log(short / long)
        relative = np.log(vol_of_vol / _rolling_median(vol_of_vol))
    return np.column_stack([slope, relative])


def describe() -> str:
    return "期限结构 ^VIX9D/^VIX3M + ^VVIX 相对 1 年中位"
