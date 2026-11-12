from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from forecast_log.bars import Bar

# Fixed before looking at the backtest score. Not tuned to these three questions.
VIX_BAND = 0.10
VIX_MIN_N = 30
NDX_VOL_WINDOW = 20
NDX_VOL_BAND = 0.25
NDX_MIN_N = 100


@dataclass(frozen=True)
class Estimate:
    probability: float
    sample_size: int
    rule: str
    median_close: float
    as_of_close: float
    horizon: int


def _prefix(bars: list[Bar] | tuple[Bar, ...], as_of: date) -> np.ndarray:
    dates = [bar.date for bar in bars]
    if dates != sorted(dates):
        raise ValueError("bars must be sorted by date")
    if as_of not in dates:
        raise ValueError(f"no bar on {as_of.isoformat()}")
    # Bars after as_of are dropped, so a later revision of the future cannot move this probability.
    closes = np.array([bar.close for bar in bars if bar.date <= as_of], dtype=float)
    if np.any(closes <= 0):
        raise ValueError("closes must be positive")
    return closes


def _vix_sample(closes: np.ndarray, horizon: int, band: float, min_n: int) -> tuple[np.ndarray, str]:
    # `closes` ends on the as-of session. The last start is H sessions earlier, so its outcome is known.
    i = len(closes) - 1
    if i < horizon:
        raise ValueError("not enough sessions for this horizon")
    start = closes[: i - horizon + 1]
    end = closes[horizon:]
    if len(start) != len(end):
        raise RuntimeError("horizon alignment")
    level = float(closes[i])
    mask = (start >= level * (1 - band)) & (start <= level * (1 + band))
    if int(mask.sum()) >= min_n:
        return end[mask], "band"
    return end, "all_history"


def estimate_vix_from_closes(
    closes: np.ndarray,
    horizon: int,
    threshold: float = 20.0,
    band: float = VIX_BAND,
    min_n: int = VIX_MIN_N,
) -> Estimate:
    sample, rule = _vix_sample(closes, horizon, band, min_n)
    return Estimate(
        probability=float(np.mean(sample > threshold)),
        sample_size=int(sample.size),
        rule=rule,
        median_close=float(np.median(sample)),
        as_of_close=float(closes[-1]),
        horizon=horizon,
    )


def vix_path(
    closes: np.ndarray,
    horizon: int,
    threshold: float = 20.0,
    band: float = VIX_BAND,
    min_n: int = VIX_MIN_N,
) -> list[dict]:
    """Level forecast for each session 1..horizon. Uses only closes through the as-of bar."""
    points = []
    for step in range(1, horizon + 1):
        sample, rule = _vix_sample(closes, step, band, min_n)
        points.append(_level_point(step, sample, float(np.mean(sample > threshold)), rule))
    return points


def estimate_vix(
    bars: list[Bar] | tuple[Bar, ...],
    as_of: date,
    horizon: int,
    threshold: float = 20.0,
    band: float = VIX_BAND,
    min_n: int = VIX_MIN_N,
) -> Estimate:
    """P(close H sessions later > threshold), from past starts near today's level."""
    return estimate_vix_from_closes(_prefix(bars, as_of), horizon, threshold, band, min_n)


def rolling_sample_std(log_returns: np.ndarray, window: int) -> np.ndarray:
    """Sample std of `window` returns ending at each close index. Earlier slots are nan."""
    totals = np.empty(len(log_returns) + 1)
    squares = np.empty(len(log_returns) + 1)
    totals[0] = 0.0
    squares[0] = 0.0
    totals[1:] = np.cumsum(log_returns)
    squares[1:] = np.cumsum(log_returns * log_returns)
    vol = np.full(len(log_returns) + 1, np.nan)
    if len(log_returns) < window:
        return vol
    end = np.arange(window, len(log_returns) + 1)
    sum_ = totals[end] - totals[end - window]
    sumsq = squares[end] - squares[end - window]
    var = (sumsq - (sum_ * sum_) / window) / (window - 1)
    vol[end] = np.sqrt(np.maximum(var, 0.0))
    return vol


def _ndx_returns(
    closes: np.ndarray,
    horizon: int,
    vol_window: int,
    vol_band: float,
    min_n: int,
) -> tuple[np.ndarray, str]:
    i = len(closes) - 1
    if i < vol_window + horizon:
        raise ValueError("not enough sessions for this horizon")
    log_close = np.log(closes)
    ahead = log_close[horizon:] - log_close[:-horizon]
    vol = rolling_sample_std(np.diff(log_close), vol_window)
    start = vol_window
    stop = i - horizon + 1
    vols = vol[start:stop]
    rets = ahead[start:stop]
    now = float(vol[i])
    mask = (vols >= now * (1 - vol_band)) & (vols <= now * (1 + vol_band))
    if int(np.count_nonzero(mask)) >= min_n:
        return rets[mask], "vol_band"
    return rets, "all_returns"


def estimate_ndx_from_closes(
    closes: np.ndarray,
    horizon: int,
    baseline: float | None = None,
    vol_window: int = NDX_VOL_WINDOW,
    vol_band: float = NDX_VOL_BAND,
    min_n: int = NDX_MIN_N,
) -> Estimate:
    spot = float(closes[-1])
    line = spot if baseline is None else float(baseline)
    sample, rule = _ndx_returns(closes, horizon, vol_window, vol_band, min_n)
    need = np.log(line) - np.log(spot)
    return Estimate(
        probability=float(np.mean(sample > need)),
        sample_size=int(sample.size),
        rule=rule,
        median_close=float(spot * np.exp(np.median(sample))),
        as_of_close=spot,
        horizon=horizon,
    )


def ndx_path(
    closes: np.ndarray,
    horizon: int,
    vol_window: int = NDX_VOL_WINDOW,
    vol_band: float = NDX_VOL_BAND,
    min_n: int = NDX_MIN_N,
) -> list[dict]:
    """Level forecast for each session 1..horizon. Uses only closes through the as-of bar."""
    spot = float(closes[-1])
    points = []
    for step in range(1, horizon + 1):
        sample, rule = _ndx_returns(closes, step, vol_window, vol_band, min_n)
        levels = spot * np.exp(sample)
        need = 0.0
        points.append(_level_point(step, levels, float(np.mean(sample > need)), rule))
    return points


def _level_point(step: int, levels: np.ndarray, probability: float, rule: str) -> dict:
    return {
        "horizon": step,
        "median": float(np.median(levels)),
        "low": float(np.quantile(levels, 0.10)),
        "high": float(np.quantile(levels, 0.90)),
        "probability": probability,
        "sample_size": int(levels.size),
        "rule": rule,
    }


def estimate_ndx(
    bars: list[Bar] | tuple[Bar, ...],
    as_of: date,
    horizon: int,
    baseline: float | None = None,
    vol_window: int = NDX_VOL_WINDOW,
    vol_band: float = NDX_VOL_BAND,
    min_n: int = NDX_MIN_N,
) -> Estimate:
    """P(close H sessions later > baseline). Baseline defaults to the as-of close."""
    return estimate_ndx_from_closes(
        _prefix(bars, as_of), horizon, baseline, vol_window, vol_band, min_n
    )
