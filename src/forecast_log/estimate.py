from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast_log.bars import Bar

# The state vector is fixed before the backtest was run, not tuned to its score.
# One week, one month, one year of change, plus the distance from the one-year midpoint.
STATE_LOOKBACKS = (5, 21, 252)
LONG_LOOKBACK = 252
NEIGHBOURS = 200
MIN_CANDIDATES = 30
Z_CLIP = 6.0


@dataclass(frozen=True)
class PathPoint:
    """One session of the forecast path, anchored at the as-of close."""

    horizon: int
    median: float
    low: float
    high: float
    probability: float


@dataclass(frozen=True)
class Estimate:
    probability: float
    sample_size: int
    rule: str
    median_close: float
    as_of_close: float
    horizon: int
    path: tuple[PathPoint, ...] = ()


def min_index(horizon: int) -> int:
    """Earliest index with enough finished analogues to forecast `horizon` sessions ahead."""
    return LONG_LOOKBACK + MIN_CANDIDATES + horizon - 1


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    out[window - 1 :] = np.median(windows, axis=1)
    return out


def state_matrix(closes: np.ndarray, lookbacks: tuple[int, ...] = STATE_LOOKBACKS) -> np.ndarray:
    """Stacked state features. Row i uses closes[0..i] only."""
    n = len(closes)
    log_close = np.log(closes)
    columns = []
    for lookback in lookbacks:
        column = np.full(n, np.nan)
        column[lookback:] = log_close[lookback:] - log_close[:-lookback]
        columns.append(column)
    midpoint = _rolling_median(closes, LONG_LOOKBACK)
    offset = np.full(n, np.nan)
    offset[LONG_LOOKBACK:] = log_close[LONG_LOOKBACK:] - np.log(midpoint[LONG_LOOKBACK:])
    columns.append(offset)
    return np.column_stack(columns)


def standardize(matrix: np.ndarray, clip: float = Z_CLIP) -> np.ndarray:
    """Expanding z-score. Row i uses the mean and std of rows 0..i only."""
    valid = ~np.isnan(matrix)
    count = np.maximum(np.cumsum(valid, axis=0), 1)
    total = np.cumsum(np.where(valid, matrix, 0.0), axis=0)
    square = np.cumsum(np.where(valid, matrix * matrix, 0.0), axis=0)
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 1e-12)
    return np.nan_to_num(np.clip((matrix - mean) / np.sqrt(variance), -clip, clip))


class AnalogModel:
    """One price series. `forecast` only ever reads closes[0..index]."""

    def __init__(self, closes: np.ndarray, lookbacks: tuple[int, ...] = STATE_LOOKBACKS):
        self.closes = np.asarray(closes, dtype=float)
        if self.closes.ndim != 1 or len(self.closes) < 2:
            raise ValueError("closes must be one-dimensional")
        if np.any(self.closes <= 0):
            raise ValueError("closes must be positive")
        self.lookbacks = lookbacks
        self.state = standardize(state_matrix(self.closes, lookbacks))

    def forecast(
        self,
        index: int,
        horizon: int,
        threshold: float,
        neighbours: int = NEIGHBOURS,
        with_path: bool = False,
    ) -> Estimate:
        """P(close H sessions later > threshold), and the whole path there.

        Each analogue is compared as a multiple of its own start, and the as-of close is
        the anchor, so the path has to come from today's level instead of restating the
        level of some past analogue.
        """
        closes = self.closes
        if not 0 <= index < len(closes):
            raise ValueError(f"index {index} is outside the series")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        lowest = min_index(horizon)
        if index < lowest:
            raise ValueError(f"index {index} is below the first usable index {lowest}")
        query = self.state[index]
        if np.isnan(query).any():
            raise ValueError("state vector is undefined at this index")

        # The analogue's outcome must already be printed, and its window must not overlap the query's.
        candidate = np.arange(LONG_LOOKBACK, index - horizon + 1)
        if candidate.size < MIN_CANDIDATES:
            raise ValueError("not enough finished analogues for this horizon")
        distance = ((self.state[candidate] - query) ** 2).sum(axis=1)
        take = min(neighbours, candidate.size)
        chosen = candidate[np.argpartition(distance, take - 1)[:take]]
        chosen.sort()
        rule = "analog" if candidate.size > take else "all_analogues"

        level = float(closes[index])
        need = float(threshold) / level
        path = ()
        if with_path:
            path = tuple(
                PathPoint(
                    horizon=step,
                    median=level * float(np.median(multiple)),
                    low=level * float(np.quantile(multiple, 0.10)),
                    high=level * float(np.quantile(multiple, 0.90)),
                    probability=float(np.mean(multiple > need)),
                )
                for step in range(1, horizon + 1)
                for multiple in (closes[chosen + step] / closes[chosen],)
            )
        last = closes[chosen + horizon] / closes[chosen]
        return Estimate(
            probability=float(np.mean(last > need)),
            sample_size=int(chosen.size),
            rule=rule,
            median_close=float(level * np.median(last)),
            as_of_close=level,
            horizon=horizon,
            path=path,
        )


def index_of(bars: list[Bar] | tuple[Bar, ...], as_of) -> int:
    for position, bar in enumerate(bars):
        if bar.date == as_of:
            return position
    raise ValueError(f"no bar on {as_of.isoformat()}")


def closes_of(bars: list[Bar] | tuple[Bar, ...]) -> np.ndarray:
    dates = [bar.date for bar in bars]
    if dates != sorted(dates):
        raise ValueError("bars must be sorted by date")
    closes = np.array([bar.close for bar in bars], dtype=float)
    if np.any(closes <= 0):
        raise ValueError("closes must be positive")
    return closes


def estimate(
    bars: list[Bar] | tuple[Bar, ...],
    as_of,
    horizon: int,
    threshold: float,
    neighbours: int = NEIGHBOURS,
    with_path: bool = True,
) -> Estimate:
    """Convenience wrapper for one series and one as-of session."""
    closes = closes_of(bars)
    return AnalogModel(closes).forecast(
        index_of(bars, as_of), horizon, threshold, neighbours, with_path
    )
