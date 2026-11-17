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
    """One price series, plus optional forward-looking columns.

    `forecast` only ever reads row `index` of the state and analogues that finished at or
    before it, so a row that carries a future value cannot leak into an earlier forecast.
    """

    def __init__(
        self,
        closes: np.ndarray,
        extra: np.ndarray | None = None,
        lookbacks: tuple[int, ...] = STATE_LOOKBACKS,
    ):
        self.closes = np.asarray(closes, dtype=float)
        if self.closes.ndim != 1 or len(self.closes) < 2:
            raise ValueError("closes must be one-dimensional")
        if np.any(self.closes <= 0):
            raise ValueError("closes must be positive")
        self.lookbacks = lookbacks
        raw = state_matrix(self.closes, lookbacks)
        if extra is not None:
            extra = np.asarray(extra, dtype=float)
            if extra.ndim != 2 or extra.shape[0] != len(self.closes):
                raise ValueError("extra must carry one row per close")
            raw = np.column_stack([raw, extra])
        # Rows with a missing input are unusable as a query and as an analogue.
        self.complete = ~np.isnan(raw).any(axis=1)
        self.state = standardize(raw)

    def first_index(self, horizon: int) -> int:
        """Earliest index with a complete state and enough finished analogues.

        Returns `len(closes)` when no index qualifies, so callers can compare ranges.
        """
        usable = np.flatnonzero(self.complete)
        usable = usable[usable >= LONG_LOOKBACK]
        if usable.size < MIN_CANDIDATES:
            return len(self.closes)
        return int(usable[MIN_CANDIDATES - 1]) + horizon

    def _build_estimate(
        self,
        index: int,
        horizon: int,
        threshold: float,
        chosen: np.ndarray,
        cand_count: int,
        with_path: bool,
    ) -> Estimate:
        closes = self.closes
        level = float(closes[index])
        need = float(threshold) / level
        rule = "analog" if cand_count > chosen.size else "all_analogues"
        path = ()
        if with_path:
            steps = np.arange(1, horizon + 1)
            multiples = closes[chosen[:, None] + steps[None, :]] / closes[chosen][:, None]
            meds = np.median(multiples, axis=0)
            q10, q90 = np.quantile(multiples, [0.10, 0.90], axis=0)
            probs = np.mean(multiples > need, axis=0)
            path = tuple(
                PathPoint(
                    horizon=int(step),
                    median=level * float(meds[k]),
                    low=level * float(q10[k]),
                    high=level * float(q90[k]),
                    probability=float(probs[k]),
                )
                for k, step in enumerate(steps)
            )
            last = multiples[:, -1]
        else:
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
        lowest = max(min_index(horizon), self.first_index(horizon))
        if index < lowest:
            raise ValueError(f"index {index} is below the first usable index {lowest}")
        if not self.complete[index]:
            raise ValueError("state vector is undefined at this index")
        query = self.state[index]

        # The analogue's outcome must already be printed, and its window must not overlap the query's.
        candidate = np.arange(LONG_LOOKBACK, index - horizon + 1)
        candidate = candidate[self.complete[candidate]]
        if candidate.size < MIN_CANDIDATES:
            raise ValueError("not enough finished analogues for this horizon")
        distance = ((self.state[candidate] - query) ** 2).sum(axis=1)
        take = min(neighbours, candidate.size)
        chosen = candidate[np.argpartition(distance, take - 1)[:take]]
        chosen.sort()
        return self._build_estimate(index, horizon, threshold, chosen, int(candidate.size), with_path)

    def forecast_batch(
        self,
        indices: list[int] | np.ndarray,
        horizon: int,
        thresholds: list[float] | np.ndarray,
        neighbours: int = NEIGHBOURS,
        with_path: bool = False,
        chunk_size: int = 192,
    ) -> list[Estimate]:
        """分块向量化批量预测，与逐行调用 `forecast()` 数值严格等价。"""
        idx_arr = np.asarray(indices, dtype=int)
        thr_arr = np.asarray(thresholds, dtype=float)
        if idx_arr.size == 0:
            return []
        n = len(self.closes)
        valid_cand = np.flatnonzero(self.complete & (np.arange(n) >= LONG_LOOKBACK))
        sc = self.state[valid_cand]
        out: list[Estimate] = []
        for b0 in range(0, idx_arr.size, chunk_size):
            b1 = min(b0 + chunk_size, idx_arr.size)
            q_idx = idx_arr[b0:b1]
            max_cand_pos = int(np.searchsorted(valid_cand, int(q_idx[-1]) - horizon, side="right"))
            sc_sub = sc[:max_cand_pos]
            diff = self.state[q_idx, None, :] - sc_sub[None, :, :]
            dist_mat = (diff * diff).sum(axis=2)
            for k, i in enumerate(q_idx):
                ii = int(i)
                limit_pos = int(np.searchsorted(valid_cand, ii - horizon, side="right"))
                if limit_pos < MIN_CANDIDATES:
                    raise ValueError("not enough finished analogues for this horizon")
                d_row = dist_mat[k, :limit_pos]
                take = min(neighbours, limit_pos)
                chosen = valid_cand[np.argpartition(d_row, take - 1)[:take]]
                chosen.sort()
                out.append(
                    self._build_estimate(ii, horizon, float(thr_arr[b0 + k]), chosen, limit_pos, with_path)
                )
        return out


def index_of(bars: list[Bar] | tuple[Bar, ...], as_of) -> int:
    for position, bar in enumerate(bars):
        if bar.date == as_of:
            return bar_pos if False else position
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
    extra: np.ndarray | None = None,
) -> Estimate:
    """Convenience wrapper for one series and one as-of session."""
    closes = closes_of(bars)
    return AnalogModel(closes, extra).forecast(
        index_of(bars, as_of), horizon, threshold, neighbours, with_path
    )
