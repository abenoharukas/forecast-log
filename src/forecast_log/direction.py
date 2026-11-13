"""指数本身未来 5 天的涨跌点（新题，和 fear index 那两页无关）。

题目（事先写死）：

    5 个交易日后，指数收盘是否比今天高。
    概率落在历史最高的 10% → 打「预测上涨」的点；最低的 10% → 打「预测下跌」的点。
    点画在到期那一天（t+5）。

方法：walk-forward 逻辑回归，只用收盘后已知的量（5/20/200 日涨跌、离 200 日均线、
20 日已实现波动、期限结构斜率、VVIX 偏离），每 21 个交易日重训一次。
分数只和「只按历史基率猜」比。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from forecast_log.bars import Series

HORIZON = 5
RETRAIN_EVERY = 21
MIN_TRAIN = 250
CLIP = (0.02, 0.98)
FLAG_QUANTILE = (0.10, 0.90)


def _logistic_fit(x: np.ndarray, y: np.ndarray, rounds: int = 25) -> np.ndarray:
    """Newton / IRLS with a small ridge so a quiet stretch cannot blow the weights up."""
    a = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(a.shape[1])
    ridge = 1e-3 * np.eye(a.shape[1])
    ridge[0, 0] = 0.0
    for _ in range(rounds):
        p = 1.0 / (1.0 + np.exp(-(a @ w)))
        weight = np.clip(p * (1 - p), 1e-6, None)
        step = np.linalg.solve(a.T @ (a * weight[:, None]) + ridge, a.T @ (y - p))
        w = w + step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _expanding_z(values: np.ndarray) -> np.ndarray:
    """Row i uses only rows 0..i, so later data cannot leak backwards."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = values[: i + 1]
        window = window[~np.isnan(window)]
        if len(window) < 30 or np.isnan(values[i]):
            continue
        sd = window.std()
        if sd <= 0:
            continue
        out[i] = (values[i] - window.mean()) / sd
    return out


@dataclass(frozen=True)
class Row:
    date: str
    close: float
    probability: float
    outcome: int | None
    target: str | None


def _returns(log_close: np.ndarray, span: int) -> np.ndarray:
    out = np.full(len(log_close), np.nan)
    out[span:] = log_close[span:] - log_close[:-span]
    return out


def _features(series: Series, matrix: np.ndarray | None) -> np.ndarray:
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    log_close = np.log(closes)
    n = len(closes)
    middle = np.full(n, np.nan)
    for i in range(199, n):
        middle[i] = closes[i - 199 : i + 1].mean()
    vol = np.full(n, np.nan)
    for i in range(20, n):
        vol[i] = float(np.std(np.diff(log_close[i - 20 : i + 1])))
    columns = [
        _returns(log_close, 5),
        _returns(log_close, 20),
        _returns(log_close, 200),
        log_close - np.log(middle),
        vol,
    ]
    # 没有期限结构数据时就不放这一列，而不是塞一列 NaN 把整行作废。
    if matrix is not None:
        columns.append(matrix[:, 0])
    return np.column_stack(columns)


def walk_forward(series: Series, matrix: np.ndarray | None) -> list[Row]:
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    n = len(closes)
    raw = _features(series, matrix)
    labels = np.full(n, np.nan)
    for i in range(n - HORIZON):
        labels[i] = 1.0 if closes[i + HORIZON] > closes[i] else 0.0
    z = np.column_stack([_expanding_z(raw[:, j]) for j in range(raw.shape[1])])
    usable = ~np.isnan(z).any(axis=1) & ~np.isnan(labels)
    dates = [bar.date.isoformat() for bar in series.bars]
    rows: list[Row] = []
    weights: np.ndarray | None = None
    last_fit = -10**9
    for i in range(n):
        if np.isnan(z[i]).any():
            continue
        ready = np.flatnonzero(usable[: i + 1])
        if ready.size < MIN_TRAIN:
            continue
        if weights is None or i - last_fit >= RETRAIN_EVERY:
            weights = _logistic_fit(z[ready], labels[ready])
            last_fit = i
        probability = float(
            np.clip(_sigmoid(float(weights[0] + z[i] @ weights[1:])), *CLIP)
        )
        target = dates[i + HORIZON] if i + HORIZON < n else None
        rows.append(
            Row(
                date=dates[i],
                close=float(closes[i]),
                probability=probability,
                outcome=None if np.isnan(labels[i]) else int(labels[i]),
                target=target,
            )
        )
    return rows


def summary(rows: list[Row]) -> dict:
    resolved = [row for row in rows if row.outcome is not None]
    if not resolved:
        return {"n": 0}
    probability = np.array([row.probability for row in resolved])
    outcome = np.array([row.outcome for row in resolved], dtype=float)
    base = float(outcome.mean())
    low, high = np.quantile(probability, FLAG_QUANTILE)

    def block(mask: np.ndarray) -> dict:
        count = int(mask.sum())
        return {
            "n": count,
            "mean_probability": round(float(probability[mask].mean()), 4) if count else None,
            "hit": round(float(outcome[mask].mean()), 4) if count else None,
        }

    return {
        "n": len(resolved),
        "horizon": HORIZON,
        "base_rate": round(base, 4),
        "brier_model": round(float(np.mean((probability - outcome) ** 2)), 4),
        "brier_base": round(float(np.mean((base - outcome) ** 2)), 4),
        "brier_half": round(float(np.mean((0.5 - outcome) ** 2)), 4),
        "mean_probability": round(float(probability.mean()), 4),
        "threshold_low": round(float(low), 4),
        "threshold_high": round(float(high), 4),
        "up_flags": block(probability >= high),
        "down_flags": block(probability <= low),
        "first": resolved[0].date,
        "last": resolved[-1].date,
    }


def document(series: Series, matrix: np.ndarray | None) -> dict:
    rows = walk_forward(series, matrix)
    stats = summary(rows)
    high = stats.get("threshold_high", 1.0)
    low = stats.get("threshold_low", 0.0)
    return {
        "question": {
            "horizon": HORIZON,
            "text": f"{HORIZON} 个交易日后，指数收盘是否比今天高",
        },
        "fetched_at": series.fetched_at,
        "time": [row.date for row in rows],
        "close": [round(row.close, 2) for row in rows],
        "probability": [round(row.probability, 4) for row in rows],
        "outcome": [row.outcome for row in rows],
        "target": [row.target for row in rows],
        "flag_up": [1 if row.probability >= high else 0 for row in rows],
        "flag_down": [1 if row.probability <= low else 0 for row in rows],
        "latest": {
            "date": rows[-1].date,
            "close": round(rows[-1].close, 2),
            "probability": round(rows[-1].probability, 4),
        },
        "summary": stats,
    }
