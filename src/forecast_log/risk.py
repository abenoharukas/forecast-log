"""风险预警：未来几个交易日内会不会出现一次大波动。

这不是点位预测。恐慌指数明天收在哪儿说不出来（日收益自相关约等于 0），
但「最近会不会不平静」有一点真信号：波动有聚集，期限结构倒挂时更容易出事。

题目（事先写死）：

    未来 5 个交易日内，^VXN 是否出现单日 |涨跌| > 15%。

判分只认 Brier，对照是当时已知的历史基准率，以及永远答 0.5。

方法：只用当天收盘后已知的四个数做逻辑回归，参数在每个预测日之前拟合，
样本只到那时为止（walk-forward），标准化也用当时的均值和标准差。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast_log.bars import Series

HORIZON = 5
MOVE = 0.15
FEATURE_NAMES = ("期限结构斜率", "^VVIX 偏离", "20 日已实现波动", "离一年中位的偏离")
RETRAIN_EVERY = 21
CLIP = (0.02, 0.98)
MIN_TRAIN = 250


@dataclass(frozen=True)
class RiskRow:
    date: str
    probability: float
    outcome: int | None
    anchor: float


def _logistic_fit(x: np.ndarray, y: np.ndarray, rounds: int = 25) -> np.ndarray:
    """Newton / IRLS with a small ridge so a quiet stretch cannot blow the weights up."""
    a = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(a.shape[1])
    ridge = 1e-3 * np.eye(a.shape[1])
    ridge[0, 0] = 0.0
    for _ in range(rounds):
        z = a @ w
        p = 1.0 / (1.0 + np.exp(-z))
        weight = np.clip(p * (1 - p), 1e-6, None)
        hessian = a.T @ (a * weight[:, None]) + ridge
        gradient = a.T @ (y - p)
        step = np.linalg.solve(hessian, gradient)
        w = w + step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _expanding_z(values: np.ndarray) -> np.ndarray:
    """Row i uses only rows 0..i, so a later stretch cannot leak into an earlier day."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = values[: i + 1]
        window = window[~np.isnan(window)]
        if len(window) < 30:
            continue
        mean = window.mean()
        sd = window.std()
        if sd <= 0 or np.isnan(values[i]):
            continue
        out[i] = (values[i] - mean) / sd
    return out


def features(series: Series, matrix: np.ndarray | None) -> np.ndarray:
    """The four inputs. All of them are printed before the session they are used on."""
    bars = series.bars
    closes = np.array([bar.close for bar in bars], dtype=float)
    log_close = np.log(closes)
    n = len(closes)
    recent = np.full(n, np.nan)
    for i in range(20, n):
        recent[i] = float(np.std(np.diff(log_close[i - 20 : i + 1])))
    rolling_mid = np.full(n, np.nan)
    for i in range(251, n):
        rolling_mid[i] = float(np.median(closes[i - 251 : i + 1]))
    offset = log_close - np.log(rolling_mid)
    if matrix is None:
        matrix = np.full((n, 2), np.nan)
    columns = [matrix[:, 0], matrix[:, 1], recent, offset]
    return np.column_stack(columns)


def outcomes(series: Series, horizon: int = HORIZON, move: float = MOVE) -> np.ndarray:
    """1 when any of the next `horizon` sessions moved more than `move`, else 0, None if unprinted."""
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    n = len(closes)
    labels: list[int | None] = []
    for i in range(n):
        window = closes[i + 1 : i + 1 + horizon]
        if len(window) < horizon:
            labels.append(None)
            continue
        labels.append(1 if float(np.max(np.abs(window / closes[i] - 1.0))) > move else 0)
    return np.array([np.nan if v is None else v for v in labels], dtype=float)


def walk_forward(series: Series, matrix: np.ndarray | None) -> list[RiskRow]:
    """One probability per session, refitting only on what had already printed."""
    raw = features(series, matrix)
    labels = outcomes(series)
    z = np.column_stack([_expanding_z(raw[:, j]) for j in range(raw.shape[1])])
    usable = ~np.isnan(z).any(axis=1) & ~np.isnan(labels)
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    rows: list[RiskRow] = []
    weights: np.ndarray | None = None
    last_fit = -10**9
    for i in range(len(series.bars)):
        if np.isnan(z[i]).any():
            continue
        ready = np.flatnonzero(usable[: i + 1])
        if ready.size < MIN_TRAIN:
            continue
        if weights is None or i - last_fit >= RETRAIN_EVERY:
            x = z[ready]
            y = labels[ready]
            if len(np.unique(y)) < 2:
                continue
            weights = _logistic_fit(x, y)
            last_fit = i
        z_i = float(weights[0] + z[i] @ weights[1:])
        probability = float(np.clip(_sigmoid(z_i), *CLIP))
        outcome = None if np.isnan(labels[i]) else int(labels[i])
        rows.append(
            RiskRow(
                date=series.bars[i].date.isoformat(),
                probability=probability,
                outcome=outcome,
                anchor=float(closes[i]),
            )
        )
    return rows


def brier(probability, outcome) -> float:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(outcome, dtype=float)
    return float(np.mean((p - y) ** 2))


def summary(rows: list[RiskRow], horizon: int = HORIZON, move: float = MOVE) -> dict:
    resolved = [row for row in rows if row.outcome is not None]
    if not resolved:
        return {"n": 0}
    probability = np.array([row.probability for row in resolved])
    outcome = np.array([row.outcome for row in resolved], dtype=float)
    base = float(outcome.mean())

    def score(p, y):
        return round(brier(p, y), 4)

    scored = {
        "n": len(resolved),
        "horizon": horizon,
        "move": move,
        "base_rate": round(base, 4),
        "brier_model": score(probability, outcome),
        "brier_base": score([base] * len(resolved), outcome),
        "brier_half": score([0.5] * len(resolved), outcome),
        "mean_probability": round(float(probability.mean()), 4),
        "first": resolved[0].date,
        "last": resolved[-1].date,
    }
    # 把模型概率按三分位切开，看高低概率组的实际发生率差多少。
    cuts = np.quantile(probability, [1 / 3, 2 / 3])
    bands = [probability <= cuts[0], (probability > cuts[0]) & (probability <= cuts[1]), probability > cuts[1]]
    scored["bands"] = [
        {
            "n": int(band.sum()),
            "mean_probability": round(float(probability[band].mean()), 4),
            "actual": round(float(outcome[band].mean()), 4),
        }
        for band in bands
        if band.any()
    ]
    return scored


def shocks(series: Series, move: float = 0.25) -> list[int]:
    """那一天自己有没有跳过 `move`。只在图上打点用，不参与概率。"""
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    change = np.abs(np.diff(closes, prepend=closes[0]) / closes) if len(closes) else []
    return [1 if float(value) > move else 0 for value in change]


def document(series: Series, matrix: np.ndarray | None) -> dict:
    rows = walk_forward(series, matrix)
    by_date = {row.date: flag for row, flag in zip(
        [RiskRow(bar.date.isoformat(), 0, None, bar.close) for bar in series.bars],
        shocks(series),
        strict=True,
    )}
    last = rows[-1]
    return {
        "question": {
            "horizon": HORIZON,
            "move": MOVE,
            "text": f"未来 {HORIZON} 个交易日内，fear index 是否出现单日 |涨跌| > {MOVE:.0%}",
        },
        "fetched_at": series.fetched_at,
        "time": [row.date for row in rows],
        "probability": [round(row.probability, 4) for row in rows],
        "outcome": [row.outcome for row in rows],
        "anchor": [round(row.anchor, 2) for row in rows],
        # 1 = 那天自己跳了 25% 以上，用来在图上点红点
        "shock": [by_date.get(row.date, 0) for row in rows],
        "latest": {
            "date": last.date,
            "probability": round(last.probability, 4),
            "anchor": last.anchor,
        },
        "summary": summary(rows),
    }
