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
from datetime import date

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


def outcomes_split(
    series: Series, horizon: int = HORIZON, move: float = MOVE
) -> tuple[np.ndarray, np.ndarray]:
    """两个方向分开数：未来 `horizon` 天内有没有涨过 move、有没有跌过 move。"""
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    n = len(closes)
    up: list[float] = []
    down: list[float] = []
    for i in range(n):
        window = closes[i + 1 : i + 1 + horizon] / closes[i] - 1.0
        if len(window) < horizon:
            up.append(np.nan)
            down.append(np.nan)
            continue
        up.append(1.0 if float(window.max()) > move else 0.0)
        down.append(1.0 if float(-window.min()) > move else 0.0)
    return np.array(up), np.array(down)


def walk_forward(
    series: Series,
    matrix: np.ndarray | None,
    labels: np.ndarray | None = None,
) -> list[RiskRow]:
    """One probability per session, refitting only on what had already printed."""
    raw = features(series, matrix)
    if labels is None:
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


def _direction(series: Series, matrix: np.ndarray | None, name: str, labels: np.ndarray,
               horizon: int, move: float, quantile: float = 0.9) -> dict:
    rows = walk_forward(series, matrix, labels)
    last = rows[-1]
    stats = summary(rows, horizon, move)
    # 打点规则：这个方向概率最高的 10% 才算「说了会涨/会跌」。阈值固定，事前可算。
    resolved = [row.probability for row in rows if row.outcome is not None]
    threshold = float(np.quantile(resolved, quantile)) if resolved else 1.0
    flagged = [row for row in rows if row.probability >= threshold]
    hits = [row for row in flagged if row.outcome == 1]
    bars = series.bars
    index = {row.date: position for position, row in enumerate(rows)}
    dates = [bar.date.isoformat() for bar in bars]
    targets = []
    for row in rows:
        position = dates.index(row.date)
        targets.append(dates[min(position + horizon, len(dates) - 1)])
    stats["threshold"] = round(threshold, 4)
    stats["flagged"] = len(flagged)
    stats["hit"] = round(len(hits) / len(flagged), 4) if flagged else None
    stats["skill"] = (
        round(1.0 - stats["brier_model"] / stats["brier_base"], 4)
        if stats.get("brier_base")
        else None
    )
    return {
        "name": name,
        "text": f"未来 {horizon} 个交易日内，是否出现单日{name}超过 {move:.0%}",
        "time": [row.date for row in rows],
        "probability": [round(row.probability, 4) for row in rows],
        "outcome": [row.outcome for row in rows],
        "anchor": [round(row.anchor, 2) for row in rows],
        "shock": _shock_for(series, rows),
        # 点画在目标日（t+horizon），不是在发出日。
        "target": targets,
        "flag": [1 if row.probability >= threshold else 0 for row in rows],
        "latest": {
            "date": last.date,
            "probability": round(last.probability, 4),
            "anchor": last.anchor,
        },
        "summary": stats,
    }


def _shock_for(series: Series, rows: list[RiskRow], move: float = 0.25) -> list[int]:
    """每一天自己有没有动过 move（涨跌都算）；图上打点用，不参与概率。"""
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    flags = [0.0] + [abs(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))]
    index = {bar.date.isoformat(): position for position, bar in enumerate(series.bars)}
    return [1 if flags[index[row.date]] > move else 0 for row in rows]


def document(series: Series, matrix: np.ndarray | None) -> dict:
    """两个方向各算一次，各自有各自的概率、结算和分数。"""
    up, down = outcomes_split(series, HORIZON, MOVE)
    return {
        "horizon": HORIZON,
        "move": MOVE,
        "fetched_at": series.fetched_at,
        "up": _direction(series, matrix, "上涨", up, HORIZON, MOVE),
        "down": _direction(series, matrix, "下跌", down, HORIZON, MOVE),
    }
