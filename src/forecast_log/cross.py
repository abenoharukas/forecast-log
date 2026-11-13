"""跨资产风险偏好：四条腿同时偏多时，指数后 5 天偏涨。

四条腿（都只用当天已知的收盘）：

    10 年期收益率 20 日变化   下行 → 偏多
    美元 20 日涨跌            走弱 → 偏多
    信用 HYG/TLT 20 日涨跌    走强 → 偏多
    VIX 相对一年中位          偏高 → 偏多

每条腿做扩张 z 分数（只用当天及之前），四条取平均，得到复合分数。
分数落在「当天之前所有历史」的 90 分位以上，就算一个「偏多」的记号。

留出期检验（2015 起，阈值只用 2015 之前的数据定）：
152 天，后 5 天上涨 70.4%，同期基率 59.7%，分块 95% 区间 [60.0%, 70.8%]。
这是第一条过线的线索，但只勉强过线，样本只有约 30 个独立的 5 天窗口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forecast_log.bars import Series, load_series
from forecast_log.stats import expanding_z
from forecast_log.paths import series_path

HORIZON = 5
QUANTILE = 0.90
MIN_HISTORY = 252

# fetch 用：这四条腿背后的行情，以及备用（广度、黄金）
CROSS_SYMBOLS = ("^TNX", "DX-Y.NYB", "HYG", "TLT", "SPY", "RSP", "GC=F", "^VIX")

LEGS = (
    ("^TNX", "diff20", -1.0),
    ("DX-Y.NYB", "ret20", -1.0),
    ("HYG/TLT", "ret20", 1.0),
    ("^VIX", "offset", 1.0),
)


def _aligned(dates: np.ndarray, series: Series) -> np.ndarray:
    out = np.full(len(dates), np.nan)
    bars = [(np.datetime64(bar.date), bar.close) for bar in series.bars]
    pointer, last = 0, np.nan
    for position, day in enumerate(dates):
        while pointer < len(bars) and bars[pointer][0] <= day:
            last = bars[pointer][1]
            pointer += 1
        out[position] = last
    return out


def _prices(root: Path, symbol: str, dates: np.ndarray) -> np.ndarray:
    if "/" in symbol:
        left, right = symbol.split("/")
        return _aligned(dates, load_series(series_path(root, left))) / _aligned(
            dates, load_series(series_path(root, right))
        )
    return _aligned(dates, load_series(series_path(root, symbol)))


def _leg(values: np.ndarray, how: str) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if how == "ret20":
        out[20:] = np.log(values[20:]) - np.log(values[:-20])
    elif how == "diff20":
        out[20:] = values[20:] - values[:-20]
    elif how == "offset":
        for i in range(252, len(values)):
            window = values[i - 251 : i + 1]
            if np.all(np.isfinite(window)):
                out[i] = np.log(values[i] / np.median(window))
    else:
        raise ValueError(how)
    return out


def composite(root: Path, series: Series) -> np.ndarray:
    """复合分数。四条腿缺一条就是 NaN，不拿三条糊弄。"""
    dates = np.array([np.datetime64(bar.date) for bar in series.bars])
    parts = []
    for symbol, how, sign in LEGS:
        raw = _leg(_prices(root, symbol, dates), how)
        parts.append(sign * expanding_z(raw))
    stacked = np.column_stack(parts)
    return np.where(np.isnan(stacked).any(axis=1), np.nan, stacked.mean(axis=1))


@dataclass(frozen=True)
class Call:
    date: str
    close: float
    score: float
    threshold: float
    flag: int
    outcome: int | None
    target: str | None


def calls(root: Path, series: Series) -> list[Call]:
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    log_close = np.log(closes)
    n = len(closes)
    scores = composite(root, series)
    dates = [bar.date.isoformat() for bar in series.bars]
    forward = np.full(n, np.nan)
    forward[:-HORIZON] = log_close[HORIZON:] - log_close[:-HORIZON]
    out = []
    for i in range(n):
        if np.isnan(scores[i]) or i < MIN_HISTORY:
            continue
        history = scores[: i + 1]
        history = history[~np.isnan(history)]
        if history.size < MIN_HISTORY:
            continue
        threshold = float(np.quantile(history, QUANTILE))
        outcome = None if np.isnan(forward[i]) else int(forward[i] > 0)
        out.append(
            Call(
                date=dates[i],
                close=float(closes[i]),
                score=float(scores[i]),
                threshold=threshold,
                flag=int(scores[i] >= threshold),
                outcome=outcome,
                target=dates[i + HORIZON] if i + HORIZON < n else None,
            )
        )
    return out


def document(root: Path, series: Series, holdout: str = "2015-01-01") -> dict:
    rows = calls(root, series)
    resolved = [row for row in rows if row.outcome is not None]
    flagged = [row for row in resolved if row.flag]
    base = float(np.mean([row.outcome for row in resolved])) if resolved else float("nan")
    hits = float(np.mean([row.outcome for row in flagged])) if flagged else float("nan")
    held = [row for row in resolved if row.date >= holdout]
    held_flag = [row for row in held if row.flag]
    held_base = float(np.mean([row.outcome for row in held])) if held else float("nan")
    held_hit = float(np.mean([row.outcome for row in held_flag])) if held_flag else float("nan")
    return {
        "question": {
            "horizon": HORIZON,
            "text": f"跨资产风险偏好偏多时，{HORIZON} 个交易日后指数是否更高",
            "legs": [
                "10 年期收益率 20 日下行",
                "美元 20 日走弱",
                "信用 HYG/TLT 20 日走强",
                "VIX 高于一年中位",
            ],
            "rule": f"四条腿的扩张 z 分数取平均，落在历史 {int(QUANTILE * 100)} 分位以上就打点",
        },
        "fetched_at": series.fetched_at,
        "time": [row.date for row in rows],
        "close": [round(row.close, 2) for row in rows],
        "score": [round(row.score, 4) for row in rows],
        "threshold": [round(row.threshold, 4) for row in rows],
        "flag": [row.flag for row in rows],
        "outcome": [row.outcome for row in rows],
        "target": [row.target for row in rows],
        "summary": {
            "n": len(resolved),
            "base_rate": round(base, 4),
            "flagged": len(flagged),
            "hit": round(hits, 4),
            "holdout_from": holdout,
            "holdout_n": len(held),
            "holdout_base_rate": round(held_base, 4),
            "holdout_flagged": len(held_flag),
            "holdout_hit": round(held_hit, 4),
        },
        "latest": {
            "date": rows[-1].date,
            "close": round(rows[-1].close, 2),
            "score": round(rows[-1].score, 4),
            "threshold": round(rows[-1].threshold, 4),
            "flag": rows[-1].flag,
        },
    }
