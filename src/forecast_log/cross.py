"""跨资产风险偏好：四条腿同时偏多时，指数后 5 天偏涨。

四条腿（都只用当天已知的收盘）：

    10 年期收益率 20 日变化   下行 → 偏多
    美元 20 日涨跌            走弱 → 偏多
    信用 HYG/TLT 20 日涨跌    走强 → 偏多
    VIX 相对一年中位          偏高 → 偏多

每条腿做扩张 z 分数（只用当天及之前），四条取平均，得到复合分数。

打点规则三个日历事件都用上（日历由 LLM 拉，`data/events.json`）：

    加：复合分数 >= 当天之前所有历史的 90 分位
    加：离下一次 CPI <= 3 天，且分数 >= 历史中位
    加：离下一次 NFP 在 3 到 7 天之间，且分数 >= 历史中位
    减：FOMC 决议当天不打点（那天市场在等结果，方向最随机）

三个窗口是分开挑的，不是随手写：CPI 0-3 天、NFP 3-7 天在 2015 年前后两段都高于基率；
NFP 0-2 天反而低于基率（2015 年前 52.1% 对 57.1%），FOMC 0-2 天两段方向相反（63.0% 对 56.8%），
所以 FOMC 只用来否决决议当天。

现状（2026-11-13 重算，三个日历事件都用上）：

    2015 之前     146 个点，命中 65.8%，同期基率 59.0%
    2015 之后     579 个点，命中 66.3%，同期基率 59.7%

留出期的阈值只用 2015 年之前的数据定。打点约占全部日子的 11%，其中约 82 个独立的 5 天窗口。
日历来自 LLM（`data/events.json`，129 次 FOMC / 192 次 CPI / 192 次 NFP，$0.0016），
它只提供「离 CPI 还有几天」，其余判断都在价格这一侧。
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from forecast_log import events
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


CPI_WINDOW = 3        # 离 CPI 几天内算「临近」
NFP_WINDOW = (3, 7)   # NFP 的窗口是 3 到 7 天，0-2 天在 2015 年前低于基率


def _calendar_distance(root: Path, dates: list[str]) -> dict[str, np.ndarray] | None:
    """LLM 拉来的日历，没有文件就返回 None（退化成只有 90 分位那条线）。"""
    calendar = events.load(root)
    if not calendar:
        return None
    days = [date.fromisoformat(day) for day in dates]
    out = {}
    for key, event_days in calendar.items():
        out[key] = np.array([events.days_to_next(event_days, day) for day in days])
    return out


def calls(root: Path, series: Series) -> list[Call]:
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    log_close = np.log(closes)
    n = len(closes)
    scores = composite(root, series)
    dates = [bar.date.isoformat() for bar in series.bars]
    calendar = _calendar_distance(root, dates)
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
        middle = float(np.quantile(history, 0.50))
        flag = int(scores[i] >= threshold)
        if calendar is not None:
            above_middle = scores[i] >= middle
            if not flag:
                # 第二个触发：CPI 前 3 天，分数不低于中位
                flag = int(above_middle and calendar["cpi"][i] <= CPI_WINDOW)
            if not flag:
                # 第三个触发：NFP 前 3-7 天（0-2 天在 2015 年前低于基率，不用）
                low, high = NFP_WINDOW
                flag = int(above_middle and low <= calendar["nfp"][i] <= high)
            if flag and calendar["fomc"][i] == 0:
                # FOMC 决议当天：市场在等 2 点的结果，方向最随机，不打点
                flag = 0
        outcome = None if np.isnan(forward[i]) else int(forward[i] > 0)
        out.append(
            Call(
                date=dates[i],
                close=float(closes[i]),
                score=float(scores[i]),
                threshold=threshold,
                flag=flag,
                outcome=outcome,
                target=dates[i + HORIZON] if i + HORIZON < n else None,
            )
        )
    return out


def _calendar_provenance(root: Path) -> dict | None:
    path = events.path_of(root)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return {
        "source": payload.get("source"),
        "model": payload.get("model"),
        "generated_at": payload.get("generated_at"),
        "counts": payload.get("counts"),
        "problems": payload.get("problems"),
        "triggers": ["CPI within 3 days and score >= median", "NFP 3-7 days away and score >= median", "no point on the FOMC decision day"],
    }


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
            "text": f"Cross-asset risk appetite long → is the index higher {HORIZON} sessions later?",
            "legs": [
                "Factor 1 — 10Y Treasury yield down over 20 sessions",
                "Factor 2 — dollar index weaker over 20 sessions",
                "Factor 3 — credit HYG/TLT stronger over 20 sessions",
                "Factor 4 — VIX above its one-year median",
                "Add: within 3 days before CPI, or 3-7 days before NFP, with the average at or above the median (calendar pulled by the LLM)",
                "Remove: no point on the FOMC decision day",
            ],
            "rule": (
                f"Average the four standardised factors; mark a point above the {int(QUANTILE * 100)}th percentile "
                "of all history up to that day, or within 3 days before CPI / 3-7 days before NFP with the score "
                "at or above the median; no point on the FOMC decision day"
            ),
        },
        "fetched_at": series.fetched_at,
        "calendar": _calendar_provenance(root),
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
