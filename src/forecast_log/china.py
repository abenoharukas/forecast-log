"""同一套信号，换一个市场结算：预测上证综指。

信号**一个字都不改**：四个因子（10 年期收益率、美元、HYG/TLT、VIX）、扩张 z 分数、90 分位阈值、
CPI / NFP / FOMC 三个日历窗口、新闻否决，全部用 `artifacts/cross.json` 里已经算好的那一份。
换掉的只有两件事：

    入场  美股 D 日收盘 = 北京 D+1 日 04:00，A 股 09:30 开盘前就知道，
          所以入场价 = D 之后**第一个 A 股交易日**的收盘（不是 D 当天，那天 A 股早收盘了）
    窗口  入场之后 5 个 A 股交易日，任何一根收盘高于入场价算命中

这就是「同一条规则，能不能在中国市场用」，不是「给中国另造一套规则」。所以这里没有国债 ETF、
没有人民币汇率、没有深证成指——那些是别的信号。

现状（信号 629 个）：

    A 股  596 个信号（同一个信号日，A 股日历上落在别的交易日），命中 77.7%，基率 76.6%，+1.1 个点
    2021 起 185 个，命中 81.1%，基率 77.0%，+4.1 个点

对比同一条信号在纳斯达克 100 上：命中 85.5%、基率 82.1%、+3.4 个点（样本外 +4.0）。
所以「同一套思路搬到 A 股」这件事，方向对但幅度小得多，2021 起的 +4.1 也只有 185 个点、标准差 ±3 个点，
不该当成结论。
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forecast_log.bars import Series, load_series
from forecast_log.paths import series_path

HORIZON = 5
HOLDOUT = "2015-01-01"
TARGET = "000001.SS"
TARGET_NAME = "Shanghai composite"

# fetch 用：这个市场只有结算标的需要拉，信号全部来自美股那一份。
CHINA_SYMBOLS = (TARGET,)


@dataclass(frozen=True)
class Entry:
    """一次「照着美股那个信号去买」的记录。"""

    date: str            # A 股入场那天（信号之后第一个 A 股交易日）
    us_date: str         # 信号本身落在哪个美股交易日
    close: float
    outcome: int | None  # 之后 5 个 A 股交易日里有没有一根收盘高于入场价
    target: str | None   # 窗口末尾那天，用来对账
    peak: float | None
    peak_at: str | None


def _sessions(series: Series) -> tuple[list[str], np.ndarray]:
    dates = [bar.date.isoformat() for bar in series.bars]
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    return dates, closes


def entries(root: Path) -> list[Entry]:
    """信号来自 `artifacts/cross.json`（美股日历），只把入场日和窗口换成 A 股的。"""
    signal_doc = json.loads((root / "artifacts" / "cross.json").read_text())
    series = load_series(series_path(root, TARGET))
    dates, closes = _sessions(series)
    n = len(dates)

    peak = np.full(n, np.nan)
    peak_at: list[str | None] = [None] * n
    for i in range(n - HORIZON):
        window = closes[i + 1 : i + 1 + HORIZON]
        if np.all(np.isfinite(window)):
            offset = int(np.argmax(window))
            peak[i] = window[offset]
            peak_at[i] = dates[i + 1 + offset]

    position = {day: index for index, day in enumerate(dates)}
    out: list[Entry] = []
    last_entry = -1
    for us_day, flag in zip(signal_doc["time"], signal_doc["flag"]):
        if not flag:
            continue
        # 第一个「严格晚于美股信号日」的 A 股交易日：那天开盘前，这个信号已经成立。
        index = int(np.searchsorted(np.array(dates), us_day, side="right"))
        if index >= n or index <= last_entry:
            continue
        last_entry = index
        day = dates[index]
        resolved = index + HORIZON < n and np.isfinite(peak[index])
        out.append(
            Entry(
                date=day,
                us_date=us_day,
                close=float(closes[index]),
                outcome=int(peak[index] > closes[index]) if resolved else None,
                target=dates[index + HORIZON] if resolved else None,
                peak=float(peak[index]) if resolved else None,
                peak_at=peak_at[index],
            )
        )
    return out


def document(root: Path) -> dict:
    """页面用的那一份：信号原封不动，换的只有入场日和窗口。"""
    signal_doc = json.loads((root / "artifacts" / "cross.json").read_text())
    series = load_series(series_path(root, TARGET))
    dates, closes = _sessions(series)
    n = len(dates)
    rows = entries(root)

    peak = np.full(n, np.nan)
    peak_at: list[str | None] = [None] * n
    for i in range(n - HORIZON):
        window = closes[i + 1 : i + 1 + HORIZON]
        if np.all(np.isfinite(window)):
            offset = int(np.argmax(window))
            peak[i] = window[offset]
            peak_at[i] = dates[i + 1 + offset]

    position = {day: index for index, day in enumerate(dates)}
    flagged = {position[row.date] for row in rows if row.outcome is not None}
    signal_of = {position[row.date]: row.us_date for row in rows if row.outcome is not None}
    # 只从美股信号文档覆盖的第一天开始画：那之前这条规则对 A 股没有任何发言权。
    first = int(np.searchsorted(np.array(dates), signal_doc["time"][0], side="left"))
    kept = list(range(first, n))

    def rate(pairs) -> float | None:
        pairs = list(pairs)
        return round(float(np.mean([hit for _, hit in pairs])), 4) if pairs else None

    def summary(low: str, high: str, picks) -> dict:
        window = [i for i in kept if low <= dates[i] < high]
        base = rate((dates[i], int(peak[i] > closes[i])) for i in window if np.isfinite(peak[i]))
        chosen = [i for i in window if i in picks and np.isfinite(peak[i])]
        hit = rate((dates[i], int(peak[i] > closes[i])) for i in chosen)
        return {
            "sessions": len([i for i in window if np.isfinite(peak[i])]),
            "signals": len(chosen),
            "hit": hit,
            "base_rate": base,
            "lift": None if hit is None or base is None else round(hit - base, 4),
        }

    # 分档：复合分数照旧是美股那一份，看它在 A 股上有没有单调性
    bands = []
    signal_days, signal_scores = signal_doc["time"], signal_doc["score"]
    scored = []
    pointer = 0
    for i in sorted(flagged):
        day = dates[i]
        while pointer + 1 < len(signal_days) and signal_days[pointer + 1] < day:
            pointer += 1
        scored.append((signal_scores[pointer], int(peak[i] > closes[i])))
    if scored:
        ordered = sorted(scored)
        size = len(ordered) // 10
        for band in range(10):
            chunk = ordered[band * size : (band + 1) * size] if band < 9 else ordered[9 * size :]
            bands.append({"band": band + 1, "hit": round(float(np.mean([hit for _, hit in chunk])), 4)})

    ndx = signal_doc["summary"]
    return {
        "instrument": TARGET,
        "name": TARGET_NAME,
        "question": {
            "horizon": HORIZON,
            "text": (
                f"Same cross-asset risk appetite long → does the {TARGET_NAME} print any close above the "
                f"entry price in the next {HORIZON} A-share sessions?"
            ),
            "factors": [
                "The signal is untouched: four US cross-asset factors, 90th percentile of expanding history",
                "Add: within 3 days before CPI, or 3-7 days before NFP, with the score at or above the median",
                "Remove: no point on the FOMC decision day, and no point on days with a strong bearish surprise",
            ],
            "rule": (
                "Nothing in the signal changes. A US signal dated D is actionable in China at the open of the "
                "first A-share session after D, because the US close lands at 04:00 Beijing time. Entry is that "
                f"session's close; the call is a hit if any of the next {HORIZON} A-share closes is above it."
            ),
        },
        "fetched_at": series.fetched_at,
        "first_session": dates[first] if kept else None,
        "last_session": dates[-1],
        "time": [dates[i] for i in kept],
        "close": [round(float(closes[i]), 2) for i in kept],
        "flag": [1 if i in flagged else 0 for i in kept],
        "us_date": [signal_of.get(i) for i in kept],
        "outcome": [int(peak[i] > closes[i]) if np.isfinite(peak[i]) else None for i in kept],
        "peak": [round(float(peak[i]), 2) if np.isfinite(peak[i]) else None for i in kept],
        "peak_at": [peak_at[i] for i in kept],
        "summary": {
            "all": summary("1900", "2100", flagged),
            "holdout": summary(HOLDOUT, "2100", flagged),
            "holdout_from": HOLDOUT,
        },
        "bands": bands,
        "reference": {
            "instrument": "^NDX",
            "signals": ndx["flagged"],
            "hit": ndx["hit"],
            "base_rate": ndx["base_rate"],
            "lift": round(ndx["hit"] - ndx["base_rate"], 4),
            "holdout_signals": ndx["holdout_flagged"],
            "holdout_hit": ndx["holdout_hit"],
            "holdout_base_rate": ndx["holdout_base_rate"],
            "holdout_lift": round(ndx["holdout_hit"] - ndx["holdout_base_rate"], 4),
        },
        "latest": {
            "date": dates[-1],
            "close": round(float(closes[-1]), 2),
        },
        "signals_seen": len(rows),
    }
