"""同一套信号，换一个市场结算：预测上证综指。

信号**一个字都不改**：四个因子（10 年期收益率、美元、HYG/TLT、VIX）、扩张 z 分数、90 分位阈值、
CPI / NFP / FOMC 三个日历窗口、新闻否决，全部用 `artifacts/cross.json` 里已经算好的那一份。
换掉的只有两件事：

    入场  美股 D 日收盘 = 北京 D+1 日 04:00，A 股 09:30 开盘前就知道，
          所以成交价 = D 之后**第一个 A 股交易日**的**开盘价**（不是收盘 —— 收盘那个价还买不到）
    窗口  入场那天加上之后 4 根收盘，共 5 根，任何一根高于入场价算命中

这就是「同一条规则，能不能在中国市场用」，不是「给中国另造一套规则」。所以这里没有国债 ETF、
没有人民币汇率、没有深证成指——那些是别的信号。

现状（开盘成交口径）：

    A 股      591 个信号，命中 81.2%，基率 80.3%，+0.9 个点
    2015 起   475 个，命中 84.0%，基率 81.3%，+2.7 个点（标准误 ±2.1，仍在噪声里）

对比同一条信号在纳斯达克 100 上：命中 86.0%、基率 83.4%、+2.6 个点。
所以「同一套思路搬到 A 股」这件事，方向对但幅度约为三分之一，且不显著。
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
    entry: float         # 入场价 = 那天的开盘价（09:30），不是收盘
    base_close: float | None  # 信号那天收盘，颜色的比较基准
    close3: float | None      # 信号后第 3 天收盘
    close5: float | None      # 信号后第 5 天收盘
    hit3: int | None          # 第 3 天收盘高于基准（绿）；`outcome` 是第 5 天（绿 + 黄）
    outcome: int | None  # 之后 5 个 A 股交易日里有没有一根收盘高于入场价
    target: str | None   # 窗口末尾那天，用来对账
    peak: float | None
    peak_at: str | None


def _sessions(
    series: Series,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int | None]]:
    dates = [bar.date.isoformat() for bar in series.bars]
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    opens = np.array([np.nan if bar.open is None else bar.open for bar in series.bars], dtype=float)
    highs = np.array([np.nan if bar.high is None else bar.high for bar in series.bars], dtype=float)
    lows = np.array([np.nan if bar.low is None else bar.low for bar in series.bars], dtype=float)
    volumes = [bar.volume for bar in series.bars]
    return dates, opens, highs, lows, closes, volumes


def entries(root: Path) -> list[Entry]:
    """信号来自数据库里的 NDX 信号（美股日历），只把入场日和窗口换成 A 股的。"""
    from forecast_log.artifacts_db import load_cross_market

    signal_doc = load_cross_market(root, "ndx")
    if signal_doc is None:
        raise FileNotFoundError("数据库中缺少 ndx 信号，请先运行 forecast-log cross")
    series = load_series(series_path(root, TARGET))
    dates, opens, _highs, _lows, closes, _volumes = _sessions(series)
    n = len(dates)

    peak = np.full(n, np.nan)
    peak_at: list[str | None] = [None] * n
    close3 = np.full(n, np.nan)
    close5 = np.full(n, np.nan)
    base_close = np.full(n, np.nan)
    # 判定基准 = 信号那天（箭头那天）的收盘价；之后第 3 天、第 5 天的收盘拿它比。
    # 买入价仍然记下（次日开盘），但它不决定颜色。
    for i in range(n - HORIZON):
        window = closes[i + 1 : i + 1 + HORIZON]
        if np.isfinite(opens[i]) and np.all(np.isfinite(window)):
            offset = int(np.argmax(window))
            peak[i] = window[offset]
            peak_at[i] = dates[i + 1 + offset]
            base_close[i] = closes[i]
            close3[i] = closes[i + 3]
            close5[i] = closes[i + 5]

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
                entry=float(opens[index]),
                base_close=float(base_close[index]) if resolved else None,
                close3=float(close3[index]) if resolved else None,
                close5=float(close5[index]) if resolved else None,
                hit3=int(close3[index] > base_close[index]) if resolved else None,
                outcome=int(close5[index] > base_close[index]) if resolved else None,
                target=dates[index + HORIZON] if resolved else None,
                peak=float(peak[index]) if resolved else None,
                peak_at=peak_at[index],
            )
        )
    return out


def _component_row(signal_doc: dict, dates: list[str], kept: list[int], i: int) -> dict:
    """这一根 A 股 K 线对应的那几列标签。

    能用的窗口 = 上一根 A 股 K 线之后、这一根之前收盘的那些美股交易日（春节、国庆可能有好几个）。
    窗口里若有一天打了点，就用**那一天**的全部列（否决列也跟着那一天，不能把整段窗口的
    FOMC 否决叠加进来）；没有打点就用最后一天。这样前端重算与后端口径一致。
    """
    us_days = signal_doc["time"]
    position = kept.index(i) if i in kept else 0
    previous = dates[kept[position - 1]] if position > 0 else None
    high = int(np.searchsorted(us_days, dates[i], side="left"))
    low = max(0, high - 1) if previous is None else int(np.searchsorted(us_days, previous, side="left"))
    if high <= low:
        return {key: 0 for key in signal_doc["components"]}
    components = signal_doc["components"]
    chosen = high - 1
    for j in range(high - 1, low - 1, -1):
        if components["composite"][j] or components["cal_cpi"][j] or components["cal_nfp"][j]:
            chosen = j
            break
    return {key: int(components[key][chosen]) for key in components}


def document(root: Path) -> dict:
    """页面用的那一份：信号原封不动，换的只有入场日和窗口。"""
    from forecast_log.artifacts_db import load_cross_market

    signal_doc = load_cross_market(root, "ndx")
    if signal_doc is None:
        raise FileNotFoundError("数据库中缺少 ndx 信号，请先运行 forecast-log cross")
    series = load_series(series_path(root, TARGET))
    dates, opens, highs, lows, closes, volumes = _sessions(series)
    n = len(dates)
    rows = entries(root)

    peak = np.full(n, np.nan)
    peak_at: list[str | None] = [None] * n
    close3 = np.full(n, np.nan)
    close5 = np.full(n, np.nan)
    base_close = np.full(n, np.nan)
    for i in range(n - HORIZON):
        window = closes[i + 1 : i + 1 + HORIZON]
        if np.isfinite(opens[i]) and np.all(np.isfinite(window)):
            offset = int(np.argmax(window))
            peak[i] = window[offset]
            peak_at[i] = dates[i + 1 + offset]
            base_close[i] = closes[i]
            close3[i] = closes[i + 3]
            close5[i] = closes[i + 5]

    position = {day: index for index, day in enumerate(dates)}
    flagged = {position[row.date] for row in rows if row.outcome is not None}
    all_flagged = {position[row.date] for row in rows}
    signal_of = {position[row.date]: row.us_date for row in rows}
    # 只从美股信号文档覆盖的第一天开始画：那之前这条规则对 A 股没有任何发言权。
    first = int(np.searchsorted(np.array(dates), signal_doc["time"][0], side="left"))
    kept = list(range(first, n))

    def rate(pairs) -> float | None:
        pairs = list(pairs)
        return round(float(np.mean([hit for _, hit in pairs])), 4) if pairs else None

    def split(indexes) -> dict:
        """命中按天拆开：三天内 / 只有第 4–5 天 / 五天都没做到。互斥，加起来是 1。"""
        indexes = [i for i in indexes if np.isfinite(peak[i])]
        if not indexes:
            return {"hit3": None, "hit45": None, "miss": None}
        three = sum(1 for i in indexes if close3[i] > base_close[i]) / len(indexes)
        later = sum(1 for i in indexes if close3[i] <= base_close[i] and close5[i] > base_close[i]) / len(indexes)
        return {"hit3": round(three, 4), "hit45": round(later, 4), "miss": round(1 - three - later, 4)}

    def summary(low: str, high: str, picks) -> dict:
        window = [i for i in kept if low <= dates[i] < high]
        usable = [i for i in window if np.isfinite(peak[i])]
        base = rate((dates[i], int(close5[i] > base_close[i])) for i in usable)
        chosen = [i for i in window if i in picks and np.isfinite(peak[i])]
        hit = rate((dates[i], int(close5[i] > base_close[i])) for i in chosen)
        return {
            "sessions": len(usable),
            "signals": len(chosen),
            "hit": hit,
            "base_rate": base,
            "lift": None if hit is None or base is None else round(hit - base, 4),
            "hit3": split(chosen)["hit3"],
            "hit45": split(chosen)["hit45"],
            "miss": split(chosen)["miss"],
            "base3": split(usable)["hit3"],
            "base45": split(usable)["hit45"],
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
        scored.append((signal_scores[pointer], int(close5[i] > base_close[i])))
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
                f"Same cross-asset risk appetite long → is the {TARGET_NAME} close {HORIZON} A-share sessions "
                "later above the basis close?"
            ),
            "factors": [
                "The signal is untouched: four US cross-asset factors, 90th percentile of expanding history",
                "Add: within 3 days before CPI, or 3-7 days before NFP, with the score at or above the median",
                "Remove: no point on the FOMC decision day, and no point on days with a strong bearish surprise",
            ],
            "rule": (
                "Nothing in the signal changes. A US signal dated D is known at 04:00 Beijing the next morning, "
                "so the first A-share session after D is the one that settles it. That session's close is the "
                f"basis of the call; the call is a hit if the close {HORIZON} A-share sessions later is above it. "
                "The open of that session is recorded as `entry` — the price you could actually pay — but it "
                "does not decide the call."
            ),
        },
        "fetched_at": series.fetched_at,
        "first_session": dates[first] if kept else None,
        "last_session": dates[-1],
        "time": [dates[i] for i in kept],
        "open": [round(float(opens[i]), 2) if np.isfinite(opens[i]) else None for i in kept],
        "high": [round(float(highs[i]), 2) if np.isfinite(highs[i]) else None for i in kept],
        "low": [round(float(lows[i]), 2) if np.isfinite(lows[i]) else None for i in kept],
        "close": [round(float(closes[i]), 2) for i in kept],
        "volume": [volumes[i] for i in kept],
        "entry": [round(float(opens[i]), 2) if i in all_flagged else None for i in kept],
        "base_close": [round(float(base_close[i]), 2) if i in flagged else None for i in kept],
        "close3": [round(float(close3[i]), 2) if i in flagged else None for i in kept],
        "close5": [round(float(close5[i]), 2) if i in flagged else None for i in kept],
        "hit3": [int(close3[i] > base_close[i]) if i in flagged else None for i in kept],
        "flag": [1 if i in all_flagged else 0 for i in kept],
        "us_date": [signal_of.get(i) for i in kept],
        "outcome": [int(close5[i] > base_close[i]) if np.isfinite(peak[i]) else None for i in kept],
        "peak": [round(float(peak[i]), 2) if np.isfinite(peak[i]) else None for i in kept],
        "peak_at": [peak_at[i] for i in kept],
        "summary": {
            "all": summary("1900", "2100", flagged),
            "holdout": summary(HOLDOUT, "2100", flagged),
            "holdout_from": HOLDOUT,
        },
        # 上下文勾选框要用的列。对齐规则：这一根 A 股 K 线所对应的美股收盘，
        # 必须是「上一根 A 股 K 线之后新产生的」——美股休市那天没有新信号，
        # 同一份美股收盘不能在连续两个 A 股交易日各下一次单。否则前端重算会多数出几笔。
        "components": {
            **{
                key: [_component_row(signal_doc, dates, kept, i)[key] for i in kept]
                for key in signal_doc["components"]
            },
            "ssec_below_ma20": [
                int(
                    i >= 19
                    and np.all(np.isfinite(closes[i - 19 : i + 1]))
                    and closes[i] < float(np.mean(closes[i - 19 : i + 1]))
                )
                for i in kept
            ],
            "ssec_gap_veto": [
                int(
                    i > 0
                    and np.isfinite(opens[i])
                    and np.isfinite(closes[i - 1])
                    and closes[i - 1] > 0
                    and (opens[i] / closes[i - 1] - 1.0) > 0.010
                )
                for i in kept
            ],
            "ssec_oversold_boost": [
                int(
                    i >= 5
                    and np.isfinite(closes[i])
                    and np.isfinite(closes[i - 5])
                    and closes[i - 5] > 0
                    and (closes[i] / closes[i - 5] - 1.0) < -0.030
                )
                for i in kept
            ],
        },
        "contexts": signal_doc["contexts"] + [
            {
                "key": "ssec_below_ma20",
                "label": "A-share trend filter — Below 20-day MA",
                "kind": "veto",
                "note": "上证综指收盘低于 20 日均线时否决买入（本土趋势过滤）",
                "available_from": dates[first] if kept else None,
            },
            {
                "key": "ssec_gap_veto",
                "label": "A-share open gap > +1.0%",
                "kind": "veto",
                "note": "A 股次日高开超过 +1.0% 时否决追高",
                "available_from": dates[first] if kept else None,
            },
            {
                "key": "ssec_oversold_boost",
                "label": "A-share 5-day pullback < -3%",
                "kind": "boost",
                "note": "上证综指 5 日跌幅超 3% 时补一个超跌反弹信号",
                "available_from": dates[first] if kept else None,
            },
        ],
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
            "open": round(float(opens[-1]), 2) if np.isfinite(opens[-1]) else None,
            "high": round(float(highs[-1]), 2) if np.isfinite(highs[-1]) else None,
            "low": round(float(lows[-1]), 2) if np.isfinite(lows[-1]) else None,
            "close": round(float(closes[-1]), 2),
            "volume": volumes[-1],
            "us_date": signal_doc["time"][int(np.searchsorted(signal_doc["time"], dates[-1], side="left")) - 1]
            if signal_doc["time"] and int(np.searchsorted(signal_doc["time"], dates[-1], side="left")) > 0
            else None,
            "flag": int(kept[-1] in all_flagged) if kept else 0,
        },
        "signals_seen": len(rows),
    }
