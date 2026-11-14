"""同样的跨资产风险偏好，搬到 A 股：四根腿换成本地输入。

这个文件回答的是「同一套思路能不能挪到上证综指」，不是「再挑一套能赢的腿」。
所以四根腿按**角色**对应过来，不是按相关性挑出来的：

    美国那份                           A 股这份
    10 年期收益率 20 日变化 下行 → 多   5 年期国债 ETF 20 日变化 上行 → 多（收益率下行）
    美元 20 日涨跌 走弱 → 多            USDCNY 20 日涨跌 下行 → 多（人民币走强）
    HYG/TLT（信用）20 日涨跌 走强 → 多  深证成指 / 上证综指 20 日涨跌 上行 → 多（小盘相对大盘）
    VIX 相对一年中位 偏高 → 多          ^VIX 相对一年中位 偏高 → 多（同一根，全球波动率）

打点规则、命中口径、90 分位阈值、扩张 z 分数全部照抄 `cross.py`：分数 >= 当天之前所有历史的
90 分位就打点，入场价 = 当天收盘，之后 5 个交易日内**任何一根收盘高于入场价**算命中。

**结论是没有结论**：369 个点，命中 77.8%，同期基率 78.4%，差 −0.7 个点；
换六种腿的组合，全样本在 −6.3 到 +3.7 个点之间来回，符号不稳定；
按复合分数分十档，命中率也不单调（第 4、5 档最高，第 8、10 档最低）。

为什么样本从 2013 开始：国债 ETF（511010.SS）2013-03 才有，这是四根腿里最短的一根。

为什么没有日历触发和新闻否决：那两层的日历（`data/events.json`）和新闻标签都是美国的 CPI / NFP /
FOMC 和英文标题，套到 A 股上是错的输入，不是「同一套思路」。所以这里只有 90 分位那一条线。
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forecast_log import cross
from forecast_log.bars import Series, load_series
from forecast_log.paths import series_path

HORIZON = cross.HORIZON
QUANTILE = cross.QUANTILE
MIN_HISTORY = cross.MIN_HISTORY

# fetch 用：结算标的 + 四根腿背后的行情
CHINA_SYMBOLS = ("000001.SS", "399001.SZ", "^HSI", "USDCNY=X", "511010.SS")

# 按角色对应，不按好坏挑
LEGS = (
    ("511010.SS", "diff20", 1.0),
    ("USDCNY=X", "ret20", -1.0),
    ("399001.SZ/000001.SS", "ret20", 1.0),
    ("^VIX", "offset", 1.0),
)

LEG_LABELS = {
    "511010.SS": "SSE 5Y govt bond ETF up (yields down)",
    "USDCNY=X": "yuan stronger (USDCNY down)",
    "399001.SZ/000001.SS": "small caps vs large caps (SZSE / SSE)",
    "^HSI": "Hang Seng up (offshore China)",
    "GC=F": "gold down",
    "^VIX": "VIX above its one-year median",
}

# 每条腿单独当信号，看得清「哪一根在起作用、往哪边」
SINGLE_LEGS = (
    ("511010.SS", "diff20", 1.0),
    ("USDCNY=X", "ret20", -1.0),
    ("399001.SZ/000001.SS", "ret20", 1.0),
    ("^HSI", "ret20", 1.0),
    ("GC=F", "ret20", -1.0),
    ("^VIX", "offset", 1.0),
)

# 换腿的组合，用来看结论是不是只挂在某一种搭配上
VARIANTS = {
    "Rates, FX, breadth, VIX": LEGS,
    "Rates, FX, Hang Seng, VIX": (
        ("511010.SS", "diff20", 1.0),
        ("USDCNY=X", "ret20", -1.0),
        ("^HSI", "ret20", 1.0),
        ("^VIX", "offset", 1.0),
    ),
    "Rates, FX, breadth, Hang Seng": (
        ("511010.SS", "diff20", 1.0),
        ("USDCNY=X", "ret20", -1.0),
        ("399001.SZ/000001.SS", "ret20", 1.0),
        ("^HSI", "ret20", 1.0),
    ),
    "FX, breadth, Hang Seng, VIX": (
        ("USDCNY=X", "ret20", -1.0),
        ("399001.SZ/000001.SS", "ret20", 1.0),
        ("^HSI", "ret20", 1.0),
        ("^VIX", "offset", 1.0),
    ),
    "FX, breadth, Hang Seng, gold": (
        ("USDCNY=X", "ret20", -1.0),
        ("399001.SZ/000001.SS", "ret20", 1.0),
        ("^HSI", "ret20", 1.0),
        ("GC=F", "ret20", -1.0),
    ),
    "FX, gold, breadth, VIX": (
        ("USDCNY=X", "ret20", -1.0),
        ("GC=F", "ret20", -1.0),
        ("399001.SZ/000001.SS", "ret20", 1.0),
        ("^VIX", "offset", 1.0),
    ),
}

# 2013 年才有国债 ETF，留出期只能往后放；2015 会只剩不到一年
HOLDOUT = "2021-01-01"


@dataclass(frozen=True)
class Call:
    date: str
    close: float
    score: float
    threshold: float
    flag: int
    outcome: int | None
    target: str | None
    peak: float | None
    peak_at: str | None


def calls(root: Path, series: Series, legs=LEGS) -> list[Call]:
    """上证日历上的信号。没有日历触发、没有新闻否决，只有 90 分位这一条线。"""
    dates = [bar.date.isoformat() for bar in series.bars]
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    scores = cross.composite(root, series, legs)
    n = len(dates)

    peak = np.full(n, np.nan)
    peak_at: list[str | None] = [None] * n
    for i in range(n - HORIZON):
        window = closes[i + 1 : i + 1 + HORIZON]
        if not np.all(np.isfinite(window)):
            continue
        offset = int(np.argmax(window))
        peak[i] = window[offset]
        peak_at[i] = dates[i + 1 + offset]

    out = []
    for i in range(n):
        if np.isnan(scores[i]) or i < MIN_HISTORY:
            continue
        history = scores[: i + 1]
        history = history[~np.isnan(history)]
        if history.size < MIN_HISTORY:
            continue
        threshold = float(np.quantile(history, QUANTILE))
        resolved = i + HORIZON < n and np.isfinite(peak[i])
        out.append(
            Call(
                date=dates[i],
                close=float(closes[i]),
                score=float(scores[i]),
                threshold=threshold,
                flag=int(scores[i] >= threshold),
                outcome=int(peak[i] > closes[i]) if resolved else None,
                target=dates[i + HORIZON] if resolved else None,
                peak=float(peak[i]) if resolved else None,
                peak_at=peak_at[i],
            )
        )
    return out


def _tally(rows: list[Call], holdout: str) -> dict:
    resolved = [row for row in rows if row.outcome is not None]
    flagged = [row for row in resolved if row.flag]
    held = [row for row in resolved if row.date >= holdout]
    held_flag = [row for row in held if row.flag]

    def mean(selected: list[Call]) -> float | None:
        return round(float(np.mean([row.outcome for row in selected])), 4) if selected else None

    return {
        "n": len(resolved),
        "base_rate": mean(resolved),
        "flagged": len(flagged),
        "hit": mean(flagged),
        "holdout_from": holdout,
        "holdout_n": len(held),
        "holdout_base_rate": mean(held),
        "holdout_flagged": len(held_flag),
        "holdout_hit": mean(held_flag),
    }


def _lift(summary: dict, key: str = "") -> float | None:
    hit = summary[f"{key}hit"]
    base = summary[f"{key}base_rate"]
    return None if hit is None or base is None else round(hit - base, 4)


def _study(root: Path, series: Series, legs, holdout: str = HOLDOUT) -> dict:
    rows = calls(root, series, legs)
    summary = _tally(rows, holdout)
    return {
        "signals": summary["flagged"],
        "hit": summary["hit"],
        "base_rate": summary["base_rate"],
        "lift": _lift(summary),
        "holdout_signals": summary["holdout_flagged"],
        "holdout_hit": summary["holdout_hit"],
        "holdout_base_rate": summary["holdout_base_rate"],
        "holdout_lift": _lift(summary, "holdout_"),
        "n": summary["n"],
        "holdout_n": summary["holdout_n"],
    }


def _deciles(rows: list[Call]) -> list[dict]:
    """复合分数的十档命中率。单调下降/上升才叫「有方向」，这里是为了给人看一眼。"""
    resolved = [row for row in rows if row.outcome is not None]
    if not resolved:
        return []
    ordered = sorted(resolved, key=lambda row: row.score)
    out = []
    size = len(ordered) // 10
    for band in range(10):
        chunk = ordered[band * size : (band + 1) * size] if band < 9 else ordered[9 * size :]
        out.append(
            {
                "band": band + 1,
                "from": round(chunk[0].score, 2),
                "to": round(chunk[-1].score, 2),
                "hit": round(float(np.mean([row.outcome for row in chunk])), 4),
            }
        )
    return out


def _imported(root: Path, series: Series) -> dict | None:
    """把那套美国信号原样搬过来：用「上一个美股收盘」（它在北京开盘前就知道）打点。

    美股 D 日收盘 = 北京 D+1 日 04:00，所以 A 股 D+1 日开盘前能看到，入场价用那天收盘。
    这是「不改造信号、只换市场」的那一版，和上面的本地四根腿是两件事。
    """
    path = root / "artifacts" / "cross.json"
    if not path.is_file():
        return None
    primary = json.loads(path.read_text())
    dates = [bar.date.isoformat() for bar in series.bars]
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    n = len(dates)
    peak = np.full(n, np.nan)
    for i in range(n - HORIZON):
        window = closes[i + 1 : i + 1 + HORIZON]
        if np.all(np.isfinite(window)):
            peak[i] = window.max()
    pointer, flag = -1, np.zeros(n, dtype=int)
    session_days = primary["time"]
    session_flags = primary["flag"]
    for i, day in enumerate(dates):
        while pointer + 1 < len(session_days) and session_days[pointer + 1] < day:
            pointer += 1
        if pointer >= 0 and session_days[pointer] < day:
            flag[i] = session_flags[pointer]
    out = []
    for i in range(n):
        if not np.isfinite(peak[i]):
            continue
        out.append((dates[i], int(flag[i]), int(peak[i] > closes[i])))

    def block(low: str, high: str) -> dict | None:
        chosen = [row for row in out if low <= row[0] < high]
        picked = [row for row in chosen if row[1]]
        if not picked:
            return None
        base = float(np.mean([row[2] for row in chosen]))
        hit = float(np.mean([row[2] for row in picked]))
        return {
            "n": len(chosen),
            "signals": len(picked),
            "hit": round(hit, 4),
            "base_rate": round(base, 4),
            "lift": round(hit - base, 4),
        }

    return {
        "source": "artifacts/cross.json",
        "entry": "the first SSEC session after the US signal day (entry = that session's close)",
        "all": block("1900", "2100"),
        "holdout": block(HOLDOUT, "2100"),
    }


def document(root: Path) -> dict:
    series = load_series(series_path(root, "000001.SS"))
    legs_doc = []
    for symbol, how, sign in SINGLE_LEGS:
        study = _study(root, series, ((symbol, how, sign),))
        legs_doc.append(
            {
                "label": LEG_LABELS.get(symbol, symbol),
                "sign": "higher is risk-on" if sign > 0 else "lower is risk-on",
                **study,
            }
        )
    variants = []
    for name, legs in VARIANTS.items():
        variants.append({"name": name, **{k: v for k, v in _study(root, series, legs).items() if k in ("signals", "hit", "base_rate", "lift", "holdout_signals", "holdout_hit", "holdout_lift")}})

    rows = calls(root, series)
    summary = _tally(rows, HOLDOUT)

    return {
        "instrument": "000001.SS",
        "name": "Shanghai composite",
        "question": {
            "horizon": HORIZON,
            "text": (
                "Cross-asset risk appetite long → does the Shanghai composite print any close above "
                f"the entry price in the next {HORIZON} sessions?"
            ),
            "legs": [
                "Factor 1 — SSE 5Y government bond ETF up over 20 sessions (yields down)",
                "Factor 2 — yuan stronger over 20 sessions (USDCNY down)",
                "Factor 3 — small caps ahead of large caps over 20 sessions (SZSE / SSE)",
                "Factor 4 — VIX above its one-year median",
            ],
            "rule": (
                f"Average the four standardised factors and mark a point above the {int(QUANTILE * 100)}th "
                "percentile of all history up to that day. The entry price is that session's close; the call is "
                f"a hit if any of the next {HORIZON} closes is above it. No CPI/NFP/FOMC layer: those calendars "
                "and the news labels are American inputs, not an A-share signal."
            ),
        },
        "fetched_at": series.fetched_at,
        "first_session": rows[0].date if rows else None,
        "last_session": rows[-1].date if rows else None,
        "time": [row.date for row in rows],
        "close": [round(row.close, 2) for row in rows],
        "score": [round(row.score, 4) for row in rows],
        "threshold": [round(row.threshold, 4) for row in rows],
        "flag": [row.flag for row in rows],
        "outcome": [row.outcome for row in rows],
        "target": [row.target for row in rows],
        "peak": [round(row.peak, 2) if row.peak is not None else None for row in rows],
        "peak_at": [row.peak_at for row in rows],
        "summary": {**summary, "holdout_from": HOLDOUT},
        "legs": legs_doc,
        "variants": variants,
        "deciles": _deciles(rows),
        "imported": _imported(root, series),
        "latest": {
            "date": rows[-1].date,
            "close": round(rows[-1].close, 2),
            "score": round(rows[-1].score, 4),
            "threshold": round(rows[-1].threshold, 4),
            "flag": rows[-1].flag,
        },
    }
