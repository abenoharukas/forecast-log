"""跨资产风险偏好：四条腿同时偏多时，指数之后 5 个交易日内更容易摸到更高价。

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

命中口径：入场价 = 信号当天收盘；之后 5 个交易日内**任何一根收盘高于入场价**就算命中
（不是「第 5 个交易日收盘更高」，也不是盘中的最高价）。

三个窗口是分开挑的，不是随手写：CPI 0-3 天、NFP 3-7 天在 2015 年前后两段都高于基率；
NFP 0-2 天低于基率（2015 年前 −2.4 个点，2015 后 −0.2 个点），
FOMC 0-2 天两段方向相反（2015 前 +3.8 个点，2015 后 −2.5 个点），所以 FOMC 只用来否决决议当天。
窗口本身是在旧口径（第 5 日收盘更高）下挑的，换口径后没有重挑，规则照旧。

现状（2026-11-15 重算，三个日历事件 + 新闻否决都用上，上面的命中口径）：

    全样本       629 个点，命中 85.5%，同期基率 82.1%
    样本外 2015+ 505 个点，命中 86.1%，同期基率 82.1%

留出期的阈值只用 2015 年之前的数据定。打点约占全部日子的 14%，其中约 101 个独立的 5 天窗口。
日历来自 LLM（`data/events.json`，129 次 FOMC / 192 次 CPI / 192 次 NFP，$0.0016），
它只提供「离 CPI 还有几天」，其余判断都在价格这一侧。
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from forecast_log import daily, events
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
    # 窗口里最高的那根收盘（页面读数用），和它落在哪天。`outcome` 就是「它比入场价高」。
    peak: float | None = None
    peak_at: str | None = None
    # 价格侧（四个因子 + 日历）自己的结论。`flag` 是再叠加新闻否决之后的结果。
    price_only: int = 0
    news_veto: int = 0
    # 逐列的信息源标签：页面上的勾选框直接对应这些列，任意组合都能在前端重算。
    composite: int = 0     # 复合分数 ≥ 90 分位
    cal_cpi: int = 0       # CPI 前 3 天且分数 ≥ 中位
    cal_nfp: int = 0       # NFP 前 3-7 天且分数 ≥ 中位
    fomc_day: int = 0      # 决议当天（否决条件）
    news_bearish: int = 0  # LLM：强利空意外（否决条件）
    news_bullish: int = 0  # LLM：强利多意外（加分条件）
    riskoff_hits: int = 0  # 正则：risk-off 概念命中数（可选否决）


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


# 页面上的勾选框清单。加一个信息源 = 在 components 里加一列 + 在这里加一条，
# 前端不用改：它按 kind 决定这一列是「触发」「否决」还是「加分」。
CONTEXTS: list[dict] = [
    {
        "key": "composite",
        "label": "Cross-asset composite (4 factors)",
        "kind": "trigger",
        "required": True,          # 基础条件：关掉它就没有任何信号，界面锁死
        "note": "z-score of yields, dollar, credit, VIX; signal above the 90th percentile",
        "available_from": None,
    },
    {
        "key": "cal_cpi",
        "label": "CPI window (3 days before)",
        "kind": "trigger",
        "note": "score at or above the median within 3 days of a CPI release",
        "available_from": "2011-01-01",
    },
    {
        "key": "cal_nfp",
        "label": "NFP window (3-7 days before)",
        "kind": "trigger",
        "note": "score at or above the median 3-7 days before payrolls",
        "available_from": "2011-01-01",
    },
    {
        "key": "fomc_day",
        "label": "FOMC decision day",
        "kind": "veto",
        "note": "no signal on the day the Fed decides",
        "available_from": "2011-01-01",
    },
    {
        "key": "news_bearish",
        "label": "News surprise (LLM)",
        "kind": "veto",
        "note": "one LLM read per day; a strong bearish surprise removes that day's signal",
        "available_from": "2002-01-01",
    },
]


def _news_labels(root: Path) -> tuple[dict[str, dict], dict]:
    """新闻侧的标签，两种来源都读：LLM 的「意外」标注优先，退回事件标签。

    这是第一个**进入 `flag` 判定**的新闻因子。它能不能留下，看 `document()` 里
    `with_news` 和 `price_only` 两套数的差 —— 页面上是并排显示的，不藏。
    """
    labels: dict[str, dict] = {}
    provenance: dict = {"source": None, "days": 0, "veto_days": 0}

    from forecast_log import surprise as surprise_module

    compact = surprise_module.load_compact(root)
    if compact:
        for day, label in compact.items():
            labels[day] = {
                "surprise": label.get("surprise", "none"),
                "direction": label.get("direction", "ambiguous"),
                "prompt_version": label.get("prompt_version", "surprise-v1"),
                "model": label.get("model", "deepseek-chat"),
            }
        provenance.update(source="artifacts/surprise.jsonl (LLM: surprise x direction)", days=len(labels))

    surprise_path = root / "artifacts" / "surprise"
    if not labels and surprise_path.is_dir():
        for path in surprise_path.glob("????-??-??.json"):
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            label = record.get("label") or {}
            labels[record["as_of"]] = {
                "surprise": label.get("surprise", "none"),
                "direction": label.get("direction", "ambiguous"),
                "prompt_version": record.get("prompt_version"),
                "model": record.get("model"),
            }
        provenance.update(source="artifacts/surprise (LLM: surprise x direction)", days=len(labels))
    provenance["veto_days"] = sum(
        1 for label in labels.values() if label["surprise"] == "strong" and label["direction"] == "bearish"
    )
    return labels, provenance


def _keyword_features(root: Path, dates: list[str]) -> dict[str, int]:
    """每天标题里命中的 risk-off 概念数。没有当天记录就是 0。"""
    from forecast_log.keywords import scan

    news_dir = root / "data" / "news"
    if not news_dir.is_dir():
        return {}
    wanted = set(dates)
    out: dict[str, int] = {}
    for path in news_dir.glob("????-??-??.json"):
        if path.stem not in wanted:
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        titles = [document.get("title", "") for document in record.get("documents", [])]
        if titles:
            out[path.stem] = len(scan(titles)["risk_off_hits"])
    return out


def calls(root: Path, series: Series) -> list[Call]:
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    n = len(closes)
    scores = composite(root, series)
    dates = [bar.date.isoformat() for bar in series.bars]
    calendar = _calendar_distance(root, dates)
    news, _ = _news_labels(root)
    keyword_features = _keyword_features(root, dates)
    # 入场价就是当天收盘；只要之后 5 个交易日里有任何一根收盘高过它就算命中。
    # `peak_at` 记下高在哪天（可能不是第 5 天），`target` 仍然是窗口末尾那天，用来对账。
    peak = np.full(n, np.nan)
    peak_at: list[str | None] = [None] * n
    for i in range(n - HORIZON):
        window = closes[i + 1 : i + 1 + HORIZON]
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
        # ---- 逐列标签：后面所有组合都从这些列算出来，不再各写一套逻辑
        label = news.get(dates[i]) or {}
        bearish = int(label.get("surprise") == "strong" and label.get("direction") == "bearish")
        bullish = int(label.get("surprise") == "strong" and label.get("direction") == "bullish")
        if calendar is None:
            cal_cpi = cal_nfp = fomc_day = 0
        else:
            above_middle = scores[i] >= middle
            low, high = NFP_WINDOW
            cal_cpi = int(above_middle and calendar["cpi"][i] <= CPI_WINDOW)
            cal_nfp = int(above_middle and low <= calendar["nfp"][i] <= high)
            fomc_day = int(calendar["fomc"][i] == 0)
        composite_flag = int(scores[i] >= threshold)

        price_only = flag
        # 默认组合 = 复合分数 + 日历触发 − FOMC 当天 − 强利空意外
        flag = int(bool(composite_flag or cal_cpi or cal_nfp) and not fomc_day and not bearish)
        resolved = i + HORIZON < n
        outcome = int(peak[i] > closes[i]) if resolved else None
        out.append(
            Call(
                date=dates[i],
                close=float(closes[i]),
                score=float(scores[i]),
                threshold=threshold,
                flag=flag,
                outcome=outcome,
                target=dates[i + HORIZON] if resolved else None,
                peak=float(peak[i]) if resolved else None,
                peak_at=peak_at[i],
                price_only=price_only,
                news_veto=int(price_only and not flag),
                composite=composite_flag,
                cal_cpi=cal_cpi,
                cal_nfp=cal_nfp,
                fomc_day=fomc_day,
                news_bearish=bearish,
                news_bullish=bullish,
                riskoff_hits=keyword_features.get(dates[i], 0),
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


def _runtime_context(root: Path) -> dict | None:
    """当日 LLM 读新闻的结果（`forecast-log daily` 写的），没有就没有这一块。

    只做展示和记账：信号由价格侧因子决定，LLM 说偏空不会把信号吞掉，只标成 contested。
    """
    record = daily.load(root)
    if record is None:
        return None
    label = record.get("label") or {}
    news = record.get("news") or {}
    return {
        "as_of": record.get("as_of"),
        "generated_at": record.get("generated_at"),
        "model": record.get("model"),
        "prompt_version": record.get("prompt_version"),
        "tone": label.get("tone"),
        "confidence": label.get("confidence"),
        "drivers": label.get("drivers") or [],
        "note": label.get("note"),
        "headlines_used": news.get("used"),
        "headlines_retrieved": news.get("retrieved"),
        "dropped_after_as_of": news.get("dropped_after_as_of"),
        "source": news.get("source"),
        "errors": news.get("errors") or [],
        "contested": record.get("contested"),
        "keywords": record.get("keywords"),
        "veto_watch": record.get("veto_watch"),
        "cost_usd": (record.get("usage") or {}).get("cost_usd"),
    }


def document(root: Path, series: Series, holdout: str = "2015-01-01") -> dict:
    rows = calls(root, series)
    resolved = [row for row in rows if row.outcome is not None]

    def tally(selected: list[Call]) -> dict:
        flagged = [row for row in selected if row.flag]
        held = [row for row in selected if row.date >= holdout]
        held_flag = [row for row in held if row.flag]
        return {
            "n": len(selected),
            "base_rate": round(float(np.mean([row.outcome for row in selected])), 4) if selected else None,
            "flagged": len(flagged),
            "hit": round(float(np.mean([row.outcome for row in flagged])), 4) if flagged else None,
            "holdout_n": len(held),
            "holdout_base_rate": round(float(np.mean([row.outcome for row in held])), 4) if held else None,
            "holdout_flagged": len(held_flag),
            "holdout_hit": round(float(np.mean([row.outcome for row in held_flag])), 4) if held_flag else None,
        }

    price_rows = [Call(**{**row.__dict__, "flag": row.price_only}) for row in resolved]
    with_news = tally(resolved)
    price_only = tally(price_rows)
    _, news_provenance = _news_labels(root)
    news_provenance["vetoed_signals"] = sum(row.news_veto for row in resolved)

    return {
        "question": {
            "horizon": HORIZON,
            "text": f"Cross-asset risk appetite long → does the index print any close above the entry price in the next {HORIZON} sessions?",
            "legs": [
                "Factor 1 — 10Y Treasury yield down over 20 sessions",
                "Factor 2 — dollar index weaker over 20 sessions",
                "Factor 3 — credit HYG/TLT stronger over 20 sessions",
                "Factor 4 — VIX above its one-year median",
                "Add: within 3 days before CPI, or 3-7 days before NFP, with the average at or above the median (calendar pulled by the LLM)",
                "Remove: no point on the FOMC decision day",
                "Remove: LLM reads a strong bearish surprise in the day's news (news veto; see the with/without table)",
            ],
            "rule": (
                f"Average the four standardised factors; mark a point above the {int(QUANTILE * 100)}th percentile "
                "of all history up to that day, or within 3 days before CPI / 3-7 days before NFP with the score "
                "at or above the median; no point on the FOMC decision day; and the news veto removes signals on "
                "days the LLM labels as a strong bearish surprise. The entry price is that session's close; the call "
                f"is a hit if any of the next {HORIZON} closes is above it."
            ),
        },
        "fetched_at": series.fetched_at,
        "calendar": _calendar_provenance(root),
        "news": news_provenance,
        "time": [row.date for row in rows],
        "close": [round(row.close, 2) for row in rows],
        "score": [round(row.score, 4) for row in rows],
        "threshold": [round(row.threshold, 4) for row in rows],
        "flag": [row.flag for row in rows],
        "flag_price_only": [row.price_only for row in rows],
        "components": {
            "composite": [row.composite for row in rows],
            "cal_cpi": [row.cal_cpi for row in rows],
            "cal_nfp": [row.cal_nfp for row in rows],
            "fomc_day": [row.fomc_day for row in rows],
            "news_bearish": [row.news_bearish for row in rows],
            "news_bullish": [row.news_bullish for row in rows],
            "riskoff_hits": [row.riskoff_hits for row in rows],
        },
        "contexts": [
            {**item, "available_from": item.get("available_from") or rows[0].date}
            if item["key"] == "composite"
            else item
            for item in CONTEXTS
        ],
        "news_veto": [row.news_veto for row in rows],
        "outcome": [row.outcome for row in rows],
        "target": [row.target for row in rows],
        "peak": [round(row.peak, 2) if row.peak is not None else None for row in rows],
        "peak_at": [row.peak_at for row in rows],
        "summary": {**with_news, "holdout_from": holdout},
        "summary_price_only": {**price_only, "holdout_from": holdout},
        "context": _runtime_context(root),
        "latest": {
            "date": rows[-1].date,
            "close": round(rows[-1].close, 2),
            "score": round(rows[-1].score, 4),
            "threshold": round(rows[-1].threshold, 4),
            "flag": rows[-1].flag,
            "flag_price_only": rows[-1].price_only,
            "news_veto": bool(rows[-1].news_veto),
        },
    }
