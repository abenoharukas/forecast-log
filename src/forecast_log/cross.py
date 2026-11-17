"""跨资产风险偏好：四个因子同时偏多时，指数之后 5 个交易日内更容易摸到更高价。

四个因子（都只用当天已知的收盘）：

    10 年期收益率 20 日变化   下行 → 偏多
    美元 20 日涨跌            走弱 → 偏多
    信用 HYG/TLT 20 日涨跌    走强 → 偏多
    VIX 相对一年中位          偏高 → 偏多

每个因子做扩张 z 分数（只用当天及之前），四个取平均，得到复合分数。

打点规则三个日历事件都用上（日历由 LLM 拉，`data/events.json`）：

    加：复合分数 >= 当天之前所有历史的 90 分位
    加：离下一次 CPI <= 3 天，且分数 >= 历史中位
    加：离下一次 NFP 在 3 到 7 天之间，且分数 >= 历史中位
    减：FOMC 决议当天不打点（那天市场在等结果，方向最随机）

成交口径：信号在 D 日**收盘后**成立，最早只能在 D+1 **开盘**成交，所以入场价 = D+1 的开盘价，
窗口 = D+1 起 5 根收盘，**任何一根高于入场价**就算命中（不是「第 5 个交易日收盘更高」，也不是盘中最高价）。
不能用 D 日自己的收盘当入场价：那个价是判断依据，同时也是已经过去的价格。

三个窗口是分开挑的，不是随手写：CPI 0-3 天、NFP 3-7 天在 2015 年前后两段都高于基率；
NFP 0-2 天低于基率（2015 年前 −2.4 个点，2015 后 −0.2 个点），
FOMC 0-2 天两段方向相反（2015 前 +3.8 个点，2015 后 −2.5 个点），所以 FOMC 只用来否决决议当天。
窗口本身是在旧口径（第 5 日收盘更高）下挑的，换口径后没有重挑，规则照旧。

现状（2026-11-14 重算，开盘成交口径，三个日历事件 + 新闻否决都用上）：

    价格侧       726 个点，命中 86.5%，同期基率 83.4%
    再加新闻否决 629 个点，命中 86.0%，同期基率 83.4%（否决反而是减分项，见 README）

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

HORIZON = 5
QUANTILE = 0.90
MIN_HISTORY = 252

# fetch 用：这四个因子背后的行情，以及备用（广度、黄金）
CROSS_SYMBOLS = ("^TNX", "DX-Y.NYB", "HYG", "TLT", "SPY", "RSP", "GC=F", "^VIX")

FACTORS = (
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
        return _aligned(dates, load_series(root, left)) / _aligned(
            dates, load_series(root, right)
        )
    return _aligned(dates, load_series(root, symbol))


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


def composite(root: Path, series: Series, factors=FACTORS) -> np.ndarray:
    """复合分数。四个因子缺一个就是 NaN，不拿三个糊弄。

    `factors` 可换：换一组跨资产因子，标准化和取平均的方式不动。
    """
    dates = np.array([np.datetime64(bar.date) for bar in series.bars])
    parts = []
    for symbol, how, sign in factors:
        raw = _leg(_prices(root, symbol, dates), how)
        parts.append(sign * expanding_z(raw))
    stacked = np.column_stack(parts)
    return np.where(np.isnan(stacked).any(axis=1), np.nan, stacked.mean(axis=1))


def _vendor_name(slug: str) -> str:
    """给页面上的人话名字：deepseek-flash → DeepSeek · flash。"""
    head, _, tail = slug.partition("-")
    pretty = {"deepseek": "DeepSeek", "qwen": "Qwen"}.get(head.split(".")[0], head)
    if head.startswith("qwen"):
        pretty = "Qwen"
        tail = head[len("qwen") :].lstrip(".") + (f"-{tail}" if tail else "")
    return f"{pretty} · {tail}" if tail else pretty


def _news_providers(root: Path, dates: list[str]) -> tuple[dict[str, list[int]], list[dict]]:
    """每个读过新闻的模型，各自一列否决位。

    标签是分批实验产出的（`artifacts/news_ab/<模型>/`），覆盖天数远少于价格序列，所以
    没标过的日子一律记 0（不否决），并把覆盖区间写进 `note`，页面上看得见。
    加这一列不改变 `flag`：默认选中的仍是原来那一个 `news_bearish`。
    """
    base = root / "artifacts" / "news_ab"
    if not base.is_dir():
        return {}, []
    columns: dict[str, list[int]] = {}
    contexts: list[dict] = []
    for folder in sorted(path for path in base.iterdir() if path.is_dir()):
        if folder.name.startswith("deepseek"):
            # DeepSeek 已由全量的 news_bearish (deepseek-chat) 代表，避免列表中重复出现两个 DeepSeek
            continue
        labels: dict[str, float] = {}
        for path in folder.glob("????-??-??.json"):
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            labels[record["as_of"]] = float(record.get("label", {}).get("severity", 0.0))
        if not labels:
            continue
        key = f"news_{folder.name}"
        columns[key] = [1 if labels.get(day, 0.0) >= 0.6 else 0 for day in dates]
        ordered = sorted(labels)
        contexts.append(
            {
                "key": key,
                "label": f"News score — {_vendor_name(folder.name)}",
                "kind": "veto",
                "note": (
                    f"severity ≥ 0.6 on a labeled day removes that day's signal; "
                    f"{len(labels)} of {len(dates)} days labeled ({ordered[0]} → {ordered[-1]}), "
                    f"unlabeled days never veto"
                ),
                "group": "news",
                "group_label": "News source",
                "available_from": ordered[0],
            }
        )
    return columns, contexts


@dataclass(frozen=True)
class Call:
    date: str
    close: float
    score: float
    threshold: float
    flag: int
    outcome: int | None
    target: str | None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: int | None = None
    # 真正能成交的那个价：信号成立之后第一个交易日的开盘价。
    entry: float | None = None
    entry_at: str | None = None
    # 第 3 根、第 5 根收盘。判定只看这两个数，不看「期间任意一根」。
    base_close: float | None = None   # 信号那天收盘，颜色的比较基准
    close3: float | None = None       # 信号后第 3 天收盘
    close5: float | None = None       # 信号后第 5 天收盘
    # `hit3` = 第 3 天收盘高于买入价（绿）。`outcome` = 第 5 天收盘高于买入价（绿 + 黄）。
    hit3: int | None = None
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
    is_closed: int = 1     # 该交易日行情是否已正式收盘（0=盘中未收盘，1=已收盘）


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
        "label": "News score — DeepSeek",
        "kind": "veto",
        "note": "one LLM read per day; a strong bearish surprise removes that day's signal (deepseek-chat)",
        "available_from": "2002-01-01",
        "group": "news",
        "group_label": "News source",
    },
    {
        "key": "news_bullish",
        "label": "News score — Strong bullish surprise",
        "kind": "boost",
        "note": "LLM reads a strong bullish surprise; adds a signal when price triggers did not fire",
        "available_from": "2002-01-01",
    },
    {
        "key": "riskoff_hits",
        "label": "Keyword scan — Risk-off headlines (≥ 2)",
        "kind": "veto",
        "threshold": 2,
        "note": "regex scan hits 2 or more risk-off concepts in daily headlines",
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


def calls(root: Path, series: Series, target: Series | None = None) -> list[Call]:
    """Signals from `series`'s calendar; outcomes from `target`'s prices when given.

    The composite, the calendar triggers and the news veto never read the price of
    the thing being predicted, so passing another instrument as `target` keeps every
    `flag` bit-for-bit and only re-scores the entries and exits. That is what makes
    the QQQ page a fair check of the same signalling rule.
    """
    dates = [bar.date.isoformat() for bar in series.bars]
    n = len(dates)
    scores = composite(root, series)
    if target is None:
        closes = np.array([bar.close for bar in series.bars], dtype=float)
        opens = np.array(
            [np.nan if bar.open is None else bar.open for bar in series.bars], dtype=float
        )
        highs = np.array(
            [np.nan if bar.high is None else bar.high for bar in series.bars], dtype=float
        )
        lows = np.array(
            [np.nan if bar.low is None else bar.low for bar in series.bars], dtype=float
        )
        volumes = [bar.volume for bar in series.bars]
        closed_flags = [1 if bar.is_closed else 0 for bar in series.bars]
    else:
        # 精确按日期取价：缺的那天就是「测不了」，不拿前一天的收盘顶替。
        def _num(val: float | None) -> float:
            return np.nan if val is None else float(val)

        price = {
            bar.date.isoformat(): (bar.open, bar.high, bar.low, bar.close, bar.volume)
            for bar in target.bars
        }
        target_closed = {bar.date.isoformat(): bool(bar.is_closed) for bar in target.bars}
        opens = np.array([_num(price.get(day, (None,) * 5)[0]) for day in dates], dtype=float)
        highs = np.array([_num(price.get(day, (None,) * 5)[1]) for day in dates], dtype=float)
        lows = np.array([_num(price.get(day, (None,) * 5)[2]) for day in dates], dtype=float)
        closes = np.array([_num(price.get(day, (None,) * 5)[3]) for day in dates], dtype=float)
        volumes = [price.get(day, (None,) * 5)[4] for day in dates]
        closed_flags = [
            1 if (bar.is_closed and target_closed.get(bar.date.isoformat(), True)) else 0
            for bar in series.bars
        ]
    calendar = _calendar_distance(root, dates)
    news, _ = _news_labels(root)
    keyword_features = _keyword_features(root, dates)
    # 信号在 D 日收盘后才成立，所以最早只能在 D+1 的**开盘**成交。
    # 入场价取 D+1 的开盘价：用 D 日自己的收盘当入场价，等于拿同一个价既做判断又做成交，
    # 收盘那一刻信号才成立，那个价已经拿不到了。
    # 窗口仍然是入场日加上之后 4 个收盘，共 5 根，任何一根高过入场价就算命中。
    # `close3` / `close5` 是第 3 根、第 5 根收盘。判定就用这两个数，不再看「期间任意一根」：
    # 第 3 天收盘高于买入价 = 绿；第 3 天没到、第 5 天到了 = 黄；第 5 天还没到 = 灰。
    entry = np.full(n, np.nan)
    entry_at: list[str | None] = [None] * n
    peak = np.full(n, np.nan)
    peak_at: list[str | None] = [None] * n
    close3 = np.full(n, np.nan)
    close5 = np.full(n, np.nan)
    base_close = np.full(n, np.nan)
    # 判定基准 = 信号那天的收盘价（你预测时屏幕上那个价）。之后的第 3 天、第 5 天收盘拿它比。
    # 买入价仍然记下（次日开盘，唯一能成交的价），但它不决定颜色。
    for i in range(n - HORIZON - 1):
        j = i + 1
        window = closes[j + 1 : j + 1 + HORIZON]
        if not np.isfinite(opens[j]) or not np.all(np.isfinite(window)):
            continue
        entry[i] = opens[j]
        entry_at[i] = dates[j]
        base_close[i] = closes[i]
        close3[i] = closes[i + 3]
        close5[i] = closes[i + 5]
        offset = int(np.argmax(window))
        peak[i] = window[offset]
        peak_at[i] = dates[j + 1 + offset]
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
        resolved = np.isfinite(peak[i]) and np.isfinite(entry[i])
        outcome = int(close5[i] > base_close[i]) if resolved else None
        out.append(
            Call(
                date=dates[i],
                open=float(opens[i]) if np.isfinite(opens[i]) else None,
                high=float(highs[i]) if np.isfinite(highs[i]) else None,
                low=float(lows[i]) if np.isfinite(lows[i]) else None,
                close=float(closes[i]),
                volume=volumes[i],
                entry=float(entry[i]) if resolved else None,
                entry_at=entry_at[i],
                base_close=float(base_close[i]) if resolved else None,
                close3=float(close3[i]) if resolved else None,
                close5=float(close5[i]) if resolved else None,
                hit3=int(close3[i] > base_close[i]) if resolved else None,
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
                is_closed=closed_flags[i],
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


def document(
    root: Path,
    series: Series,
    holdout: str = "2015-01-01",
    instrument: str | None = None,
    target: Series | None = None,
) -> dict:
    """The study for one instrument. The signals never read this instrument's prices.

    The composite is built from the four cross-asset factors only, so swapping the
    target (NDX -> QQQ) leaves `flag` untouched and changes only the outcomes.
    """
    label = instrument or series.symbol
    rows = calls(root, series, target)
    resolved = [row for row in rows if row.outcome is not None]

    def split(rows: list[Call]) -> dict:
        """命中按天拆开：三天内 / 只有第 4–5 天 / 五天都没做到。互斥，加起来是 1。"""
        if not rows:
            return {"hit3": None, "hit45": None, "miss": None}
        three = float(np.mean([1 if row.hit3 == 1 else 0 for row in rows]))
        later = float(np.mean([1 if (row.outcome == 1 and row.hit3 == 0) else 0 for row in rows]))
        return {"hit3": round(three, 4), "hit45": round(later, 4), "miss": round(1 - three - later, 4)}

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
            # 三天内 / 只有第 4–5 天：信号侧与基率并排，避免只报「五天命中」那个合计数。
            "hit3": split(flagged)["hit3"],
            "hit45": split(flagged)["hit45"],
            "miss": split(flagged)["miss"],
            "base3": split(selected)["hit3"],
            "base45": split(selected)["hit45"],
        }

    price_rows = [Call(**{**row.__dict__, "flag": row.price_only}) for row in resolved]
    with_news = tally(resolved)
    price_only = tally(price_rows)
    _, news_provenance = _news_labels(root)
    news_provenance["vetoed_signals"] = sum(row.news_veto for row in resolved)
    news_columns, news_contexts = _news_providers(root, [row.date for row in rows])

    extra_columns: dict[str, list[int]] = {}
    extra_contexts: list[dict] = []
    if target is not None:
        closes_arr = np.array([row.close for row in rows], dtype=float)
        below_ma20: list[int] = []
        gap_veto: list[int] = []
        for idx, row in enumerate(rows):
            if idx >= 19 and np.all(np.isfinite(closes_arr[idx - 19 : idx + 1])):
                ma20 = float(np.mean(closes_arr[idx - 19 : idx + 1]))
                below_ma20.append(int(closes_arr[idx] < ma20))
            else:
                below_ma20.append(0)
            if row.entry is not None and row.close and np.isfinite(row.close):
                gap_veto.append(int((row.entry / row.close - 1.0) > 0.015))
            else:
                gap_veto.append(0)
        extra_columns["qqq_below_ma20"] = below_ma20
        extra_columns["qqq_gap_veto"] = gap_veto
        extra_contexts.extend([
            {
                "key": "qqq_below_ma20",
                "label": f"{label} trend filter — Below 20-day MA",
                "kind": "veto",
                "note": f"veto signal when {label} closes below its 20-session moving average",
                "available_from": rows[0].date if rows else None,
            },
            {
                "key": "qqq_gap_veto",
                "label": f"{label} overnight gap > +1.5%",
                "kind": "veto",
                "note": f"veto entry when {label} next-session open gaps up more than 1.5% above signal close",
                "available_from": rows[0].date if rows else None,
            },
        ])

    return {
        "instrument": label,
        "question": {
            "horizon": HORIZON,
            "text": (
                f"Cross-asset risk appetite long → is the {label} close {HORIZON} sessions later "
                "above the basis close?"
            ),
            "factors": [
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
                "days the LLM labels as a strong bearish surprise. A point is issued on a close, so that close "
                "is the basis of the call: the close 3 sessions later and the close 5 sessions later are compared "
                f"against it, and the call is a hit when the day-{HORIZON} close is above the basis. The next "
                "session's open is recorded as `entry` — the price you could actually pay — but it does not "
                "decide the call."
            ),
        },
        "fetched_at": series.fetched_at,
        "first_session": rows[0].date if rows else None,
        "last_session": rows[-1].date if rows else None,
        "calendar": _calendar_provenance(root),
        "news": news_provenance,
        "time": [row.date for row in rows],
        "open": [round(row.open, 2) if row.open is not None else None for row in rows],
        "high": [round(row.high, 2) if row.high is not None else None for row in rows],
        "low": [round(row.low, 2) if row.low is not None else None for row in rows],
        "close": [round(row.close, 2) if np.isfinite(row.close) else None for row in rows],
        "volume": [row.volume for row in rows],
        "entry": [round(row.entry, 2) if row.entry is not None else None for row in rows],
        "entry_at": [row.entry_at for row in rows],
        "base_close": [round(row.base_close, 2) if row.base_close is not None else None for row in rows],
        "close3": [round(row.close3, 2) if row.close3 is not None else None for row in rows],
        "close5": [round(row.close5, 2) if row.close5 is not None else None for row in rows],
        "hit3": [row.hit3 for row in rows],
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
            **news_columns,
            **extra_columns,
        },
        "contexts": [
            {**item, "available_from": item.get("available_from") or rows[0].date}
            if item["key"] == "composite"
            else item
            for item in CONTEXTS
        ]
        + news_contexts
        + extra_contexts,
        "news_veto": [row.news_veto for row in rows],
        "outcome": [row.outcome for row in rows],
        "target": [row.target for row in rows],
        "peak": [round(row.peak, 2) if row.peak is not None else None for row in rows],
        "peak_at": [row.peak_at for row in rows],
        "is_closed": [row.is_closed for row in rows],
        "summary": {**with_news, "holdout_from": holdout},
        "summary_price_only": {**price_only, "holdout_from": holdout},
        "context": _runtime_context(root),
        "latest": {
            "date": rows[-1].date,
            "open": round(rows[-1].open, 2) if rows[-1].open is not None else None,
            "high": round(rows[-1].high, 2) if rows[-1].high is not None else None,
            "low": round(rows[-1].low, 2) if rows[-1].low is not None else None,
            "close": round(rows[-1].close, 2) if np.isfinite(rows[-1].close) else None,
            "volume": rows[-1].volume,
            "score": round(rows[-1].score, 4),
            "threshold": round(rows[-1].threshold, 4),
            "flag": rows[-1].flag,
            "flag_price_only": rows[-1].price_only,
            "news_veto": bool(rows[-1].news_veto),
            "is_closed": bool(rows[-1].is_closed),
        },
    }


def _instrument_row(doc: dict) -> dict:
    summary = doc["summary"]
    hit, base = summary["hit"], summary["base_rate"]
    held_hit, held_base = summary["holdout_hit"], summary["holdout_base_rate"]
    return {
        "instrument": doc["instrument"],
        "signals": summary["flagged"],
        "precision": hit,
        "base_rate": base,
        "lift": None if hit is None or base is None else round(hit - base, 4),
        "holdout_signals": summary["holdout_flagged"],
        "holdout_precision": held_hit,
        "holdout_base_rate": held_base,
        "holdout_lift": None if held_hit is None or held_base is None else round(held_hit - held_base, 4),
    }


def comparison(primary: dict, other: dict) -> dict:
    """One set of signals, two instruments: precision side by side, plus where they disagree.

    Everything that decides *when* to buy (composite, calendar, news veto) is computed
    without the target's own prices, so the two documents must flag the same sessions.
    `same_flags` says whether that held; if it ever stops holding, the comparison is void.
    """
    if primary["time"] != other["time"]:
        raise ValueError("documents are on different session calendars")
    disagreements = []
    for index, day in enumerate(primary["time"]):
        if not primary["flag"][index]:
            continue
        if primary["outcome"][index] is None or other["outcome"][index] is None:
            continue
        if primary["outcome"][index] == other["outcome"][index]:
            continue
        disagreements.append(
            {
                "date": day,
                "target": primary["target"][index],
                "a_return": round(primary["peak"][index] / primary["entry"][index] - 1, 6),
                "b_return": round(other["peak"][index] / other["entry"][index] - 1, 6),
                "a_hit": primary["outcome"][index],
                "b_hit": other["outcome"][index],
            }
        )
    return {
        "a": primary["instrument"],
        "b": other["instrument"],
        "rows": [_instrument_row(primary), _instrument_row(other)],
        "same_days": primary["time"] == other["time"],
        "same_flags": primary["flag"] == other["flag"],
        "disagreements": disagreements,
    }
