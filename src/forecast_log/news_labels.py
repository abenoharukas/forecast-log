"""让 LLM 在运行时干它真正擅长的那件事：把一天的标题读成「事件」，不是读成「关键词」。

为什么不是正则：正则只知道 `oil` 出现了，不知道是「油价崩了」还是「油价崩了但已见底」，
也不知道一条「央行紧急降息」和一条「央行日程公布」的严重程度差多少。这是 LLM 的活。

为什么不让它预测涨跌：盲答实验里它 Brier 0.2293，约等于抛硬币。所以它的输出是**事件
标签**，不是概率；涨跌仍由价格侧四个因子决定。标签能不能进规则，由同一套窗口检验决定
（两段都超过基率、且 bootstrap 区间不跨 0）——和 CPI / NFP 用的是同一把尺子。
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from forecast_log.agent.env import merged_env
from forecast_log.agent.llm import LLM, OpenAICompatLLM

PROMPT_VERSION = "news-label-v1"
BATCH_DAYS = 10
MAX_TITLES_PER_DAY = 25

EVENT_TYPES = (
    "monetary_policy",  # 央行、利率、QE、紧急降息
    "inflation",        # CPI / PPI / 通胀预期
    "growth",           # GDP、就业、PMI、衰退
    "credit",           # 银行、违约、流动性
    "geopolitics",      # 战争、制裁、袭击
    "trade",            # 关税、贸易协定
    "markets",          # 指数、波动率、资金流
    "other",
)
DIRECTIONS = ("supportive", "adverse", "mixed")

SYSTEM = """You convert one day of news headlines into a small structured label.

You are NOT forecasting. Never output a probability for any asset. Output only the JSON object.

Schema:
{"days": [{"as_of": "YYYY-MM-DD",
           "regime": "risk_on" | "neutral" | "risk_off",
           "severity": 0.0-1.0,
           "novelty": 0.0-1.0,
           "events": [{"type": <one of """ + json.dumps(list(EVENT_TYPES)) + """,
                       "direction": "supportive" | "adverse" | "mixed",
                       "severity": 0.0-1.0,
                       "evidence": "<a phrase copied from the headlines>"}],
           "note": "<= 160 chars"}]}

Rules:
- One entry per input day, in the same order, with the same as_of date.
- severity = how much this day's news would move risk appetite (0 = routine, 1 = crisis).
- novelty = how much of it is new information vs continuation of a known story.
- At most 3 events per day; omit the day's events entirely if it is routine noise (empty list).
- Regime is about risk appetite implied by the news, not about prices going up.
- Copy evidence phrases from the headlines; do not invent facts not present."""


def label_dir(root: Path) -> Path:
    return root / "artifacts" / "news_labels"


def day_path(root: Path, day: str) -> Path:
    return label_dir(root) / f"{day}.json"


def read_day_titles(root: Path, days: list[date]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for day in days:
        path = root / "data" / "news" / f"{day.isoformat()}.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        titles = [document.get("title", "") for document in record.get("documents", []) if document.get("title")]
        if titles:
            out[day.isoformat()] = titles[:MAX_TITLES_PER_DAY]
    return out


def _prompt(batch: dict[str, list[str]]) -> str:
    lines = ["Label each day below. One JSON object with a `days` array, same order.", ""]
    for as_of, titles in batch.items():
        lines.append(f"### {as_of}")
        for title in titles:
            lines.append(f"- {title[:220]}")
        lines.append("")
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


def _parse_days(content: str, expected: list[str]) -> list[dict]:
    text = (content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in response: {text[:160]!r}")
    body = json.loads(text[start : end + 1])
    days = body.get("days")
    if not isinstance(days, list):
        raise ValueError("missing `days` array")
    cleaned = []
    for entry in days:
        as_of = str(entry.get("as_of", "")).strip()
        if as_of not in expected:
            continue
        regime = str(entry.get("regime", "neutral")).lower()
        events = []
        for event in (entry.get("events") or [])[:3]:
            kind = str(event.get("type", "other")).lower()
            events.append(
                {
                    "type": kind if kind in EVENT_TYPES else "other",
                    "direction": str(event.get("direction", "mixed")).lower()
                    if str(event.get("direction", "")).lower() in DIRECTIONS
                    else "mixed",
                    "severity": _clamp(event.get("severity")),
                    "evidence": str(event.get("evidence", ""))[:200],
                }
            )
        cleaned.append(
            {
                "as_of": as_of,
                "regime": regime if regime in ("risk_on", "neutral", "risk_off") else "neutral",
                "severity": _clamp(entry.get("severity")),
                "novelty": _clamp(entry.get("novelty")),
                "events": events,
                "note": str(entry.get("note", ""))[:160],
            }
        )
    return cleaned


def _clamp(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(max(number, 0.0), 1.0), 3)


def label_batch(root: Path, batch: dict[str, list[str]], llm: LLM) -> tuple[list[dict], dict]:
    reply = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _prompt(batch)},
        ],
        temperature=0.1,
        max_tokens=8000,
    )
    usage = reply.usage.to_json()
    usage["model"] = getattr(llm, "name", "unknown")
    return _parse_days(reply.content or "", list(batch)), usage


def pending(root: Path, days: list[date]) -> list[date]:
    return [day for day in days if not day_path(root, day.isoformat()).is_file()]


def run(
    root: Path,
    days: list[date],
    llm: LLM | None = None,
    batch_size: int = BATCH_DAYS,
    limit: int | None = None,
    workers: int = 1,
    progress=None,
) -> dict:
    """按批标注，逐天落盘。中断重跑只补没标的那些天。"""
    todo = pending(root, days)
    if limit is not None:
        todo = todo[:limit]
    stats = {"todo": len(todo), "days": 0, "batches": 0, "failed": 0, "cost_usd": 0.0, "errors": []}
    if not todo:
        return stats
    if llm is None:
        llm = OpenAICompatLLM.from_env(merged_env(root))

    titles = read_day_titles(root, todo)
    keys = list(titles)
    chunks = [{key: titles[key] for key in keys[start : start + batch_size]} for start in range(0, len(keys), batch_size)]
    label_dir(root).mkdir(parents=True, exist_ok=True)

    def one(chunk: dict[str, list[str]]) -> tuple[dict[str, list[str]], tuple[list[dict], dict] | Exception]:
        try:
            return chunk, label_batch(root, chunk, llm)
        except Exception as error:  # 一批失败不该毁掉整次运行
            return chunk, error

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for chunk, result in pool.map(one, chunks):
            if isinstance(result, Exception):
                stats["failed"] += len(chunk)
                stats["errors"].append(f"{min(chunk)}..: {type(result).__name__}: {result}")
                continue
            labeled, usage = result
            stats["batches"] += 1
            stats["cost_usd"] += usage.get("cost_usd", 0.0)
            for entry in labeled:
                record = {
                    "as_of": entry["as_of"],
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "prompt_version": PROMPT_VERSION,
                    "model": usage.get("model"),
                    "label": {key: value for key, value in entry.items() if key != "as_of"},
                    "headlines_used": len(chunk.get(entry["as_of"], [])),
                }
                day_path(root, entry["as_of"]).write_text(json.dumps(record, ensure_ascii=False))
                stats["days"] += 1
            if progress:
                progress(stats)
    return stats


def export_jsonl(root: Path, path: Path | None = None) -> Path:
    """逐日标签压成一行一天。进 git 的是这份；逐日文件留在本地做审计。"""
    target = path or (root / "artifacts" / "news_labels.jsonl")
    rows = load(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for day in sorted(rows):
            handle.write(json.dumps({"as_of": day, **rows[day].get("label", {})}, ensure_ascii=False) + "\n")
    return target


def load(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in label_dir(root).glob("????-??-??.json"):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out[record["as_of"]] = record
    return out


# 标定表：把 LLM 的 severity 换成一个「5 日涨跌幅该放多大」的倍数。
#
# 怎么来的：2013–2026、3481 天，按 severity 四分位看 5 日后 |NDX 涨跌幅|：
#   1.69% → 1.91% → 2.01% → 2.52%，全样本均值 2.03% ⇒ 倍数 0.83 / 0.94 / 0.99 / 1.24
# 它有多可信：corr(severity, |ret|) = 0.13；控制掉 trailing 20 日已实现波动率之后，
#   偏相关仍有 0.07（t≈4.2），在波动率三分组内 high−low 分别是 +0.15 / +0.47 / +0.80 个点。
# 它不是什么：**不是方向信号**。regime=risk_off 与上涨率的关系 −0.9 个点、区间跨 0；
#   severity 分档的上涨率 63.3 / 56.0 / 60.0 / 60.5 也不单调。所以只用它缩放区间宽度。
# 稳定性：2015 后 0.86 / 0.89 / 0.99 / 1.26（单调）；2013–2014 那两年 0.97 / 1.02 / 0.94 / 1.07（噪声）。
SIZE_BINS = ((0.30, 0.83), (0.40, 0.94), (0.50, 0.99), (1.01, 1.24))
SIZE_SOURCE = "quartiles of |5d NDX return| by severity, 2013-2026 (n=3481); post-2015 stable, 2013-14 noisy"


def size_multiplier(severity: float) -> float:
    """severity → 5 日波动幅度倍数。中位数日 = 1.0。"""
    for ceiling, factor in SIZE_BINS:
        if severity < ceiling:
            return factor
    return SIZE_BINS[-1][1]


def flatten(label: dict) -> dict:
    """把一天的结构化标签变成可统计的几个数。"""
    events = label.get("events") or []
    adverse = [event for event in events if event["direction"] == "adverse"]
    supportive = [event for event in events if event["direction"] == "supportive"]
    return {
        "regime": label.get("regime", "neutral"),
        "severity": label.get("severity", 0.0),
        "novelty": label.get("novelty", 0.0),
        "events": len(events),
        "adverse": len(adverse),
        "supportive": len(supportive),
        "max_adverse": max((event["severity"] for event in adverse), default=0.0),
        "max_supportive": max((event["severity"] for event in supportive), default=0.0),
        "types": sorted({event["type"] for event in events}),
        "size_multiplier": size_multiplier(label.get("severity", 0.0)),
    }
