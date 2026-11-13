"""LLM 负责的那一段：把价格里没有的事件日历拉下来。

价格序列不知道「下周三有 FOMC」，人知道，而且这类信息提前公布、可以事后核对。所以它不进
模型的状态向量，而是单独存成一份日历，由 `cross.py` 折成「离下一次会议还有几天」这样的腿。

存 `data/events.json`，带 prompt、模型、时间戳、token 成本和校验结果：事后能看出它是问出来的
还是编的（每年次数、是否工作日、是否在范围内）。

    forecast-log events          # 重新拉一次（覆盖 data/events.json）
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from forecast_log.agent.llm import LLM
from forecast_log.agent_cli import build_llm

NY = ZoneInfo("America/New_York")

PROMPT = """List the scheduled US macro event dates from 2011-01-01 through 2026-12-31 that
volatility traders plan around. Return JSON only, no prose, in this shape:

{"fomc": ["YYYY-MM-DD", ...], "cpi": ["YYYY-MM-DD", ...], "nfp": ["YYYY-MM-DD", ...]}

Rules:
- "fomc": the second day of each scheduled FOMC meeting (the day the statement is released).
- "cpi": the release date of the monthly CPI report.
- "nfp": the release date of the monthly jobs report.
- Only real dates. If you are unsure about a month, omit it rather than guess.
- Include unscheduled emergency meetings if they happened (2020-03-15)."""

EXPECTED_PER_YEAR = {"fomc": (1, 9), "cpi": (10, 13), "nfp": (11, 13)}


def path_of(root: Path) -> Path:
    return root / "data" / "events.json"


def fetch(root: Path, llm: LLM | None = None, now: datetime | None = None) -> dict:
    """问一次 LLM，校验，落盘。返回写下的记录。"""
    llm = llm or build_llm(root)
    reply = llm.chat(
        [
            {
                "role": "system",
                "content": "You are a precise financial calendar assistant. JSON only.",
            },
            {"role": "user", "content": PROMPT},
        ],
        temperature=0.0,
        max_tokens=8000,
    )
    text = (reply.content or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise RuntimeError(f"模型没有返回 JSON：{text[:400]}")
    payload = json.loads(match.group(0))

    calendar: dict[str, list[str]] = {}
    problems: list[str] = []
    for key, per_year in EXPECTED_PER_YEAR.items():
        days = sorted({date.fromisoformat(item) for item in payload.get(key, [])})
        if not days:
            problems.append(f"{key}: 空")
        counts: dict[int, int] = {}
        for day in days:
            counts[day.year] = counts.get(day.year, 0) + 1
            if day.weekday() > 4:
                problems.append(f"{key}: {day} 是周末")
            if not 2011 <= day.year <= 2026:
                problems.append(f"{key}: {day} 超出范围")
        low, high = per_year
        for year, count in sorted(counts.items()):
            if not low <= count <= high:
                problems.append(f"{key}: {year} 有 {count} 次，预期 {low}-{high}")
        calendar[key] = [day.isoformat() for day in days]

    record = {
        "generated_at": (now or datetime.now(NY)).astimezone(NY).isoformat(timespec="seconds"),
        "source": "llm",
        "model": getattr(llm, "name", "?"),
        "prompt": PROMPT,
        "usage": reply.usage.to_json(),
        "counts": {key: len(value) for key, value in calendar.items()},
        "problems": problems,
        "calendar": calendar,
    }
    out = path_of(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return record


def load(root: Path) -> dict[str, list[date]] | None:
    """日历，没有文件时返回 None，调用方自己决定要不要退化成四条腿。"""
    path = path_of(root)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())["calendar"]
    return {key: sorted(date.fromisoformat(day) for day in days) for key, days in payload.items()}


def days_to_next(days: list[date], day: date) -> float:
    for event in days:
        if event >= day:
            return float((event - day).days)
    return float("nan")
