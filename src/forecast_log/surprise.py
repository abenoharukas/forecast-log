"""新闻里的「意外」，而不是「出现」。

之前测「今天有没有 `inflation` 这个词」，那等于把预期内和超预期混在一起——市场定价的是
后者。这个模块只问一件事：**这一天的标题里，有没有一条新发生的、超出此前预期的事情，
会改变美股未来 5 个交易日的方向。**

输出只有两类东西：
- `surprise`：none / mild / strong（是否超出预期）
- `direction`：bullish / bearish / ambiguous（对美股偏多还是偏空）

它仍然不是概率，也不预测点位。标签能不能用，由「信号日上的精确率」检验决定。
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from forecast_log.agent.env import merged_env
from forecast_log.agent.llm import LLM, OpenAICompatLLM

PROMPT_VERSION = "surprise-v1"
BATCH_DAYS = 12
MAX_TITLES = 22

SYSTEM = """You extract market-moving surprises from a day's news headlines. You never forecast prices.

For each day decide:
- "surprise": "none" | "mild" | "strong"
  = is there a NEW development that was not already expected? Routine reporting, scheduled
    data releases matching expectations, and follow-ups on old stories are "none".
- "direction": "bullish" | "bearish" | "ambiguous"
  = direction for US equities over the next week if this surprise is the dominant driver.
- "evidence": one headline phrase, copied verbatim, that carries the surprise.
- "category": "monetary" | "inflation" | "growth" | "credit" | "geopolitics" | "trade" | "other"

Rules:
- Judge surprise from the text only. Phrases like "as expected", "unchanged", "in line with
  forecasts", "had been widely anticipated" mean "none".
- Words like "unexpectedly", "surprise", "shock", "first time since", "emergency",
  "worse than expected", "beat estimates" indicate surprise — record the direction carefully.
- A day with only geopolitical violence is usually "ambiguous" unless it moves oil or credit.
- Do not output probabilities. Do not mention price levels.

Return JSON: {"days": [{"as_of": "YYYY-MM-DD", "surprise": ..., "direction": ..., "category": ..., "evidence": "..."}]}"""


def label_dir(root: Path) -> Path:
    return root / "artifacts" / "surprise"


def day_path(root: Path, day: str) -> Path:
    return label_dir(root) / f"{day}.json"


def _titles(root: Path, day: str) -> list[str]:
    from forecast_log.db import get_news_fetch

    try:
        cached = get_news_fetch(root, date.fromisoformat(day))
    except ValueError:
        cached = None
    if cached and cached.get("documents"):
        titles = [str(d.get("title", "")).strip() for d in cached["documents"] if d.get("title")]
        if titles:
            return titles[:MAX_TITLES]

    path = root / "data" / "news" / f"{day}.json"
    if not path.is_file():
        return []
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return [d.get("title", "") for d in record.get("documents", []) if d.get("title")][:MAX_TITLES]


def _prompt(batch: dict[str, list[str]]) -> str:
    lines = ["Extract the surprise for each day. Same order, one object per day.", ""]
    for day, titles in batch.items():
        lines.append(f"### {day}")
        lines.extend(f"- {title[:200]}" for title in titles)
        lines.append("")
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


def _parse(content: str, expected: list[str]) -> list[dict]:
    text = content or ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    body = json.loads(text[start : end + 1])
    out = []
    for entry in body.get("days") or []:
        day = str(entry.get("as_of", "")).strip()
        if day not in expected:
            continue
        surprise = str(entry.get("surprise", "none")).lower()
        direction = str(entry.get("direction", "ambiguous")).lower()
        out.append(
            {
                "as_of": day,
                "surprise": surprise if surprise in ("none", "mild", "strong") else "none",
                "direction": direction if direction in ("bullish", "bearish", "ambiguous") else "ambiguous",
                "category": str(entry.get("category", "other"))[:20],
                "evidence": str(entry.get("evidence", ""))[:200],
            }
        )
    return out


def run(
    root: Path,
    days: list[date],
    llm: LLM | None = None,
    workers: int = 5,
    limit: int | None = None,
    overwrite_days: set[str] | None = None,
) -> dict:
    overwrite = overwrite_days or set()
    todo = [day for day in days if day.isoformat() in overwrite or not day_path(root, day.isoformat()).is_file()]
    if limit:
        todo = todo[:limit]
    stats = {"todo": len(todo), "days": 0, "failed": 0, "cost_usd": 0.0, "errors": []}
    if not todo:
        return stats
    if llm is None:
        env = merged_env(root)
        if env.get("DEEPSEEK_API_KEY"):
            from forecast_log import providers

            llm = providers.build(env.get("DEEPSEEK_MODEL") or "deepseek-chat", root, env=env)
        else:
            llm = OpenAICompatLLM.from_env(env)
    batch: dict[str, list[str]] = {}
    for day in todo:
        titles = _titles(root, day.isoformat())
        if titles:
            batch[day.isoformat()] = titles
    keys = list(batch)
    chunks = [{k: batch[k] for k in keys[i : i + BATCH_DAYS]} for i in range(0, len(keys), BATCH_DAYS)]
    label_dir(root).mkdir(parents=True, exist_ok=True)

    def one(chunk: dict[str, list[str]]):
        try:
            reply = llm.chat(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _prompt(chunk)},
                ],
                temperature=0.0,
                max_tokens=6000,
            )
            return chunk, reply
        except Exception as error:
            return chunk, error

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for chunk, result in pool.map(one, chunks):
            if isinstance(result, Exception):
                stats["failed"] += len(chunk)
                stats["errors"].append(f"{min(chunk)}: {type(result).__name__}: {result}")
                continue
            try:
                labeled = _parse(result.content or "", list(chunk))
            except Exception as error:
                stats["failed"] += len(chunk)
                stats["errors"].append(f"{min(chunk)}: parse {error}")
                continue
            stats["cost_usd"] += result.usage.cost_usd
            for entry in labeled:
                day_path(root, entry["as_of"]).write_text(
                    json.dumps(
                        {
                            "as_of": entry["as_of"],
                            "prompt_version": PROMPT_VERSION,
                            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "model": getattr(llm, "name", "?"),
                            "label": {k: v for k, v in entry.items() if k != "as_of"},
                        },
                        ensure_ascii=False,
                    )
                )
                stats["days"] += 1
    return stats


def export_jsonl(root: Path, path: Path | None = None) -> Path:
    """把逐日文件压成一行一天，进 git 的那份。

    逐日原文（含 prompt 版本、模型、生成时间）留在 `artifacts/surprise/`，不进 git；
    真正被判分用到的是 `surprise` × `direction` 两个字段，这份 JSONL 就够复现。
    """
    target = path or (root / "artifacts" / "surprise.jsonl")
    rows = load(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for day in sorted(rows):
            label = rows[day]
            handle.write(json.dumps({"as_of": day, **label}, ensure_ascii=False) + "\n")
    return target


def load_compact(root: Path) -> dict[str, dict]:
    path = root / "artifacts" / "surprise.jsonl"
    if not path.is_file():
        return load(root)
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        day = row.pop("as_of")
        out[day] = row
    return out


def load(root: Path) -> dict[str, dict]:
    out = {}
    for path in label_dir(root).glob("????-??-??.json"):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out[record["as_of"]] = record["label"]
    return out
