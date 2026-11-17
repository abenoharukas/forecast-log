"""每天收盘后跑一次：抓当天的新闻，让 LLM 说一句"今天是什么风险偏好"，写进文件。

和 `events.py` 的区别，这个模块就是为这个区别存在：

- `events.py` 是**离线**的。日历是历史事实，拉一次够用，LLM 只做信息整理。
- 这个模块是**运行时**的。每个交易日跑一次，读的是当天已经发生的新闻，产出当天的标签，
  存进 `artifacts/daily.json`，points 页面在信号卡里显示它。

三条纪律：

1. **不预测价格**。LLM 只回答"今天的新闻在讲什么、偏多还是偏空、有多确定"，不回答涨跌。
   涨跌仍由 `cross.py` 四个价格因子决定。LLM 说偏空、信号触发，页面标成 `contested`，
   不吞掉信号 —— 先记账，等样本够了再谈能不能用。
2. **不偷看**。新闻只取 `published_at <= 当天` 的；晚于当天的标题一律丢掉并计数，
   和 `agent/gate.py` 的 as-of 闸门同一个标准。
3. **能重放**。每条记录存模型、prompt 版本、token 成本、标题原文、丢弃条数。
   `data/news/gdelt-YYYY-MM-DD.json` 是缓存，同一天重跑不重复请求。
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from xml.etree import ElementTree
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from forecast_log import events, keywords
from forecast_log.agent.env import merged_env
from forecast_log.agent.llm import LLM, OpenAICompatLLM

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
# GDELT 是唯一能给历史日期取新闻的免费源，但它的免费额度按 IP 限流，经常直接 429。
# 所以后面挂一组 RSS：只有最近条目，但天天有新内容，够「收盘后跑一次」用。
FEEDS = (
    ("wsj-markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("fed-monetary", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    ("sec-press", "https://www.sec.gov/news/pressreleases.rss"),
)
QUERIES = (
    'nasdaq',
    '"federal reserve" inflation',
)
RETRY_AFTER_S = 25.0
LOOKBACK_DAYS = 2
MAX_RECORDS = 20
RATE_LIMIT_S = 5.5
PROMPT_VERSION = "daily-v1"
MAX_HEADLINES_KEPT = 60
TONES = ("risk-on", "neutral", "risk-off")

ARTIFACT = Path("artifacts") / "daily.json"          # 最新一条，页面读这个
DAILY_DIR = Path("artifacts") / "daily"              # 每天一个文件，只写不覆盖
LOG = Path("data") / "daily.jsonl"                   # 追加日志，不进 git


def artifact_path(root: Path) -> Path:
    return root / ARTIFACT


def day_path(root: Path, day: date) -> Path:
    """当天那一份，写一次就不再动：别的时间点的记录不会被后一天盖掉。"""
    return root / DAILY_DIR / f"{day.isoformat()}.json"


# ------------------------------------------------------------------ retrieval


def _doc(article: dict) -> dict | None:
    title = (article.get("title") or "").strip()
    url = (article.get("url") or "").strip()
    seen = (article.get("seendate") or "").strip()  # 20261112T134500Z
    if not title or not url or len(seen) < 8:
        return None
    try:
        published = datetime.strptime(seen[:8], "%Y%m%d").date()
    except ValueError:
        return None
    return {
        "id": hashlib.sha1(url.encode()).hexdigest()[:12],
        "published_at": published.isoformat(),
        "title": title[:400],
        "text": title[:400],
        "url": url,
        "domain": article.get("domain", ""),
    }


def _gdelt(
    query: str, start: date, end: date, timeout: float = 20.0, attempts: int = 3
) -> tuple[list[dict], str]:
    params = urllib.parse.urlencode(
        {
            "query": query,
            "mode": "artlist",
            "maxrecords": MAX_RECORDS,
            "format": "json",
            "startdatetime": f"{start:%Y%m%d}000000",
            "enddatetime": f"{end:%Y%m%d}235959",
        }
    )
    request = urllib.request.Request(f"{GDELT}?{params}", headers={"User-Agent": "forecast-log/1.0"})
    raw, last = "", ""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code}"
            if error.code != 429 or attempt == attempts:
                return [], last
            time.sleep(RETRY_AFTER_S)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = f"{type(error).__name__}: {error}"
            if attempt == attempts:
                return [], last
            time.sleep(3.0)
    else:
        return [], last or "no response"
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        # GDELT 用纯文本 200 回应限流，不是 JSON
        return [], raw.strip()[:200] or "unparseable response"
    articles = body.get("articles") or []
    documents = [doc for doc in (_doc(article) for article in articles) if doc]
    return documents, ""


def _rss(url: str, timeout: float = 20.0) -> tuple[list[dict], str]:
    request = urllib.request.Request(url, headers={"User-Agent": "forecast-log/1.0 (+forecasting agent)"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
        return [], f"{type(error).__name__}: {error}"
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as error:
        return [], f"unparseable feed: {error}"
    documents = []
    for item in root.iter():
        if item.tag.split("}")[-1] not in ("item", "entry"):
            continue
        fields = {child.tag.split("}")[-1]: (child.text or "").strip() for child in item}
        title = fields.get("title", "")
        link = fields.get("link", "") or fields.get("guid", "")
        published = _parse_rss_date(fields.get("pubDate") or fields.get("updated") or fields.get("published") or "")
        if not title or not link or published is None:
            continue
        documents.append(
            {
                "id": hashlib.sha1((link + title).encode()).hexdigest()[:12],
                "published_at": published.isoformat(),
                "title": title[:400],
                "text": title[:400],
                "url": link,
                "domain": url.split("/")[2],
            }
        )
    return documents, ""


def _parse_rss_date(text: str) -> date | None:
    text = text.strip()
    if not text:
        return None
    for pattern in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:len("Fri, 13 Nov 2026 20:15:00 +0000") + 6], pattern).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def fetch_news(root: Path, day: date, lookback: int = LOOKBACK_DAYS, use_cache: bool = True) -> dict:
    """当天为止的标题。先问 GDELT（能按日期取历史），被限流就退回 RSS。

    缓存到 SQLite（`data/forecast.db`），同一天重跑不发请求；旧 JSON 目录仍可作回退读取。
    """
    from forecast_log.db import get_news_fetch, save_news_fetch

    if use_cache:
        cached = get_news_fetch(root, day)
        if cached is not None:
            return cached
        cache = root / "data" / "news" / f"{day.isoformat()}.json"
        if cache.is_file():
            record = json.loads(cache.read_text())
            return {**record, "cached": True}

    documents: list[dict] = []
    errors: list[str] = []
    providers: list[str] = []
    for index, query in enumerate(QUERIES):
        found, error = _gdelt(query, day - timedelta(days=lookback), day)
        documents.extend(found)
        if found:
            providers.append("gdelt")
        if error:
            errors.append(f"gdelt {query!r}: {error}")
        if index + 1 < len(QUERIES):
            time.sleep(RATE_LIMIT_S)

    for name, url in FEEDS:
        found, error = _rss(url)
        documents.extend(found)
        if found:
            providers.append(name)
        if error:
            errors.append(f"{name}: {error}")

    seen: set[str] = set()
    unique = []
    for document in sorted(documents, key=lambda d: d["published_at"], reverse=True):
        if document["id"] in seen:
            continue
        seen.add(document["id"])
        unique.append(document)

    record = {
        "as_of": day.isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "+".join(dict.fromkeys(providers)) or "none",
        "providers": list(dict.fromkeys(providers)),
        "queries": list(QUERIES),
        "documents": unique[: MAX_RECORDS * len(QUERIES)],
        "errors": errors,
        "cached": False,
    }
    save_news_fetch(root, record)
    return record


def gate(documents: list[dict], day: date) -> tuple[list[dict], int]:
    """只留 `published_at <= day` 的标题，返回 (留下的, 丢掉的条数)。"""
    kept, dropped = [], 0
    for document in documents:
        try:
            published = date.fromisoformat(str(document.get("published_at"))[:10])
        except ValueError:
            dropped += 1
            continue
        if published > day:
            dropped += 1
            continue
        kept.append(document)
    return kept, dropped


# --------------------------------------------------------------------- LLM


def _readings(root: Path, day: date) -> dict:
    """当天价格侧的数字，给 LLM 当上下文，也用来判断它和信号是否一致。"""
    from forecast_log.artifacts_db import load_cross_market

    out: dict = {"as_of": day.isoformat()}
    cross = load_cross_market(root, "ndx")
    if cross is not None:
        latest = cross.get("latest") or {}
        if latest.get("date") == day.isoformat():
            out.update(
                {
                    "composite": latest.get("score"),
                    "trigger": latest.get("threshold"),
                    "signal": bool(latest.get("flag")),
                }
            )
    calendar = events.load(root)
    if calendar is not None:
        iso = [day.isoformat()]
        for key in ("cpi", "nfp", "fomc"):
            distance = events.days_to_next(calendar.get(key, []), day)
            out[f"days_to_{key}"] = None if distance != distance else round(float(distance), 1)
    return out


SYSTEM = """You read news headlines and output one JSON object. Nothing else.

You are NOT forecasting prices. You answer only: what is today's news flow doing to risk
appetite, and how confident are you from these headlines alone.

Output exactly this shape:
{"tone": "risk-on" | "neutral" | "risk-off",
 "confidence": 0.0-1.0,
 "drivers": ["<= 3 short phrases, each traceable to a headline you were given>"],
 "note": "<= 200 chars, plain English, no price prediction>"}

Rules:
- Use only the headlines and numbers provided. No outside knowledge of future events.
- If the headlines are thin, off-topic, or contradictory, say "neutral" with confidence <= 0.35.
- Never state or imply a probability for the index going up or down."""


def _prompt(readings: dict, documents: list[dict], day: date) -> str:
    lines = [
        f"Session: {day.isoformat()} (US close already happened).",
        f"Price-side readings on that session: {json.dumps(readings, ensure_ascii=False)}",
        "",
        f"Headlines published on or before {day.isoformat()} ({len(documents)} total, newest first):",
    ]
    for document in documents[:24]:
        lines.append(f"- [{document['published_at']}] {document['title']}")
    if not documents:
        lines.append("- (no headlines were retrieved)")
    lines.append("")
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


def _parse(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    body = json.loads(text[start : end + 1])
    tone = str(body.get("tone", "")).strip().lower()
    if tone not in TONES:
        raise ValueError(f"tone must be one of {TONES}, got {tone!r}")
    try:
        confidence = float(body.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    drivers = [str(item)[:120] for item in (body.get("drivers") or [])][:3]
    return {
        "tone": tone,
        "confidence": round(min(max(confidence, 0.0), 1.0), 3),
        "drivers": drivers,
        "note": str(body.get("note", ""))[:200],
    }


def classify(documents: list[dict], readings: dict, day: date, llm: LLM) -> tuple[dict, dict]:
    reply = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _prompt(readings, documents, day)},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    usage = reply.usage.to_json()
    usage["model"] = getattr(llm, "name", "unknown")
    return _parse(reply.content or ""), usage


# ------------------------------------------------------------------ the run


def run(
    root: Path,
    day: date | None = None,
    llm: LLM | None = None,
    use_cache: bool = True,
) -> dict:
    """跑一次。返回记录，同时写 `artifacts/daily.json` 和追加 `data/daily.jsonl`。"""
    day = day or date.today()
    news = fetch_news(root, day, use_cache=use_cache)
    kept, dropped = gate(news["documents"], day)
    readings = _readings(root, day)

    if llm is None:
        llm = OpenAICompatLLM.from_env(merged_env(root))
    label, usage = classify(kept, readings, day, llm)

    features = keywords.scan([document["title"] for document in kept])
    record = {
        "as_of": day.isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "+".join(dict.fromkeys(providers)) or "none",
        "providers": list(dict.fromkeys(providers)),
        "queries": list(QUERIES),
        "documents": unique[: MAX_RECORDS * len(QUERIES)],
        "errors": errors,
        "cached": False,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record


def gate(documents: list[dict], day: date) -> tuple[list[dict], int]:
    """只留 `published_at <= day` 的标题，返回 (留下的, 丢掉的条数)。"""
    kept, dropped = [], 0
    for document in documents:
        try:
            published = date.fromisoformat(str(document.get("published_at"))[:10])
        except ValueError:
            dropped += 1
            continue
        if published > day:
            dropped += 1
            continue
        kept.append(document)
    return kept, dropped


# --------------------------------------------------------------------- LLM


def _readings(root: Path, day: date) -> dict:
    """当天价格侧的数字，给 LLM 当上下文，也用来判断它和信号是否一致。"""
    from forecast_log.artifacts_db import load_cross_market

    out: dict = {"as_of": day.isoformat()}
    cross = load_cross_market(root, "ndx")
    if cross is not None:
        latest = cross.get("latest") or {}
        if latest.get("date") == day.isoformat():
            out.update(
                {
                    "composite": latest.get("score"),
                    "trigger": latest.get("threshold"),
                    "signal": bool(latest.get("flag")),
                }
            )
    calendar = events.load(root)
    if calendar is not None:
        iso = [day.isoformat()]
        for key in ("cpi", "nfp", "fomc"):
            distance = events.days_to_next(calendar.get(key, []), day)
            out[f"days_to_{key}"] = None if distance != distance else round(float(distance), 1)
    return out


SYSTEM = """You read news headlines and output one JSON object. Nothing else.

You are NOT forecasting prices. You answer only: what is today's news flow doing to risk
appetite, and how confident are you from these headlines alone.

Output exactly this shape:
{"tone": "risk-on" | "neutral" | "risk-off",
 "confidence": 0.0-1.0,
 "drivers": ["<= 3 short phrases, each traceable to a headline you were given>"],
 "note": "<= 200 chars, plain English, no price prediction>"}

Rules:
- Use only the headlines and numbers provided. No outside knowledge of future events.
- If the headlines are thin, off-topic, or contradictory, say "neutral" with confidence <= 0.35.
- Never state or imply a probability for the index going up or down."""


def _prompt(readings: dict, documents: list[dict], day: date) -> str:
    lines = [
        f"Session: {day.isoformat()} (US close already happened).",
        f"Price-side readings on that session: {json.dumps(readings, ensure_ascii=False)}",
        "",
        f"Headlines published on or before {day.isoformat()} ({len(documents)} total, newest first):",
    ]
    for document in documents[:24]:
        lines.append(f"- [{document['published_at']}] {document['title']}")
    if not documents:
        lines.append("- (no headlines were retrieved)")
    lines.append("")
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


def _parse(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    body = json.loads(text[start : end + 1])
    tone = str(body.get("tone", "")).strip().lower()
    if tone not in TONES:
        raise ValueError(f"tone must be one of {TONES}, got {tone!r}")
    try:
        confidence = float(body.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    drivers = [str(item)[:120] for item in (body.get("drivers") or [])][:3]
    return {
        "tone": tone,
        "confidence": round(min(max(confidence, 0.0), 1.0), 3),
        "drivers": drivers,
        "note": str(body.get("note", ""))[:200],
    }


def classify(documents: list[dict], readings: dict, day: date, llm: LLM) -> tuple[dict, dict]:
    reply = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _prompt(readings, documents, day)},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    usage = reply.usage.to_json()
    usage["model"] = getattr(llm, "name", "unknown")
    return _parse(reply.content or ""), usage


# ------------------------------------------------------------------ the run


def run(
    root: Path,
    day: date | None = None,
    llm: LLM | None = None,
    use_cache: bool = True,
) -> dict:
    """跑一次。返回记录，同时写 `artifacts/daily.json` 和追加 `data/daily.jsonl`。"""
    day = day or date.today()
    news = fetch_news(root, day, use_cache=use_cache)
    kept, dropped = gate(news["documents"], day)
    readings = _readings(root, day)

    if llm is None:
        llm = OpenAICompatLLM.from_env(merged_env(root))
    label, usage = classify(kept, readings, day, llm)

    features = keywords.scan([document["title"] for document in kept])
    # 否决票只在「信号日 + risk-off 词 ≥2」时提示。样本 250 个、只有 1.4 个标准误，
    # 所以只标不改，见 README「新闻关键字」。
    veto_watch = bool(readings.get("signal")) and len(features["risk_off_hits"]) >= 2
    contested = bool(readings.get("signal")) and label["tone"] == "risk-off" and label["confidence"] >= 0.7
    record = {
        "as_of": day.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "model": getattr(llm, "name", "unknown"),
        "usage": usage,
        "news": {
            "source": news["source"],
            "cached": news["cached"],
            "retrieved": len(news["documents"]),
            "used": len(kept),
            "dropped_after_as_of": dropped,
            "errors": news["errors"],
        },
        # 用过的标题全留，不只前几条：`data/news/` 只是缓存（进了 .gitignore），
        # 真正要能复查的是这个文件。每天 40 条上下，一个月也就几百 KB。
        "headlines": [
            {
                "published_at": document["published_at"],
                "title": document["title"],
                "url": document["url"],
                "domain": document.get("domain", ""),
            }
            for document in kept[:MAX_HEADLINES_KEPT]
        ],
        "headlines_not_shown": max(0, len(kept) - MAX_HEADLINES_KEPT),
        "readings": readings,
        "label": label,
        "keywords": features,
        "veto_watch": veto_watch,
        "contested": contested,
    }

    path = artifact_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    frozen = day_path(root, day)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    log = root / LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps({k: v for k, v in record.items() if k != "headlines"}, ensure_ascii=False) + "\n")
    return record


def load(root: Path, day: date | None = None) -> dict | None:
    """`artifacts/daily.json`（或指定某天的那份），不存在就是 None。

    页面照常显示，只是没有这一行。
    """
    path = day_path(root, day) if day is not None else artifact_path(root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
