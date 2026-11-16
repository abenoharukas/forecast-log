"""Wikipedia 的 Portal:Current events，当作 2013 年以来的每日新闻底稿。

为什么不用 GDELT 的历史接口：它能按日期查，但免费额度按 IP 限流，从这台机器上一个
窗口要 8 分钟才失败一次（全是 429），按周补完 13 年要几十小时。Wikipedia 每天一页、
没有 key、一天 0.9 秒，2013 年至今都在。

代价要写清楚：它是**世界新闻摘要**（含经济/市场条目），不是行情快讯。对「今天新闻流
什么调子」够用，对「某只股票的公告」不够用。升级路径写在 README。
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

WIKI = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "forecast-log/1.0 (daily news archive; research use)"

_BULLET = re.compile(r"^(?P<stars>\*+)\s*(?P<body>.*)$")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_LINK = re.compile(r"\[\[(?P<target>[^\]|]+)(?:\|(?P<label>[^\]]+))?\]\]")
_EXTLINK = re.compile(r"\[(?:https?|ftp)://[^\]]*\]")
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S)
_TAG = re.compile(r"<[^>]+>")


def day_dir(root: Path) -> Path:
    return root / "data" / "news"


def page_title(day: date) -> str:
    return f"Portal:Current_events/{day.year}_{day:%B}_{day.day}"


def _clean(text: str) -> str:
    text = _REF.sub("", text)
    for _ in range(3):  # {{a|{{b}}}} 这种嵌套，清三轮就干净了
        text = _TEMPLATE.sub("", text)
    text = _LINK.sub(lambda m: (m.group("label") or m.group("target")).strip(), text)
    text = _EXTLINK.sub("", text)
    text = _TAG.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", text).strip(" ,;:.")


def wiki_page(day: date, timeout: float = 30.0) -> tuple[str, str]:
    params = urllib.parse.urlencode(
        {
            "action": "parse",
            "page": page_title(day),
            "format": "json",
            "prop": "wikitext",
            "formatversion": "2",
            "redirects": "1",
        }
    )
    request = urllib.request.Request(f"{WIKI}?{params}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as error:
        return "", f"HTTP {error.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return "", f"{type(error).__name__}: {error}"
    except json.JSONDecodeError:
        return "", "unparseable response"
    if "error" in body:
        return "", f"api: {body['error'].get('code', '')}"
    return body.get("parse", {}).get("wikitext", "") or "", ""


def wiki_batch(days: list[date], timeout: float = 60.0) -> tuple[dict[str, str], str]:
    """一次取多天（MediaWiki 上限 50 个标题）。一天一次请求会被限流，一批 50 天不会。"""
    titles = [page_title(day) for day in days]
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "revisions",
            "rvslots": "main",
            "rvprop": "content",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
        }
    )
    request = urllib.request.Request(f"{WIKI}?{params}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as error:
        return {}, f"HTTP {error.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {}, f"{type(error).__name__}: {error}"
    except json.JSONDecodeError:
        return {}, "unparseable response"
    if "error" in body:
        return {}, f"api: {body['error'].get('code', '')}"
    normalised = {
        item["from"]: item["to"] for item in (body.get("query", {}).get("normalized") or [])
    }
    out: dict[str, str] = {}
    for page in body.get("query", {}).get("pages", []):
        if page.get("missing"):
            continue
        content = ((page.get("revisions") or [{}])[0].get("slots", {}).get("main", {}) or {}).get("content", "")
        if content:
            out[page["title"]] = content
    return {**out, **{norm: out[target] for norm, target in normalised.items() if target in out}}, ""


def parse_current_events(wikitext: str) -> list[dict]:
    """一天的页面 → 条目列表。`*` / `**` / `***` 的层级留成 depth，后面筛市场条目要用。"""
    items = []
    for line in wikitext.splitlines():
        match = _BULLET.match(line.strip())
        if not match:
            continue
        text = _clean(match.group("body"))
        if len(text) < 12:
            continue
        if text.lower().startswith(("see also", "references", "external links")):
            continue
        items.append({"depth": len(match.group("stars")), "title": text[:400]})
    return items


def fetch_day(root: Path, day: date, pace: float = 0.0, attempts: int = 4) -> dict:
    """一天一页，写入 SQLite；数据库或旧 JSON 已经在就直接返回。

    只有真的取到页面才落库：429 / 网络错误不写，这样下次重跑会重试那一天，
    不会把「限流」永久记成「这天没新闻」。
    """
    from forecast_log.db import get_news_fetch, save_news_fetch

    cached = get_news_fetch(root, day)
    if cached is not None:
        return cached
    path = day_dir(root) / f"{day.isoformat()}.json"
    if path.is_file():
        return json.loads(path.read_text())
    wikitext, error = "", ""
    for attempt in range(1, attempts + 1):
        wikitext, error = wiki_page(day)
        if wikitext:
            break
        if error.startswith("HTTP 429"):
            time.sleep(4.0 * attempt)
        elif error.startswith("HTTP"):
            break
        else:
            time.sleep(1.5 * attempt)
    items = parse_current_events(wikitext) if wikitext else []
    record = {
        "as_of": day.isoformat(),
        "source": "wikipedia-current-events",
        "page": page_title(day),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": [
            {
                "title": item["title"],
                "text": item["title"],
                "url": f"https://en.wikipedia.org/wiki/{page_title(day).replace(' ', '_')}",
                "domain": "en.wikipedia.org",
                "published_at": day.isoformat(),
                "depth": item["depth"],
            }
            for item in items
        ],
        "status": "ok" if items else "empty",
        "error": error,
    }
    if error and not wikitext:
        return record
    save_news_fetch(root, record)
    if pace:
        time.sleep(pace)
    return record


def backfill_batch(root: Path, days: list[date], batch: int = 40, pace: float = 1.5) -> dict:
    """按批补：一批 40 天一次请求，写完按天落库。中断重跑只补缺的那些天。"""
    from forecast_log.db import existing_news_dates, save_news_fetch

    have_db = existing_news_dates(root)
    todo = [day for day in days if day.isoformat() not in have_db and not (day_dir(root) / f"{day.isoformat()}.json").is_file()]
    stats = {"requested": len(days), "already": len(days) - len(todo), "ok": 0, "empty": 0, "failed": 0, "errors": []}
    for start in range(0, len(todo), batch):
        group = todo[start : start + batch]
        for attempt in range(1, 4):
            contents, error = wiki_batch(group)
            if contents or error.startswith("api:"):
                break
            time.sleep(5.0 * attempt)
        if not contents:
            stats["failed"] += len(group)
            stats["errors"].append(f"{group[0]}..{group[-1]}: {error}")
            continue
        for day in group:
            key = page_title(day)
            wikitext = contents.get(key) or contents.get(key.replace("_", " "))
            if not wikitext:
                stats["failed"] += 1
                stats["errors"].append(f"{day}: not in batch response")
                continue
            items = parse_current_events(wikitext)
            record = {
                "as_of": day.isoformat(),
                "source": "wikipedia-current-events",
                "page": key,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "documents": [
                    {
                        "title": item["title"],
                        "text": item["title"],
                        "url": f"https://en.wikipedia.org/wiki/{key.replace(' ', '_')}",
                        "domain": "en.wikipedia.org",
                        "published_at": day.isoformat(),
                        "depth": item["depth"],
                    }
                    for item in items
                ],
                "status": "ok" if items else "empty",
                "error": "",
            }
            save_news_fetch(root, record)
            stats["ok" if items else "empty"] += 1
        time.sleep(pace)
    return stats


def backfill_days(root: Path, days: list[date], workers: int = 3) -> dict:
    """并发补多天。按天入库、按天跳过，中断了重跑只补缺的那几天。"""
    from concurrent.futures import ThreadPoolExecutor
    from forecast_log.db import existing_news_dates

    have_db = existing_news_dates(root)
    todo = [day for day in days if day.isoformat() not in have_db and not (day_dir(root) / f"{day.isoformat()}.json").is_file()]
    stats = {
        "requested": len(days),
        "already": len(days) - len(todo),
        "ok": 0,
        "empty": 0,
        "failed": 0,
        "errors": [],
    }
    if not todo:
        return stats
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for record in pool.map(lambda day: fetch_day(root, day), todo):
            if record["status"] == "ok":
                stats["ok"] += 1
            elif record["status"] == "empty":
                stats["empty"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(f"{record['as_of']}: {record.get('error', '')}")
    return stats


def trading_days(root: Path, symbol: str = "^NDX") -> list[date]:
    """有行情的那些天，回填和覆盖率都按这个口径算，别按日历天。"""
    from forecast_log.bars import load_series
    from forecast_log.paths import series_path

    series = load_series(series_path(root, symbol))
    return [bar.date for bar in series.bars]


def coverage(root: Path, days: list[date] | None = None) -> dict:
    """按年统计有新闻的天数；给了交易日列表，就把缺哪几天也算出来。"""
    from forecast_log.db import existing_news_dates, news_coverage_stats

    by_year = news_coverage_stats(root)
    have = existing_news_dates(root)
    if not by_year and day_dir(root).is_dir():
        for path in sorted(day_dir(root).glob("????-??-??.json")):
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if record.get("documents"):
                by_year[path.stem[:4]] = by_year.get(path.stem[:4], 0) + 1
        have = {path.stem for path in day_dir(root).glob("????-??-??.json")}
    out: dict = {"days_with_news": sum(by_year.values()), "by_year": dict(sorted(by_year.items()))}
    if days is not None:
        missing = [day.isoformat() for day in days if day.isoformat() not in have]
        out["trading_days"] = len(days)
        out["trading_days_with_news"] = len(days) - len(missing)
        out["missing_count"] = len(missing)
        out["missing_first"] = missing[:10]
    return out
