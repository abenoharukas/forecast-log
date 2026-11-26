"""实时新闻与 5 日预测参考资料聚合层。

职责：
1. 并发拉取多路金融/宏观 RSS（华尔街日报市场、CNBC 市场与全球、美联储货币政策、SEC 公告、
   Yahoo Finance 纳指/QQQ 与中概/A股），并按需用 Wikipedia Current Events 一次性批量补齐
   最近缺失的历史空窗日（最多回溯 14 天），全部写入 SQLite `news` 表。
2. 组装 5 日涨跌预测所需的其他参考资料：
   - 宏观事件日历倒计时（CPI / NFP / FOMC 及当前是否处于信号触发/否决窗口）
   - 跨资产参考指标快照（10Y 美债、美元 DXY、信用 HYG/TLT、VIX、VXN、黄金、大盘广度）
   - 标题关键字情绪扫描（risk-on / risk-off 命中与否决观察标记）
   - 最新 LLM 日度情绪解读（`artifacts/daily.json`，若存在）
3. 提供 15 分钟自动冷却判断：页面每次打开自动调用 `/api/news/refresh` 时，若今日新闻已新鲜入库则
   毫秒级返回缓存；若新闻过期或用户手动点击「获取新闻」（`force=1`）则立即并发抓取并落库。
"""

from __future__ import annotations

import email.utils
import hashlib
import math
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from forecast_log import daily, events, keywords, news_ab, news_labels, polymarket, providers, surprise, wiki_news
from forecast_log.agent.env import merged_env
from forecast_log.db import (
    existing_news_dates,
    get_series,
    latest_news_status,
    load_recent_news,
    save_news_fetch,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
USER_AGENT = "forecast-log/1.0 (+market news & macro reference)"
NEWS_COOLDOWN_SECONDS = 900  # 15 分钟自动刷新冷却
MAX_GAP_BACKFILL_DAYS = 14
FEED_TIMEOUT_SECONDS = 8.0

_NEWS_REFRESH_LOCK = threading.Lock()
_LLM_STATE_LOCK = threading.Lock()
_LLM_SCORING_STATE: dict[Path, dict[str, Any]] = {}

# (source_id, url, category, label)
LIVE_FEEDS: tuple[tuple[str, str, str, str], ...] = (
    ("wsj-markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "us", "WSJ Markets"),
    (
        "cnbc-markets",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
        "us",
        "CNBC Markets",
    ),
    (
        "cnbc-world",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
        "global",
        "CNBC World",
    ),
    ("fed-monetary", "https://www.federalreserve.gov/feeds/press_monetary.xml", "macro", "Fed Monetary"),
    ("sec-press", "https://www.sec.gov/news/pressreleases.rss", "macro", "SEC Press"),
    (
        "yahoo-ndx",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5ENDX,QQQ,SPY&region=US&lang=en-US",
        "us",
        "Yahoo US/NDX",
    ),
    (
        "yahoo-ssec",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=000001.SS,FXI,KWEB&region=US&lang=en-US",
        "china",
        "Yahoo China/A-Shares",
    ),
)

SOURCE_CATEGORY_MAP = {name: cat for name, _, cat, _ in LIVE_FEEDS}


def _parse_iso_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _parse_feed_datetime(raw: str) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(text)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, IndexError):
        pass
    dt = _parse_iso_dt(text)
    if dt is not None:
        return dt.isoformat(timespec="seconds")
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10]).isoformat()
        except ValueError:
            return None
    return None


def _extract_link(item: ElementTree.Element) -> str:
    for child in item:
        tag = child.tag.split("}")[-1]
        if tag == "link":
            href = (child.attrib.get("href") or "").strip()
            if href:
                return href
            text = (child.text or "").strip()
            if text:
                return text
    for child in item:
        if child.tag.split("}")[-1] == "guid":
            text = (child.text or "").strip()
            if text.startswith("http"):
                return text
    return ""


def _fetch_rss(name: str, url: str, timeout: float = FEED_TIMEOUT_SECONDS) -> tuple[list[dict], str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    try:
        xml_root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        return [], f"parse error: {exc}"

    domain = url.split("/")[2] if "://" in url else name
    docs: list[dict] = []
    for node in xml_root.iter():
        tag = node.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        title = ""
        pub_raw = ""
        for child in node:
            ctag = child.tag.split("}")[-1]
            if ctag == "title" and not title:
                title = (child.text or "").strip()
            elif ctag in ("pubDate", "updated", "published", "date") and not pub_raw:
                pub_raw = (child.text or "").strip()
        link = _extract_link(node)
        published_at = _parse_feed_datetime(pub_raw)
        if not title or not link or not published_at:
            continue
        link_domain = link.split("/")[2] if "://" in link else domain
        doc_id = hashlib.sha1(f"{name}|{link}|{title}".encode("utf-8")).hexdigest()[:12]
        docs.append(
            {
                "id": doc_id,
                "published_at": published_at,
                "title": " ".join(title.split())[:400],
                "text": " ".join(title.split())[:400],
                "url": link,
                "domain": link_domain,
                "source": name,
            }
        )
    return docs, ""


def _backfill_recent_wiki_gap(root: Path, today: date, max_days: int = MAX_GAP_BACKFILL_DAYS) -> tuple[list[str], str]:
    """用单次 MediaWiki 批量请求补齐最近缺失的每日新闻记录（最多回溯 max_days 天）。"""
    have = existing_news_dates(root)
    # 只检查最近 max_days 天内（不含今天，今天主要靠实时财经 RSS + 昨日百科）
    start = today - timedelta(days=max_days)
    missing = [
        start + timedelta(days=offset)
        for offset in range(max_days)
        if (start + timedelta(days=offset)).isoformat() not in have
    ]
    if not missing:
        return [], ""

    contents, error = wiki_news.wiki_batch(missing, timeout=12.0)
    if not contents:
        return [], error or "empty wiki batch"

    filled: list[str] = []
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for day in missing:
        key = wiki_news.page_title(day)
        wikitext = contents.get(key) or contents.get(key.replace("_", " "))
        if not wikitext:
            continue
        items = wiki_news.parse_current_events(wikitext)
        if not items:
            continue
        record = {
            "as_of": day.isoformat(),
            "source": "wikipedia-current-events",
            "page": key,
            "fetched_at": fetched_at,
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
            "status": "ok",
            "error": "",
        }
        save_news_fetch(root, record)
        filled.append(day.isoformat())
    return filled, ""


def needs_news_refresh(
    root: Path,
    cooldown_sec: int = NEWS_COOLDOWN_SECONDS,
    now: datetime | None = None,
) -> bool:
    """判断新闻表是否需要自动抓取（无今日新闻、今日为空、或距上次抓取超过冷却期）。"""
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    today_bj = now_utc.astimezone(SHANGHAI).date().isoformat()
    status = latest_news_status(root)
    latest_as_of = status.get("latest_as_of")
    if not latest_as_of or latest_as_of < today_bj:
        return True
    if int(status.get("latest_day_docs") or 0) <= 0:
        return True
    fetched_dt = _parse_iso_dt(status.get("latest_fetched_at"))
    if fetched_dt is None:
        return True
    age = (now_utc - fetched_dt).total_seconds()
    return age >= cooldown_sec


def _classify_item(doc: dict) -> dict:
    title = str(doc.get("title") or "")
    hits = [name for name, pat in keywords.COMPILED.items() if pat.search(title)]
    off_hits = [h for h in hits if h in keywords.RISK_OFF]
    on_hits = [h for h in hits if h in keywords.RISK_ON]
    if off_hits and not on_hits:
        sentiment = "risk-off"
    elif on_hits and not off_hits:
        sentiment = "risk-on"
    elif off_hits and on_hits:
        sentiment = "mixed"
    else:
        sentiment = "neutral"

    source = str(doc.get("source") or "")
    domain = str(doc.get("domain") or "").lower()
    category = "global"
    if "yahoo-ssec" in source or any(k in title.lower() for k in ("china", "chinese", "beijing", "pboc", "shanghai", "csi", "hang seng", "alibaba", "tencent")):
        category = "china"
    elif "fed-monetary" in source or "sec-press" in source or "federalreserve.gov" in domain or "sec.gov" in domain:
        category = "macro"
    elif "wikipedia" in source or "wikipedia.org" in domain:
        category = "global"
    elif any(s in source for s in ("wsj", "cnbc-markets", "yahoo-ndx")) or any(d in domain for d in ("wsj.com", "cnbc.com", "yahoo.com")):
        category = "us"

    return {
        **doc,
        "category": category,
        "tags": hits[:5],
        "sentiment": sentiment,
    }


def _macro_calendar_reference(root: Path, today: date) -> dict:
    """读取 data/events.json，计算距下一次 CPI / NFP / FOMC 的天数及 5 日预测窗口提示。"""
    cal = events.load(root)
    if not cal:
        return {"available": False, "as_of": today.isoformat(), "events": []}

    specs = (
        ("cpi", "CPI 通胀数据", "CPI 前 0–3 天且复合分 ≥ 中位触发买入增补"),
        ("nfp", "NFP 非农就业", "NFP 前 3–7 天且复合分 ≥ 中位触发买入增补"),
        ("fomc", "FOMC 联储决议", "FOMC 决议当天否决所有买卖信号"),
    )
    items: list[dict] = []
    for key, label, rule_hint in specs:
        dates_list = cal.get(key) or []
        next_day = next((d for d in dates_list if d >= today), None)
        days_away = (next_day - today).days if next_day is not None else None
        in_window = False
        window_status = "常规期"
        if days_away is not None:
            if key == "cpi" and 0 <= days_away <= 3:
                in_window = True
                window_status = "触发窗口（临近 CPI）"
            elif key == "nfp" and 3 <= days_away <= 7:
                in_window = True
                window_status = "触发窗口（NFP 前 3–7 天）"
            elif key == "fomc" and days_away == 0:
                in_window = True
                window_status = "否决窗口（FOMC 决议日）"
            elif key == "fomc" and 1 <= days_away <= 3:
                window_status = "临近决议（观望留意）"
        items.append(
            {
                "key": key,
                "label": label,
                "next_date": next_day.isoformat() if next_day else None,
                "days_away": days_away,
                "in_window": in_window,
                "window_status": window_status,
                "rule_hint": rule_hint,
            }
        )
    return {
        "available": True,
        "as_of": today.isoformat(),
        "events": items,
    }


def _pct_change(latest: float, prev: float) -> float | None:
    if prev == 0 or not math.isfinite(latest) or not math.isfinite(prev):
        return None
    return round((latest / prev - 1.0) * 100.0, 2)


def _cross_assets_reference(root: Path) -> list[dict]:
    """从 bars 表汇总 5 日涨跌预测的核心跨资产参考指标快照。"""
    out: list[dict] = []

    def _single(symbol: str, label: str, unit: str, bullish_when: str) -> dict | None:
        series = get_series(root, symbol)
        if series is None or not series.bars:
            return None
        bars = series.bars
        last = bars[-1]
        c0 = float(last.close)
        c1 = float(bars[-2].close) if len(bars) >= 2 else None
        c5 = float(bars[-6].close) if len(bars) >= 6 else None
        c20 = float(bars[-21].close) if len(bars) >= 21 else None

        if unit == "diff":
            chg1 = round(c0 - c1, 3) if c1 is not None else None
            chg5 = round(c0 - c5, 3) if c5 is not None else None
            chg20 = round(c0 - c20, 3) if c20 is not None else None
        else:
            chg1 = _pct_change(c0, c1) if c1 is not None else None
            chg5 = _pct_change(c0, c5) if c5 is not None else None
            chg20 = _pct_change(c0, c20) if c20 is not None else None

        hint = "中性"
        if symbol in ("^VIX", "^VXN") and len(bars) >= 60:
            window = sorted(float(b.close) for b in bars[-252:])
            med = window[len(window) // 2]
            ratio = (c0 / med - 1.0) * 100.0 if med else 0.0
            hint = f"较年均中位 {ratio:+.1f}%" + ("（偏高→恐慌反向偏多）" if ratio > 5 else "（偏低→平静）" if ratio < -5 else "")
        elif chg20 is not None:
            if bullish_when == "down":
                hint = "20日下行（偏多因子）" if chg20 < 0 else "20日上行（偏空压力）"
            elif bullish_when == "up":
                hint = "20日走强（偏多因子）" if chg20 > 0 else "20日走弱（偏空压力）"

        return {
            "symbol": symbol,
            "label": label,
            "date": last.date.isoformat(),
            "close": round(c0, 2),
            "unit": unit,
            "chg_1d": chg1,
            "chg_5d": chg5,
            "chg_20d": chg20,
            "is_closed": bool(last.is_closed),
            "hint": hint,
        }

    for spec in (
        ("^TNX", "10Y 美债收益率", "diff", "down"),
        ("DX-Y.NYB", "美元指数 DXY", "pct", "down"),
        ("^VIX", "标普波动率 VIX", "pct", "high"),
        ("^VXN", "纳指波动率 VXN", "pct", "high"),
        ("GC=F", "黄金期货 GC=F", "pct", "neutral"),
    ):
        row = _single(*spec)
        if row:
            out.append(row)

    # HYG/TLT 信用风险偏好比（核心因子 3）
    hyg = get_series(root, "HYG")
    tlt = get_series(root, "TLT")
    if hyg and tlt and hyg.bars and tlt.bars:
        tlt_by_date = {b.date: float(b.close) for b in tlt.bars}
        ratios = [(b.date, float(b.close) / tlt_by_date[b.date]) for b in hyg.bars if b.date in tlt_by_date and tlt_by_date[b.date] > 0]
        if ratios:
            last_d, r0 = ratios[-1]
            r1 = ratios[-2][1] if len(ratios) >= 2 else None
            r5 = ratios[-6][1] if len(ratios) >= 6 else None
            r20 = ratios[-21][1] if len(ratios) >= 21 else None
            c20 = _pct_change(r0, r20) if r20 is not None else None
            out.insert(
                2,
                {
                    "symbol": "HYG/TLT",
                    "label": "信用风险偏好 (HYG/TLT)",
                    "date": last_d.isoformat(),
                    "close": round(r0, 4),
                    "unit": "pct",
                    "chg_1d": _pct_change(r0, r1) if r1 is not None else None,
                    "chg_5d": _pct_change(r0, r5) if r5 is not None else None,
                    "chg_20d": c20,
                    "is_closed": bool(hyg.bars[-1].is_closed and tlt.bars[-1].is_closed),
                    "hint": "20日走强（偏多因子）" if (c20 or 0) > 0 else "20日走弱（避险偏空）",
                },
            )
    return out


def _daily_llm_summary(root: Path) -> dict | None:
    record = daily.load(root)
    if not record:
        return None
    label = record.get("label") or {}
    return {
        "as_of": record.get("as_of"),
        "generated_at": record.get("generated_at"),
        "model": record.get("model"),
        "tone": label.get("tone"),
        "confidence": label.get("confidence"),
        "drivers": label.get("drivers") or [],
        "note": label.get("note") or "",
        "contested": bool(record.get("contested")),
    }


def _resolve_llm_models(root: Path) -> tuple[str, str, dict[str, str]]:
    env = merged_env(root)
    ds_model = env.get("DEEPSEEK_MODEL") or "deepseek-chat"
    qw_model = env.get("QWEN_MODEL") or "qwen3.8-flash"
    return ds_model, qw_model, env


def _get_llm_state(root: Path) -> dict[str, Any]:
    key = root.resolve()
    with _LLM_STATE_LOCK:
        st = _LLM_SCORING_STATE.get(key)
        if st is None:
            st = {
                "running": False,
                "signals_recomputed": False,
                "last_updated_at": "",
                "errors": [],
                "pending_deepseek": 0,
                "pending_qwen": 0,
            }
            _LLM_SCORING_STATE[key] = st
        return dict(st)


def _recent_news_dates(root: Path, today: date, days_back: int = MAX_GAP_BACKFILL_DAYS) -> list[date]:
    cutoff = (today - timedelta(days=max(1, days_back))).isoformat()
    have = existing_news_dates(root)
    out: list[date] = []
    for raw in sorted(have):
        if raw >= cutoff and raw <= today.isoformat():
            try:
                out.append(date.fromisoformat(raw))
            except ValueError:
                continue
    return out


def _pending_llm_days(
    root: Path,
    today: date,
    days_back: int = MAX_GAP_BACKFILL_DAYS,
    force_today: bool = False,
) -> tuple[list[date], list[date], str, str, dict[str, str]]:
    ds_model, qw_model, env = _resolve_llm_models(root)
    candidates = _recent_news_dates(root, today, days_back=days_back)
    today_iso = today.isoformat()

    ds_todo: list[date] = []
    if env.get("DEEPSEEK_API_KEY"):
        for day in candidates:
            iso = day.isoformat()
            if (force_today and iso == today_iso) or not surprise.day_path(root, iso).is_file():
                ds_todo.append(day)

    qw_todo: list[date] = []
    if env.get("QWEN_API_KEY"):
        qw_dir = news_ab.ab_dir(root, news_ab.slug_of(qw_model))
        for day in candidates:
            iso = day.isoformat()
            if (force_today and iso == today_iso) or not (qw_dir / f"{iso}.json").is_file():
                qw_todo.append(day)

    return ds_todo, qw_todo, ds_model, qw_model, env


def _recompute_signals_safe(root: Path) -> bool:
    """当新闻或 LLM 新闻标签更新后，重算 ndx/qqq/ssec 的信号与因素位图。"""
    from forecast_log import artifacts_db
    from forecast_log.bars import has_series
    from forecast_log.commands import cross_command

    if not has_series(root, "^NDX"):
        return False
    artifacts_db._NEWS_PROVENANCE_CACHE.pop(root.resolve(), None)
    cross_command(root)
    return True


def score_recent_news_with_llms(
    root: Path,
    days_back: int = MAX_GAP_BACKFILL_DAYS,
    force_today: bool = False,
    recompute_signals: bool = True,
) -> dict[str, Any]:
    """用 DeepSeek（surprise × direction）与 Qwen（regime / severity / events）并发解读近期新闻并重算信号。"""
    now_sh = datetime.now(SHANGHAI)
    today_bj = now_sh.date()
    today_iso = today_bj.isoformat()
    ds_todo, qw_todo, ds_model, qw_model, env = _pending_llm_days(
        root, today_bj, days_back=days_back, force_today=force_today
    )

    errors: list[str] = []
    ds_scored = 0
    qw_scored = 0
    overwrite = {today_iso} if force_today else set()

    def _run_ds() -> tuple[int, list[str]]:
        if not ds_todo:
            return 0, []
        llm = providers.build(ds_model, root, env=env)
        st = surprise.run(root, ds_todo, llm=llm, workers=2, overwrite_days=overwrite)
        if st.get("days", 0) > 0:
            surprise.export_jsonl(root)
        return int(st.get("days", 0)), list(st.get("errors") or [])

    def _run_qw() -> tuple[int, list[str]]:
        if not qw_todo:
            return 0, []
        llm = providers.build(qw_model, root, env=env)
        st = news_ab.label(
            root,
            qw_todo,
            qw_model,
            llm,
            workers=2,
            overwrite_days=overwrite,
            batch_days=4,
        )
        return int(st.get("days", 0)), list(st.get("errors") or [])

    if ds_todo or qw_todo:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = {}
            if ds_todo:
                futs[pool.submit(_run_ds)] = "deepseek"
            if qw_todo:
                futs[pool.submit(_run_qw)] = "qwen"
            for fut in as_completed(futs):
                who = futs[fut]
                try:
                    cnt, errs = fut.result()
                    if who == "deepseek":
                        ds_scored = cnt
                    else:
                        qw_scored = cnt
                    errors.extend(f"{who}: {e}" for e in errs)
                except Exception as exc:
                    errors.append(f"{who}: {exc}")

    did_recompute = False
    if recompute_signals and (ds_scored > 0 or qw_scored > 0):
        try:
            did_recompute = _recompute_signals_safe(root)
        except Exception as exc:
            errors.append(f"recompute_signals: {exc}")

    ts = datetime.now(SHANGHAI).isoformat(timespec="seconds")
    key = root.resolve()
    with _LLM_STATE_LOCK:
        _LLM_SCORING_STATE[key] = {
            "running": False,
            "signals_recomputed": did_recompute,
            "last_updated_at": ts,
            "errors": errors[:6],
            "pending_deepseek": 0,
            "pending_qwen": 0,
        }
    return {
        "deepseek_scored": ds_scored,
        "qwen_scored": qw_scored,
        "signals_recomputed": did_recompute,
        "errors": errors,
    }


def trigger_llm_news_scoring(
    root: Path,
    days_back: int = MAX_GAP_BACKFILL_DAYS,
    force_today: bool = False,
    sync: bool = False,
) -> dict[str, Any]:
    """按需触发 DeepSeek + Qwen 解读近期新闻；默认后台异步执行，不阻塞页面首屏。"""
    today_bj = datetime.now(SHANGHAI).date()
    ds_todo, qw_todo, _, _, _ = _pending_llm_days(
        root, today_bj, days_back=days_back, force_today=force_today
    )
    if not ds_todo and not qw_todo:
        return _get_llm_state(root)

    key = root.resolve()
    with _LLM_STATE_LOCK:
        cur = _LLM_SCORING_STATE.get(key)
        if cur and cur.get("running"):
            return dict(cur)
        _LLM_SCORING_STATE[key] = {
            "running": True,
            "signals_recomputed": False,
            "last_updated_at": (cur or {}).get("last_updated_at", ""),
            "errors": [],
            "pending_deepseek": len(ds_todo),
            "pending_qwen": len(qw_todo),
        }

    if sync:
        score_recent_news_with_llms(root, days_back=days_back, force_today=force_today, recompute_signals=True)
        return _get_llm_state(root)

    def _worker() -> None:
        try:
            score_recent_news_with_llms(root, days_back=days_back, force_today=force_today, recompute_signals=True)
        except Exception as exc:
            with _LLM_STATE_LOCK:
                st = _LLM_SCORING_STATE.get(key) or {}
                st["running"] = False
                st["errors"] = [str(exc)]
                _LLM_SCORING_STATE[key] = st

    threading.Thread(target=_worker, daemon=True).start()
    return _get_llm_state(root)


def load_llm_indicators(root: Path) -> dict[str, Any]:
    """读取 DeepSeek 与 Qwen 的最新新闻打分指标及近 7 日对比。"""
    ds_model, qw_model, _ = _resolve_llm_models(root)
    state = _get_llm_state(root)

    # DeepSeek: 合并 surprise.jsonl 与逐日 artifacts/surprise/*.json
    ds_map: dict[str, dict] = {}
    try:
        ds_map.update(surprise.load_compact(root))
        ds_map.update(surprise.load(root))
    except Exception:
        pass

    # Qwen: 读取 artifacts/news_ab/<qwen_slug>/*.json
    qw_slug = news_ab.slug_of(qw_model)
    qw_raw: dict[str, dict] = {}
    try:
        qw_raw = news_ab.load_labeler(root, qw_slug)
    except Exception:
        pass

    def _fmt_ds(day: str, row: dict) -> dict:
        sur = str(row.get("surprise") or "none")
        direction = str(row.get("direction") or "ambiguous")
        return {
            "as_of": day,
            "model": row.get("model") or ds_model,
            "surprise": sur,
            "direction": direction,
            "category": str(row.get("category") or "other"),
            "evidence": str(row.get("evidence") or ""),
            "veto": sur == "strong" and direction == "bearish",
            "boost": sur == "strong" and direction == "bullish",
        }

    def _fmt_qw(day: str, rec: dict) -> dict:
        lbl = rec.get("label") or {}
        sev = float(lbl.get("severity") or 0.0)
        return {
            "as_of": day,
            "model": rec.get("model") or qw_model,
            "regime": str(lbl.get("regime") or "neutral"),
            "severity": round(sev, 3),
            "novelty": round(float(lbl.get("novelty") or 0.0), 3),
            "size_multiplier": news_labels.size_multiplier(sev),
            "veto": sev >= 0.6,
            "note": str(lbl.get("note") or ""),
            "events": list(lbl.get("events") or [])[:3],
        }

    latest_ds_day = max(ds_map) if ds_map else None
    latest_qw_day = max(qw_raw) if qw_raw else None

    all_days = sorted(set(ds_map) | set(qw_raw), reverse=True)[:7]
    recent_days = [
        {
            "as_of": d,
            "deepseek": _fmt_ds(d, ds_map[d]) if d in ds_map else None,
            "qwen": _fmt_qw(d, qw_raw[d]) if d in qw_raw else None,
        }
        for d in all_days
    ]

    latest_date = max(d for d in (latest_ds_day, latest_qw_day) if d) if (latest_ds_day or latest_qw_day) else None
    return {
        "running": bool(state.get("running")),
        "signals_recomputed": bool(state.get("signals_recomputed")),
        "last_updated_at": state.get("last_updated_at") or "",
        "pending_deepseek": int(state.get("pending_deepseek") or 0),
        "pending_qwen": int(state.get("pending_qwen") or 0),
        "latest_date": latest_date,
        "deepseek": ({**_fmt_ds(latest_ds_day, ds_map[latest_ds_day]), "total_days": len(ds_map)} if latest_ds_day else None),
        "qwen": ({**_fmt_qw(latest_qw_day, qw_raw[latest_qw_day]), "total_days": len(qw_raw)} if latest_qw_day else None),
        "recent_days": recent_days,
        "errors": list(state.get("errors") or []),
    }


def load_news_bundle(
    root: Path,
    market: str = "ndx",
    limit: int = 50,
    cached: bool = True,
    fetch_meta: dict[str, Any] | None = None,
) -> dict:
    """从本地数据库与参考文件读取最新新闻 + 5日预测参考资料（含 DeepSeek × Qwen 新闻指标）。"""
    now_sh = datetime.now(SHANGHAI)
    today_bj = now_sh.date()
    status = latest_news_status(root)
    raw_items = load_recent_news(root, limit=limit, days=14)
    items = [_classify_item(item) for item in raw_items]

    # 对最近的新闻标题跑关键字风险扫描
    scan_titles = [item["title"] for item in items[:40]]
    kw = keywords.scan(scan_titles)
    kw["veto_watch"] = len(kw["risk_off_hits"]) >= 2

    sources_seen: list[str] = []
    for item in items:
        for src in str(item.get("source") or "").split("+"):
            src = src.strip()
            if src and src != "none" and src not in sources_seen:
                sources_seen.append(src)

    meta = fetch_meta or {}
    return {
        "ok": True,
        "cached": cached,
        "market": market,
        "beijing_date": today_bj.isoformat(),
        "refreshed_at": meta.get("refreshed_at") or status.get("latest_fetched_at") or now_sh.isoformat(timespec="seconds"),
        "news": {
            "latest_as_of": status.get("latest_as_of"),
            "fetched_at": status.get("latest_fetched_at") or "",
            "total_rows": status.get("total_rows") or 0,
            "latest_day_docs": status.get("latest_day_docs") or 0,
            "recent_days": status.get("recent_days") or [],
            "sources": meta.get("providers") or sources_seen,
            "backfilled_days": meta.get("backfilled_days") or [],
            "new_count": meta.get("new_count", 0),
            "errors": meta.get("errors") or [],
            "items": items,
        },
        "keywords": kw,
        "llm_indicators": load_llm_indicators(root),
        "macro_calendar": _macro_calendar_reference(root, today_bj),
        "cross_assets": _cross_assets_reference(root),
        "polymarket": polymarket.load_cached_polymarket(root),
        "daily_llm": _daily_llm_summary(root),
    }


def refresh_news(
    root: Path,
    force: bool = False,
    market: str = "ndx",
    limit: int = 50,
    cooldown_sec: int = NEWS_COOLDOWN_SECONDS,
    score_llm: bool = True,
    sync_llm: bool = False,
) -> dict:
    """刷新新闻并落库，自动触发 DeepSeek + Qwen 新闻解读与信号重算，返回完整参考资料包。"""
    with _NEWS_REFRESH_LOCK:
        if not force and not needs_news_refresh(root, cooldown_sec=cooldown_sec):
            if polymarket.needs_polymarket_refresh(root, cooldown_sec=cooldown_sec):
                try:
                    polymarket.refresh_polymarket(root, force=False, cooldown_sec=cooldown_sec)
                except Exception:
                    pass
            if score_llm:
                trigger_llm_news_scoring(root, force_today=False, sync=sync_llm)
            return load_news_bundle(root, market=market, limit=limit, cached=True)

        now_sh = datetime.now(SHANGHAI)
        today_bj = now_sh.date()
        fetched_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

        all_docs: list[dict] = []
        providers_ok: list[str] = []
        errors: list[str] = []
        backfilled_days: list[str] = []

        with ThreadPoolExecutor(max_workers=len(LIVE_FEEDS) + 2) as pool:
            feed_futs = {
                pool.submit(_fetch_rss, name, url): name
                for name, url, _, _ in LIVE_FEEDS
            }
            wiki_fut = pool.submit(_backfill_recent_wiki_gap, root, today_bj)
            poly_fut = pool.submit(polymarket.refresh_polymarket, root, force, cooldown_sec)

            for fut in as_completed(feed_futs):
                name = feed_futs[fut]
                try:
                    docs, err = fut.result()
                    if docs:
                        providers_ok.append(name)
                        all_docs.extend(docs)
                    if err:
                        errors.append(f"{name}: {err}")
                except Exception as exc:
                    errors.append(f"{name}: {exc}")

            try:
                filled, wiki_err = wiki_fut.result()
                backfilled_days = filled
                if filled:
                    providers_ok.append("wikipedia-current-events")
                if wiki_err:
                    errors.append(f"wikipedia: {wiki_err}")
            except Exception as exc:
                errors.append(f"wikipedia: {exc}")

            try:
                poly_res = poly_fut.result()
                if poly_res.get("available"):
                    providers_ok.append("polymarket")
                for perr in (poly_res.get("errors") or [])[:2]:
                    errors.append(f"polymarket: {perr}")
            except Exception as exc:
                errors.append(f"polymarket: {exc}")

        seen_ids: set[str] = set()
        seen_urls: set[str] = set()
        unique_docs: list[dict] = []
        for doc in sorted(all_docs, key=lambda d: d.get("published_at") or "", reverse=True):
            did = doc["id"]
            url_key = (doc.get("url") or "").lower()
            if did in seen_ids or (url_key and url_key in seen_urls):
                continue
            seen_ids.add(did)
            if url_key:
                seen_urls.add(url_key)
            unique_docs.append(doc)

        if unique_docs:
            by_source: dict[str, list[dict]] = {}
            for doc in unique_docs[:120]:
                by_source.setdefault(doc.get("source") or "rss", []).append(doc)
            for src_name, src_docs in by_source.items():
                save_news_fetch(
                    root,
                    {
                        "as_of": today_bj.isoformat(),
                        "source": src_name,
                        "page": "",
                        "fetched_at": fetched_utc,
                        "documents": src_docs,
                    },
                )

        # 新新闻入库后先立即重算一次关键字风险因子，再触发 DeepSeek + Qwen 指标打分
        if unique_docs or backfilled_days:
            try:
                _recompute_signals_safe(root)
            except Exception:
                pass

        if score_llm:
            trigger_llm_news_scoring(root, force_today=force, sync=sync_llm)

        return load_news_bundle(
            root,
            market=market,
            limit=limit,
            cached=False,
            fetch_meta={
                "refreshed_at": now_sh.isoformat(timespec="seconds"),
                "providers": list(dict.fromkeys(providers_ok)),
                "backfilled_days": backfilled_days,
                "new_count": len(unique_docs[:120]),
                "errors": errors,
            },
        )
