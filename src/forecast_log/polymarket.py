"""Polymarket 预测市场赔率聚合与按日历史回溯（宏观利率 / 关税中国 / 美股指数）。

使用 Polymarket 公开只读接口（无需 API Key）：
- Gamma API (`https://gamma-api.polymarket.com`)：检索活跃或指定历史交易日在线的盘口元数据
- CLOB API (`https://clob.polymarket.com/prices-history`)：回溯历史信号日收盘时刻 (`21:15 UTC`) 的真实隐含赔率与 24h 变化，杜绝结算后未来函数泄漏
- 实时缓存：`artifacts/polymarket.json`（默认 15 分钟冷却）
- 历史按日缓存：`artifacts/polymarket_days/<YYYY-MM-DD>.json`（一次拉取永久命中）
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")
GAMMA_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
USER_AGENT = "forecast-log/1.0 (+polymarket macro reference)"
POLYMARKET_COOLDOWN_SECONDS = 900
FETCH_TIMEOUT_SECONDS = 8.0
HIST_TIMEOUT_SECONDS = 5.0
MIN_POLYMARKET_DATE = "2023-01-01"

_POLY_LOCK = threading.Lock()
_DAY_LOCK = threading.Lock()

CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("macro", "宏观利率", ("Fed", "rate cut", "recession", "inflation")),
    ("china_tariff", "关税中国", ("tariff", "China")),
    ("us_index", "美股指数", ("S&P 500", "Nasdaq")),
)

HISTORICAL_TAG_SPECS: tuple[tuple[str, str], ...] = (
    ("fed-rates", "macro"),
    ("finance", "macro"),
    ("tariffs", "china_tariff"),
    ("china", "china_tariff"),
    ("stocks", "us_index"),
)

CATEGORY_LABELS = {key: label for key, label, _ in CATEGORY_SPECS}

# 过滤与金融/宏观无关的娱乐、体育或纯地缘军事噪音
_NOISE_RE = re.compile(
    r"\b(nba|nfl|mlb|nhl|ufc|ncaa|premier league|champions league|world cup|"
    r"grammy|oscar|emmy|taylor swift|mrbeast|kardashian|bachelor|gta\s*6|"
    r"taiwan|military clash|blockade|iran strike|strikes iran|ukraine ceasefire|pope|election winner|nominee)\b",
    re.IGNORECASE,
)

_CAT_MATCHERS: dict[str, re.Pattern[str]] = {
    "macro": re.compile(
        r"\b(fed|fomc|interest rate|rate cut|rate hike|bps|cpi|inflation|pce|recession|gdp|powell|treasury|yield)\b",
        re.IGNORECASE,
    ),
    "china_tariff": re.compile(
        r"\b(tariff|tariffs|china|chinese|beijing|pboc|xi jinping|trade war|trade deal|import duty|ai chip|export control|sanctions on china)\b",
        re.IGNORECASE,
    ),
    "us_index": re.compile(
        r"\b(s&p\s*500|spx|spy|nasdaq|ndx|qqq|dow jones|russell 2000|stock market|wall street|all[- ]time high)\b",
        re.IGNORECASE,
    ),
}


def cache_path(root: Path) -> Path:
    return root / "artifacts" / "polymarket.json"


def day_cache_path(root: Path, as_of: str) -> Path:
    return root / "artifacts" / "polymarket_days" / f"{as_of}.json"


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


def _parse_json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return []
    return []


def _to_float(val: Any, default: float | None = None) -> float | None:
    if val is None or val == "":
        return default
    try:
        num = float(val)
        return num if num == num else default
    except (TypeError, ValueError):
        return default


def _fetch_search(query: str, timeout: float = FETCH_TIMEOUT_SECONDS) -> tuple[list[dict], str]:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "events_status": "active",
            "limit_per_type": 8,
        }
    )
    url = f"{GAMMA_SEARCH_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    events = payload.get("events")
    return (events if isinstance(events, list) else []), ""


def _fetch_events_on_date(tag_slug: str, as_of: str, timeout: float = HIST_TIMEOUT_SECONDS) -> tuple[list[dict], str]:
    params = urllib.parse.urlencode(
        {
            "tag_slug": tag_slug,
            "order": "volume",
            "ascending": "false",
            "start_date_max": f"{as_of}T21:15:00Z",
            "end_date_min": f"{as_of}T00:00:00Z",
            "limit": 6,
        }
    )
    url = f"{GAMMA_EVENTS_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return (payload if isinstance(payload, list) else []), ""


def _fetch_clob_price_as_of(token_id: str, cutoff_ts: int, timeout: float = HIST_TIMEOUT_SECONDS) -> tuple[float | None, float | None]:
    """查询 CLOB 历史价格序列，返回截至 cutoff_ts 的隐含概率及前 24h 概率变化。"""
    start_ts = cutoff_ts - 86400 * 4
    params = urllib.parse.urlencode(
        {
            "market": token_id,
            "startTs": start_ts,
            "endTs": cutoff_ts,
            "fidelity": 60,
        }
    )
    url = f"{CLOB_HISTORY_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None, None

    pts = [
        (int(pt["t"]), float(pt["p"]))
        for pt in (payload.get("history") or [])
        if isinstance(pt, dict) and pt.get("t") is not None and pt.get("p") is not None and int(pt["t"]) <= cutoff_ts
    ]
    if not pts:
        return None, None
    pts.sort(key=lambda x: x[0])
    latest_t, latest_p = pts[-1]
    target_prev = latest_t - 86400
    prev_pt = min(pts, key=lambda x: abs(x[0] - target_prev))
    delta_24h = round(latest_p - prev_pt[1], 4) if abs(prev_pt[0] - target_prev) <= 86400 * 1.5 and prev_pt[0] < latest_t else None
    return round(latest_p, 4), delta_24h


def _classify_event(title: str, default_cat: str) -> str | None:
    if not title or _NOISE_RE.search(title):
        return None
    for cat in ("china_tariff", "us_index", "macro"):
        if _CAT_MATCHERS[cat].search(title):
            return cat
    return None


def _extract_market_summary(market: dict) -> dict[str, Any] | None:
    if market.get("closed") is True or market.get("active") is False:
        return None
    outcomes = [str(o) for o in _parse_json_list(market.get("outcomes"))]
    prices_raw = _parse_json_list(market.get("outcomePrices"))
    prices = [_to_float(p) for p in prices_raw]
    if not prices or prices[0] is None:
        return None

    yes_idx = 0
    for idx, name in enumerate(outcomes):
        if name.strip().lower() == "yes" and idx < len(prices) and prices[idx] is not None:
            yes_idx = idx
            break
    prob = prices[yes_idx]
    if prob is None:
        return None

    label = (
        str(market.get("groupItemTitle") or "").strip()
        or str(market.get("question") or "").strip()
    )
    outcome_name = outcomes[yes_idx] if yes_idx < len(outcomes) else "Yes"
    delta_24h = _to_float(market.get("oneDayPriceChange"))
    vol_24h = _to_float(market.get("volume24hr"), 0.0) or 0.0

    return {
        "question": str(market.get("question") or label).strip(),
        "label": label,
        "outcome": outcome_name,
        "prob": round(prob, 4),
        "delta_24h": round(delta_24h, 4) if delta_24h is not None else None,
        "volume_24h": round(vol_24h, 2),
    }


def _normalize_event(ev: dict, category: str) -> dict[str, Any] | None:
    if ev.get("closed") is True or ev.get("active") is False:
        return None
    title = str(ev.get("title") or "").strip()
    slug = str(ev.get("slug") or "").strip()
    if not title or not slug:
        return None

    vol_24h = _to_float(ev.get("volume24hr"), 0.0) or 0.0
    liquidity = _to_float(ev.get("liquidity"), 0.0) or 0.0
    if vol_24h <= 0 and liquidity <= 0:
        return None

    raw_markets = ev.get("markets") or []
    parsed_markets: list[dict[str, Any]] = []
    for m in raw_markets:
        if not isinstance(m, dict):
            continue
        item = _extract_market_summary(m)
        if item is not None:
            parsed_markets.append(item)
    if not parsed_markets:
        return None

    parsed_markets.sort(key=lambda x: (x["prob"], x["volume_24h"]), reverse=True)
    lead = parsed_markets[0]
    top_outcomes = parsed_markets[:2] if len(parsed_markets) > 1 else parsed_markets[:1]

    return {
        "id": str(ev.get("id") or slug),
        "slug": slug,
        "title": title,
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "url": f"https://polymarket.com/event/{slug}",
        "volume_24h": round(vol_24h, 2),
        "liquidity": round(liquidity, 2),
        "end_date": str(ev.get("endDate") or "")[:10],
        "lead_prob": lead["prob"],
        "lead_label": lead["label"] if len(parsed_markets) > 1 else lead["outcome"],
        "delta_24h": lead["delta_24h"],
        "outcomes": top_outcomes,
    }


def load_cached_polymarket(root: Path) -> dict[str, Any]:
    path = cache_path(root)
    if not path.is_file():
        return {
            "available": False,
            "fetched_at": "",
            "categories": {k: CATEGORY_LABELS[k] for k, _, _ in CATEGORY_SPECS},
            "items": [],
            "errors": [],
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "available": False,
        "fetched_at": "",
        "categories": {k: CATEGORY_LABELS[k] for k, _, _ in CATEGORY_SPECS},
        "items": [],
        "errors": [],
    }


def needs_polymarket_refresh(
    root: Path,
    cooldown_sec: int = POLYMARKET_COOLDOWN_SECONDS,
    now: datetime | None = None,
) -> bool:
    cached = load_cached_polymarket(root)
    if not cached.get("available") or not cached.get("items"):
        return True
    dt = _parse_iso_dt(cached.get("fetched_at"))
    if dt is None:
        return True
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return (now_utc - dt).total_seconds() >= cooldown_sec


def refresh_polymarket(
    root: Path,
    force: bool = False,
    cooldown_sec: int = POLYMARKET_COOLDOWN_SECONDS,
    per_category: int = 4,
) -> dict[str, Any]:
    """拉取宏观利率、关税中国、美股指数三大类活跃 Polymarket 盘口并缓存到本地。"""
    with _POLY_LOCK:
        if not force and not needs_polymarket_refresh(root, cooldown_sec=cooldown_sec):
            return load_cached_polymarket(root)

        queries: list[tuple[str, str]] = []
        for cat, _, terms in CATEGORY_SPECS:
            for term in terms:
                queries.append((cat, term))

        seen: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            futs = {pool.submit(_fetch_search, term): (cat, term) for cat, term in queries}
            for fut in as_completed(futs):
                default_cat, term = futs[fut]
                try:
                    evs, err = fut.result()
                    if err:
                        errors.append(f"{term}: {err}")
                        continue
                    for ev in evs:
                        if not isinstance(ev, dict):
                            continue
                        cat = _classify_event(str(ev.get("title") or ""), default_cat)
                        if not cat:
                            continue
                        norm = _normalize_event(ev, cat)
                        if not norm:
                            continue
                        prev = seen.get(norm["id"])
                        if prev is None or norm["volume_24h"] > prev["volume_24h"]:
                            seen[norm["id"]] = norm
                except Exception as exc:
                    errors.append(f"{term}: {exc}")

        if not seen:
            existing = load_cached_polymarket(root)
            if existing.get("items"):
                return {**existing, "errors": errors[:4]}
            return {
                "available": False,
                "fetched_at": "",
                "categories": {k: CATEGORY_LABELS[k] for k, _, _ in CATEGORY_SPECS},
                "items": [],
                "errors": errors[:4],
            }

        selected: list[dict[str, Any]] = []
        for cat, _, _ in CATEGORY_SPECS:
            cat_items = [it for it in seen.values() if it["category"] == cat]
            cat_items.sort(key=lambda x: (x["volume_24h"], x["liquidity"]), reverse=True)
            selected.extend(cat_items[:per_category])

        selected.sort(key=lambda x: x["volume_24h"], reverse=True)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {
            "available": True,
            "fetched_at": fetched_at,
            "categories": {k: CATEGORY_LABELS[k] for k, _, _ in CATEGORY_SPECS},
            "items": selected,
            "errors": errors[:4],
        }
        path = cache_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload


def load_polymarket_as_of(
    root: Path,
    as_of: str,
    allow_network: bool = True,
) -> dict[str, Any]:
    """回溯指定信号日 `as_of` 美股收盘时刻（21:15 UTC）的 Polymarket 宏观/关税/美股盘口赔率。

    严格使用 CLOB `prices-history` 截至 `as_of` 收盘前的历史点位，绝不读取已结算盘口的终局 0/1 价格。
    """
    try:
        day_obj = date.fromisoformat(str(as_of)[:10])
    except ValueError:
        return {"available": False, "as_of": as_of, "mode": "invalid_date", "note": "日期格式无效", "items": []}

    day_iso = day_obj.isoformat()
    if day_iso < MIN_POLYMARKET_DATE:
        return {
            "available": False,
            "as_of": day_iso,
            "mode": "pre_polymarket",
            "note": f"该信号日（{day_iso}）早于 Polymarket 宏观与利率 CLOB 盘口覆盖期（2023+）。",
            "items": [],
        }

    today_ny = datetime.now(NEW_YORK).date().isoformat()
    if day_iso >= today_ny:
        live = refresh_polymarket(root, force=False) if allow_network and needs_polymarket_refresh(root) else load_cached_polymarket(root)
        return {
            "available": bool(live.get("available") and live.get("items")),
            "as_of": day_iso,
            "mode": "live",
            "fetched_at": live.get("fetched_at") or "",
            "note": "",
            "items": (live.get("items") or [])[:6],
        }

    dpath = day_cache_path(root, day_iso)
    if dpath.is_file():
        try:
            cached_doc = json.loads(dpath.read_text(encoding="utf-8"))
            if isinstance(cached_doc, dict):
                return cached_doc
        except (OSError, json.JSONDecodeError):
            pass

    if not allow_network:
        return {
            "available": False,
            "as_of": day_iso,
            "mode": "offline",
            "note": "离线模式未缓存该日 Polymarket 历史赔率。",
            "items": [],
        }

    with _DAY_LOCK:
        if dpath.is_file():
            try:
                return json.loads(dpath.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        cutoff_dt = datetime(day_obj.year, day_obj.month, day_obj.day, 21, 15, 0, tzinfo=timezone.utc)
        cutoff_ts = int(cutoff_dt.timestamp())

        candidates: dict[str, tuple[str, dict]] = {}
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=len(HISTORICAL_TAG_SPECS)) as pool:
            futs = {
                pool.submit(_fetch_events_on_date, tag, day_iso): (tag, default_cat)
                for tag, default_cat in HISTORICAL_TAG_SPECS
            }
            for fut in as_completed(futs):
                tag, default_cat = futs[fut]
                try:
                    evs, err = fut.result()
                    if err:
                        errors.append(f"{tag}: {err}")
                        continue
                    for ev in evs:
                        if not isinstance(ev, dict):
                            continue
                        cat = _classify_event(str(ev.get("title") or ""), default_cat)
                        if not cat:
                            continue
                        ev_id = str(ev.get("id") or ev.get("slug") or "")
                        if not ev_id:
                            continue
                        vol = _to_float(ev.get("volume"), 0.0) or 0.0
                        prev = candidates.get(ev_id)
                        if prev is None or vol > (_to_float(prev[1].get("volume"), 0.0) or 0.0):
                            candidates[ev_id] = (cat, ev)
                except Exception as exc:
                    errors.append(f"{tag}: {exc}")

        picked_events: list[tuple[str, dict]] = []
        for cat, _, _ in CATEGORY_SPECS:
            cat_evs = [(c, ev) for c, ev in candidates.values() if c == cat]
            cat_evs.sort(key=lambda x: _to_float(x[1].get("volume"), 0.0) or 0.0, reverse=True)
            picked_events.extend(cat_evs[:2])

        # 为每个候选事件提取前 3 个高成交子盘口并通过 CLOB 回溯当日赔率
        token_jobs: list[tuple[str, str, dict, dict, str, str]] = []
        for cat, ev in picked_events[:6]:
            ev_id = str(ev.get("id") or ev.get("slug"))
            raw_mkts = [m for m in (ev.get("markets") or []) if isinstance(m, dict)]
            raw_mkts.sort(key=lambda m: _to_float(m.get("volume"), 0.0) or 0.0, reverse=True)
            for m in raw_mkts[:3]:
                outcomes = [str(o) for o in _parse_json_list(m.get("outcomes"))]
                clob_ids = [str(t) for t in _parse_json_list(m.get("clobTokenIds"))]
                if not clob_ids:
                    continue
                yes_idx = 0
                for idx, name in enumerate(outcomes):
                    if name.strip().lower() == "yes" and idx < len(clob_ids):
                        yes_idx = idx
                        break
                tok = clob_ids[yes_idx] if yes_idx < len(clob_ids) else clob_ids[0]
                out_name = outcomes[yes_idx] if yes_idx < len(outcomes) else "Yes"
                label = (
                    str(m.get("groupItemTitle") or "").strip()
                    or (out_name if len(raw_mkts) == 1 else str(m.get("question") or "").strip())
                )
                token_jobs.append((ev_id, cat, ev, m, tok, label))

        resolved_by_event: dict[str, list[dict[str, Any]]] = {}
        if token_jobs:
            with ThreadPoolExecutor(max_workers=min(8, len(token_jobs))) as pool:
                fut_map = {
                    pool.submit(_fetch_clob_price_as_of, tok, cutoff_ts): (ev_id, cat, ev, m, label)
                    for ev_id, cat, ev, m, tok, label in token_jobs
                }
                for fut in as_completed(fut_map):
                    ev_id, cat, ev, m, label = fut_map[fut]
                    prob, d24 = fut.result()
                    if prob is None:
                        continue
                    resolved_by_event.setdefault(ev_id, []).append(
                        {
                            "cat": cat,
                            "ev": ev,
                            "label": label,
                            "question": str(m.get("question") or ev.get("title") or "").strip(),
                            "prob": prob,
                            "delta_24h": d24,
                            "mkt_vol": _to_float(m.get("volume"), 0.0) or 0.0,
                        }
                    )

        items: list[dict[str, Any]] = []
        for cat, ev in picked_events[:6]:
            ev_id = str(ev.get("id") or ev.get("slug"))
            hits = resolved_by_event.get(ev_id) or []
            if not hits:
                continue
            hits.sort(key=lambda x: (x["prob"], x["mkt_vol"]), reverse=True)
            lead = hits[0]
            slug = str(ev.get("slug") or "")
            total_vol = _to_float(ev.get("volume"), 0.0) or 0.0
            items.append(
                {
                    "id": ev_id,
                    "slug": slug,
                    "title": str(ev.get("title") or "").strip(),
                    "category": cat,
                    "category_label": CATEGORY_LABELS.get(cat, cat),
                    "url": f"https://polymarket.com/event/{slug}" if slug else "",
                    "volume_total": round(total_vol, 2),
                    "end_date": str(ev.get("endDate") or "")[:10],
                    "lead_prob": lead["prob"],
                    "lead_label": lead["label"],
                    "delta_24h": lead["delta_24h"],
                }
            )

        items.sort(key=lambda x: x.get("volume_total") or 0.0, reverse=True)
        doc = {
            "available": bool(items),
            "as_of": day_iso,
            "mode": "historical_clob",
            "cutoff_utc": f"{day_iso}T21:15:00Z",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "" if items else f"未检索到 {day_iso} 当日有 CLOB 历史成交记录的宏观/关税/指数盘口。",
            "items": items,
            "errors": errors[:3],
        }
        # 仅在网络正常（无全量网络失败）时写入按日缓存，避免断网时把空结果焊死
        if items or not errors:
            dpath.parent.mkdir(parents=True, exist_ok=True)
            dpath.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc
