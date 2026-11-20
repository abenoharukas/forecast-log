"""看板数据读写层：全部基于 `forecast.db` 的 4 张核心表（`bars`、`signals`、`news`、`agent_eval`）。

- Cross 信号页（`/ndx`、`/qqq`、`/a-shares`）：只把纯信号列存进 `signals` 表，查询时通过 `LEFT JOIN bars`
  关联客观 OHLCV，并在内存中毫秒级聚合出命中率汇总、对比表、分位数分箱及静态筛选项配置。
- VXN 波动率页（`/vxn`）：不落任何 `timeline_*` 表，直接基于 `bars` 向量化现算并按最新 K 线指纹缓存。
- Agent 评测页（`/agent`）：直接读写扁平表 `agent_eval`（一行对应一道题×一个预测模型），实时计算 Brier/LogLoss/ECE。
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from forecast_log.db import SIGNAL_COLUMNS, _extract_agent_rows_from_doc, connect, db_path

PRICE_FLOAT_COLUMNS = ("open", "high", "low", "close")

SESSION_COLUMNS = (
    *PRICE_FLOAT_COLUMNS,
    "volume",
    *SIGNAL_COLUMNS,
    "is_closed",
)

DEFAULT_INSTRUMENTS = {
    "ndx": "^NDX",
    "qqq": "QQQ",
    "ssec": "000001.SS",
}

CROSS_SCALAR_ROOTS = (
    "summary",
    "summary_price_only",
    "context",
    "comparison",
    "bands",
    "reference",
    "signals_seen",
    "news",
    "latest",
)

HOLDOUT_DEFAULT = "2015-01-01"

_TIMELINE_MEM_CACHE: dict[Path, tuple[tuple, dict]] = {}
_NEWS_PROVENANCE_CACHE: dict[Path, tuple[int, dict]] = {}


# ------------------------------------------------------------------ D1 同步辅助（保留供 publish_d1.py 调用）


def classify(value: Any) -> tuple[str, str | None]:
    if value is None:
        return "null", None
    if isinstance(value, bool):
        return "bool", "1" if value else "0"
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        return "num", repr(value)
    return "str", str(value)


def walk(prefix: str, value: Any, out: list[tuple[str, str, str | None]], sep: str = ".") -> None:
    if isinstance(value, dict):
        if not value:
            out.append((prefix, "obj", None))
        for key, inner in value.items():
            walk(f"{prefix}{sep}{key}" if prefix else str(key), inner, out, sep=sep)
    elif isinstance(value, list):
        if not value:
            out.append((prefix, "list", None))
        for index, inner in enumerate(value):
            walk(f"{prefix}{sep}{index}", inner, out, sep=sep)
    else:
        kind, text = classify(value)
        out.append((prefix, kind, text))


# ------------------------------------------------------------------ Cross 静态定义与实时聚合


def _question_meta(market: str, instrument: str) -> dict[str, Any]:
    if market == "ssec":
        from forecast_log.china import HORIZON, TARGET_NAME

        return {
            "name": TARGET_NAME,
            "question": {
                "horizon": HORIZON,
                "text": (
                    f"Same cross-asset risk appetite long → is the {TARGET_NAME} close {HORIZON} A-share sessions "
                    "later above the basis close?"
                ),
                "factors": [
                    "The signal is untouched: four US cross-asset factors, 90th percentile of expanding history",
                    "Add: within 3 days before CPI, or 3-7 days before NFP, with the score at or above the median",
                    "Remove: no point on the FOMC decision day, and no point on days with a strong bearish surprise",
                ],
                "rule": (
                    "Nothing in the signal changes. A US signal dated D is known at 04:00 Beijing the next morning, "
                    "so the first A-share session after D is the one that settles it. That session's close is the "
                    f"basis of the call; the call is a hit if the close {HORIZON} A-share sessions later is above it. "
                    "The open of that session is recorded as `entry` — the price you could actually pay — but it "
                    "does not decide the call."
                ),
            },
        }

    from forecast_log.cross import HORIZON, QUANTILE

    return {
        "name": None,
        "question": {
            "horizon": HORIZON,
            "text": (
                f"Cross-asset risk appetite long → is the {instrument} close {HORIZON} sessions later "
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
    }


def _contexts_for_market(
    root: Path,
    market: str,
    instrument: str,
    dates: list[str],
    signal_dates: list[str] | None = None,
) -> list[dict[str, Any]]:
    from forecast_log.cross import CONTEXTS, _news_providers

    first_day = dates[0] if dates else None
    ref_dates = signal_dates if signal_dates is not None else dates
    ref_first_day = ref_dates[0] if ref_dates else first_day
    base = [
        {**item, "available_from": item.get("available_from") or ref_first_day}
        if item["key"] == "composite"
        else dict(item)
        for item in CONTEXTS
    ]
    _, news_contexts = _news_providers(root, ref_dates)
    extra: list[dict[str, Any]] = []
    if market == "qqq":
        extra.extend(
            [
                {
                    "key": "qqq_below_ma20",
                    "label": f"{instrument} trend filter — Below 20-day MA",
                    "kind": "veto",
                    "note": f"veto signal when {instrument} closes below its 20-session moving average",
                    "available_from": first_day,
                },
                {
                    "key": "qqq_gap_veto",
                    "label": f"{instrument} overnight gap > +1.5%",
                    "kind": "veto",
                    "note": f"veto entry when {instrument} next-session open gaps up more than 1.5% above signal close",
                    "available_from": first_day,
                },
            ]
        )
    elif market == "ssec":
        extra.extend(
            [
                {
                    "key": "ssec_below_ma20",
                    "label": "A-share trend filter — Below 20-day MA",
                    "kind": "veto",
                    "note": "上证综指收盘低于 20 日均线时否决买入（本土趋势过滤）",
                    "available_from": first_day,
                },
                {
                    "key": "ssec_gap_veto",
                    "label": "A-share open gap > +1.0%",
                    "kind": "veto",
                    "note": "A 股次日高开超过 +1.0% 时否决追高",
                    "available_from": first_day,
                },
                {
                    "key": "ssec_oversold_boost",
                    "label": "A-share 5-day pullback < -3%",
                    "kind": "boost",
                    "note": "上证综指 5 日跌幅超 3% 时补一个超跌反弹信号",
                    "available_from": first_day,
                },
            ]
        )

    cleaned: list[dict[str, Any]] = []
    for item in [*base, *news_contexts, *extra]:
        out: dict[str, Any] = {"key": item["key"]}
        for field in ("label", "kind", "group", "group_label", "note", "available_from"):
            if item.get(field):
                out[field] = item[field]
        if item.get("required"):
            out["required"] = True
        if item.get("threshold") is not None:
            out["threshold"] = item["threshold"]
        cleaned.append(out)
    return cleaned


def _cached_news_provenance(root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    surprise_file = resolved_root / "artifacts" / "surprise.jsonl"
    mtime = surprise_file.stat().st_mtime_ns if surprise_file.is_file() else -1
    cached = _NEWS_PROVENANCE_CACHE.get(resolved_root)
    if cached is not None and cached[0] == mtime:
        return dict(cached[1])
    from forecast_log.cross import _news_labels

    _, prov = _news_labels(root)
    _NEWS_PROVENANCE_CACHE[resolved_root] = (mtime, dict(prov))
    return dict(prov)


def _compute_us_tally(rows: list[Any], flag_col: str, holdout: str = HOLDOUT_DEFAULT) -> dict[str, Any]:
    resolved = [r for r in rows if r["outcome"] is not None]

    def _split(sample: list[Any]) -> dict[str, float | None]:
        if not sample:
            return {"hit3": None, "hit45": None, "miss": None}
        three = sum(1 for r in sample if r["hit3"] == 1) / len(sample)
        later = sum(1 for r in sample if r["outcome"] == 1 and r["hit3"] == 0) / len(sample)
        return {
            "hit3": round(three, 4),
            "hit45": round(later, 4),
            "miss": round(1.0 - three - later, 4),
        }

    flagged = [r for r in resolved if r[flag_col]]
    held = [r for r in resolved if r["date"] >= holdout]
    held_flag = [r for r in held if r[flag_col]]
    f_split = _split(flagged)
    b_split = _split(resolved)
    return {
        "n": len(resolved),
        "base_rate": round(sum(int(r["outcome"]) for r in resolved) / len(resolved), 4) if resolved else None,
        "flagged": len(flagged),
        "hit": round(sum(int(r["outcome"]) for r in flagged) / len(flagged), 4) if flagged else None,
        "holdout_n": len(held),
        "holdout_base_rate": round(sum(int(r["outcome"]) for r in held) / len(held), 4) if held else None,
        "holdout_flagged": len(held_flag),
        "holdout_hit": round(sum(int(r["outcome"]) for r in held_flag) / len(held_flag), 4) if held_flag else None,
        "hit3": f_split["hit3"],
        "hit45": f_split["hit45"],
        "miss": f_split["miss"],
        "base3": b_split["hit3"],
        "base45": b_split["hit45"],
        "holdout_from": holdout,
    }


def _compute_us_sell_tally(rows: list[Any], holdout: str = HOLDOUT_DEFAULT) -> dict[str, Any]:
    usable = [r for r in rows if r["base_close"] is not None and r["close5"] is not None]

    def _down5(r: Any) -> int:
        return 1 if float(r["close5"]) < float(r["base_close"]) else 0

    def _down3(r: Any) -> int:
        return 1 if (r["close3"] is not None and float(r["close3"]) < float(r["base_close"])) else 0

    def _split(sample: list[Any]) -> dict[str, float | None]:
        if not sample:
            return {"hit3": None, "hit45": None, "miss": None}
        three = sum(_down3(r) for r in sample) / len(sample)
        later = sum(1 for r in sample if _down5(r) == 1 and _down3(r) == 0) / len(sample)
        return {"hit3": round(three, 4), "hit45": round(later, 4), "miss": round(1.0 - three - later, 4)}

    flagged = [r for r in usable if r["sell_flag"]]
    held = [r for r in usable if r["date"] >= holdout]
    held_flag = [r for r in held if r["sell_flag"]]
    f_split = _split(flagged)
    b_split = _split(usable)
    return {
        "n": len(usable),
        "base_rate": round(sum(_down5(r) for r in usable) / len(usable), 4) if usable else None,
        "flagged": len(flagged),
        "hit": round(sum(_down5(r) for r in flagged) / len(flagged), 4) if flagged else None,
        "holdout_n": len(held),
        "holdout_base_rate": round(sum(_down5(r) for r in held) / len(held), 4) if held else None,
        "holdout_flagged": len(held_flag),
        "holdout_hit": round(sum(_down5(r) for r in held_flag) / len(held_flag), 4) if held_flag else None,
        "hit3": f_split["hit3"],
        "hit45": f_split["hit45"],
        "miss": f_split["miss"],
        "base3": b_split["hit3"],
        "base45": b_split["hit45"],
        "holdout_from": holdout,
    }


def _compute_ssec_sell_summary(rows: list[Any], holdout: str = HOLDOUT_DEFAULT) -> dict[str, Any]:
    n = len(rows)

    def _pair(idx: int) -> tuple[float, float, float] | None:
        r = rows[idx]
        base = r["base_close"] if r["base_close"] is not None else r["close"]
        c3 = r["close3"] if r["close3"] is not None else (rows[idx + 3]["close"] if idx + 3 < n else None)
        c5 = r["close5"] if r["close5"] is not None else (rows[idx + 5]["close"] if idx + 5 < n else None)
        if base is None or c5 is None:
            return None
        return float(base), (float(c3) if c3 is not None else float(base)), float(c5)

    def _split(indices: list[int]) -> dict[str, float | None]:
        if not indices:
            return {"hit3": None, "hit45": None, "miss": None}
        three = 0
        later = 0
        for idx in indices:
            p = _pair(idx)
            if p is None:
                continue
            base, c3, c5 = p
            if c3 < base:
                three += 1
            elif c5 < base:
                later += 1
        t_rate = three / len(indices)
        l_rate = later / len(indices)
        return {"hit3": round(t_rate, 4), "hit45": round(l_rate, 4), "miss": round(1.0 - t_rate - l_rate, 4)}

    def _window(low: str) -> dict[str, Any]:
        usable = [i for i, r in enumerate(rows) if r["date"] >= low and r["outcome"] is not None and _pair(i) is not None]
        chosen = [i for i in usable if rows[i]["sell_flag"]]
        base = round(sum(1 for i in usable if _pair(i)[2] < _pair(i)[0]) / len(usable), 4) if usable else None
        hit = round(sum(1 for i in chosen if _pair(i)[2] < _pair(i)[0]) / len(chosen), 4) if chosen else None
        cs = _split(chosen)
        us = _split(usable)
        return {
            "sessions": len(usable),
            "signals": len(chosen),
            "hit": hit,
            "base_rate": base,
            "lift": None if hit is None or base is None else round(hit - base, 4),
            "hit3": cs["hit3"],
            "hit45": cs["hit45"],
            "miss": cs["miss"],
            "base3": us["hit3"],
            "base45": us["hit45"],
        }

    return {"all": _window("1900"), "holdout": _window(holdout), "holdout_from": holdout}


def _compute_ssec_summary_and_bands(rows: list[Any], ndx_doc: dict | None, holdout: str = HOLDOUT_DEFAULT) -> tuple[dict, list, dict | None, int]:
    n = len(rows)

    def _hit3_at(idx: int) -> bool:
        r = rows[idx]
        if r["hit3"] is not None:
            return int(r["hit3"]) == 1
        if idx + 3 < n and r["close"] is not None and rows[idx + 3]["close"] is not None:
            return float(rows[idx + 3]["close"]) > float(r["close"])
        return False

    def _hit5_at(idx: int) -> bool:
        r = rows[idx]
        if r["outcome"] is not None:
            return int(r["outcome"]) == 1
        if idx + 5 < n and r["close"] is not None and rows[idx + 5]["close"] is not None:
            return float(rows[idx + 5]["close"]) > float(r["close"])
        return False

    def _split(indices: list[int]) -> dict[str, float | None]:
        if not indices:
            return {"hit3": None, "hit45": None, "miss": None}
        three = sum(1 for idx in indices if _hit3_at(idx)) / len(indices)
        later = sum(1 for idx in indices if (not _hit3_at(idx)) and _hit5_at(idx)) / len(indices)
        return {
            "hit3": round(three, 4),
            "hit45": round(later, 4),
            "miss": round(1.0 - three - later, 4),
        }

    def _window_summary(low: str) -> dict[str, Any]:
        usable = [i for i, r in enumerate(rows) if r["date"] >= low and r["outcome"] is not None]
        chosen = [i for i in usable if rows[i]["flag"]]
        base = round(sum(int(rows[i]["outcome"]) for i in usable) / len(usable), 4) if usable else None
        hit = round(sum(int(rows[i]["outcome"]) for i in chosen) / len(chosen), 4) if chosen else None
        c_split = _split(chosen)
        u_split = _split(usable)
        return {
            "sessions": len(usable),
            "signals": len(chosen),
            "hit": hit,
            "base_rate": base,
            "lift": None if hit is None or base is None else round(hit - base, 4),
            "hit3": c_split["hit3"],
            "hit45": c_split["hit45"],
            "miss": c_split["miss"],
            "base3": u_split["hit3"],
            "base45": u_split["hit45"],
        }

    bands: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    if ndx_doc is not None and ndx_doc.get("time") and ndx_doc.get("score"):
        us_days = ndx_doc["time"]
        us_scores = ndx_doc["score"]
        scored = []
        pointer = 0
        for r in rows:
            if not r["flag"] or r["outcome"] is None:
                continue
            day = r["date"]
            while pointer + 1 < len(us_days) and us_days[pointer + 1] < day:
                pointer += 1
            if us_scores[pointer] is not None:
                scored.append((float(us_scores[pointer]), int(r["outcome"])))
        if scored:
            ordered = sorted(scored)
            size = len(ordered) // 10
            if size > 0:
                for band in range(10):
                    chunk = ordered[band * size : (band + 1) * size] if band < 9 else ordered[9 * size :]
                    bands.append(
                        {"band": band + 1, "hit": round(sum(h for _, h in chunk) / len(chunk), 4)}
                    )
        ns = ndx_doc.get("summary") or {}
        if ns.get("hit") is not None and ns.get("base_rate") is not None:
            reference = {
                "instrument": "^NDX",
                "signals": ns["flagged"],
                "hit": ns["hit"],
                "base_rate": ns["base_rate"],
                "lift": round(ns["hit"] - ns["base_rate"], 4),
                "holdout_signals": ns["holdout_flagged"],
                "holdout_hit": ns["holdout_hit"],
                "holdout_base_rate": ns["holdout_base_rate"],
                "holdout_lift": round(ns["holdout_hit"] - ns["holdout_base_rate"], 4)
                if ns.get("holdout_hit") is not None and ns.get("holdout_base_rate") is not None
                else None,
            }

    signals_seen = sum(1 for r in rows if r["flag"])
    summary = {
        "all": _window_summary("1900"),
        "holdout": _window_summary(holdout),
        "holdout_from": holdout,
    }
    return summary, bands, reference, signals_seen


def build_cross_bundle(doc: dict, market: str) -> dict:
    from forecast_log.bars import is_session_closed

    nodes: list[tuple[str, str, str | None]] = []
    for key in CROSS_SCALAR_ROOTS:
        if key in doc and doc[key] is not None:
            walk(key, doc[key], nodes)
    if doc.get("calendar") is not None:
        walk("calendar", doc["calendar"], nodes)
    if isinstance(doc.get("question"), dict) and "factors" in doc["question"]:
        walk("question.factors", doc["question"]["factors"], nodes)

    components = doc.get("components") or {}
    contexts_in = doc.get("contexts") or []
    ordered_keys = [c["key"] for c in contexts_in] if contexts_in else list(components.keys())
    for k in components:
        if k not in ordered_keys:
            ordered_keys.append(k)
    positions = {key: i for i, key in enumerate(ordered_keys)}

    instrument = str(doc.get("instrument") or DEFAULT_INSTRUMENTS.get(market, market))
    tz_name = "Asia/Shanghai" if (market == "ssec" or instrument.endswith((".SS", ".SZ"))) else "America/New_York"
    fetched_at = doc.get("fetched_at") or ""
    times = doc.get("time") or []
    latest_block = doc.get("latest") if isinstance(doc.get("latest"), dict) else {}

    sessions = []
    for index, day in enumerate(times):
        mask = 0
        for key, slot in positions.items():
            series = components.get(key) or []
            if index < len(series) and series[index]:
                mask |= int(series[index]) << (4 * slot)
        row: dict[str, Any] = {"date": day, "factors_packed": mask}
        is_latest_day = day == latest_block.get("date")
        for column in SESSION_COLUMNS:
            series = doc.get(column)
            val = series[index] if isinstance(series, list) and index < len(series) else None
            if val is None and is_latest_day and column in latest_block:
                val = latest_block[column]
                if isinstance(val, bool) and column in ("flag", "flag_price_only", "news_veto"):
                    val = 1 if val else 0
            row[column] = val
        if row.get("is_closed") is None:
            from datetime import date as _date

            row["is_closed"] = 1 if is_session_closed(_date.fromisoformat(day), tz_name, fetched_at or None) else 0
        else:
            row["is_closed"] = 1 if row["is_closed"] else 0
        sessions.append(row)

    contexts = [
        {
            "position": position,
            "key": ctx["key"],
            "label": ctx.get("label"),
            "kind": ctx.get("kind"),
            "required": 1 if ctx.get("required") else 0,
            "threshold": ctx.get("threshold"),
            "group_name": ctx.get("group"),
            "group_label": ctx.get("group_label"),
            "note": ctx.get("note"),
            "available_from": ctx.get("available_from"),
        }
        for position, ctx in enumerate(contexts_in)
    ]

    question = doc.get("question") or {}
    first_session = doc.get("first_session") or (times[0] if times else None)
    last_session = doc.get("last_session") or (times[-1] if times else None)
    return {
        "market": market,
        "row": {
            "market": market,
            "instrument": instrument,
            "name": doc.get("name"),
            "horizon": question.get("horizon"),
            "question": question.get("text"),
            "rule": question.get("rule"),
            "first_session": first_session,
            "last_session": last_session,
            "fetched_at": fetched_at,
        },
        "nodes": nodes,
        "contexts": contexts,
        "factor_keys": [[i, key] for i, key in enumerate(ordered_keys)],
        "sessions": sessions,
    }


def save_cross_market(root: Path, market: str, doc: dict) -> int:
    """仅将纯信号列增量写入独立的 `signals` 表，不重复存 OHLCV，也不生成任何配置碎表。"""
    bundle = build_cross_bundle(doc, market)
    fetched_at = str(bundle["row"]["fetched_at"] or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    conn = connect(db_path(root))
    try:
        existing = {
            row["date"]: row
            for row in conn.execute(
                """
                SELECT s.date, s.flag, s.sell_flag, s.sell_threshold, s.score, s.outcome, s.factors_packed, s.fetched_at,
                       COALESCE(b.is_closed, 1) AS is_closed
                FROM signals s
                LEFT JOIN bars b ON b.symbol = ? AND b.date = s.date
                WHERE s.market = ?
                """,
                (bundle["row"]["instrument"], market),
            ).fetchall()
        }
        all_sessions = bundle["sessions"]
        dirty_indices: set[int] = set()
        for idx, s in enumerate(all_sessions):
            prev = existing.get(s["date"])
            if (
                prev is None
                or prev["is_closed"] == 0
                or not s.get("is_closed", 1)
                or (prev["outcome"] is None and s.get("outcome") is not None)
                or (prev["sell_flag"] is None and s.get("sell_flag") is not None)
                or (prev["sell_threshold"] is None and s.get("sell_threshold") is not None)
            ):
                for back in range(max(0, idx - 7), idx + 1):
                    dirty_indices.add(back)

        for idx in range(max(0, len(all_sessions) - 7), len(all_sessions)):
            s = all_sessions[idx]
            prev = existing.get(s["date"])
            if (
                prev is None
                or prev["flag"] != s.get("flag")
                or prev["sell_flag"] != s.get("sell_flag")
                or prev["sell_threshold"] != s.get("sell_threshold")
                or prev["score"] != s.get("score")
                or prev["outcome"] != s.get("outcome")
                or prev["factors_packed"] != s.get("factors_packed")
                or prev["fetched_at"] != fetched_at
            ):
                dirty_indices.add(idx)

        if not dirty_indices and all_sessions:
            # 至少刷新最后一行上的 fetched_at 水位线，确保 _needs_refresh 能读到最新重算时刻
            dirty_indices.add(len(all_sessions) - 1)

        sessions_to_write = [all_sessions[i] for i in sorted(dirty_indices)]
        if sessions_to_write:
            cols = ("market", "date", *SIGNAL_COLUMNS, "factors_packed", "fetched_at")
            placeholders = ",".join("?" for _ in cols)
            updates = ", ".join(f"{c} = excluded.{c}" for c in (*SIGNAL_COLUMNS, "factors_packed", "fetched_at"))
            with conn:
                conn.executemany(
                    f"""
                    INSERT INTO signals ({",".join(cols)})
                    VALUES ({placeholders})
                    ON CONFLICT(market, date) DO UPDATE SET {updates}
                    """,
                    [
                        tuple([market, s["date"], *[s.get(c) for c in SIGNAL_COLUMNS], s["factors_packed"], fetched_at])
                        for s in sessions_to_write
                    ],
                )
        return len(sessions_to_write)
    finally:
        conn.close()


def load_cross_market(root: Path, market: str, _include_cross_ref: bool = True) -> dict | None:
    path = db_path(root)
    if not path.is_file():
        return None
    instrument = DEFAULT_INSTRUMENTS.get(market, market)
    conn = connect(path)
    try:
        rows = conn.execute(
            """
            SELECT
                s.*,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                COALESCE(b.is_closed, 1) AS is_closed,
                COALESCE(NULLIF(s.fetched_at, ''), b.fetched_at, '') AS eff_fetched_at
            FROM signals s
            LEFT JOIN bars b ON b.symbol = ? AND b.date = s.date
            WHERE s.market = ?
            ORDER BY s.date ASC
            """,
            (instrument, market),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    qmeta = _question_meta(market, instrument)
    dates = [str(r["date"]) for r in rows]
    ndx_doc = (
        load_cross_market(root, "ndx", _include_cross_ref=False)
        if (_include_cross_ref and market in ("qqq", "ssec"))
        else None
    )
    contexts = _contexts_for_market(
        root,
        market,
        instrument,
        dates,
        signal_dates=(ndx_doc["time"] if (market == "ssec" and ndx_doc) else None),
    )

    doc: dict[str, Any] = {}
    if qmeta.get("name"):
        doc["name"] = qmeta["name"]
    doc["instrument"] = instrument
    doc["question"] = qmeta["question"]
    doc["fetched_at"] = str(rows[-1]["eff_fetched_at"] or "")
    doc["first_session"] = dates[0]
    doc["last_session"] = dates[-1]
    doc["time"] = dates

    for column in SESSION_COLUMNS:
        if column in PRICE_FLOAT_COLUMNS:
            values = [round(float(r[column]), 2) if r[column] is not None else None for r in rows]
        else:
            values = [r[column] for r in rows]
        if any(v is not None for v in values):
            doc[column] = values

    doc["components"] = {
        ctx["key"]: [(int(r["factors_packed"]) >> (4 * pos)) & 0xF for r in rows]
        for pos, ctx in enumerate(contexts)
    }
    if market == "ssec":
        if ndx_doc and ndx_doc.get("time") and (ndx_doc.get("components") or {}).get("sell_composite"):
            us_days = ndx_doc["time"]
            us_sell_comp = ndx_doc["components"]["sell_composite"]
            ssec_sell_comp: list[int] = []
            for idx, day in enumerate(dates):
                prev_day = dates[idx - 1] if idx > 0 else None
                high = int(np.searchsorted(us_days, day, side="left"))
                low = max(0, high - 1) if prev_day is None else int(np.searchsorted(us_days, prev_day, side="left"))
                hit = 0
                for j in range(high - 1, low - 1, -1):
                    if us_sell_comp[j]:
                        hit = 1
                        break
                ssec_sell_comp.append(hit)
            doc["components"]["sell_composite"] = ssec_sell_comp
        else:
            doc["components"]["sell_composite"] = [int(r["sell_flag"] or 0) for r in rows]
    else:
        doc["components"]["sell_composite"] = [
            1 if (r["score"] is not None and r["sell_threshold"] is not None and float(r["score"]) <= float(r["sell_threshold"])) else 0
            for r in rows
        ]
    doc["contexts"] = contexts

    last = rows[-1]
    if market == "ssec":
        summary, bands, reference, signals_seen = _compute_ssec_summary_and_bands(rows, ndx_doc)
        doc["summary"] = summary
        doc["summary_sell"] = _compute_ssec_sell_summary(rows)
        doc["bands"] = bands
        if reference is not None:
            doc["reference"] = reference
        doc["signals_seen"] = signals_seen
        doc["latest"] = {
            "date": last["date"],
            "open": round(float(last["open"]), 2) if last["open"] is not None else None,
            "high": round(float(last["high"]), 2) if last["high"] is not None else None,
            "low": round(float(last["low"]), 2) if last["low"] is not None else None,
            "close": round(float(last["close"]), 2) if last["close"] is not None else None,
            "volume": last["volume"],
            "us_date": last["us_date"]
            or (
                ndx_doc["time"][int(np.searchsorted(ndx_doc["time"], last["date"], side="left")) - 1]
                if ndx_doc and ndx_doc.get("time") and int(np.searchsorted(ndx_doc["time"], last["date"], side="left")) > 0
                else None
            ),
            "flag": int(last["flag"] or 0),
            "sell_flag": int(last["sell_flag"] or 0),
            "is_closed": bool(last["is_closed"]),
        }
    else:
        from forecast_log.cross import _calendar_provenance, _runtime_context, comparison

        doc["summary"] = _compute_us_tally(rows, "flag")
        doc["summary_price_only"] = _compute_us_tally(rows, "flag_price_only")
        doc["summary_sell"] = _compute_us_sell_tally(rows)
        cal_prov = _calendar_provenance(root)
        if cal_prov is not None:
            doc["calendar"] = cal_prov
        news_prov = _cached_news_provenance(root)
        news_prov["vetoed_signals"] = sum(int(r["news_veto"] or 0) for r in rows if r["outcome"] is not None)
        doc["news"] = news_prov
        rt_ctx = _runtime_context(root)
        if rt_ctx is not None:
            doc["context"] = rt_ctx
        if market == "qqq" and _include_cross_ref:
            ndx_doc = load_cross_market(root, "ndx", _include_cross_ref=False)
            if ndx_doc is not None and ndx_doc.get("time") == doc["time"]:
                doc["comparison"] = comparison(ndx_doc, doc)
        doc["latest"] = {
            "date": last["date"],
            "open": round(float(last["open"]), 2) if last["open"] is not None else None,
            "high": round(float(last["high"]), 2) if last["high"] is not None else None,
            "low": round(float(last["low"]), 2) if last["low"] is not None else None,
            "close": round(float(last["close"]), 2) if last["close"] is not None else None,
            "volume": last["volume"],
            "score": round(float(last["score"]), 4) if last["score"] is not None else None,
            "threshold": round(float(last["threshold"]), 4) if last["threshold"] is not None else None,
            "sell_threshold": round(float(last["sell_threshold"]), 4) if last["sell_threshold"] is not None else None,
            "flag": int(last["flag"] or 0) if last["flag"] is not None else 0,
            "sell_flag": int(last["sell_flag"] or 0) if last["sell_flag"] is not None else 0,
            "flag_price_only": int(last["flag_price_only"] or 0) if last["flag_price_only"] is not None else 0,
            "news_veto": bool(last["news_veto"]),
            "is_closed": bool(last["is_closed"]),
        }

    return doc


def cross_fetched_at(root: Path, market: str) -> str | None:
    path = db_path(root)
    if not path.is_file():
        return None
    instrument = DEFAULT_INSTRUMENTS.get(market, market)
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT fetched_at FROM signals WHERE market = ? ORDER BY date DESC LIMIT 1",
            (market,),
        ).fetchone()
        if row and row["fetched_at"]:
            return str(row["fetched_at"])
        bar_row = conn.execute(
            "SELECT fetched_at FROM bars WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (instrument,),
        ).fetchone()
        return str(bar_row["fetched_at"]) if (bar_row and bar_row["fetched_at"]) else None
    finally:
        conn.close()


def cross_sessions_summary(root: Path) -> dict[str, str | None]:
    path = db_path(root)
    if not path.is_file():
        return {}
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT market, MAX(date) AS last_session FROM signals GROUP BY market"
        ).fetchall()
        return {str(row["market"]): row["last_session"] for row in rows}
    finally:
        conn.close()


def cross_market_status(root: Path) -> dict[str, dict[str, Any]]:
    """基于 `signals` 与 `bars` 返回各市场最新交易日、抓取时间、是否已收盘及未收盘日期。"""
    path = db_path(root)
    if not path.is_file():
        return {}
    conn = connect(path)
    try:
        markets = [
            str(r["market"])
            for r in conn.execute("SELECT DISTINCT market FROM signals ORDER BY market ASC").fetchall()
        ]
        out: dict[str, dict[str, Any]] = {}
        for mkt in markets:
            inst = DEFAULT_INSTRUMENTS.get(mkt, mkt)
            sig_row = conn.execute(
                "SELECT date, fetched_at FROM signals WHERE market = ? ORDER BY date DESC LIMIT 1",
                (mkt,),
            ).fetchone()
            last_bar = conn.execute(
                "SELECT date, close, is_closed, fetched_at FROM bars WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                (inst,),
            ).fetchone()
            unclosed = [
                str(r["date"])
                for r in conn.execute(
                    "SELECT date FROM bars WHERE symbol = ? AND is_closed = 0 ORDER BY date ASC",
                    (inst,),
                ).fetchall()
            ]
            out[mkt] = {
                "instrument": inst,
                "last_session": last_bar["date"] if last_bar else (sig_row["date"] if sig_row else None),
                "last_close": round(float(last_bar["close"]), 2) if (last_bar and last_bar["close"] is not None) else None,
                "fetched_at": (sig_row["fetched_at"] if (sig_row and sig_row["fetched_at"]) else None)
                or (last_bar["fetched_at"] if last_bar else None),
                "is_closed": bool(last_bar["is_closed"]) if last_bar is not None else True,
                "unclosed_dates": unclosed,
            }
        return out
    finally:
        conn.close()


# ------------------------------------------------------------------ VXN timeline（纯读 bars 实时计算 + 内存缓存）


def build_timeline_bundle(doc: dict) -> dict:
    closes = []
    for sym in ("vxn", "ndx"):
        block = doc.get(sym) or {}
        for idx, (d, c) in enumerate(zip(block.get("time") or [], block.get("close") or [])):
            closes.append((sym, idx, d, c))

    rolling = []
    nodes: list[tuple[str, str, str | None]] = []
    for sym in ("vxn", "ndx"):
        r = (doc.get("rolling") or {}).get(sym) or {}
        for extra_k, extra_v in r.items():
            if extra_k not in ("time", "as_of", "anchor", "median", "low", "high", "actual"):
                walk(f"rolling.{sym}.{extra_k}", extra_v, nodes)
        times = r.get("time") or []
        for idx in range(len(times)):
            rolling.append(
                (
                    sym,
                    idx,
                    times[idx],
                    r["as_of"][idx] if idx < len(r.get("as_of") or []) else None,
                    r["anchor"][idx] if idx < len(r.get("anchor") or []) else None,
                    r["median"][idx] if idx < len(r.get("median") or []) else None,
                    r["low"][idx] if idx < len(r.get("low") or []) else None,
                    r["high"][idx] if idx < len(r.get("high") or []) else None,
                    r["actual"][idx] if idx < len(r.get("actual") or []) else None,
                )
            )

    daily = []
    for sym in ("vxn", "ndx"):
        d = (doc.get("daily") or {}).get(sym) or {}
        for extra_k, extra_v in d.items():
            if extra_k not in ("time", "median", "low", "high", "actual", "previous", "risk"):
                walk(f"daily.{sym}.{extra_k}", extra_v, nodes)
        times = d.get("time") or []
        for idx in range(len(times)):
            daily.append(
                (
                    sym,
                    idx,
                    times[idx],
                    d["median"][idx] if idx < len(d.get("median") or []) else None,
                    d["low"][idx] if idx < len(d.get("low") or []) else None,
                    d["high"][idx] if idx < len(d.get("high") or []) else None,
                    d["actual"][idx] if idx < len(d.get("actual") or []) else None,
                    d["previous"][idx] if idx < len(d.get("previous") or []) else None,
                    d["risk"][idx] if idx < len(d.get("risk") or []) else None,
                )
            )

    ladder = []
    lad = (doc.get("ladder") or {}).get("vxn") or {}
    for key in ("horizons", "pending", "summary"):
        if key in lad:
            walk(f"ladder.vxn.{key}", lad[key], nodes)
    for h_str, col in (lad.get("columns") or {}).items():
        h = int(h_str)
        med = col.get("median") or []
        low = col.get("low") or []
        high = col.get("high") or []
        for idx in range(len(med)):
            ladder.append(
                (
                    "vxn",
                    h,
                    idx,
                    med[idx],
                    low[idx] if idx < len(low) else None,
                    high[idx] if idx < len(high) else None,
                )
            )

    if "forward" in doc:
        walk("forward", doc["forward"], nodes)

    return {
        "closes": closes,
        "rolling": rolling,
        "daily": daily,
        "ladder": ladder,
        "nodes": nodes,
    }


def _timeline_fingerprint(vxn, ndx) -> tuple:
    return (
        len(vxn.bars),
        vxn.bars[-1].date.isoformat(),
        vxn.bars[-1].close,
        vxn.bars[-1].is_closed,
        len(ndx.bars),
        ndx.bars[-1].date.isoformat(),
        ndx.bars[-1].close,
        ndx.bars[-1].is_closed,
    )


def save_timeline(root: Path, doc: dict) -> None:
    """时间轴不再向 SQLite 写入任何冗余表，仅更新当前进程内存缓存。"""
    from forecast_log.bars import has_series, load_series

    if has_series(root, "^VXN") and has_series(root, "^NDX"):
        vxn = load_series(root, "^VXN")
        ndx = load_series(root, "^NDX")
        _TIMELINE_MEM_CACHE[root.resolve()] = (_timeline_fingerprint(vxn, ndx), doc)


def load_timeline(root: Path) -> dict | None:
    """直接基于 `bars` 实时向量化计算 VXN 时间轴，并在进程内按最新 K 线指纹缓存。"""
    from forecast_log.backtest import PATH_HORIZON, forecast_rows, rolling_paths, timeline_document
    from forecast_log.bars import has_series, load_series
    from forecast_log.features import load_matrix

    if not (has_series(root, "^VXN") and has_series(root, "^NDX")):
        return None

    resolved_root = root.resolve()
    vxn = load_series(root, "^VXN")
    ndx = load_series(root, "^NDX")
    fp = _timeline_fingerprint(vxn, ndx)

    cached = _TIMELINE_MEM_CACHE.get(resolved_root)
    if cached is not None and cached[0] == fp:
        return cached[1]

    extra = load_matrix(root, [bar.date for bar in vxn.bars])
    rows = forecast_rows(vxn, ndx, extra)
    paths = {
        "vxn": rolling_paths(vxn, PATH_HORIZON, extra),
        "ndx": rolling_paths(ndx, PATH_HORIZON),
    }
    doc = timeline_document(vxn, ndx, rows, extra, paths)
    _TIMELINE_MEM_CACHE[resolved_root] = (fp, doc)
    return doc


# ------------------------------------------------------------------ Agent eval（扁平表 agent_eval）


def build_agent_bundle(doc: dict, dataset: str) -> dict:
    nodes: list[tuple[str, str, str | None]] = []
    for key, val in doc.items():
        if key in ("n", "scored_on_shared", "generated_at"):
            continue
        walk(key, val, nodes, sep="\x1f")
    return {
        "dataset": dataset,
        "row": {
            "dataset": dataset,
            "n": doc.get("n"),
            "scored_on_shared": doc.get("scored_on_shared"),
            "generated_at": doc.get("generated_at"),
        },
        "nodes": nodes,
    }


def save_agent_eval(root: Path, dataset: str, doc: dict) -> None:
    rows = _extract_agent_rows_from_doc(dataset, doc)
    if not rows:
        return
    conn = connect(db_path(root))
    try:
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO agent_eval (
                    dataset, question_id, predictor, symbol, as_of, as_of_close,
                    resolve_session, horizon, family, threshold, question_text, outcome,
                    probability, cost_usd_total, cost_usd_per_forecast, latency_p50_s,
                    latency_p95_s, tool_calls_mean, gate_hits, failures,
                    generated_at, config_model, models_merged
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    finally:
        conn.close()


def load_agent_eval(root: Path, dataset: str) -> dict | None:
    from forecast_log.agent.eval import brier, ece, log_loss, reliability

    path = db_path(root)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT * FROM agent_eval WHERE dataset = ? ORDER BY as_of ASC, horizon ASC, question_id ASC",
            (dataset,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    first = rows[0]
    generated_at = str(first["generated_at"] or "")
    config_model = str(first["config_model"] or "")
    merged_list = [m for m in str(first["models_merged"] or "").split(",") if m]

    question_meta: dict[str, dict[str, Any]] = {}
    pred_answers: dict[str, dict[str, float]] = {}
    pred_telemetry: dict[str, sqlite3.Row] = {}

    for r in rows:
        qid = str(r["question_id"])
        pname = str(r["predictor"])
        if qid not in question_meta:
            question_meta[qid] = {
                "as_of": r["as_of"],
                "as_of_close": r["as_of_close"],
                "family": r["family"],
                "horizon": int(r["horizon"]),
                "outcome": int(r["outcome"]),
                "resolve_session": r["resolve_session"],
                "symbol": r["symbol"],
                "text": r["question_text"],
                "threshold": float(r["threshold"]),
            }
        pred_answers.setdefault(pname, {})
        pred_telemetry.setdefault(pname, r)
        if r["probability"] is not None:
            pred_answers[pname][qid] = float(r["probability"])

    bench = list(question_meta.keys())
    alive_sets = [set(ans.keys()) for ans in pred_answers.values() if ans]
    shared_ids = set.intersection(*alive_sets) if alive_sets else set()
    ordered_shared = sorted(shared_ids, key=lambda k: (question_meta[k]["resolve_session"], k))
    outcomes = {qid: question_meta[qid]["outcome"] for qid in ordered_shared}

    predictors_out: dict[str, dict[str, Any]] = {}
    for pname in sorted(pred_answers.keys()):
        ans = pred_answers[pname]
        tel = pred_telemetry[pname]
        if not ans or not ordered_shared:
            predictors_out[pname] = {
                "brier": None,
                "log_loss": None,
                "ece": None,
                "hit_direction": None,
                "mean_probability": None,
                "observed_rate": round(statistics.fmean(outcomes.values()), 4) if outcomes else None,
                "cost_usd_total": float(tel["cost_usd_total"] or 0.0),
                "cost_usd_per_forecast": tel["cost_usd_per_forecast"],
                "latency_p50_s": tel["latency_p50_s"],
                "latency_p95_s": tel["latency_p95_s"],
                "tool_calls_mean": float(tel["tool_calls_mean"] or 0.0),
                "gate_hits": int(tel["gate_hits"] or 0),
                "failures": int(tel["failures"] or 0),
                "reliability": [],
                "answers": {},
                "pairs": {},
                "by_family": {},
                "n": 0,
            }
            continue

        pairs_list = [(ans[qid], question_meta[qid]["outcome"]) for qid in ordered_shared]
        families = sorted({question_meta[qid]["family"] for qid in ordered_shared})
        by_family = {}
        for fam in families:
            fpairs = [(ans[qid], question_meta[qid]["outcome"]) for qid in ordered_shared if question_meta[qid]["family"] == fam]
            if fpairs:
                by_family[fam] = {
                    "n": len(fpairs),
                    "brier": round(brier(fpairs), 4),
                    "observed": round(statistics.fmean(o for _, o in fpairs), 3),
                }

        predictors_out[pname] = {
            "brier": round(brier(pairs_list), 4),
            "log_loss": round(log_loss(pairs_list), 4),
            "ece": round(ece(pairs_list), 4),
            "hit_direction": round(
                statistics.fmean(1.0 if (p >= 0.5) == (o == 1) else 0.0 for p, o in pairs_list),
                4,
            ),
            "mean_probability": round(statistics.fmean(p for p, _ in pairs_list), 4),
            "observed_rate": round(statistics.fmean(o for _, o in pairs_list), 4),
            "cost_usd_total": float(tel["cost_usd_total"] or 0.0),
            "cost_usd_per_forecast": tel["cost_usd_per_forecast"],
            "latency_p50_s": tel["latency_p50_s"],
            "latency_p95_s": tel["latency_p95_s"],
            "tool_calls_mean": float(tel["tool_calls_mean"] or 0.0),
            "gate_hits": int(tel["gate_hits"] or 0),
            "failures": int(tel["failures"] or 0),
            "reliability": reliability(pairs_list),
            "answers": {qid: round(p, 6) for qid, p in ans.items()},
            "pairs": {qid: round(ans[qid], 6) for qid in ordered_shared},
            "by_family": by_family,
            "n": len(ordered_shared),
        }

    references: dict[str, Any] = {}
    if ordered_shared:
        half_pairs = [(0.5, question_meta[qid]["outcome"]) for qid in ordered_shared]
        references = {
            "constant_0.5": {"brier": round(brier(half_pairs), 4), "log_loss": round(log_loss(half_pairs), 4)},
            "observed_rate": round(statistics.fmean(question_meta[qid]["outcome"] for qid in ordered_shared), 4),
        }

    modes = sorted({m["family"] for m in question_meta.values()})
    symbol = str(first["symbol"] or "^VXN")
    return {
        "n": len(ordered_shared),
        "scored_on_shared": len(ordered_shared),
        "generated_at": generated_at,
        "bench": bench,
        "questions": ordered_shared,
        "outcomes": outcomes,
        "question_meta": question_meta,
        "predictors": predictors_out,
        "references": references,
        "config": {
            "symbol": symbol,
            "limit": len(bench),
            "modes": modes,
            "model": config_model,
            "models_merged": merged_list,
        },
    }
