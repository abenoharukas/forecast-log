"""信号箭头逐日溯源面板数据聚合（`GET /api/signal-detail`）。

针对任意一个 BUY / SELL 信号箭头，按严格无未来函数口径（`<= signal_date`）汇总：
1. 跨资产定量因子与本土过滤指标快照（`^TNX`、`DX-Y.NYB`、`HYG/TLT`、`^VIX` 及 QQQ/A股均线跳空）
2. 宏观事件日历倒计时窗口（当日距下一次 `CPI`、`NFP`、`FOMC` 天数及窗口状态）
3. 当日参考新闻列表（按财经/宏观相关性优先排序）、风险/利好关键字命中及 DeepSeek × Qwen 历史打分
4. Polymarket 预测市场截至信号当日收盘（`21:15 UTC`）的隐含赔率与 24h 变化
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
import math
from pathlib import Path
from typing import Any

from forecast_log import events, keywords, news_ab, news_labels, polymarket, surprise
from forecast_log.agent.env import merged_env
from forecast_log.db import connect, db_path, get_news_fetch, get_series


def _bar_index_le(bars: tuple, target_day: date) -> int:
    dates = [b.date for b in bars]
    pos = bisect_right(dates, target_day) - 1
    return pos


def _pct(curr: float, prev: float) -> float | None:
    if not prev or not math.isfinite(curr) or not math.isfinite(prev):
        return None
    return round((curr / prev - 1.0) * 100.0, 2)


def _cross_factors_as_of(root: Path, sig_day: date, market: str, mkt_day: date) -> list[dict[str, Any]]:
    """计算截至 `sig_day` 当日已知的 4 个核心跨资产因子与本土辅助指标（零未来函数）。"""
    items: list[dict[str, Any]] = []

    tnx = get_series(root, "^TNX")
    if tnx and tnx.bars:
        idx = _bar_index_le(tnx.bars, sig_day)
        if idx >= 0:
            c0 = float(tnx.bars[idx].close)
            c20 = float(tnx.bars[idx - 20].close) if idx >= 20 else None
            d20 = round(c0 - c20, 3) if c20 is not None else None
            items.append(
                {
                    "symbol": "^TNX",
                    "label": "10Y 美债收益率",
                    "date": tnx.bars[idx].date.isoformat(),
                    "close": round(c0, 3),
                    "metric": f"20d {d20:+.2f}" if d20 is not None else "—",
                    "tone": "bull" if (d20 is not None and d20 < 0) else "bear" if (d20 is not None and d20 > 0) else "neutral",
                    "note": "收益率下行 → 估值偏多" if (d20 is not None and d20 < 0) else "收益率上行 → 贴现偏空",
                }
            )

    dxy = get_series(root, "DX-Y.NYB")
    if dxy and dxy.bars:
        idx = _bar_index_le(dxy.bars, sig_day)
        if idx >= 0:
            c0 = float(dxy.bars[idx].close)
            p20 = _pct(c0, float(dxy.bars[idx - 20].close)) if idx >= 20 else None
            items.append(
                {
                    "symbol": "DX-Y.NYB",
                    "label": "美元指数 DXY",
                    "date": dxy.bars[idx].date.isoformat(),
                    "close": round(c0, 2),
                    "metric": f"20d {p20:+.2f}%" if p20 is not None else "—",
                    "tone": "bull" if (p20 is not None and p20 < 0) else "bear" if (p20 is not None and p20 > 0) else "neutral",
                    "note": "美元走弱 → 风险偏好偏多" if (p20 is not None and p20 < 0) else "美元走强 → 流动性收紧",
                }
            )

    hyg = get_series(root, "HYG")
    tlt = get_series(root, "TLT")
    if hyg and tlt and hyg.bars and tlt.bars:
        tlt_map = {b.date: float(b.close) for b in tlt.bars if b.date <= sig_day and float(b.close) > 0}
        ratios = [(b.date, float(b.close) / tlt_map[b.date]) for b in hyg.bars if b.date in tlt_map]
        if ratios:
            d0, r0 = ratios[-1]
            p20 = _pct(r0, ratios[-21][1]) if len(ratios) >= 21 else None
            items.append(
                {
                    "symbol": "HYG/TLT",
                    "label": "信用利差比 HYG/TLT",
                    "date": d0.isoformat(),
                    "close": round(r0, 4),
                    "metric": f"20d {p20:+.2f}%" if p20 is not None else "—",
                    "tone": "bull" if (p20 is not None and p20 > 0) else "bear" if (p20 is not None and p20 < 0) else "neutral",
                    "note": "高收益债跑赢长债 → 信用偏多" if (p20 is not None and p20 > 0) else "信用利差走弱 → 避险偏空",
                }
            )

    vix = get_series(root, "^VIX")
    if vix and vix.bars:
        idx = _bar_index_le(vix.bars, sig_day)
        if idx >= 0:
            c0 = float(vix.bars[idx].close)
            vs_med = None
            if idx >= 60:
                start = max(0, idx - 251)
                win = sorted(float(b.close) for b in vix.bars[start : idx + 1])
                med = win[len(win) // 2]
                vs_med = _pct(c0, med)
            items.append(
                {
                    "symbol": "^VIX",
                    "label": "恐慌指数 VIX",
                    "date": vix.bars[idx].date.isoformat(),
                    "close": round(c0, 2),
                    "metric": f"vs年均中位 {vs_med:+.1f}%" if vs_med is not None else "—",
                    "tone": "bull" if (vs_med is not None and vs_med > 0) else "neutral",
                    "note": "高于年均中位 → 恐慌溢价补偿偏多" if (vs_med is not None and vs_med > 0) else "低于年均中位 → 波动平静",
                }
            )

    if market == "qqq":
        qqq = get_series(root, "QQQ")
        if qqq and qqq.bars:
            idx = _bar_index_le(qqq.bars, sig_day)
            if idx >= 19:
                c0 = float(qqq.bars[idx].close)
                ma20 = sum(float(b.close) for b in qqq.bars[idx - 19 : idx + 1]) / 20.0
                diff_ma = _pct(c0, ma20)
                items.append(
                    {
                        "symbol": "QQQ·MA20",
                        "label": "QQQ 20日均线位置",
                        "date": qqq.bars[idx].date.isoformat(),
                        "close": round(ma20, 2),
                        "metric": f"偏离 {diff_ma:+.2f}%" if diff_ma is not None else "—",
                        "tone": "bull" if (diff_ma or 0) >= 0 else "bear",
                        "note": "站上 20 日均线" if (diff_ma or 0) >= 0 else "跌破 20 日均线（可选趋势否决）",
                    }
                )
    elif market == "ssec":
        ssec = get_series(root, "000001.SS")
        if ssec and ssec.bars:
            idx = _bar_index_le(ssec.bars, mkt_day)
            if idx >= 19:
                c0 = float(ssec.bars[idx].close)
                ma20 = sum(float(b.close) for b in ssec.bars[idx - 19 : idx + 1]) / 20.0
                diff_ma = _pct(c0, ma20)
                items.append(
                    {
                        "symbol": "SSEC·MA20",
                        "label": "上证 20日均线位置",
                        "date": ssec.bars[idx].date.isoformat(),
                        "close": round(ma20, 2),
                        "metric": f"偏离 {diff_ma:+.2f}%" if diff_ma is not None else "—",
                        "tone": "bull" if (diff_ma or 0) >= 0 else "bear",
                        "note": "站上 20 日均线" if (diff_ma or 0) >= 0 else "低于 20 日均线（可选趋势否决）",
                    }
                )
    return items


def _calendar_as_of(root: Path, sig_day: date) -> list[dict[str, Any]]:
    cal = events.load(root)
    if not cal:
        return []
    out: list[dict[str, Any]] = []
    for key, label in (("cpi", "CPI 通胀"), ("nfp", "NFP 非农"), ("fomc", "FOMC 决议")):
        days_list = cal.get(key) or []
        dist = events.days_to_next(days_list, sig_day)
        next_dt = next((d.isoformat() for d in days_list if d >= sig_day), None)
        status = "常规期"
        active_kind = ""
        if dist <= 3 and key == "cpi":
            status = "买入触发窗口 (0-3d)"
            active_kind = "trigger"
        elif 3 <= dist <= 7 and key == "nfp":
            status = "买入触发窗口 (3-7d)"
            active_kind = "trigger"
        elif dist == 0 and key == "fomc":
            status = "决议日否决窗口"
            active_kind = "veto"
        out.append(
            {
                "key": key,
                "label": label,
                "next_date": next_dt,
                "days_away": int(dist) if dist < 900 else None,
                "status": status,
                "active_kind": active_kind,
            }
        )
    return out


def _llm_labels_on_day(root: Path, day_iso: str) -> dict[str, Any]:
    ds_rec: dict[str, Any] | None = None
    day_file = surprise.day_path(root, day_iso)
    if day_file.is_file():
        try:
            raw = json.loads(day_file.read_text(encoding="utf-8"))
            lbl = raw.get("label") or {}
            sur = str(lbl.get("surprise") or "none")
            direc = str(lbl.get("direction") or "ambiguous")
            ds_rec = {
                "model": raw.get("model") or "deepseek-chat",
                "surprise": sur,
                "direction": direc,
                "category": str(lbl.get("category") or ""),
                "evidence": str(lbl.get("evidence") or ""),
                "veto": sur == "strong" and direc == "bearish",
                "boost": sur == "strong" and direc == "bullish",
            }
        except Exception:
            ds_rec = None
    if ds_rec is None:
        compact = surprise.load_compact(root)
        if day_iso in compact:
            row = compact[day_iso]
            sur = str(row.get("surprise") or "none")
            direc = str(row.get("direction") or "ambiguous")
            ds_rec = {
                "model": row.get("model") or "deepseek-chat",
                "surprise": sur,
                "direction": direc,
                "category": str(row.get("category") or ""),
                "evidence": str(row.get("evidence") or ""),
                "veto": sur == "strong" and direc == "bearish",
                "boost": sur == "strong" and direc == "bullish",
            }

    qw_rec: dict[str, Any] | None = None
    env = merged_env(root)
    qw_model = env.get("QWEN_MODEL") or "qwen3.8-flash"
    qw_file = news_ab.ab_dir(root, news_ab.slug_of(qw_model)) / f"{day_iso}.json"
    if not qw_file.is_file():
        ab_root = root / "artifacts" / "news_ab"
        if ab_root.is_dir():
            for cand in sorted(ab_root.glob(f"*qwen*/{day_iso}.json")):
                qw_file = cand
                break
    if qw_file.is_file():
        try:
            raw = json.loads(qw_file.read_text(encoding="utf-8"))
            lbl = raw.get("label") or {}
            sev = float(lbl.get("severity") or 0.0)
            qw_rec = {
                "model": raw.get("model") or qw_model,
                "regime": str(lbl.get("regime") or "neutral"),
                "severity": round(sev, 3),
                "size_multiplier": news_labels.size_multiplier(sev),
                "veto": sev >= 0.6,
                "note": str(lbl.get("note") or ""),
            }
        except Exception:
            qw_rec = None

    return {"deepseek": ds_rec, "qwen": qw_rec}


import json  # noqa: E402


def _news_on_day(root: Path, sig_day: date, market: str, mkt_day: date, limit: int = 10) -> dict[str, Any]:
    target_days = [sig_day]
    if market == "ssec" and mkt_day != sig_day:
        target_days.append(mkt_day)

    raw_docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in target_days:
        rec = get_news_fetch(root, d)
        if not rec or not rec.get("documents"):
            continue
        src_default = str(rec.get("source") or "")
        for doc in rec["documents"]:
            title = str(doc.get("title") or "").strip()
            if not title:
                continue
            domain = str(doc.get("domain") or "")
            if "wikipedia.org" in domain.lower() and len(title) < 36 and "." not in title:
                continue
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            hits = [name for name, pat in keywords.COMPILED.items() if pat.search(title)]
            off_hits = [h for h in hits if h in keywords.RISK_OFF]
            on_hits = [h for h in hits if h in keywords.RISK_ON]
            sentiment = (
                "risk-off"
                if off_hits and not on_hits
                else "risk-on"
                if on_hits and not off_hits
                else "mixed"
                if off_hits and on_hits
                else "neutral"
            )
            domain = str(doc.get("domain") or "")
            is_fin = any(k in domain.lower() for k in ("wsj", "cnbc", "yahoo", "federalreserve", "sec.gov", "reuters", "bloomberg"))
            relevance = (3 if is_fin else 0) + len(hits) * 2
            raw_docs.append(
                {
                    "as_of": d.isoformat(),
                    "title": title,
                    "url": str(doc.get("url") or ""),
                    "domain": domain or src_default,
                    "tags": hits[:4],
                    "sentiment": sentiment,
                    "_rel": relevance,
                }
            )

    all_titles = [d["title"] for d in raw_docs]
    kw_scan = keywords.scan(all_titles) if all_titles else {"items": 0, "hits": [], "risk_off_hits": [], "risk_on_hits": []}
    raw_docs.sort(key=lambda x: x["_rel"], reverse=True)
    top_items = [{k: v for k, v in item.items() if k != "_rel"} for item in raw_docs[:limit]]

    return {
        "as_of": sig_day.isoformat(),
        "total_count": len(raw_docs),
        "keywords": {
            "risk_off_hits": kw_scan.get("risk_off_hits") or [],
            "risk_on_hits": kw_scan.get("risk_on_hits") or [],
            "hits": (kw_scan.get("hits") or [])[:8],
        },
        "llm": _llm_labels_on_day(root, sig_day.isoformat()),
        "items": top_items,
    }


def load_signal_detail(
    root: Path,
    market: str,
    date_str: str,
    us_date_str: str | None = None,
    fetch_poly: bool = True,
) -> dict[str, Any] | None:
    """加载单个信号日（`date_str`）的全部参考依据、当日新闻与 Polymarket 赔率。"""
    mkt = market if market in ("ndx", "qqq", "ssec") else "ndx"
    try:
        mkt_day = date.fromisoformat(str(date_str)[:10])
    except ValueError:
        return None

    sig_iso = mkt_day.isoformat()
    sig_row: dict[str, Any] | None = None
    path = db_path(root)
    if path.is_file():
        conn = connect(path)
        try:
            row = conn.execute(
                "SELECT * FROM signals WHERE market = ? AND date = ?",
                (mkt, mkt_day.isoformat()),
            ).fetchone()
            if row is not None:
                sig_row = dict(row)
                if mkt == "ssec" and (us_date_str or row["us_date"]):
                    sig_iso = str(us_date_str or row["us_date"])[:10]
            if mkt == "ssec" and sig_iso != mkt_day.isoformat():
                us_row = conn.execute(
                    "SELECT score, threshold, sell_threshold FROM signals WHERE market = 'ndx' AND date = ?",
                    (sig_iso,),
                ).fetchone()
                if us_row is not None and sig_row is not None:
                    sig_row["score"] = us_row["score"]
                    sig_row["threshold"] = us_row["threshold"]
                    sig_row["sell_threshold"] = us_row["sell_threshold"]
        finally:
            conn.close()

    if mkt == "ssec" and us_date_str:
        sig_iso = str(us_date_str)[:10]

    try:
        sig_day = date.fromisoformat(sig_iso)
    except ValueError:
        sig_day = mkt_day

    return {
        "ok": True,
        "market": mkt,
        "market_date": mkt_day.isoformat(),
        "signal_date": sig_day.isoformat(),
        "signal_row": {
            "score": round(float(sig_row["score"]), 4) if (sig_row and sig_row.get("score") is not None) else None,
            "threshold": round(float(sig_row["threshold"]), 4) if (sig_row and sig_row.get("threshold") is not None) else None,
            "sell_threshold": round(float(sig_row["sell_threshold"]), 4) if (sig_row and sig_row.get("sell_threshold") is not None) else None,
            "entry": round(float(sig_row["entry"]), 2) if (sig_row and sig_row.get("entry") is not None) else None,
            "entry_at": sig_row.get("entry_at") if sig_row else None,
            "base_close": round(float(sig_row["base_close"]), 2) if (sig_row and sig_row.get("base_close") is not None) else None,
            "close3": round(float(sig_row["close3"]), 2) if (sig_row and sig_row.get("close3") is not None) else None,
            "close5": round(float(sig_row["close5"]), 2) if (sig_row and sig_row.get("close5") is not None) else None,
            "outcome": sig_row.get("outcome") if sig_row else None,
            "hit3": sig_row.get("hit3") if sig_row else None,
        },
        "cross_factors": _cross_factors_as_of(root, sig_day, mkt, mkt_day),
        "macro_calendar": _calendar_as_of(root, sig_day),
        "news": _news_on_day(root, sig_day, mkt, mkt_day),
        "polymarket": polymarket.load_polymarket_as_of(root, sig_day.isoformat(), allow_network=fetch_poly),
    }
