"""SQLite / D1 共享的看板数据读写层：cross 信号、VXN timeline、agent 评测结果。

本地全部存进 `data/forecast.db`，线上同步到 Cloudflare D1 的同名表，
前后端和脚本不再读写 `artifacts/*.json`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forecast_log.db import connect, db_path

SESSION_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "entry",
    "entry_at",
    "base_close",
    "close3",
    "close5",
    "hit3",
    "flag",
    "flag_price_only",
    "news_veto",
    "score",
    "threshold",
    "outcome",
    "target",
    "peak",
    "peak_at",
    "us_date",
    "is_closed",
)

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


def cast(kind: str, value: str | None) -> Any:
    if kind == "list":
        return []
    if kind == "obj":
        return {}
    if value is None:
        return None
    if kind == "num":
        return float(value)
    if kind == "int":
        return int(value)
    if kind == "bool":
        return value in ("1", "true")
    return value


def set_path(root: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cursor: Any = root
    for i in range(len(parts) - 1):
        key: Any = int(parts[i]) if parts[i].isdigit() else parts[i]
        if isinstance(cursor, list):
            while len(cursor) <= key:
                cursor.append(None)
        if cursor[key] is None or (isinstance(cursor, dict) and key not in cursor):
            cursor[key] = [] if parts[i + 1].isdigit() else {}
        elif isinstance(cursor, dict) and key not in cursor:
            cursor[key] = [] if parts[i + 1].isdigit() else {}
        cursor = cursor[key]
    last: Any = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
    if isinstance(cursor, list):
        while len(cursor) <= last:
            cursor.append(None)
    cursor[last] = value


def _set_path_clean(root: dict, path: str, value: Any) -> None:
    parts = path.split("\x1f") if "\x1f" in path else path.split(".")
    cursor: Any = root
    for i in range(len(parts) - 1):
        part = parts[i]
        next_is_idx = parts[i + 1].isdigit()
        if part.isdigit():
            idx = int(part)
            while len(cursor) <= idx:
                cursor.append([] if next_is_idx else {})
            if cursor[idx] is None:
                cursor[idx] = [] if next_is_idx else {}
            cursor = cursor[idx]
        else:
            if part not in cursor or cursor[part] is None:
                cursor[part] = [] if next_is_idx else {}
            cursor = cursor[part]
    last = parts[-1]
    if last.isdigit():
        idx = int(last)
        while len(cursor) <= idx:
            cursor.append(None)
        cursor[idx] = value
    else:
        cursor[last] = value


# ------------------------------------------------------------------ Cross 信号


def build_cross_bundle(doc: dict, market: str) -> dict:
    from forecast_log.bars import is_session_closed

    nodes: list[tuple[str, str, str | None]] = []
    for key in CROSS_SCALAR_ROOTS:
        if key in doc:
            walk(key, doc[key], nodes)
    if "calendar" in doc:
        walk("calendar", doc["calendar"], nodes)
    if isinstance(doc.get("question"), dict) and "factors" in doc["question"]:
        walk("question.factors", doc["question"]["factors"], nodes)

    components = doc.get("components") or {}
    factor_keys = list(components.keys())
    positions = {key: i for i, key in enumerate(factor_keys)}

    instrument = str(doc.get("instrument") or "")
    tz_name = "Asia/Shanghai" if (market == "ssec" or instrument.endswith((".SS", ".SZ"))) else "America/New_York"
    fetched_at = doc.get("fetched_at")

    sessions = []
    for index, day in enumerate(doc["time"]):
        mask = 0
        for key, slot in positions.items():
            series = components.get(key) or []
            if index < len(series) and series[index]:
                mask |= int(series[index]) << (4 * slot)
        row = {"date": day, "factors_packed": mask}
        for column in SESSION_COLUMNS:
            series = doc.get(column)
            row[column] = series[index] if isinstance(series, list) and index < len(series) else None
        if row.get("is_closed") is None:
            from datetime import date as _date

            row["is_closed"] = 1 if is_session_closed(_date.fromisoformat(day), tz_name, fetched_at) else 0
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
        for position, ctx in enumerate(doc.get("contexts") or [])
    ]

    question = doc.get("question") or {}
    return {
        "market": market,
        "row": {
            "market": market,
            "instrument": doc.get("instrument"),
            "name": doc.get("name"),
            "horizon": question.get("horizon"),
            "question": question.get("text"),
            "rule": question.get("rule"),
            "first_session": doc.get("first_session"),
            "last_session": doc.get("last_session"),
            "fetched_at": doc.get("fetched_at"),
        },
        "nodes": nodes,
        "contexts": contexts,
        "factor_keys": [[i, key] for i, key in enumerate(factor_keys)],
        "sessions": sessions,
    }


def save_cross_market(root: Path, market: str, doc: dict) -> int:
    """增量更新信号表：绝不删除历史行，只插入新增交易日、更新未收盘（is_closed=0）行或刚完成结算的尾部行。"""
    bundle = build_cross_bundle(doc, market)
    conn = connect(db_path(root))
    try:
        existing = {
            row["date"]: row
            for row in conn.execute(
                "SELECT date, open, high, low, close, volume, flag, outcome, is_closed FROM session WHERE market = ?",
                (market,),
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
            ):
                # 若某一天之前未收盘或新入库，其前 7 个交易日的入场价/3日/5日结算窗口可能依赖这一天的价格，一并重算覆盖
                for back in range(max(0, idx - 7), idx + 1):
                    dirty_indices.add(back)

        # 尾部最近 7 个交易日若 OHLCV、信号或收盘标记发生变化，也一并增量同步
        for idx in range(max(0, len(all_sessions) - 7), len(all_sessions)):
            s = all_sessions[idx]
            prev = existing.get(s["date"])
            if (
                prev is None
                or prev["close"] != s.get("close")
                or prev["open"] != s.get("open")
                or prev["high"] != s.get("high")
                or prev["low"] != s.get("low")
                or prev["volume"] != s.get("volume")
                or prev["flag"] != s.get("flag")
                or prev["outcome"] != s.get("outcome")
                or prev["is_closed"] != s.get("is_closed")
            ):
                dirty_indices.add(idx)

        sessions_to_write = [all_sessions[i] for i in sorted(dirty_indices)]

        with conn:
            r = bundle["row"]
            conn.execute(
                """
                INSERT OR REPLACE INTO market (market, instrument, name, horizon, question, rule, first_session, last_session, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["market"],
                    r["instrument"],
                    r["name"],
                    r["horizon"],
                    r["question"],
                    r["rule"],
                    r["first_session"],
                    r["last_session"],
                    r["fetched_at"],
                ),
            )
            if sessions_to_write:
                cols = ("market", "date", *SESSION_COLUMNS, "factors_packed")
                placeholders = ",".join("?" for _ in cols)
                conn.executemany(
                    f"INSERT OR REPLACE INTO session ({','.join(cols)}) VALUES ({placeholders})",
                    [tuple([market] + [s.get(c) for c in cols[1:]]) for s in sessions_to_write],
                )
            conn.executemany(
                "INSERT OR REPLACE INTO factor_key (market, position, key) VALUES (?, ?, ?)",
                [(market, pos, key) for pos, key in bundle["factor_keys"]],
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO contexts (market, position, key, label, kind, required, threshold, group_name, group_label, note, available_from)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        market,
                        c["position"],
                        c["key"],
                        c["label"],
                        c["kind"],
                        c["required"],
                        c["threshold"],
                        c["group_name"],
                        c["group_label"],
                        c["note"],
                        c["available_from"],
                    )
                    for c in bundle["contexts"]
                ],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO market_node (market, path, kind, value) VALUES (?, ?, ?, ?)",
                [(market, path, kind, value) for path, kind, value in bundle["nodes"]],
            )
        return len(sessions_to_write)
    finally:
        conn.close()


def load_cross_market(root: Path, market: str) -> dict | None:
    path = db_path(root)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        meta = conn.execute("SELECT * FROM market WHERE market = ?", (market,)).fetchone()
        sessions = conn.execute("SELECT * FROM session WHERE market = ? ORDER BY date", (market,)).fetchall()
        if meta is None or not sessions:
            return None
        nodes = conn.execute("SELECT path, kind, value FROM market_node WHERE market = ?", (market,)).fetchall()
        contexts = conn.execute("SELECT * FROM contexts WHERE market = ? ORDER BY position", (market,)).fetchall()
        factors = conn.execute("SELECT position, key FROM factor_key WHERE market = ? ORDER BY position", (market,)).fetchall()

        doc: dict[str, Any] = {}
        if meta["name"]:
            doc["name"] = meta["name"]
        doc["instrument"] = meta["instrument"]
        doc["question"] = {"horizon": meta["horizon"], "text": meta["question"], "rule": meta["rule"]}
        doc["fetched_at"] = meta["fetched_at"]
        doc["first_session"] = meta["first_session"]
        doc["last_session"] = meta["last_session"]
        doc["time"] = [row["date"] for row in sessions]

        for column in SESSION_COLUMNS:
            values = [row[column] for row in sessions]
            if any(v is not None for v in values):
                doc[column] = values

        doc["components"] = {}
        for f in factors:
            pos = f["position"]
            key = f["key"]
            doc["components"][key] = [(row["factors_packed"] >> (4 * pos)) & 0xF for row in sessions]

        doc["contexts"] = []
        for row in contexts:
            out: dict[str, Any] = {"key": row["key"]}
            if row["label"]:
                out["label"] = row["label"]
            if row["kind"]:
                out["kind"] = row["kind"]
            if row["required"]:
                out["required"] = True
            if row["threshold"] is not None:
                out["threshold"] = row["threshold"]
            if row["group_name"]:
                out["group"] = row["group_name"]
                out["group_label"] = row["group_label"]
            if row["note"]:
                out["note"] = row["note"]
            if row["available_from"]:
                out["available_from"] = row["available_from"]
            doc["contexts"].append(out)

        for row in nodes:
            _set_path_clean(doc, row["path"], cast(row["kind"], row["value"]))

        if isinstance(doc.get("latest"), dict) and sessions:
            doc["latest"]["is_closed"] = bool(sessions[-1]["is_closed"])

        return doc
    finally:
        conn.close()


def cross_fetched_at(root: Path, market: str) -> str | None:
    path = db_path(root)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        row = conn.execute("SELECT fetched_at FROM market WHERE market = ?", (market,)).fetchone()
        return row["fetched_at"] if row else None
    finally:
        conn.close()


def cross_sessions_summary(root: Path) -> dict[str, str | None]:
    path = db_path(root)
    if not path.is_file():
        return {}
    conn = connect(path)
    try:
        rows = conn.execute("SELECT market, last_session FROM market").fetchall()
        return {row["market"]: row["last_session"] for row in rows}
    finally:
        conn.close()


def unclosed_sessions_summary(root: Path) -> dict[str, list[str]]:
    """返回数据库中各市场尚未收盘（is_closed = 0）的交易日列表：{market: [date, ...]}。"""
    path = db_path(root)
    if not path.is_file():
        return {}
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT market, date FROM session WHERE is_closed = 0 ORDER BY market ASC, date ASC"
        ).fetchall()
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(str(row["market"]), []).append(str(row["date"]))
        return out
    finally:
        conn.close()


def cross_market_status(root: Path) -> dict[str, dict[str, Any]]:
    """返回各市场最新交易日、抓取时间、是否已收盘以及所有未收盘交易日清单。"""
    path = db_path(root)
    if not path.is_file():
        return {}
    conn = connect(path)
    try:
        markets = conn.execute("SELECT market, instrument, last_session, fetched_at FROM market").fetchall()
        out: dict[str, dict[str, Any]] = {}
        for m in markets:
            mkt = str(m["market"])
            last_row = conn.execute(
                "SELECT date, close, is_closed FROM session WHERE market = ? ORDER BY date DESC LIMIT 1",
                (mkt,),
            ).fetchone()
            unclosed = [
                str(r["date"])
                for r in conn.execute(
                    "SELECT date FROM session WHERE market = ? AND is_closed = 0 ORDER BY date ASC",
                    (mkt,),
                ).fetchall()
            ]
            out[mkt] = {
                "instrument": m["instrument"],
                "last_session": last_row["date"] if last_row else m["last_session"],
                "last_close": last_row["close"] if last_row else None,
                "fetched_at": m["fetched_at"],
                "is_closed": bool(last_row["is_closed"]) if last_row is not None else True,
                "unclosed_dates": unclosed,
            }
        return out
    finally:
        conn.close()


# ------------------------------------------------------------------ VXN timeline


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


def save_timeline(root: Path, doc: dict) -> None:
    """增量更新 timeline 各表：追加新增交易日并刷新尾部最近 5 个交易日（覆盖盘中未收盘记录），绝不清空历史表。"""
    bundle = build_timeline_bundle(doc)
    conn = connect(db_path(root))
    try:
        close_max = {
            r["symbol"]: r["m"]
            for r in conn.execute("SELECT symbol, MAX(idx) AS m FROM timeline_close GROUP BY symbol").fetchall()
            if r["m"] is not None
        }
        rolling_max = {
            r["symbol"]: r["m"]
            for r in conn.execute("SELECT symbol, MAX(idx) AS m FROM timeline_rolling GROUP BY symbol").fetchall()
            if r["m"] is not None
        }
        daily_max = {
            r["symbol"]: r["m"]
            for r in conn.execute("SELECT symbol, MAX(idx) AS m FROM timeline_daily GROUP BY symbol").fetchall()
            if r["m"] is not None
        }
        ladder_max = {
            r["horizon"]: r["m"]
            for r in conn.execute("SELECT horizon, MAX(idx) AS m FROM timeline_ladder GROUP BY horizon").fetchall()
            if r["m"] is not None
        }

        new_closes = [r for r in bundle["closes"] if r[1] >= max(0, close_max.get(r[0], -1) - 5)]
        new_rolling = [r for r in bundle["rolling"] if r[1] >= max(0, rolling_max.get(r[0], -1) - 5)]
        new_daily = [r for r in bundle["daily"] if r[1] >= max(0, daily_max.get(r[0], -1) - 5)]
        new_ladder = [r for r in bundle["ladder"] if r[2] >= max(0, ladder_max.get(r[1], -1) - 5)]

        with conn:
            if new_closes:
                conn.executemany(
                    "INSERT OR REPLACE INTO timeline_close (symbol, idx, date, close) VALUES (?, ?, ?, ?)",
                    new_closes,
                )
            if new_rolling:
                conn.executemany(
                    "INSERT OR REPLACE INTO timeline_rolling (symbol, idx, time, as_of, anchor, median, low, high, actual) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    new_rolling,
                )
            if new_daily:
                conn.executemany(
                    "INSERT OR REPLACE INTO timeline_daily (symbol, idx, time, median, low, high, actual, previous, risk) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    new_daily,
                )
            if new_ladder:
                conn.executemany(
                    "INSERT OR REPLACE INTO timeline_ladder (symbol, horizon, idx, median, low, high) VALUES (?, ?, ?, ?, ?, ?)",
                    new_ladder,
                )
            conn.executemany(
                "INSERT OR REPLACE INTO timeline_node (path, kind, value) VALUES (?, ?, ?)",
                bundle["nodes"],
            )
    finally:
        conn.close()


def load_timeline(root: Path) -> dict | None:
    path = db_path(root)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        closes = conn.execute("SELECT symbol, date, close FROM timeline_close ORDER BY symbol, idx").fetchall()
        if not closes:
            return None
        rolling = conn.execute("SELECT * FROM timeline_rolling ORDER BY symbol, idx").fetchall()
        daily = conn.execute("SELECT * FROM timeline_daily ORDER BY symbol, idx").fetchall()
        ladder = conn.execute("SELECT * FROM timeline_ladder ORDER BY horizon, idx").fetchall()
        nodes = conn.execute("SELECT path, kind, value FROM timeline_node WHERE path NOT LIKE 'questions.%'").fetchall()

        doc: dict[str, Any] = {
            "rolling": {"vxn": {}, "ndx": {}},
            "ladder": {"vxn": {"columns": {}}},
            "forward": {},
            "vxn": {"time": [], "close": []},
            "ndx": {"time": [], "close": []},
            "daily": {"vxn": {}, "ndx": {}},
        }

        for row in closes:
            sym = row["symbol"]
            doc[sym]["time"].append(row["date"])
            doc[sym]["close"].append(row["close"])

        for sym in ("vxn", "ndx"):
            rows = [r for r in rolling if r["symbol"] == sym]
            doc["rolling"][sym] = {
                "time": [r["time"] for r in rows],
                "as_of": [r["as_of"] for r in rows],
                "anchor": [r["anchor"] for r in rows],
                "median": [r["median"] for r in rows],
                "low": [r["low"] for r in rows],
                "high": [r["high"] for r in rows],
                "actual": [r["actual"] for r in rows],
            }
            drows = [r for r in daily if r["symbol"] == sym]
            doc["daily"][sym] = {
                "time": [r["time"] for r in drows],
                "median": [r["median"] for r in drows],
                "low": [r["low"] for r in drows],
                "high": [r["high"] for r in drows],
                "actual": [r["actual"] for r in drows],
                "previous": [r["previous"] for r in drows],
                "risk": [r["risk"] for r in drows],
            }

        cols = doc["ladder"]["vxn"]["columns"]
        for row in ladder:
            h = str(row["horizon"])
            if h not in cols:
                cols[h] = {"median": [], "low": [], "high": []}
            cols[h]["median"].append(row["median"])
            cols[h]["low"].append(row["low"])
            cols[h]["high"].append(row["high"])

        for row in nodes:
            _set_path_clean(doc, row["path"], cast(row["kind"], row["value"]))

        return doc
    finally:
        conn.close()


# ------------------------------------------------------------------ Agent eval


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
    bundle = build_agent_bundle(doc, dataset)
    conn = connect(db_path(root))
    try:
        with conn:
            r = bundle["row"]
            conn.execute(
                "INSERT OR REPLACE INTO agent_dataset (dataset, n, scored_on_shared, generated_at) VALUES (?, ?, ?, ?)",
                (r["dataset"], r["n"], r["scored_on_shared"], r["generated_at"]),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO agent_node (dataset, path, kind, value) VALUES (?, ?, ?, ?)",
                [(dataset, path, kind, value) for path, kind, value in bundle["nodes"]],
            )
    finally:
        conn.close()


def load_agent_eval(root: Path, dataset: str) -> dict | None:
    path = db_path(root)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        meta = conn.execute("SELECT * FROM agent_dataset WHERE dataset = ?", (dataset,)).fetchone()
        if meta is None:
            return None
        nodes = conn.execute("SELECT path, kind, value FROM agent_node WHERE dataset = ?", (dataset,)).fetchall()
        doc: dict[str, Any] = {
            "n": meta["n"],
            "scored_on_shared": meta["scored_on_shared"],
            "generated_at": meta["generated_at"],
        }
        for row in nodes:
            _set_path_clean(doc, row["path"], cast(row["kind"], row["value"]))
        return doc
    finally:
        conn.close()
