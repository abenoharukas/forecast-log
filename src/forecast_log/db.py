"""SQLite 极简存储层：全库仅保留 5 张纯粹的业务实体表（`bars`、`signals`、`news`、`agent_eval`、`chat_messages`）。

5 张核心表职责：
1. `bars`          —— 全品种客观日线行情（OHLCV、是否收盘 `is_closed`、抓取时间戳 `fetched_at`）
2. `signals`       —— 跨资产买入信号与结算结果（不含重复 OHLCV，读取时按品种关联 `bars`）
3. `news`          —— 每日新闻标题、正文与抓取来源记录
4. `agent_eval`    —— LLM 盲测做题结果扁平表（一行对应一道题×一个预测模型）
5. `chat_messages` —— 多模型（DeepSeek × Qwen）与用户的群聊研判历史记录

设计约束：
- 不使用外键（FOREIGN KEY），靠联合主键 + CHECK 约束 + 索引保证数据完整性。
- 开启 WAL 模式，支持多线程刷新与查询并发读写。
- 严格增量追加，绝不删除历史已完成交易日。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forecast_log.bars import Series

SIGNAL_COLUMNS = (
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
)

OBSOLETE_TABLES = (
    "series_meta",
    "session",
    "market",
    "contexts",
    "factor_key",
    "market_node",
    "news_fetches",
    "news_items",
    "timeline_close",
    "timeline_daily",
    "timeline_rolling",
    "timeline_ladder",
    "timeline_node",
    "agent_dataset",
    "agent_node",
)


def db_path(root: Path) -> Path:
    return root / "data" / "forecast.db"


def timezone_for_symbol(symbol: str) -> str:
    return "Asia/Shanghai" if symbol.endswith((".SS", ".SZ")) else "America/New_York"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS bars (
            symbol     TEXT NOT NULL,
            date       TEXT NOT NULL,
            open       REAL CHECK (open IS NULL OR open > 0),
            high       REAL CHECK (high IS NULL OR high > 0),
            low        REAL CHECK (low IS NULL OR low > 0),
            close      REAL NOT NULL CHECK (close > 0),
            volume     INTEGER CHECK (volume IS NULL OR volume >= 0),
            is_closed  INTEGER NOT NULL DEFAULT 1,
            fetched_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (symbol, date),
            CHECK (high IS NULL OR low IS NULL OR low <= high)
        );

        CREATE INDEX IF NOT EXISTS idx_bars_date
            ON bars(date);

        CREATE TABLE IF NOT EXISTS signals (
            market          TEXT NOT NULL,
            date            TEXT NOT NULL,
            entry           REAL,
            entry_at        TEXT,
            base_close      REAL,
            close3          REAL,
            close5          REAL,
            hit3            INTEGER,
            flag            INTEGER,
            flag_price_only INTEGER,
            news_veto       INTEGER,
            score           REAL,
            threshold       REAL,
            outcome         INTEGER,
            target          TEXT,
            peak            REAL,
            peak_at         TEXT,
            us_date         TEXT,
            factors_packed  INTEGER NOT NULL DEFAULT 0,
            fetched_at      TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (market, date)
        );

        CREATE TABLE IF NOT EXISTS news (
            as_of        TEXT NOT NULL,
            doc_id       TEXT NOT NULL DEFAULT '',
            published_at TEXT NOT NULL DEFAULT '',
            title        TEXT NOT NULL DEFAULT '',
            text         TEXT NOT NULL DEFAULT '',
            url          TEXT NOT NULL DEFAULT '',
            domain       TEXT NOT NULL DEFAULT '',
            depth        INTEGER,
            source       TEXT NOT NULL DEFAULT '',
            page         TEXT NOT NULL DEFAULT '',
            fetched_at   TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (as_of, doc_id)
        );

        CREATE INDEX IF NOT EXISTS idx_news_published_at
            ON news(published_at);

        CREATE INDEX IF NOT EXISTS idx_news_domain
            ON news(domain);

        CREATE TABLE IF NOT EXISTS agent_eval (
            dataset               TEXT NOT NULL,
            question_id           TEXT NOT NULL,
            predictor             TEXT NOT NULL,
            symbol                TEXT NOT NULL,
            as_of                 TEXT NOT NULL,
            as_of_close           REAL,
            resolve_session       TEXT NOT NULL,
            horizon               INTEGER NOT NULL,
            family                TEXT NOT NULL,
            threshold             REAL NOT NULL,
            question_text         TEXT NOT NULL,
            outcome               INTEGER NOT NULL,
            probability           REAL,
            cost_usd_total        REAL NOT NULL DEFAULT 0.0,
            cost_usd_per_forecast REAL,
            latency_p50_s         REAL,
            latency_p95_s         REAL,
            tool_calls_mean       REAL NOT NULL DEFAULT 0.0,
            gate_hits             INTEGER NOT NULL DEFAULT 0,
            failures              INTEGER NOT NULL DEFAULT 0,
            generated_at          TEXT NOT NULL DEFAULT '',
            config_model          TEXT NOT NULL DEFAULT '',
            models_merged         TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (dataset, question_id, predictor)
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            thread_id  TEXT NOT NULL,
            seq        INTEGER NOT NULL,
            market     TEXT NOT NULL,
            as_of      TEXT NOT NULL DEFAULT '',
            role       TEXT NOT NULL CHECK (role IN ('context', 'user', 'deepseek', 'qwen')),
            model      TEXT NOT NULL DEFAULT '',
            stance     TEXT NOT NULL DEFAULT '',
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (thread_id, seq)
        );

        CREATE INDEX IF NOT EXISTS idx_chat_messages_market_created
            ON chat_messages(market, created_at DESC, thread_id, seq);
        """
    )
    _migrate_to_four_tables(conn)


def _cast_legacy_node(kind: str, value: str | None) -> Any:
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


def _set_legacy_path(root: dict, path: str, value: Any) -> None:
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


def _extract_agent_rows_from_doc(dataset: str, doc: dict) -> list[tuple]:
    generated_at = str(doc.get("generated_at") or "")
    config = doc.get("config") or {}
    config_model = str(config.get("model") or "")
    models_merged = ",".join(str(m) for m in (config.get("models_merged") or []))
    meta_map = doc.get("question_meta") or {}
    outcomes_map = doc.get("outcomes") or {}
    bench = doc.get("bench") or list(meta_map.keys())
    predictors = doc.get("predictors") or {}

    rows: list[tuple] = []
    for qid in bench:
        qmeta = meta_map.get(qid) or {}
        symbol = str(qmeta.get("symbol") or config.get("symbol") or "^VXN")
        as_of = str(qmeta.get("as_of") or "")
        as_of_close = float(qmeta["as_of_close"]) if qmeta.get("as_of_close") is not None else None
        resolve_session = str(qmeta.get("resolve_session") or "")
        horizon = int(qmeta.get("horizon") or 0)
        family = str(qmeta.get("family") or "level")
        threshold = float(qmeta.get("threshold") or 0.0)
        qtext = str(qmeta.get("text") or qid)
        outcome = int(qmeta["outcome"] if qmeta.get("outcome") is not None else outcomes_map.get(qid, 0))

        for pred_name, pdata in predictors.items():
            answers = pdata.get("answers") or {}
            pairs = pdata.get("pairs") or {}
            prob_raw = answers.get(qid, pairs.get(qid))
            prob = float(prob_raw) if prob_raw is not None else None
            rows.append(
                (
                    dataset,
                    qid,
                    pred_name,
                    symbol,
                    as_of,
                    as_of_close,
                    resolve_session,
                    horizon,
                    family,
                    threshold,
                    qtext,
                    outcome,
                    prob,
                    float(pdata.get("cost_usd_total") or 0.0),
                    float(pdata["cost_usd_per_forecast"]) if pdata.get("cost_usd_per_forecast") is not None else None,
                    float(pdata["latency_p50_s"]) if pdata.get("latency_p50_s") is not None else None,
                    float(pdata["latency_p95_s"]) if pdata.get("latency_p95_s") is not None else None,
                    float(pdata.get("tool_calls_mean") or 0.0),
                    int(pdata.get("gate_hits") or 0),
                    int(pdata.get("failures") or 0),
                    generated_at,
                    config_model,
                    models_merged,
                )
            )
    return rows


def _migrate_to_four_tables(conn: sqlite3.Connection) -> None:
    """将旧版最多 15 张碎表无损收敛为 `bars`、`signals`、`news`、`agent_eval` 四张表。"""
    existing_tables = {
        str(r["name"])
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    bar_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(bars)").fetchall()}
    has_obsolete = bool(existing_tables & set(OBSOLETE_TABLES))
    bars_needs_cleanup = "fetched_at" not in bar_cols or "factors_packed" in bar_cols or "entry" in bar_cols

    if not has_obsolete and not bars_needs_cleanup:
        return

    with conn:
        # 1. 若旧库中信号列在 `bars`（或 `session`）表内，先无损迁入独立的 `signals` 纯信号表
        if "factors_packed" in bar_cols and "market" in existing_tables:
            sig_cols_csv = ", ".join(SIGNAL_COLUMNS)
            b_sig_cols_csv = ", ".join(f"b.{c}" for c in SIGNAL_COLUMNS)
            conn.execute(
                f"""
                INSERT OR IGNORE INTO signals (
                    market, date, {sig_cols_csv}, factors_packed, fetched_at
                )
                SELECT
                    m.market, b.date, {b_sig_cols_csv},
                    COALESCE(b.factors_packed, 0),
                    COALESCE(m.fetched_at, '')
                FROM market m
                JOIN bars b ON b.symbol = m.instrument
                WHERE (m.first_session IS NULL OR b.date >= m.first_session)
                  AND (m.last_session IS NULL OR b.date <= m.last_session)
                """
            )
        if "session" in existing_tables:
            sess_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(session)").fetchall()}
            sig_cols_csv = ", ".join(SIGNAL_COLUMNS)
            s_exprs = ", ".join(f"s.{c}" if c in sess_cols else "NULL" for c in SIGNAL_COLUMNS)
            fetched_join = "LEFT JOIN market m ON m.market = s.market" if "market" in existing_tables else ""
            fetched_expr = "COALESCE(m.fetched_at, '')" if "market" in existing_tables else "''"
            conn.execute(
                f"""
                INSERT OR IGNORE INTO signals (
                    market, date, {sig_cols_csv}, factors_packed, fetched_at
                )
                SELECT
                    s.market, s.date, {s_exprs},
                    COALESCE(s.factors_packed, 0),
                    {fetched_expr}
                FROM session s
                {fetched_join}
                """
            )

        # 2. 清理 `bars` 表至纯净的 9 列股价结构（吸收 `series_meta.fetched_at`）
        if bars_needs_cleanup:
            conn.execute(
                """
                CREATE TABLE _bars_clean (
                    symbol     TEXT NOT NULL,
                    date       TEXT NOT NULL,
                    open       REAL CHECK (open IS NULL OR open > 0),
                    high       REAL CHECK (high IS NULL OR high > 0),
                    low        REAL CHECK (low IS NULL OR low > 0),
                    close      REAL NOT NULL CHECK (close > 0),
                    volume     INTEGER CHECK (volume IS NULL OR volume >= 0),
                    is_closed  INTEGER NOT NULL DEFAULT 1,
                    fetched_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (symbol, date),
                    CHECK (high IS NULL OR low IS NULL OR low <= high)
                )
                """
            )
            has_meta = "series_meta" in existing_tables
            open_expr = "b.open" if "open" in bar_cols else "NULL"
            high_expr = "b.high" if "high" in bar_cols else "NULL"
            low_expr = "b.low" if "low" in bar_cols else "NULL"
            vol_expr = "b.volume" if "volume" in bar_cols else "NULL"
            closed_expr = "COALESCE(b.is_closed, 1)" if "is_closed" in bar_cols else "1"
            if "fetched_at" in bar_cols and has_meta:
                fa_expr = "COALESCE(NULLIF(b.fetched_at, ''), m.fetched_at, '')"
            elif has_meta:
                fa_expr = "COALESCE(m.fetched_at, '')"
            elif "fetched_at" in bar_cols:
                fa_expr = "COALESCE(b.fetched_at, '')"
            else:
                fa_expr = "''"
            meta_join = "LEFT JOIN series_meta m ON m.symbol = b.symbol" if has_meta else ""

            conn.execute(
                f"""
                INSERT INTO _bars_clean (symbol, date, open, high, low, close, volume, is_closed, fetched_at)
                SELECT b.symbol, b.date, {open_expr}, {high_expr}, {low_expr}, b.close, {vol_expr}, {closed_expr}, {fa_expr}
                FROM bars b
                {meta_join}
                WHERE b.close IS NOT NULL
                """
            )
            conn.execute("DROP TABLE bars")
            conn.execute("ALTER TABLE _bars_clean RENAME TO bars")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bars_date ON bars(date)")

        # 3. 合并 `news_items` 与 `news_fetches` 至单张 `news` 表（含 0 条新闻的已抓取占位日）
        if "news_items" in existing_tables:
            has_nf = "news_fetches" in existing_tables
            nf_join = "LEFT JOIN news_fetches f ON f.as_of = i.as_of" if has_nf else ""
            src_expr = "COALESCE(f.source, '')" if has_nf else "''"
            page_expr = "COALESCE(f.page, '')" if has_nf else "''"
            nfa_expr = "COALESCE(f.fetched_at, '')" if has_nf else "''"
            conn.execute(
                f"""
                INSERT OR IGNORE INTO news (
                    as_of, doc_id, published_at, title, text, url, domain, depth, source, page, fetched_at
                )
                SELECT
                    i.as_of, i.doc_id, i.published_at, i.title, i.text, i.url, i.domain, i.depth,
                    {src_expr}, {page_expr}, {nfa_expr}
                FROM news_items i
                {nf_join}
                """
            )
            if has_nf:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO news (
                        as_of, doc_id, published_at, title, text, url, domain, depth, source, page, fetched_at
                    )
                    SELECT
                        f.as_of, '', '', '', '', '', '', NULL,
                        COALESCE(f.source, ''), COALESCE(f.page, ''), COALESCE(f.fetched_at, '')
                    FROM news_fetches f
                    LEFT JOIN news_items i ON f.as_of = i.as_of
                    WHERE i.as_of IS NULL
                    """
                )

        # 4. 将 `agent_dataset` + `agent_node` 还原并迁入扁平 `agent_eval` 表
        if "agent_dataset" in existing_tables and "agent_node" in existing_tables:
            ds_rows = conn.execute("SELECT * FROM agent_dataset").fetchall()
            for ds_meta in ds_rows:
                ds_name = str(ds_meta["dataset"])
                doc: dict[str, Any] = {
                    "n": ds_meta["n"],
                    "scored_on_shared": ds_meta["scored_on_shared"],
                    "generated_at": ds_meta["generated_at"],
                }
                node_rows = conn.execute(
                    "SELECT path, kind, value FROM agent_node WHERE dataset = ?",
                    (ds_name,),
                ).fetchall()
                for nr in node_rows:
                    _set_legacy_path(doc, str(nr["path"]), _cast_legacy_node(str(nr["kind"]), nr["value"]))
                agent_rows = _extract_agent_rows_from_doc(ds_name, doc)
                if agent_rows:
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
                        agent_rows,
                    )

        # 5. 移除全部 11+ 张旧配置碎表、重复表与 EAV 节点表
        for obsolete in OBSOLETE_TABLES:
            if obsolete in existing_tables:
                conn.execute(f"DROP TABLE IF EXISTS {obsolete}")


# ------------------------------------------------------------------ 股价序列（bars）


def upsert_series(root: Path, series: Series) -> None:
    from forecast_log.bars import is_session_closed, validate_bars

    validate_bars(series.bars)
    tz_name = series.timezone or timezone_for_symbol(series.symbol)
    fetched_at = series.fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = connect(db_path(root))
    try:
        max_row = conn.execute(
            "SELECT MAX(date) AS max_date, COUNT(*) AS cnt FROM bars WHERE symbol = ?",
            (series.symbol,),
        ).fetchone()
        max_date = max_row["max_date"] if max_row and max_row["cnt"] else None
        unclosed_dates = {
            str(r["date"])
            for r in conn.execute(
                "SELECT date FROM bars WHERE symbol = ? AND is_closed = 0",
                (series.symbol,),
            ).fetchall()
        }
        # 已有历史数据时：插入新日期 + 覆盖尾部最近 5 个交易日 + 强制重刷所有之前未收盘（is_closed=0）的日期
        tail_cutoff = (
            series.bars[max(0, len(series.bars) - 5)].date.isoformat()
            if series.bars
            else max_date
        )
        bars_to_write = (
            [
                b
                for b in series.bars
                if b.date.isoformat() >= max_date
                or b.date.isoformat() >= tail_cutoff
                or b.date.isoformat() in unclosed_dates
                or not b.is_closed
            ]
            if max_date is not None
            else list(series.bars)
        )
        rows = [
            (
                series.symbol,
                bar.date.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                1 if (bar.is_closed and is_session_closed(bar.date, tz_name, fetched_at)) else 0,
                fetched_at,
            )
            for bar in bars_to_write
        ]
        if rows:
            with conn:
                conn.executemany(
                    """
                    INSERT INTO bars (symbol, date, open, high, low, close, volume, is_closed, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, date) DO UPDATE SET
                        open       = excluded.open,
                        high       = excluded.high,
                        low        = excluded.low,
                        close      = excluded.close,
                        volume     = excluded.volume,
                        is_closed  = excluded.is_closed,
                        fetched_at = excluded.fetched_at
                    """,
                    rows,
                )
    finally:
        conn.close()


def get_series(root: Path, symbol: str) -> Series | None:
    from forecast_log.bars import Bar, Series, validate_bars

    path = db_path(root)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        bar_rows = conn.execute(
            """
            SELECT date, open, high, low, close, volume, is_closed, fetched_at
            FROM bars
            WHERE symbol = ?
            ORDER BY date ASC
            """,
            (symbol,),
        ).fetchall()
        if not bar_rows:
            return None
        fetched_at = ""
        for r in reversed(bar_rows):
            if r["fetched_at"]:
                fetched_at = str(r["fetched_at"])
                break
        bars = tuple(
            Bar(
                date=date.fromisoformat(row["date"]),
                close=float(row["close"]),
                open=float(row["open"]) if row["open"] is not None else None,
                high=float(row["high"]) if row["high"] is not None else None,
                low=float(row["low"]) if row["low"] is not None else None,
                volume=int(row["volume"]) if row["volume"] is not None else None,
                is_closed=bool(row["is_closed"]) if row["is_closed"] is not None else True,
            )
            for row in bar_rows
        )
        series = Series(
            symbol=symbol,
            timezone=timezone_for_symbol(symbol),
            fetched_at=fetched_at,
            bars=bars,
        )
        validate_bars(series.bars)
        return series
    finally:
        conn.close()


def has_series_in_db(root: Path, symbol: str) -> bool:
    path = db_path(root)
    if not path.is_file():
        return False
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT 1 FROM bars WHERE symbol = ? LIMIT 1",
            (symbol,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def list_db_symbols(root: Path) -> list[str]:
    path = db_path(root)
    if not path.is_file():
        return []
    conn = connect(path)
    try:
        rows = conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol ASC").fetchall()
        return [str(row["symbol"]) for row in rows]
    finally:
        conn.close()


def unclosed_bars_summary(root: Path) -> dict[str, list[str]]:
    """返回数据库中所有尚未收盘（is_closed = 0）的 K 线：{symbol: [date, ...]}。"""
    path = db_path(root)
    if not path.is_file():
        return {}
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT symbol, date FROM bars WHERE is_closed = 0 ORDER BY symbol ASC, date ASC"
        ).fetchall()
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(str(row["symbol"]), []).append(str(row["date"]))
        return out
    finally:
        conn.close()


def _parse_iso_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def refresh_check_summary(
    root: Path,
    now: datetime | None = None,
    intraday_cooldown_sec: int = 300,
) -> dict[str, object]:
    """直接基于 `bars` 表区分已过收盘的残缺行（overdue_bars）与冷却期外的盘中行（stale_intraday_bars）。"""
    from forecast_log.bars import is_session_closed

    now_dt = now if now is not None else datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    path = db_path(root)
    if not path.is_file():
        return {
            "needs_refresh": True,
            "overdue_bars": {},
            "stale_intraday_bars": {},
            "active_intraday_bars": {},
        }

    conn = connect(path)
    try:
        bar_rows = conn.execute(
            """
            SELECT symbol, date, fetched_at
            FROM bars
            WHERE is_closed = 0
            ORDER BY symbol ASC, date ASC
            """
        ).fetchall()

        overdue_bars: dict[str, list[str]] = {}
        active_intraday_bars: dict[str, list[str]] = {}
        stale_intraday_bars: dict[str, list[str]] = {}

        for r in bar_rows:
            sym = str(r["symbol"])
            d_str = str(r["date"])
            tz_name = timezone_for_symbol(sym)
            if is_session_closed(date.fromisoformat(d_str), tz_name, now_dt):
                overdue_bars.setdefault(sym, []).append(d_str)
            else:
                active_intraday_bars.setdefault(sym, []).append(d_str)
                fetched_dt = _parse_iso_dt(r["fetched_at"])
                age = (now_dt - fetched_dt.astimezone(timezone.utc)).total_seconds() if fetched_dt else 1e9
                if age >= intraday_cooldown_sec:
                    stale_intraday_bars.setdefault(sym, []).append(d_str)

        return {
            "needs_refresh": bool(overdue_bars or stale_intraday_bars),
            "overdue_bars": overdue_bars,
            "stale_intraday_bars": stale_intraday_bars,
            "active_intraday_bars": active_intraday_bars,
        }
    finally:
        conn.close()


# ------------------------------------------------------------------ 新闻存储（news 单表）


def _make_doc_id(as_of: str, index: int, doc: dict) -> str:
    raw_id = (doc.get("id") or "").strip()
    if raw_id:
        return raw_id
    seed = f"{as_of}|{index}|{doc.get('url', '')}|{doc.get('title', '')}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def save_news_fetch(root: Path, record: dict) -> None:
    as_of = str(record["as_of"])
    source = str(record.get("source") or "none")
    page = str(record.get("page") or "")
    fetched_at = str(record.get("fetched_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))

    documents = record.get("documents") or []
    if documents:
        item_rows = [
            (
                as_of,
                _make_doc_id(as_of, idx, doc),
                str(doc.get("published_at") or as_of),
                str(doc.get("title") or ""),
                str(doc.get("text") or doc.get("title") or ""),
                str(doc.get("url") or ""),
                str(doc.get("domain") or ""),
                int(doc["depth"]) if doc.get("depth") is not None else None,
                source,
                page,
                fetched_at,
            )
            for idx, doc in enumerate(documents)
        ]
    else:
        # 当天无新闻时写入 doc_id='' 占位行，避免后续重复抓取该空日
        item_rows = [(as_of, "", "", "", "", "", "", None, source, page, fetched_at)]

    conn = connect(db_path(root))
    try:
        with conn:
            if documents:
                conn.execute("DELETE FROM news WHERE as_of = ? AND doc_id = ''", (as_of,))
            conn.executemany(
                """
                INSERT INTO news (
                    as_of, doc_id, published_at, title, text, url, domain, depth, source, page, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(as_of, doc_id) DO UPDATE SET
                    published_at = excluded.published_at,
                    title        = excluded.title,
                    text         = excluded.text,
                    url          = excluded.url,
                    domain       = excluded.domain,
                    depth        = excluded.depth,
                    source       = excluded.source,
                    page         = excluded.page,
                    fetched_at   = excluded.fetched_at
                """,
                item_rows,
            )
    finally:
        conn.close()


def get_news_fetch(root: Path, day: date) -> dict | None:
    path = db_path(root)
    if not path.is_file():
        return None
    as_of = day.isoformat()
    conn = connect(path)
    try:
        items = conn.execute(
            """
            SELECT doc_id, published_at, title, text, url, domain, depth, source, page, fetched_at
            FROM news
            WHERE as_of = ?
            ORDER BY published_at DESC, rowid ASC
            """,
            (as_of,),
        ).fetchall()
        if not items:
            return None
        first = items[0]
        documents = []
        for row in items:
            if not row["doc_id"]:
                continue
            doc = {
                "id": row["doc_id"],
                "published_at": row["published_at"],
                "title": row["title"],
                "text": row["text"],
                "url": row["url"],
                "domain": row["domain"],
            }
            if row["depth"] is not None:
                doc["depth"] = row["depth"]
            documents.append(doc)
        return {
            "as_of": as_of,
            "source": first["source"] or "none",
            "page": first["page"] or "",
            "providers": [p for p in str(first["source"] or "").split("+") if p and p != "none"],
            "queries": [],
            "errors": [],
            "status": "ok",
            "error": "",
            "fetched_at": first["fetched_at"] or "",
            "documents": documents,
            "cached": True,
        }
    finally:
        conn.close()


def has_news_fetch(root: Path, day: date) -> bool:
    path = db_path(root)
    if not path.is_file():
        return False
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT 1 FROM news WHERE as_of = ? LIMIT 1",
            (day.isoformat(),),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def existing_news_dates(root: Path) -> set[str]:
    path = db_path(root)
    if not path.is_file():
        return set()
    conn = connect(path)
    try:
        rows = conn.execute("SELECT DISTINCT as_of FROM news").fetchall()
        return {str(row["as_of"]) for row in rows}
    finally:
        conn.close()


def news_coverage_stats(root: Path) -> dict[str, int]:
    path = db_path(root)
    if not path.is_file():
        return {}
    conn = connect(path)
    try:
        rows = conn.execute(
            """
            SELECT substr(as_of, 1, 4) AS yr, COUNT(DISTINCT as_of) AS cnt
            FROM news
            WHERE doc_id != ''
            GROUP BY yr
            ORDER BY yr ASC
            """
        ).fetchall()
        return {str(row["yr"]): int(row["cnt"]) for row in rows}
    finally:
        conn.close()


def load_all_news_items(root: Path) -> list[dict]:
    path = db_path(root)
    if not path.is_file():
        return []
    conn = connect(path)
    try:
        rows = conn.execute(
            """
            SELECT as_of, doc_id, published_at, title, text, url, domain, depth
            FROM news
            WHERE doc_id != ''
            ORDER BY as_of ASC, rowid ASC
            """
        ).fetchall()
        out = []
        for row in rows:
            doc = {
                "as_of": row["as_of"],
                "id": row["doc_id"],
                "published_at": row["published_at"],
                "title": row["title"],
                "text": row["text"],
                "url": row["url"],
                "domain": row["domain"],
            }
            if row["depth"] is not None:
                doc["depth"] = row["depth"]
            out.append(doc)
        return out
    finally:
        conn.close()


# ------------------------------------------------------------------ 多模型群聊存储（chat_messages）


def append_chat_message(
    root: Path,
    thread_id: str,
    market: str,
    as_of: str,
    role: str,
    content: str,
    model: str = "",
    stance: str = "",
    created_at: str | None = None,
) -> dict:
    ts = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = connect(db_path(root))
    try:
        with conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM chat_messages WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            seq = int(row["next_seq"]) if row else 1
            conn.execute(
                """
                INSERT INTO chat_messages (
                    thread_id, seq, market, as_of, role, model, stance, content, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (thread_id, seq, market, as_of, role, model, stance, content, ts),
            )
        return {
            "thread_id": thread_id,
            "seq": seq,
            "market": market,
            "as_of": as_of,
            "role": role,
            "model": model,
            "stance": stance,
            "content": content,
            "created_at": ts,
        }
    finally:
        conn.close()


def get_chat_thread(root: Path, thread_id: str) -> list[dict]:
    path = db_path(root)
    if not path.is_file():
        return []
    conn = connect(path)
    try:
        rows = conn.execute(
            """
            SELECT thread_id, seq, market, as_of, role, model, stance, content, created_at
            FROM chat_messages
            WHERE thread_id = ?
            ORDER BY seq ASC
            """,
            (thread_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_chat_threads(root: Path, market: str, limit: int = 20) -> list[dict]:
    path = db_path(root)
    if not path.is_file():
        return []
    conn = connect(path)
    try:
        rows = conn.execute(
            """
            SELECT
                thread_id,
                market,
                MAX(as_of) AS as_of,
                MIN(created_at) AS created_at,
                MAX(created_at) AS updated_at,
                COUNT(*) AS message_count
            FROM chat_messages
            WHERE market = ?
            GROUP BY thread_id, market
            ORDER BY updated_at DESC, thread_id DESC
            LIMIT ?
            """,
            (market, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
