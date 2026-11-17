"""SQLite 存储层：统一管理股价序列（bars）与新闻（news）的持久化。

设计约束：
- 不使用外键（FOREIGN KEY），靠联合主键 + CHECK 约束 + 索引保证数据完整性。
- 开启 WAL 模式，支持 serve.py 多线程刷新与查询并发读写。
- 所有时间序列与新闻读写均通过本模块进行，原 JSON 文件保留作回滚备份。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forecast_log.bars import Series


def db_path(root: Path) -> Path:
    return root / "data" / "forecast.db"


def symbol_slug(symbol: str) -> str:
    slug = symbol.lstrip("^").lower().replace("/", "_")
    if not slug:
        raise ValueError("symbol must not be empty")
    return slug


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
        CREATE TABLE IF NOT EXISTS series_meta (
            symbol      TEXT PRIMARY KEY,
            slug        TEXT NOT NULL,
            timezone    TEXT NOT NULL,
            fetched_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_series_meta_slug
            ON series_meta(slug);

        CREATE TABLE IF NOT EXISTS bars (
            symbol    TEXT NOT NULL,
            date      TEXT NOT NULL,
            open      REAL CHECK (open IS NULL OR open > 0),
            high      REAL CHECK (high IS NULL OR high > 0),
            low       REAL CHECK (low IS NULL OR low > 0),
            close     REAL NOT NULL CHECK (close > 0),
            volume    INTEGER CHECK (volume IS NULL OR volume >= 0),
            is_closed INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (symbol, date),
            CHECK (high IS NULL OR low IS NULL OR low <= high)
        );

        CREATE INDEX IF NOT EXISTS idx_bars_date
            ON bars(date);

        CREATE TABLE IF NOT EXISTS news_fetches (
            as_of           TEXT PRIMARY KEY,
            source          TEXT NOT NULL,
            page            TEXT NOT NULL DEFAULT '',
            providers_json  TEXT NOT NULL DEFAULT '[]',
            queries_json    TEXT NOT NULL DEFAULT '[]',
            errors_json     TEXT NOT NULL DEFAULT '[]',
            status          TEXT NOT NULL DEFAULT 'ok',
            error           TEXT NOT NULL DEFAULT '',
            fetched_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_news_fetches_status
            ON news_fetches(status);

        CREATE TABLE IF NOT EXISTS news_items (
            as_of         TEXT NOT NULL,
            doc_id        TEXT NOT NULL,
            published_at  TEXT NOT NULL,
            title         TEXT NOT NULL,
            text          TEXT NOT NULL,
            url           TEXT NOT NULL DEFAULT '',
            domain        TEXT NOT NULL DEFAULT '',
            depth         INTEGER,
            PRIMARY KEY (as_of, doc_id)
        );

        CREATE INDEX IF NOT EXISTS idx_news_items_published_at
            ON news_items(published_at);

        CREATE INDEX IF NOT EXISTS idx_news_items_domain
            ON news_items(domain);

        CREATE INDEX IF NOT EXISTS idx_news_items_as_of_depth
            ON news_items(as_of, depth);

        CREATE TABLE IF NOT EXISTS market (
            market        TEXT PRIMARY KEY,
            instrument    TEXT NOT NULL,
            name          TEXT,
            horizon       INTEGER,
            question      TEXT,
            rule          TEXT,
            first_session TEXT,
            last_session  TEXT,
            fetched_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS session (
            market          TEXT NOT NULL,
            date            TEXT NOT NULL,
            open            REAL,
            high            REAL,
            low             REAL,
            close           REAL,
            volume          INTEGER,
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
            is_closed       INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (market, date)
        );

        CREATE INDEX IF NOT EXISTS session_by_market_date ON session (market, date);

        CREATE TABLE IF NOT EXISTS factor_key (
            market   TEXT NOT NULL,
            position INTEGER NOT NULL,
            key      TEXT NOT NULL,
            PRIMARY KEY (market, position)
        );

        CREATE TABLE IF NOT EXISTS contexts (
            market         TEXT NOT NULL,
            position       INTEGER NOT NULL,
            key            TEXT NOT NULL,
            label          TEXT,
            kind           TEXT,
            required       INTEGER,
            threshold      REAL,
            group_name     TEXT,
            group_label    TEXT,
            note           TEXT,
            available_from TEXT,
            PRIMARY KEY (market, position)
        );

        CREATE TABLE IF NOT EXISTS market_node (
            market TEXT NOT NULL,
            path   TEXT NOT NULL,
            kind   TEXT NOT NULL,
            value  TEXT,
            PRIMARY KEY (market, path)
        );

        CREATE TABLE IF NOT EXISTS timeline_close (
            symbol TEXT NOT NULL,
            idx    INTEGER NOT NULL,
            date   TEXT NOT NULL,
            close  REAL NOT NULL,
            PRIMARY KEY (symbol, idx)
        );

        CREATE TABLE IF NOT EXISTS timeline_rolling (
            symbol TEXT NOT NULL,
            idx    INTEGER NOT NULL,
            time   TEXT NOT NULL,
            as_of  TEXT,
            anchor REAL,
            median REAL,
            low    REAL,
            high   REAL,
            actual REAL,
            PRIMARY KEY (symbol, idx)
        );

        CREATE TABLE IF NOT EXISTS timeline_daily (
            symbol   TEXT NOT NULL,
            idx      INTEGER NOT NULL,
            time     TEXT NOT NULL,
            median   REAL,
            low      REAL,
            high     REAL,
            actual   REAL,
            previous REAL,
            risk     INTEGER,
            PRIMARY KEY (symbol, idx)
        );

        CREATE TABLE IF NOT EXISTS timeline_ladder (
            symbol  TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            idx     INTEGER NOT NULL,
            median  REAL,
            low     REAL,
            high    REAL,
            PRIMARY KEY (symbol, horizon, idx)
        );

        CREATE TABLE IF NOT EXISTS timeline_node (
            path  TEXT NOT NULL PRIMARY KEY,
            kind  TEXT NOT NULL,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_dataset (
            dataset          TEXT PRIMARY KEY,
            n                INTEGER,
            scored_on_shared INTEGER,
            generated_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_node (
            dataset TEXT NOT NULL,
            path    TEXT NOT NULL,
            kind    TEXT NOT NULL,
            value   TEXT,
            PRIMARY KEY (dataset, path)
        );
        """
    )
    _migrate_is_closed(conn)


def _migrate_is_closed(conn: sqlite3.Connection) -> None:
    """为旧版数据库平滑补充 is_closed 列，并根据最后抓取时间戳校正尾部未收盘行。"""
    from forecast_log.bars import is_session_closed

    bar_cols = {row["name"] for row in conn.execute("PRAGMA table_info(bars)").fetchall()}
    session_cols = {row["name"] for row in conn.execute("PRAGMA table_info(session)").fetchall()}
    added = False
    if "is_closed" not in bar_cols:
        conn.execute("ALTER TABLE bars ADD COLUMN is_closed INTEGER NOT NULL DEFAULT 1")
        added = True
    if "is_closed" not in session_cols:
        conn.execute("ALTER TABLE session ADD COLUMN is_closed INTEGER NOT NULL DEFAULT 1")
        added = True

    if added:
        with conn:
            metas = conn.execute("SELECT symbol, timezone, fetched_at FROM series_meta").fetchall()
            for meta in metas:
                sym = meta["symbol"]
                tz_name = meta["timezone"] or ("Asia/Shanghai" if sym.endswith((".SS", ".SZ")) else "America/New_York")
                fetched_at = meta["fetched_at"]
                tail_bars = conn.execute(
                    "SELECT date FROM bars WHERE symbol = ? ORDER BY date DESC LIMIT 5",
                    (sym,),
                ).fetchall()
                for row in tail_bars:
                    closed = 1 if is_session_closed(date.fromisoformat(row["date"]), tz_name, fetched_at) else 0
                    if not closed:
                        conn.execute(
                            "UPDATE bars SET is_closed = 0 WHERE symbol = ? AND date = ?",
                            (sym, row["date"]),
                        )

            markets = conn.execute("SELECT market, instrument, fetched_at FROM market").fetchall()
            for mkt in markets:
                market_key = mkt["market"]
                inst = mkt["instrument"] or ""
                tz_name = "Asia/Shanghai" if (market_key == "ssec" or inst.endswith((".SS", ".SZ"))) else "America/New_York"
                fetched_at = mkt["fetched_at"]
                tail_sessions = conn.execute(
                    "SELECT date FROM session WHERE market = ? ORDER BY date DESC LIMIT 5",
                    (market_key,),
                ).fetchall()
                for row in tail_sessions:
                    closed = 1 if is_session_closed(date.fromisoformat(row["date"]), tz_name, fetched_at) else 0
                    if not closed:
                        conn.execute(
                            "UPDATE session SET is_closed = 0 WHERE market = ? AND date = ?",
                            (market_key, row["date"]),
                        )


# ------------------------------------------------------------------ 股价序列


def upsert_series(root: Path, series: Series) -> None:
    from forecast_log.bars import is_session_closed, validate_bars

    validate_bars(series.bars)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    slug = symbol_slug(series.symbol)
    tz_name = series.timezone or ("Asia/Shanghai" if series.symbol.endswith((".SS", ".SZ")) else "America/New_York")
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
                1 if (bar.is_closed and is_session_closed(bar.date, tz_name, series.fetched_at)) else 0,
            )
            for bar in bars_to_write
        ]
        with conn:
            conn.execute(
                """
                INSERT INTO series_meta (symbol, slug, timezone, fetched_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    slug       = excluded.slug,
                    timezone   = excluded.timezone,
                    fetched_at = excluded.fetched_at,
                    updated_at = excluded.updated_at
                """,
                (series.symbol, slug, tz_name, series.fetched_at, now_iso),
            )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO bars (symbol, date, open, high, low, close, volume, is_closed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, date) DO UPDATE SET
                        open      = excluded.open,
                        high      = excluded.high,
                        low       = excluded.low,
                        close     = excluded.close,
                        volume    = excluded.volume,
                        is_closed = excluded.is_closed
                    """,
                    rows,
                )
    finally:
        conn.close()


def _row_to_series(conn: sqlite3.Connection, meta: sqlite3.Row) -> Series:
    from forecast_log.bars import Bar, Series, validate_bars

    symbol = meta["symbol"]
    bar_rows = conn.execute(
        """
        SELECT date, open, high, low, close, volume, is_closed
        FROM bars
        WHERE symbol = ?
        ORDER BY date ASC
        """,
        (symbol,),
    ).fetchall()
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
        timezone=meta["timezone"],
        fetched_at=meta["fetched_at"],
        bars=bars,
    )
    validate_bars(series.bars)
    return series


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


def get_series(root: Path, symbol: str) -> Series | None:
    path = db_path(root)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        meta = conn.execute(
            "SELECT symbol, timezone, fetched_at FROM series_meta WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        if meta is None:
            return None
        return _row_to_series(conn, meta)
    finally:
        conn.close()


def get_series_by_slug(root: Path, slug: str) -> Series | None:
    path = db_path(root)
    if not path.is_file():
        return None
    conn = connect(path)
    try:
        meta = conn.execute(
            "SELECT symbol, timezone, fetched_at FROM series_meta WHERE slug = ?",
            (slug,),
        ).fetchone()
        if meta is None:
            return None
        return _row_to_series(conn, meta)
    finally:
        conn.close()


def has_series_in_db(root: Path, symbol: str) -> bool:
    path = db_path(root)
    if not path.is_file():
        return False
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT 1 FROM series_meta WHERE symbol = ? OR slug = ? LIMIT 1",
            (symbol, symbol_slug(symbol)),
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
        rows = conn.execute("SELECT symbol FROM series_meta ORDER BY symbol ASC").fetchall()
        return [str(row["symbol"]) for row in rows]
    finally:
        conn.close()


# ------------------------------------------------------------------ 新闻存储


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
    providers_json = json.dumps(record.get("providers") or [], ensure_ascii=False)
    queries_json = json.dumps(record.get("queries") or [], ensure_ascii=False)
    errors_json = json.dumps(record.get("errors") or [], ensure_ascii=False)
    status = str(record.get("status") or "ok")
    error = str(record.get("error") or "")
    fetched_at = str(record.get("fetched_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))

    documents = record.get("documents") or []
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
        )
        for idx, doc in enumerate(documents)
    ]

    conn = connect(db_path(root))
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO news_fetches (
                    as_of, source, page, providers_json, queries_json,
                    errors_json, status, error, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(as_of) DO UPDATE SET
                    source         = excluded.source,
                    page           = excluded.page,
                    providers_json = excluded.providers_json,
                    queries_json   = excluded.queries_json,
                    errors_json    = excluded.errors_json,
                    status         = excluded.status,
                    error          = excluded.error,
                    fetched_at     = excluded.fetched_at
                """,
                (as_of, source, page, providers_json, queries_json, errors_json, status, error, fetched_at),
            )
            conn.executemany(
                """
                INSERT INTO news_items (
                    as_of, doc_id, published_at, title, text, url, domain, depth
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(as_of, doc_id) DO UPDATE SET
                    published_at = excluded.published_at,
                    title        = excluded.title,
                    text         = excluded.text,
                    url          = excluded.url,
                    domain       = excluded.domain,
                    depth        = excluded.depth
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
        meta = conn.execute(
            "SELECT * FROM news_fetches WHERE as_of = ?",
            (as_of,),
        ).fetchone()
        if meta is None:
            return None
        items = conn.execute(
            """
            SELECT doc_id, published_at, title, text, url, domain, depth
            FROM news_items
            WHERE as_of = ?
            ORDER BY published_at DESC, rowid ASC
            """,
            (as_of,),
        ).fetchall()
        documents = []
        for row in items:
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
            "as_of": meta["as_of"],
            "source": meta["source"],
            "page": meta["page"],
            "providers": json.loads(meta["providers_json"]),
            "queries": json.loads(meta["queries_json"]),
            "errors": json.loads(meta["errors_json"]),
            "status": meta["status"],
            "error": meta["error"],
            "fetched_at": meta["fetched_at"],
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
            "SELECT 1 FROM news_fetches WHERE as_of = ? LIMIT 1",
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
        rows = conn.execute("SELECT as_of FROM news_fetches").fetchall()
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
            FROM news_items
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
            FROM news_items
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
