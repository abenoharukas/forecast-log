"""Sync local SQLite (`data/forecast.db`) into Supabase PostgreSQL via `psql` COPY.

Uses standard library Python + `psql` (`COPY ... FROM STDIN` into a transaction-scoped
`TEMP TABLE ... ON COMMIT DROP`, followed by `INSERT ... ON CONFLICT DO UPDATE`), so:
  - Zero extra Python dependencies (no psycopg install needed)
  - Works with both Supabase Session/Direct Pooler (5432) and Transaction Pooler (6543)
  - Full 250k-row seed completes in seconds and is completely idempotent

Usage:
    python3 scripts/sync_supabase.py --schema            # apply deploy/schema.pg.sql + incremental sync
    python3 scripts/sync_supabase.py --schema --full     # apply schema + full upsert of all 5 tables
    python3 scripts/sync_supabase.py --schema-only       # only apply deploy/schema.pg.sql
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "forecast.db"
PG_SCHEMA = ROOT / "deploy" / "schema.pg.sql"

MARKETS = {"ndx": "^NDX", "qqq": "QQQ", "ssec": "000001.SS"}

TABLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "bars": (
        ("symbol", "date", "open", "high", "low", "close", "volume", "is_closed", "fetched_at"),
        ("symbol", "date"),
    ),
    "signals": (
        (
            "market",
            "date",
            "entry",
            "entry_at",
            "base_close",
            "close3",
            "close5",
            "hit3",
            "flag",
            "flag_price_only",
            "sell_flag",
            "sell_threshold",
            "news_veto",
            "score",
            "threshold",
            "outcome",
            "target",
            "peak",
            "peak_at",
            "us_date",
            "factors_packed",
            "fetched_at",
        ),
        ("market", "date"),
    ),
    "news": (
        (
            "as_of",
            "doc_id",
            "published_at",
            "title",
            "text",
            "url",
            "domain",
            "depth",
            "source",
            "page",
            "fetched_at",
        ),
        ("as_of", "doc_id"),
    ),
    "agent_eval": (
        (
            "dataset",
            "question_id",
            "predictor",
            "symbol",
            "as_of",
            "as_of_close",
            "resolve_session",
            "horizon",
            "family",
            "threshold",
            "question_text",
            "outcome",
            "probability",
            "cost_usd_total",
            "cost_usd_per_forecast",
            "latency_p50_s",
            "latency_p95_s",
            "tool_calls_mean",
            "gate_hits",
            "failures",
            "generated_at",
            "config_model",
            "models_merged",
        ),
        ("dataset", "question_id", "predictor"),
    ),
    "chat_messages": (
        ("thread_id", "seq", "market", "as_of", "role", "model", "stance", "content", "created_at"),
        ("thread_id", "seq"),
    ),
}

NULL_TOKEN = "__FORECAST_LOG_NULL__"


def _read_env_key(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip("\"'")
    return ""


def resolve_db_url(cli_url: str) -> str:
    if cli_url:
        return cli_url.strip()
    for key in ("SUPABASE_DIRECT_URL", "SUPABASE_DB_URL", "DATABASE_URL"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    for env_file in (ROOT / ".env", ROOT / ".dev.vars"):
        for key in ("SUPABASE_DIRECT_URL", "SUPABASE_DB_URL", "DATABASE_URL"):
            val = _read_env_key(env_file, key)
            if val:
                return val
    return ""


def redact_url(url: str) -> str:
    return re.sub(r"://([^:@/]+):([^@/]*)@", r"://\1:***@", url)


def ensure_sslmode(url: str) -> str:
    if re.search(r"@(localhost|127\.0\.0\.1|\[::1\])[:/]", url, re.I):
        return url
    if "sslmode=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sslmode=require"


def psql_bin() -> str:
    path = shutil.which("psql")
    if not path:
        raise SystemExit("缺少 psql 命令（macOS 可运行：brew install libpq 或 brew install postgresql）")
    return path


def run_psql(url: str, *, sql: str | None = None, file_path: Path | None = None, stdin_bytes: bytes | None = None) -> str:
    cmd = [psql_bin(), ensure_sslmode(url), "-X", "-q", "-v", "ON_ERROR_STOP=1"]
    if file_path is not None:
        cmd.extend(["-f", str(file_path)])
    elif sql is not None:
        cmd.extend(["-At", "-F", "\t", "-c", sql])
    env = dict(os.environ)
    env.setdefault("PGCONNECT_TIMEOUT", "20")
    proc = subprocess.run(
        cmd,
        input=stdin_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"psql 执行失败（exit {proc.returncode}）：\n{err}")
    return proc.stdout.decode("utf-8", errors="replace")


def apply_schema(url: str) -> None:
    if not PG_SCHEMA.is_file():
        raise SystemExit(f"缺少建表 SQL：{PG_SCHEMA}")
    print(f"应用 Postgres 表结构 ({PG_SCHEMA.relative_to(ROOT)}) ...", flush=True)
    run_psql(url, file_path=PG_SCHEMA)
    print("表结构就绪 ✓", flush=True)


def query_rows(url: str, sql: str) -> list[list[str]]:
    out = run_psql(url, sql=sql)
    rows: list[list[str]] = []
    for line in out.splitlines():
        if line == "":
            continue
        rows.append(line.split("\t"))
    return rows


def remote_state(url: str) -> dict:
    bars = {}
    for row in query_rows(url, "SELECT symbol, COUNT(*), COALESCE(MAX(date), '') FROM bars GROUP BY symbol"):
        bars[row[0]] = {"count": int(row[1]), "max_date": row[2] or None}

    markets = {}
    for row in query_rows(url, "SELECT market, COUNT(*), COALESCE(MAX(date), '') FROM signals GROUP BY market"):
        markets[row[0]] = {"count": int(row[1]), "max_date": row[2] or None}

    news_rows = query_rows(url, "SELECT COUNT(*), COALESCE(MAX(as_of), '') FROM news")
    news = {"count": int(news_rows[0][0]), "max_as_of": news_rows[0][1] or None} if news_rows else {"count": 0, "max_as_of": None}

    agents = {}
    for row in query_rows(url, "SELECT dataset, COUNT(*), COALESCE(MAX(generated_at), '') FROM agent_eval GROUP BY dataset"):
        agents[row[0]] = {"count": int(row[1]), "generated_at": row[2] or ""}

    chat_rows = query_rows(url, "SELECT COUNT(*), COALESCE(MAX(created_at), '') FROM chat_messages")
    chat = {"count": int(chat_rows[0][0]), "max_created_at": chat_rows[0][1] or None} if chat_rows else {"count": 0, "max_created_at": None}

    return {"bars": bars, "markets": markets, "news": news, "agents": agents, "chat": chat}


def _csv_cell(value) -> str:
    if value is None:
        return NULL_TOKEN
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if "\x00" in text:
        text = text.replace("\x00", "")
    return text


def upsert_batch(url: str, table: str, columns: tuple[str, ...], pk: tuple[str, ...], rows: list[sqlite3.Row]) -> None:
    col_list = ", ".join(columns)
    pk_list = ", ".join(pk)
    non_pk = [c for c in columns if c not in pk]
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk)

    buf = io.StringIO()
    buf.write("BEGIN;\n")
    buf.write(f"CREATE TEMP TABLE _stage (LIKE {table}) ON COMMIT DROP;\n")
    buf.write(f"COPY _stage ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '{NULL_TOKEN}');\n")
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        writer.writerow([_csv_cell(row[col]) for col in columns])
    buf.write("\\.\n")
    buf.write(
        f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _stage "
        f"ON CONFLICT ({pk_list}) DO UPDATE SET {update_clause};\n"
    )
    buf.write("COMMIT;\n")

    run_psql(url, stdin_bytes=buf.getvalue().encode("utf-8"))


def push_table(
    conn: sqlite3.Connection,
    url: str,
    table: str,
    where: str,
    params: tuple,
    batch_size: int,
    label: str | None = None,
) -> int:
    columns, pk = TABLES[table]
    sql = f"SELECT {', '.join(columns)} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    cursor = conn.execute(sql, params)
    total = 0
    tag = label or table
    while True:
        chunk = cursor.fetchmany(batch_size)
        if not chunk:
            break
        upsert_batch(url, table, columns, pk, chunk)
        total += len(chunk)
        print(f"\r  {tag}: {total} 行", end="", flush=True)
    if total:
        print(f"\r  {tag}: {total} 行 ✓", flush=True)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local SQLite (data/forecast.db) into Supabase PostgreSQL")
    parser.add_argument("--url", default="", help="Postgres connection URL (default: SUPABASE_DIRECT_URL / SUPABASE_DB_URL)")
    parser.add_argument("--schema", action="store_true", help="Apply deploy/schema.pg.sql before syncing")
    parser.add_argument("--schema-only", action="store_true", help="Only apply deploy/schema.pg.sql and exit")
    parser.add_argument("--full", action="store_true", help="Full upsert of all 5 tables (default: incremental if bars non-empty)")
    parser.add_argument("--truncate", action="store_true", help="Truncate all 5 tables before full load (implies --full)")
    parser.add_argument("--skip-news", action="store_true", help="Skip syncing news table")
    parser.add_argument("--batch", type=int, default=15_000, help="Rows per COPY transaction (default: 15000)")
    args = parser.parse_args()

    url = resolve_db_url(args.url)
    if not url:
        raise SystemExit("缺少数据库连接串：请传入 --url 或在 .env 中设置 SUPABASE_DB_URL / SUPABASE_DIRECT_URL")

    print(f"目标 Postgres: {redact_url(url)}", flush=True)

    if args.schema or args.schema_only:
        apply_schema(url)
    if args.schema_only:
        return

    if not DB.is_file():
        raise SystemExit(f"缺少本地数据库：{DB}（可先运行 make gh-db-pull）")

    state = remote_state(url)
    remote_bar_total = sum(v["count"] for v in state["bars"].values())
    full_mode = args.full or args.truncate or remote_bar_total == 0

    if args.truncate:
        print("清空远端 5 张表 (TRUNCATE) ...", flush=True)
        run_psql(url, sql="TRUNCATE TABLE bars, signals, news, agent_eval, chat_messages;")
        state = {"bars": {}, "markets": {}, "news": {"count": 0, "max_as_of": None}, "agents": {}, "chat": {"count": 0, "max_created_at": None}}

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    try:
        if full_mode:
            print("模式：全量同步 (COPY + UPSERT)", flush=True)
            push_table(conn, url, "bars", "", (), args.batch)
            push_table(conn, url, "signals", "", (), args.batch)
            if not args.skip_news:
                push_table(conn, url, "news", "", (), min(args.batch, 10_000))
            push_table(conn, url, "agent_eval", "", (), args.batch)
            push_table(conn, url, "chat_messages", "", (), args.batch)
        else:
            print(f"模式：增量同步（远端已有 {remote_bar_total} 行 bars）", flush=True)
            remote_bars = state["bars"]
            remote_markets = state["markets"]
            remote_news = state["news"]
            remote_agents = state["agents"]
            remote_chat = state["chat"]

            # 1. bars: tail 20 days per symbol (or full if new symbol)
            for (symbol,) in conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol"):
                remote_max = (remote_bars.get(symbol) or {}).get("max_date")
                if remote_max:
                    cutoff = (date.fromisoformat(remote_max) - timedelta(days=20)).isoformat()
                    push_table(conn, url, "bars", "symbol = ? AND date >= ?", (symbol, cutoff), args.batch, label=f"bars[{symbol}]")
                else:
                    push_table(conn, url, "bars", "symbol = ?", (symbol,), args.batch, label=f"bars[{symbol}]")

            # 2. signals: tail 20 days + unresolved rows
            for market in MARKETS:
                remote_max = (remote_markets.get(market) or {}).get("max_date")
                if remote_max:
                    cutoff = (date.fromisoformat(remote_max) - timedelta(days=20)).isoformat()
                    push_table(
                        conn,
                        url,
                        "signals",
                        "market = ? AND (date >= ? OR outcome IS NULL)",
                        (market, cutoff),
                        args.batch,
                        label=f"signals[{market}]",
                    )
                else:
                    push_table(conn, url, "signals", "market = ?", (market,), args.batch, label=f"signals[{market}]")

            # 3. news: full if empty, else last 3 days
            if not args.skip_news:
                remote_as_of = remote_news.get("max_as_of")
                if not remote_as_of and not remote_news.get("count"):
                    push_table(conn, url, "news", "", (), min(args.batch, 10_000))
                else:
                    cutoff = (date.fromisoformat(remote_as_of) - timedelta(days=2)).isoformat() if remote_as_of else "1900-01-01"
                    push_table(conn, url, "news", "as_of >= ?", (cutoff,), min(args.batch, 10_000))

            # 4. agent_eval: resend dataset when count or generated_at changed
            for dataset in ("deepseek", "qwen"):
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt, MAX(generated_at) AS generated_at FROM agent_eval WHERE dataset = ?",
                    (dataset,),
                ).fetchone()
                if not row or not row["cnt"]:
                    continue
                remote = remote_agents.get(dataset) or {}
                if remote.get("count") == row["cnt"] and remote.get("generated_at") == (row["generated_at"] or ""):
                    print(f"  agent_eval[{dataset}]: 已是最新，跳过", flush=True)
                    continue
                push_table(conn, url, "agent_eval", "dataset = ?", (dataset,), args.batch, label=f"agent_eval[{dataset}]")

            # 5. chat_messages: resend after watermark
            local_chat = conn.execute(
                "SELECT COUNT(*) AS cnt, MAX(created_at) AS max_created_at FROM chat_messages"
            ).fetchone()
            if local_chat and local_chat["cnt"] and (
                remote_chat.get("count") != local_chat["cnt"]
                or (remote_chat.get("max_created_at") or "") != (local_chat["max_created_at"] or "")
            ):
                cutoff = remote_chat.get("max_created_at") or "1900-01-01"
                push_table(conn, url, "chat_messages", "created_at >= ?", (cutoff,), args.batch)
    finally:
        conn.close()

    final = remote_state(url)
    print(
        "Supabase 同步完成 ✓  "
        f"bars={sum(v['count'] for v in final['bars'].values())} "
        f"signals={sum(v['count'] for v in final['markets'].values())} "
        f"news={final['news']['count']} "
        f"agent_eval={sum(v['count'] for v in final['agents'].values())} "
        f"chat={final['chat']['count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
