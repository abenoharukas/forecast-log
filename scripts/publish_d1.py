"""Push new/changed rows from local SQLite (`data/forecast.db`) into Cloudflare D1.

The deployed database uses the same five tables as the local one:

    bars / signals / news / agent_eval / chat_messages

The first load is a full seed (`scripts/deploy_cf.sh` → `wrangler d1 execute --file`).
This script is the incremental path used afterwards: it asks the Worker for D1's water
marks (`/api/state`) and only uploads the tail that can have changed — recent bars,
unsettled signals, the newest news days, agent reports whose `generated_at` moved, and
new chat messages. Completed history is never re-sent.

    make publish                       # against local `wrangler dev`
    python3 scripts/publish_d1.py --url https://forecast.lumixraku.org --token ...
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "forecast.db"

MARKETS = {"ndx": "^NDX", "qqq": "QQQ", "ssec": "000001.SS"}

SIGNAL_COLUMNS = (
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
)
BAR_COLUMNS = ("symbol", "date", "open", "high", "low", "close", "volume", "is_closed", "fetched_at")
NEWS_COLUMNS = (
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
)
AGENT_COLUMNS = (
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
)
CHAT_COLUMNS = ("thread_id", "seq", "market", "as_of", "role", "model", "stance", "content", "created_at")

OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": os.environ["INGEST_PROXY"], "https": os.environ["INGEST_PROXY"]})
    if os.environ.get("INGEST_PROXY")
    else urllib.request.ProxyHandler({})
)


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def sql_for(table: str, columns: tuple[str, ...], rows: list[sqlite3.Row]) -> str:
    values = ",".join(
        "(" + ",".join(literal(row[column]) for column in columns) + ")" for row in rows
    )
    return f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES {values}"


def chunks(seq, size):
    for index in range(0, len(seq), size):
        yield seq[index : index + size]


def post(url: str, token: str, payload: dict, retries: int = 5) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    delay = 1.5
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url.rstrip("/") + "/api/ingest",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "forecast-publish/1.0",
            },
        )
        try:
            with OPENER.open(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            if not body.get("ok"):
                raise RuntimeError(f"写入失败：{body}")
            return body
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}")
            if exc.code in (401, 403):
                break
        except Exception as exc:
            last_err = exc
        if attempt < retries:
            time.sleep(delay)
            delay *= 1.8
    raise SystemExit(f"请求 /api/ingest 失败：{last_err}")


def get_state(url: str, token: str, retries: int = 6) -> dict:
    delay = 1.5
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url.rstrip("/") + "/api/state",
                method="GET",
                headers={"User-Agent": "forecast-publish/1.0"},
            )
            with OPENER.open(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
            if body.get("ok"):
                return body
            last_err = RuntimeError(f"/api/state 返回异常: {body}")
        except Exception as exc:
            last_err = exc
        if attempt < retries:
            time.sleep(delay)
            delay *= 1.6
    raise SystemExit(f"无法读取 D1 状态：{last_err}")


def push(conn: sqlite3.Connection, url: str, token: str, table: str, columns: tuple[str, ...], where: str, params: tuple, batch: int) -> int:
    cursor = conn.execute(f"SELECT {','.join(columns)} FROM {table} WHERE {where}", params)
    rows = cursor.fetchall()
    if not rows:
        return 0
    done = 0
    for part in chunks(rows, batch):
        post(url, token, {"op": "sql", "sql": sql_for(table, columns, part)})
        done += len(part)
        print(f"\r  {table}: {done}/{len(rows)}", end="", flush=True)
    print(f"\r  {table}: {done}/{len(rows)} ✓", flush=True)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally push local SQLite deltas into Cloudflare D1")
    parser.add_argument("--url", default=os.environ.get("WORKER_URL", "http://127.0.0.1:8787"))
    parser.add_argument("--token", default=os.environ.get("INGEST_TOKEN", ""))
    parser.add_argument("--batch", type=int, default=400, help="rows per INSERT statement")
    parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("缺少 INGEST_TOKEN（本地放 .dev.vars，线上用 wrangler secret put）")
    if not DB.is_file():
        raise SystemExit("缺少 data/forecast.db")

    state = get_state(args.url, args.token)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    try:
        remote_bars = state.get("bars") or {}
        remote_markets = state.get("markets") or {}
        remote_news = state.get("news") or {}
        remote_agents = state.get("agents") or {}
        remote_chat = state.get("chat") or {}

        # 1. bars: 每个品种只回传尾部 15 个交易日（覆盖 is_closed / OHLC 盘中修订）
        for symbol in [row[0] for row in conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol")]:
            remote_max = (remote_bars.get(symbol) or {}).get("max_date")
            if remote_max:
                cutoff = (date.fromisoformat(remote_max) - timedelta(days=20)).isoformat()
                push(conn, args.url, args.token, "bars", BAR_COLUMNS, "symbol = ? AND date >= ?", (symbol, cutoff), args.batch)
            else:
                push(conn, args.url, args.token, "bars", BAR_COLUMNS, "symbol = ?", (symbol,), args.batch)

        # 2. signals: 尾部 15 天 + 尚未结算的行
        for market in MARKETS:
            remote_max = (remote_markets.get(market) or {}).get("max_date")
            if remote_max:
                cutoff = (date.fromisoformat(remote_max) - timedelta(days=20)).isoformat()
                push(
                    conn,
                    args.url,
                    args.token,
                    "signals",
                    SIGNAL_COLUMNS,
                    "market = ? AND (date >= ? OR outcome IS NULL)",
                    (market, cutoff),
                    args.batch,
                )
            else:
                push(conn, args.url, args.token, "signals", SIGNAL_COLUMNS, "market = ?", (market,), args.batch)

        # 3. news: 最新一天整体重发（同一天可能补抓了文档）
        if not args.skip_news:
            remote_as_of = remote_news.get("max_as_of")
            if not remote_as_of and not remote_news.get("cnt"):
                print("news: D1 还是空的，请用 make cf 做一次全量 seed（跳过增量）", flush=True)
            else:
                cutoff = (date.fromisoformat(remote_as_of) - timedelta(days=2)).isoformat() if remote_as_of else "1900-01-01"
                push(conn, args.url, args.token, "news", NEWS_COLUMNS, "as_of >= ?", (cutoff,), max(args.batch, 100))

        # 4. agent_eval: generated_at 变了就整份重发（每份约 1 千行）
        for dataset in ("deepseek", "qwen"):
            row = conn.execute(
                "SELECT COUNT(*) AS cnt, MAX(generated_at) AS generated_at FROM agent_eval WHERE dataset = ?",
                (dataset,),
            ).fetchone()
            if not row or not row["cnt"]:
                continue
            remote = remote_agents.get(dataset) or {}
            if remote.get("count") == row["cnt"] and remote.get("generated_at") == row["generated_at"]:
                print(f"agent_eval/{dataset}: 已是最新，跳过", flush=True)
                continue
            push(
                conn,
                args.url,
                args.token,
                "agent_eval",
                AGENT_COLUMNS,
                "dataset = ?",
                (dataset,),
                max(args.batch, 100),
            )

        # 5. chat_messages: 量很小，水位线之后全发
        local_chat = conn.execute(
            "SELECT COUNT(*) AS cnt, MAX(created_at) AS max_created_at FROM chat_messages"
        ).fetchone()
        if local_chat and local_chat["cnt"] and (
            remote_chat.get("cnt") != local_chat["cnt"]
            or remote_chat.get("max_created_at") != local_chat["max_created_at"]
        ):
            cutoff = remote_chat.get("max_created_at") or "1900-01-01"
            push(conn, args.url, args.token, "chat_messages", CHAT_COLUMNS, "created_at >= ?", (cutoff,), 100)
    finally:
        conn.close()

    print("D1 增量同步完成", flush=True)
    print("提示：/api/timeline 与 cross 元数据随静态资源发布；数据变化后请跑 make cf（或至少 make static 后 wrangler deploy）", flush=True)


if __name__ == "__main__":
    main()
