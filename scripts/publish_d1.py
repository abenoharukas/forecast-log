"""Push dashboard data from local SQLite (`data/forecast.db`) into Cloudflare D1.

Strictly incremental and append-only:
- Queries D1's current state (`/api/state`) to inspect existing row counts and max dates/indices.
- Never truncates or deletes existing tables (`DELETE FROM` is never used).
- Completed historical trading days are never re-uploaded; only new trading days (or newly
  settled tail outcomes) are inserted via `INSERT OR IGNORE` / `INSERT OR REPLACE`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast_log.artifacts_db import (  # noqa: E402
    SESSION_COLUMNS,
    build_agent_bundle,
    build_cross_bundle,
    build_timeline_bundle,
    load_agent_eval,
    load_cross_market,
    load_timeline,
)


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": os.environ["INGEST_PROXY"], "https": os.environ["INGEST_PROXY"]})
    if os.environ.get("INGEST_PROXY")
    else urllib.request.ProxyHandler({})
)


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


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def get_state(url: str, token: str, retries: int = 6) -> dict:
    """获取 D1 当前各表水位线；刚部署完 Worker 时自动重试，绝不静默返回空状态导致全量重传。"""
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
            if body.get("ok") and isinstance(body.get("markets"), dict):
                return body
            last_err = RuntimeError(f"/api/state 返回异常: {body}")
        except Exception as exc:
            last_err = exc
        if attempt < retries:
            time.sleep(delay)
            delay *= 1.6
    # 兜底通过 /api/ingest 的 state 操作获取
    return post(url, token, {"op": "state"})


def push_cross(url: str, token: str, market: str, doc: dict, batch: int, remote_info: dict | None = None) -> None:
    bundle = build_cross_bundle(doc, market)
    all_rows = bundle["sessions"]
    remote_info = remote_info or {}
    remote_count = remote_info.get("count") or 0
    max_date = remote_info.get("max_date")
    first_unresolved = remote_info.get("first_unresolved")
    remote_node_count = remote_info.get("node_count") or 0

    local_last_date = all_rows[-1]["date"] if all_rows else None

    if remote_count >= len(all_rows) and max_date == local_last_date and remote_node_count >= len(bundle["nodes"]):
        print(f"cross/{market}: D1 已是最新（{remote_count} 个交易日，最新 {max_date}），跳过写入", flush=True)
        return

    if remote_count > 0 and max_date:
        # 只插入比 D1 最新日期更新的交易日，以及之前未结算（outcome IS NULL）但本地现已结算的尾部行
        rows = [
            r
            for r in all_rows
            if r["date"] > max_date
            or (first_unresolved and r["date"] >= first_unresolved and r.get("outcome") is not None)
        ]
    else:
        rows = all_rows

    columns = ("market", "date", *SESSION_COLUMNS, "factors_packed")
    if rows:
        step = max(batch, 300)
        done = 0
        total = len(rows)
        for part in chunks(rows, step):
            values = ",".join(
                "(" + literal(market) + "," + ",".join(literal(row.get(column)) for column in columns[1:]) + ")"
                for row in part
            )
            post(url, token, {"op": "sql", "sql": f"INSERT OR REPLACE INTO session ({','.join(columns)}) VALUES {values}"})
            done += len(part)
            print(f"\r  cross/{market}: {done}/{total}", end="", flush=True)
        print(f"\r  cross/{market}: {done}/{total} ✓", flush=True)

    row = bundle["row"]
    post(
        url,
        token,
        {
            "op": "sql",
            "sql": "INSERT OR REPLACE INTO market ({}) VALUES ({})".format(
                ",".join(row.keys()), ",".join(literal(v) for v in row.values())
            ),
        },
    )
    for part in chunks(bundle["nodes"], 200):
        post(
            url,
            token,
            {
                "op": "sql",
                "sql": "INSERT OR REPLACE INTO market_node (market, path, kind, value) VALUES "
                + ",".join(
                    f"({literal(market)},{literal(path)},{literal(kind)},{literal(value)})"
                    for path, kind, value in part
                ),
            },
        )
    post(
        url,
        token,
        {
            "op": "sql",
            "sql": "INSERT OR REPLACE INTO contexts (market, position, key, label, kind, required, threshold, group_name, group_label, note, available_from) VALUES "
            + ",".join(
                "("
                + ",".join(
                    literal(ctx[field])
                    for field in (
                        "market",
                        "position",
                        "key",
                        "label",
                        "kind",
                        "required",
                        "threshold",
                        "group_name",
                        "group_label",
                        "note",
                        "available_from",
                    )
                )
                + ")"
                for ctx in ({**ctx, "market": market} for ctx in bundle["contexts"])
            ),
        },
    )
    post(
        url,
        token,
        {
            "op": "sql",
            "sql": "INSERT OR REPLACE INTO factor_key (market, position, key) VALUES "
            + ",".join(f"({literal(market)},{position},{literal(key)})" for position, key in bundle["factor_keys"]),
        },
    )
    print(f"cross/{market}: 增量写入 {len(rows)} 个交易日（D1 共 {max(remote_count, len(all_rows))} 天）", flush=True)


def push_timeline(url: str, token: str, doc: dict, batch: int, remote_tl: dict | None = None) -> None:
    bundle = build_timeline_bundle(doc)
    remote_tl = remote_tl or {}

    close_max = remote_tl.get("close") or {}
    rolling_max = remote_tl.get("rolling") or {}
    daily_max = remote_tl.get("daily") or {}
    ladder_max = remote_tl.get("ladder") or {}
    remote_node_count = remote_tl.get("node_count") or 0

    closes = [r for r in bundle["closes"] if r[1] > close_max.get(r[0], -1)]
    rolling = [r for r in bundle["rolling"] if r[1] > rolling_max.get(r[0], -1)]
    daily = [r for r in bundle["daily"] if r[1] > daily_max.get(r[0], -1)]
    ladder = [r for r in bundle["ladder"] if r[2] > ladder_max.get(str(r[1]), -1)]

    if not any([closes, rolling, daily, ladder]) and remote_node_count >= len(bundle["nodes"]):
        print("timeline: D1 已是最新数据，跳过写入", flush=True)
        return

    tl_batch = max(batch, 500)

    def _push_rows(label: str, rows: list, sql_prefix: str) -> None:
        if not rows:
            return
        total = len(rows)
        done = 0
        for part in chunks(rows, tl_batch):
            values = ",".join("(" + ",".join(literal(v) for v in row) + ")" for row in part)
            post(url, token, {"op": "sql", "sql": f"{sql_prefix} {values}"})
            done += len(part)
            print(f"\r  timeline/{label}: {done}/{total}", end="", flush=True)
        print(f"\r  timeline/{label}: {done}/{total} ✓", flush=True)

    _push_rows("close", closes, "INSERT OR IGNORE INTO timeline_close (symbol, idx, date, close) VALUES")
    _push_rows(
        "rolling",
        rolling,
        "INSERT OR IGNORE INTO timeline_rolling (symbol, idx, time, as_of, anchor, median, low, high, actual) VALUES",
    )
    _push_rows(
        "daily",
        daily,
        "INSERT OR IGNORE INTO timeline_daily (symbol, idx, time, median, low, high, actual, previous, risk) VALUES",
    )
    _push_rows(
        "ladder",
        ladder,
        "INSERT OR IGNORE INTO timeline_ladder (symbol, horizon, idx, median, low, high) VALUES",
    )

    nodes = bundle["nodes"]
    if nodes:
        done = 0
        total = len(nodes)
        for part in chunks(nodes, 300):
            values = ",".join(f"({literal(p)},{literal(k)},{literal(v)})" for p, k, v in part)
            post(url, token, {"op": "sql", "sql": f"INSERT OR REPLACE INTO timeline_node (path, kind, value) VALUES {values}"})
            done += len(part)
            print(f"\r  timeline/node: {done}/{total}", end="", flush=True)
        print(f"\r  timeline/node: {done}/{total} ✓", flush=True)

    print(
        f"timeline: 增量同步 {len(closes)} 收盘、{len(rolling)} rolling、"
        f"{len(daily)} daily、{len(ladder)} ladder 到 D1",
        flush=True,
    )


def push_agent(url: str, token: str, dataset: str, doc: dict, remote_agent: dict | None = None) -> None:
    bundle = build_agent_bundle(doc, dataset)
    remote_agent = remote_agent or {}
    if (
        remote_agent.get("generated_at") == doc.get("generated_at")
        and (remote_agent.get("node_count") or 0) >= len(bundle["nodes"])
    ):
        print(f"agent/{dataset}: D1 已是最新数据（{doc.get('generated_at')}），跳过写入", flush=True)
        return

    r = bundle["row"]
    post(
        url,
        token,
        {
            "op": "sql",
            "sql": "INSERT OR REPLACE INTO agent_dataset (dataset, n, scored_on_shared, generated_at) VALUES "
            f"({literal(r['dataset'])},{literal(r['n'])},{literal(r['scored_on_shared'])},{literal(r['generated_at'])})",
        },
    )
    nodes = bundle["nodes"]
    done = 0
    total = len(nodes)
    for part in chunks(nodes, 300):
        values = ",".join(f"({literal(dataset)},{literal(p)},{literal(k)},{literal(v)})" for p, k, v in part)
        post(url, token, {"op": "sql", "sql": f"INSERT OR REPLACE INTO agent_node (dataset, path, kind, value) VALUES {values}"})
        done += len(part)
        print(f"\r  agent/{dataset}: {done}/{total}", end="", flush=True)
    print(f"\r  agent/{dataset}: {done}/{total} ✓", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="把本地 SQLite 里的看板数据增量追加到 Cloudflare D1（绝不清空历史表）")
    parser.add_argument("--url", default=os.environ.get("WORKER_URL", "http://127.0.0.1:8787"))
    parser.add_argument("--token", default=os.environ.get("INGEST_TOKEN", ""))
    parser.add_argument("--market", default="ndx,qqq,ssec")
    parser.add_argument("--skip-timeline", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--batch", type=int, default=300, help="每条 INSERT 写多少行")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("缺少 INGEST_TOKEN（本地放 .dev.vars，线上用 wrangler secret put）")

    state = get_state(args.url, args.token)
    remote_markets = state.get("markets") or {}
    remote_tl = state.get("timeline") or {}
    remote_agents = state.get("agents") or {}

    for market in [m.strip() for m in args.market.split(",") if m.strip()]:
        doc = load_cross_market(ROOT, market)
        if doc is None:
            raise SystemExit(f"data/forecast.db 中缺少 {market} 信号，先跑 forecast-log cross")
        push_cross(args.url, args.token, market, doc, args.batch, remote_markets.get(market))

    if not args.skip_timeline:
        tl = load_timeline(ROOT)
        if tl is not None:
            push_timeline(args.url, args.token, tl, args.batch, remote_tl)

    if not args.skip_agent:
        for dataset in ("deepseek", "qwen"):
            ag = load_agent_eval(ROOT, dataset)
            if ag is not None:
                push_agent(args.url, args.token, dataset, ag, remote_agents.get(dataset))

    print("D1 增量同步完成", flush=True)


if __name__ == "__main__":
    main()
