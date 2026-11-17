"""Push dashboard data from local SQLite (`data/forecast.db`) into Cloudflare D1.

Reads directly from `data/forecast.db` (no `artifacts/*.json` files needed) and pushes:
- cross signals (`ndx`, `qqq`, `ssec`)
- VXN timeline (`timeline`)
- agent evaluation (`deepseek`, `qwen`)

    INGEST_TOKEN=... python3 scripts/publish_d1.py --url https://<worker>
    INGEST_TOKEN=... python3 scripts/publish_d1.py --url http://127.0.0.1:8787   # wrangler dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def post(url: str, token: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/ingest",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} {exc.reason}: {detail}") from exc
    if not body.get("ok"):
        raise SystemExit(f"写入失败：{body}")
    return body


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def get_state(url: str) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/state",
        method="GET",
        headers={"User-Agent": "forecast-publish/1.0"},
    )
    try:
        with OPENER.open(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def push_cross(url: str, token: str, market: str, doc: dict, batch: int, remote_info: dict | None = None, force: bool = False) -> None:
    bundle = build_cross_bundle(doc, market)
    all_rows = bundle["sessions"]
    remote_count = (remote_info or {}).get("count") or 0
    first_unresolved = (remote_info or {}).get("first_unresolved")
    max_date = (remote_info or {}).get("max_date")

    if not force and remote_count > 0 and (first_unresolved or max_date):
        cutoff = first_unresolved or max_date
        tail_cutoff = all_rows[-10]["date"] if len(all_rows) >= 10 else ""
        if tail_cutoff and tail_cutoff < cutoff:
            cutoff = tail_cutoff
        rows = [r for r in all_rows if r["date"] >= cutoff]
        post(url, token, {"market": market, "op": "clear_meta"})
    else:
        rows = all_rows
        post(url, token, {"market": market, "op": "clear"})

    columns = ("market", "date", *SESSION_COLUMNS, "factors_packed")
    for part in chunks(rows, max(batch, 300)):
        values = ",".join(
            "(" + literal(market) + "," + ",".join(literal(row.get(column)) for column in columns[1:]) + ")"
            for row in part
        )
        post(url, token, {"op": "sql", "sql": f"INSERT OR REPLACE INTO session ({','.join(columns)}) VALUES {values}"})

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
    for part in chunks(bundle["nodes"], 80):
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
    print(f"cross/{market}: {len(rows)} 个交易日、{len(bundle['nodes'])} 个字段已写入 D1")


def push_timeline(url: str, token: str, doc: dict, batch: int, remote_tl: dict | None = None, force: bool = False) -> None:
    bundle = build_timeline_bundle(doc)
    remote_tl = remote_tl or {}
    if force:
        post(url, token, {"op": "clear_timeline"})
        remote_tl = {}

    close_max = remote_tl.get("close") or {}
    rolling_max = remote_tl.get("rolling") or {}
    daily_max = remote_tl.get("daily") or {}
    ladder_max = remote_tl.get("ladder") or {}

    closes = [r for r in bundle["closes"] if r[1] > close_max.get(r[0], -1)]
    rolling = [r for r in bundle["rolling"] if r[1] > rolling_max.get(r[0], -1)]
    daily = [r for r in bundle["daily"] if r[1] > daily_max.get(r[0], -1)]
    ladder = [r for r in bundle["ladder"] if r[2] > ladder_max.get(str(r[1]), -1)]

    if not any([closes, rolling, daily, ladder]):
        print("timeline: D1 已是最新数据，0 行需要增量写入", flush=True)
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

    _push_rows("close", closes, "INSERT OR REPLACE INTO timeline_close (symbol, idx, date, close) VALUES")
    _push_rows(
        "rolling",
        rolling,
        "INSERT OR REPLACE INTO timeline_rolling (symbol, idx, time, as_of, anchor, median, low, high, actual) VALUES",
    )
    _push_rows(
        "daily",
        daily,
        "INSERT OR REPLACE INTO timeline_daily (symbol, idx, time, median, low, high, actual, previous, risk) VALUES",
    )
    _push_rows(
        "ladder",
        ladder,
        "INSERT OR REPLACE INTO timeline_ladder (symbol, horizon, idx, median, low, high) VALUES",
    )

    for part in chunks(bundle["nodes"], 80):
        values = ",".join(f"({literal(p)},{literal(k)},{literal(v)})" for p, k, v in part)
        post(url, token, {"op": "sql", "sql": f"INSERT OR REPLACE INTO timeline_node (path, kind, value) VALUES {values}"})

    print(
        f"timeline: 增量同步 {len(closes)} 收盘、{len(rolling)} rolling、"
        f"{len(daily)} daily、{len(ladder)} ladder 到 D1",
        flush=True,
    )


def push_agent(url: str, token: str, dataset: str, doc: dict, remote_agent: dict | None = None, force: bool = False) -> None:
    if not force and remote_agent and remote_agent.get("generated_at") == doc.get("generated_at"):
        print(f"agent/{dataset}: D1 已存在最新数据（{doc.get('generated_at')}），跳过重复录入", flush=True)
        return
    bundle = build_agent_bundle(doc, dataset)
    post(url, token, {"op": "clear_agent", "dataset": dataset})
    r = bundle["row"]
    post(
        url,
        token,
        {
            "op": "sql",
            "sql": "INSERT INTO agent_dataset (dataset, n, scored_on_shared, generated_at) VALUES "
            f"({literal(r['dataset'])},{literal(r['n'])},{literal(r['scored_on_shared'])},{literal(r['generated_at'])})",
        },
    )
    for part in chunks(bundle["nodes"], 80):
        values = ",".join(f"({literal(dataset)},{literal(p)},{literal(k)},{literal(v)})" for p, k, v in part)
        post(url, token, {"op": "sql", "sql": f"INSERT OR REPLACE INTO agent_node (dataset, path, kind, value) VALUES {values}"})
    print(f"agent/{dataset}: {len(bundle['nodes'])} 个字段已写入 D1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="把本地 SQLite 里的看板数据增量同步到 Cloudflare D1")
    parser.add_argument("--url", default=os.environ.get("WORKER_URL", "http://127.0.0.1:8787"))
    parser.add_argument("--token", default=os.environ.get("INGEST_TOKEN", ""))
    parser.add_argument("--market", default="ndx,qqq,ssec")
    parser.add_argument("--skip-timeline", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--force", action="store_true", help="强制全量重写")
    parser.add_argument("--batch", type=int, default=100, help="每条 INSERT 写多少行")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("缺少 INGEST_TOKEN（本地放 .dev.vars，线上用 wrangler secret put）")

    state = get_state(args.url)
    remote_markets = state.get("markets") or {}
    remote_tl = state.get("timeline") or {}
    remote_agents = state.get("agents") or {}

    for market in [m.strip() for m in args.market.split(",") if m.strip()]:
        doc = load_cross_market(ROOT, market)
        if doc is None:
            raise SystemExit(f"data/forecast.db 中缺少 {market} 信号，先跑 forecast-log cross")
        push_cross(args.url, args.token, market, doc, args.batch, remote_markets.get(market), args.force)

    if not args.skip_timeline:
        tl = load_timeline(ROOT)
        if tl is not None:
            push_timeline(args.url, args.token, tl, args.batch, remote_tl, args.force)

    if not args.skip_agent:
        for dataset in ("deepseek", "qwen"):
            ag = load_agent_eval(ROOT, dataset)
            if ag is not None:
                push_agent(args.url, args.token, dataset, ag, remote_agents.get(dataset), args.force)

    print("D1 同步完成", flush=True)


if __name__ == "__main__":
    main()
