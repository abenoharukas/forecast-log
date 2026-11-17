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


def push_cross(url: str, token: str, market: str, doc: dict, batch: int) -> None:
    bundle = build_cross_bundle(doc, market)
    post(url, token, {"market": market, "op": "clear"})

    columns = ("market", "date", *SESSION_COLUMNS, "factors_packed")
    rows = bundle["sessions"]
    for part in chunks(rows, batch):
        values = ",".join(
            "(" + literal(market) + "," + ",".join(literal(row.get(column)) for column in columns[1:]) + ")"
            for row in part
        )
        post(url, token, {"op": "sql", "sql": f"INSERT INTO session ({','.join(columns)}) VALUES {values}"})

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


def push_timeline(url: str, token: str, doc: dict, batch: int) -> None:
    bundle = build_timeline_bundle(doc)
    post(url, token, {"op": "clear_timeline"})

    for part in chunks(bundle["closes"], batch):
        values = ",".join(f"({literal(s)},{i},{literal(d)},{literal(c)})" for s, i, d, c in part)
        post(url, token, {"op": "sql", "sql": f"INSERT INTO timeline_close (symbol, idx, date, close) VALUES {values}"})

    for part in chunks(bundle["rolling"], batch):
        values = ",".join("(" + ",".join(literal(v) for v in row) + ")" for row in part)
        post(
            url,
            token,
            {
                "op": "sql",
                "sql": "INSERT INTO timeline_rolling (symbol, idx, time, as_of, anchor, median, low, high, actual) VALUES "
                + values,
            },
        )

    for part in chunks(bundle["daily"], batch):
        values = ",".join("(" + ",".join(literal(v) for v in row) + ")" for row in part)
        post(
            url,
            token,
            {
                "op": "sql",
                "sql": "INSERT INTO timeline_daily (symbol, idx, time, median, low, high, actual, previous, risk) VALUES "
                + values,
            },
        )

    for part in chunks(bundle["ladder"], batch):
        values = ",".join("(" + ",".join(literal(v) for v in row) + ")" for row in part)
        post(
            url,
            token,
            {"op": "sql", "sql": f"INSERT INTO timeline_ladder (symbol, horizon, idx, median, low, high) VALUES {values}"},
        )

    for part in chunks(bundle["nodes"], 80):
        values = ",".join(f"({literal(p)},{literal(k)},{literal(v)})" for p, k, v in part)
        post(url, token, {"op": "sql", "sql": f"INSERT OR REPLACE INTO timeline_node (path, kind, value) VALUES {values}"})

    print(
        f"timeline: {len(bundle['closes'])} 收盘、{len(bundle['rolling'])} rolling、"
        f"{len(bundle['daily'])} daily、{len(bundle['ladder'])} ladder 已写入 D1"
    )


def push_agent(url: str, token: str, dataset: str, doc: dict) -> None:
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
    print(f"agent/{dataset}: {len(bundle['nodes'])} 个字段已写入 D1")


def main() -> None:
    parser = argparse.ArgumentParser(description="把本地 SQLite 里的看板数据同步到 Cloudflare D1")
    parser.add_argument("--url", default=os.environ.get("WORKER_URL", "http://127.0.0.1:8787"))
    parser.add_argument("--token", default=os.environ.get("INGEST_TOKEN", ""))
    parser.add_argument("--market", default="ndx,qqq,ssec")
    parser.add_argument("--skip-timeline", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--batch", type=int, default=100, help="每条 INSERT 写多少行")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("缺少 INGEST_TOKEN（本地放 .dev.vars，线上用 wrangler secret put）")

    for market in [m.strip() for m in args.market.split(",") if m.strip()]:
        doc = load_cross_market(ROOT, market)
        if doc is None:
            raise SystemExit(f"data/forecast.db 中缺少 {market} 信号，先跑 forecast-log cross")
        push_cross(args.url, args.token, market, doc, args.batch)

    if not args.skip_timeline:
        tl = load_timeline(ROOT)
        if tl is not None:
            push_timeline(args.url, args.token, tl, args.batch)

    if not args.skip_agent:
        for dataset in ("deepseek", "qwen"):
            ag = load_agent_eval(ROOT, dataset)
            if ag is not None:
                push_agent(args.url, args.token, dataset, ag)

    print("D1 已更新")


if __name__ == "__main__":
    main()
