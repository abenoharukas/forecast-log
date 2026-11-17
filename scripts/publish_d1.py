"""Push the computed signals into Cloudflare D1, so the deployed site reads a database.

The Python pipeline still computes `artifacts/cross*.json` locally -- that is where the
numpy work happens. This script only moves the numbers into D1; the Worker then serves
`/api/cross/<market>` straight out of the tables, and no JSON file is published.

    INGEST_TOKEN=... python3 scripts/publish_d1.py --url https://<worker>
    INGEST_TOKEN=... python3 scripts/publish_d1.py --url http://127.0.0.1:8787   # wrangler dev

Layout it writes:

    market       one row per tab: instrument, question, rule, first/last session
    session      one row per trading day: OHLCV + derived signal columns
    factor_key   which bit of session.factors_mask is which switch
    contexts     the switch definitions the page renders
    market_node  every nested value as one scalar row, addressed by a dotted path

`/api/ingest` is the only writer, so local and production go through the same code path.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MARKETS = {"ndx": "cross.json", "qqq": "cross_qqq.json", "ssec": "cross_ssec.json"}

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
)

# 这些顶层键不进 market 行，拆成路径存 market_node
SCALAR_ROOTS = (
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


def classify(value):
    if value is None:
        return "null", None
    if isinstance(value, bool):
        return "bool", "1" if value else "0"
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        return "num", repr(value)
    return "str", str(value)


def walk(prefix: str, value, out: list[tuple[str, str, str | None]]) -> None:
    """Nested dict/list -> one row per scalar, path carries the structure."""
    if isinstance(value, dict):
        if not value:
            out.append((prefix, "obj", None))
        for key, inner in value.items():
            walk(f"{prefix}.{key}" if prefix else str(key), inner, out)
    elif isinstance(value, list):
        if not value:
            out.append((prefix, "list", None))  # 空数组也要留个位置，否则结构就少一层
        for index, inner in enumerate(value):
            walk(f"{prefix}.{index}", inner, out)
    else:
        kind, text = classify(value)
        out.append((prefix, kind, text))


def build(doc: dict, market: str) -> dict:
    nodes: list[tuple[str, str, str | None]] = []
    for key in SCALAR_ROOTS:
        if key in doc:
            walk(key, doc[key], nodes)
    if "calendar" in doc:
        walk("calendar", doc["calendar"], nodes)
    if isinstance(doc.get("question"), dict):
        # question.horizon/text/rule 已经是 market 表的列，这里只补 factors 数组
        if "factors" in doc["question"]:
            walk("question.factors", doc["question"]["factors"], nodes)

    components = doc.get("components") or {}
    factor_keys = list(components.keys())
    positions = {key: i for i, key in enumerate(factor_keys)}

    sessions = []
    for index, day in enumerate(doc["time"]):
        # 每个开关占 4 bit（取值 0..15，riskoff_hits 最大是 5），11 个开关 = 44 bit，
        # 仍在 JS 安全整数范围内，所以一条 session 行就能装下全部 factors
        mask = 0
        for key, slot in positions.items():
            series = components.get(key) or []
            if index < len(series) and series[index]:
                mask |= int(series[index]) << (4 * slot)
        row = {"date": day, "factors_packed": mask}
        for column in SESSION_COLUMNS:
            series = doc.get(column)
            row[column] = series[index] if isinstance(series, list) and index < len(series) else None
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


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


# 默认不走环境里的 http_proxy：目标通常是本机 wrangler 或 Cloudflare，
# 走代理反而连不上（本机代理挂掉时表现为 502）。
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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with OPENER.open(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not body.get("ok"):
        raise SystemExit(f"写入失败：{body}")
    return body


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def push(url: str, token: str, bundle: dict, batch: int) -> None:
    market = bundle["market"]
    post(url, token, {"market": market, "op": "clear"})

    columns = ("market", "date", *SESSION_COLUMNS, "factors_packed")
    rows = bundle["sessions"]
    for part in chunks(rows, batch):
        values = ",".join(
            "(" + literal(market) + "," + ",".join(literal(row.get(column)) for column in columns[1:]) + ")"
            for row in part
        )
        post(url, token, {"market": market, "op": "sql", "sql": f"INSERT INTO session ({','.join(columns)}) VALUES {values}"})

    row = bundle["row"]
    post(
        url,
        token,
        {
            "market": market,
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
                "market": market,
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
            "market": market,
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
            "market": market,
            "op": "sql",
            "sql": "INSERT OR REPLACE INTO factor_key (market, position, key) VALUES "
            + ",".join(f"({literal(market)},{position},{literal(key)})" for position, key in bundle["factor_keys"]),
        },
    )
    print(f"{market}: {len(rows)} 个交易日、{len(bundle['nodes'])} 个字段已写入 D1")


def main() -> None:
    parser = argparse.ArgumentParser(description="把算好的信号写进 Cloudflare D1")
    parser.add_argument("--url", default=os.environ.get("WORKER_URL", "http://127.0.0.1:8787"))
    parser.add_argument("--token", default=os.environ.get("INGEST_TOKEN", ""))
    parser.add_argument("--market", default="ndx,qqq,ssec")
    parser.add_argument("--batch", type=int, default=100, help="每条 INSERT 写多少行")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("缺少 INGEST_TOKEN（本地放 .dev.vars，线上用 wrangler secret put）")

    for market in [m.strip() for m in args.market.split(",") if m.strip()]:
        path = ROOT / "artifacts" / MARKETS[market]
        if not path.is_file():
            raise SystemExit(f"缺少 {path}，先跑 forecast-log cross")
        push(args.url, args.token, build(json.loads(path.read_text()), market), args.batch)
    print("D1 已更新")


if __name__ == "__main__":
    main()
