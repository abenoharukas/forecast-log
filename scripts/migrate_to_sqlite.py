"""一次性把现有的 data/raw/*.json 和 data/news/*.json 导入 data/forecast.db。

不删除任何原 JSON 文件，方便回滚对照。重复执行是幂等的（ON CONFLICT DO UPDATE）。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from forecast_log.bars import Bar, Series
from forecast_log.db import connect, db_path, save_news_fetch, upsert_series
from forecast_log.paths import repo_root


def migrate(root: Path) -> dict[str, int]:
    conn = connect(db_path(root))
    conn.close()

    raw_files = sorted((root / "data" / "raw").glob("*.json"))
    raw_count = 0
    bar_count = 0
    for path in raw_files:
        payload = json.loads(path.read_text())
        bars = tuple(
            Bar(
                date=date.fromisoformat(row["date"]),
                close=float(row["close"]),
                open=float(row["open"]) if row.get("open") is not None else None,
                high=float(row["high"]) if row.get("high") is not None else None,
                low=float(row["low"]) if row.get("low") is not None else None,
                volume=int(row["volume"]) if row.get("volume") is not None else None,
            )
            for row in payload["bars"]
        )
        series = Series(
            symbol=payload["symbol"],
            timezone=payload["timezone"],
            fetched_at=payload["fetched_at"],
            bars=bars,
        )
        upsert_series(root, series)
        raw_count += 1
        bar_count += len(bars)

    news_files = sorted((root / "data" / "news").glob("????-??-??.json"))
    news_days = 0
    news_items = 0
    for path in news_files:
        record = json.loads(path.read_text())
        if "as_of" not in record:
            record["as_of"] = path.stem
        save_news_fetch(root, record)
        news_days += 1
        news_items += len(record.get("documents") or [])

    return {
        "series_files": raw_count,
        "bars": bar_count,
        "news_days": news_days,
        "news_items": news_items,
    }


if __name__ == "__main__":
    root = repo_root()
    stats = migrate(root)
    print(f"导入完成 -> {db_path(root)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
