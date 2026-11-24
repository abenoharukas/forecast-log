#!/usr/bin/env python3
"""Build the derived payloads the Worker cannot recompute cheaply at the edge.

The five D1 tables are the source of truth and stay byte-identical to the local
`data/forecast.db`. Two kinds of data are *derived* and therefore built here, at deploy
time, with the same Python code the local server uses:

- `_meta/cross-<market>.json`: question text, context switch definitions, calendar and
  news provenance. They come from static config plus `artifacts/`, not from the DB, and
  the per-day series that accompanies them *is* read from D1 by the Worker.
- `_meta/timeline.json`: the VXN page's fan charts are ~900 lines of numpy (analogue
  model over 40 years of bars). Porting that to JavaScript would invite numeric drift,
  so it is computed once here and served as an asset.

Everything lands in `deploy/dist/_meta/` and is uploaded with the static assets.

    python3 scripts/build_d1_payloads.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast_log.artifacts_db import load_cross_market, load_timeline  # noqa: E402

OUT = ROOT / "deploy" / "dist" / "_meta"

META_KEYS = ("name", "instrument", "question", "contexts", "calendar", "news", "context")
MARKETS = ("ndx", "qqq", "ssec")


def write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.1f} KB", flush=True)


def main() -> None:
    if not (ROOT / "data" / "forecast.db").is_file():
        raise SystemExit("missing data/forecast.db")

    for market in MARKETS:
        doc = load_cross_market(ROOT, market)
        if doc is None:
            raise SystemExit(f"no {market} signals in data/forecast.db; run `forecast-log cross` first")
        write(f"cross-{market}.json", {key: doc[key] for key in META_KEYS if key in doc})

    timeline = load_timeline(ROOT)
    if timeline is None:
        raise SystemExit("cannot build timeline: missing ^VXN/^NDX bars")
    write("timeline.json", timeline)

    events_file = ROOT / "data" / "events.json"
    if events_file.is_file():
        events_doc = json.loads(events_file.read_text(encoding="utf-8"))
        write("events.json", {"calendar": events_doc.get("calendar") or {}, "generated_at": events_doc.get("generated_at")})

    from forecast_log.news_feed import load_llm_indicators

    write("news-indicators.json", load_llm_indicators(ROOT))


if __name__ == "__main__":
    main()
