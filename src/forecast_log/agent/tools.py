"""The agent's tools. Every one of them is gated, and every one of them returns JSON.

Design notes worth defending in a review:

- Tools raise `LookAheadError` instead of clipping. The model sees the error in the
  transcript, corrects itself, and the trace keeps the evidence that the gate fired.
- `analogue_forecast` is the old model, demoted from "the answer" to "one input". The
  agent has to decide whether to trust it, which is the actual skill being hired for.
- News search is behind an interface with a null implementation. The repo has no
  licensed feed and no search key, so `search_news` returns an empty corpus and says so.
  The agent then has to say it is pricing from prices alone, instead of inventing a
  headline. Wiring a real provider is one class; the eval deliberately measures the
  price-only path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from forecast_log.agent.gate import AsOf, LookAheadError, check_bars
from forecast_log.bars import Series, has_series, load_series
from forecast_log.db import db_path, list_db_symbols, load_all_news_items
from forecast_log.estimate import AnalogModel
from forecast_log.paths import series_path

MAX_DOC_CHARS = 4000


class UnknownTool(RuntimeError):
    pass


class UnknownSymbol(RuntimeError):
    pass


@dataclass
class ToolOutcome:
    name: str
    payload: dict
    error: str | None = None
    latency_s: float = 0.0
    gate_hit: bool = False

    def as_message(self, limit: int) -> str:
        body = {"error": self.error} if self.error else self.payload
        text = json.dumps(body, default=str, ensure_ascii=False)
        if len(text) > limit:
            text = text[:limit] + f'… [truncated, {len(text) - limit} chars]'
        return text


def available_symbols(root: Path) -> list[str]:
    found = set(list_db_symbols(root))
    raw_dir = root / "data" / "raw"
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*.json")):
            try:
                found.add(str(json.loads(path.read_text())["symbol"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return sorted(found)


@dataclass
class Toolbox:
    root: Path
    as_of: AsOf
    news_dir: Path | None = None
    _series: dict[str, Series] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.news_dir is None:
            self.news_dir = self.root / "data" / "news"

    # ---------------------------------------------------------------- plumbing

    def specs(self) -> list[dict]:
        """JSON Schema for each tool, in the shape `/chat/completions` expects."""
        return [_spec(name, description, schema) for name, description, schema in SPECS]

    def call(self, name: str, arguments: dict) -> ToolOutcome:
        started = time.perf_counter()
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                raise UnknownTool(f"no tool named {name!r}; available: {', '.join(s[0] for s in SPECS)}")
            payload = handler(**arguments)
        except LookAheadError as error:
            return ToolOutcome(
                name=name,
                payload={},
                error=str(error),
                latency_s=time.perf_counter() - started,
                gate_hit=True,
            )
        except (UnknownSymbol, TypeError, ValueError) as error:
            return ToolOutcome(
                name=name,
                payload={},
                error=f"{type(error).__name__}: {error}",
                latency_s=time.perf_counter() - started,
            )
        return ToolOutcome(name=name, payload=payload, latency_s=time.perf_counter() - started)

    def series(self, symbol: str) -> Series:
        symbol = str(symbol)
        if symbol not in self._series:
            if not has_series(self.root, symbol):
                raise UnknownSymbol(
                    f"{symbol!r} is not fetched; available: {', '.join(available_symbols(self.root))}"
                )
            self._series[symbol] = load_series(series_path(self.root, symbol))
        return self._series[symbol]

    def visible(self, symbol: str) -> tuple[list[date], np.ndarray]:
        """Sessions and closes at or before the as-of session. The only data path."""
        series = self.series(symbol)
        bars = [bar for bar in series.bars if bar.date <= self.as_of.session]
        if len(bars) < 2:
            raise ValueError(f"{symbol} has no sessions at or before {self.as_of.session}")
        self.as_of.check(bars[-1].date, f"{symbol} series")
        return [bar.date for bar in bars], np.array([bar.close for bar in bars], dtype=float)

    # ------------------------------------------------------------------- tools

    def _tool_price_history(self, symbol: str, sessions: int = 60) -> dict:
        sessions = max(2, min(int(sessions), 756))
        days, closes = self.visible(symbol)
        take = min(sessions, len(days))
        rows = [
            {"date": day.isoformat(), "close": round(float(close), 4)}
            for day, close in zip(days[-take:], closes[-take:])
        ]
        check_bars(rows, self.as_of, "price_history")
        return {
            "symbol": symbol,
            "sessions": take,
            "rows": rows,
            "note": "daily closes, unadjusted; the last row is the as-of session",
        }

    def _tool_series_stats(self, symbol: str, window: int = 252) -> dict:
        window = max(30, min(int(window), 1000))
        days, closes = self.visible(symbol)
        take = min(window, len(closes))
        recent = closes[-take:]
        returns = np.diff(np.log(recent))
        level = float(closes[-1])

        def change(lookback: int) -> float | None:
            if len(closes) <= lookback:
                return None
            return round(float(closes[-1] / closes[-1 - lookback] - 1), 4)

        return {
            "symbol": symbol,
            "as_of": days[-1].isoformat(),
            "last": round(level, 4),
            "window_sessions": take,
            "mean": round(float(recent.mean()), 4),
            "median": round(float(np.median(recent)), 4),
            "p10": round(float(np.quantile(recent, 0.10)), 4),
            "p90": round(float(np.quantile(recent, 0.90)), 4),
            "z_of_last": round(float((level - recent.mean()) / (recent.std() or 1.0)), 3),
            "change_5": change(5),
            "change_21": change(21),
            "change_252": change(252),
            # In percent per year: 24.5 means 24.5%, which is the unit the model reasons in.
            "realized_vol_21_pct_annualized": (
                round(float(returns[-21:].std(ddof=1) * np.sqrt(252) * 100), 1)
                if len(returns) >= 21
                else None
            ),
        }

    def _tool_base_rate(self, symbol: str, threshold: float, horizon: int) -> dict:
        days, closes = self.visible(symbol)
        horizon = int(horizon)
        threshold = float(threshold)
        if horizon < 1:
            raise ValueError("horizon must be at least 1 session")
        # An analogue counts only once its outcome is printed at or before the as-of session.
        usable = len(closes) - horizon
        if usable < 30:
            raise ValueError("not enough history to measure a base rate")
        hit = closes[horizon:] > threshold
        hit = hit[:usable]
        return {
            "symbol": symbol,
            "question": f"close more than {threshold} exactly {horizon} sessions later",
            "sessions_measured": int(usable),
            "from": days[0].isoformat(),
            "to": days[-horizon - 1].isoformat(),
            "base_rate": round(float(hit.mean()), 4),
            "note": "unconditional rate over printed history only; no state conditioning",
        }

    def _tool_analogue_forecast(self, symbol: str, threshold: float, horizon: int) -> dict:
        days, closes = self.visible(symbol)
        horizon = int(horizon)
        model = AnalogModel(closes)
        index = len(closes) - 1
        lowest = max(model.first_index(horizon), 0)
        if index < lowest:
            raise ValueError(
                f"not enough printed history for a {horizon}-session analogue forecast "
                f"(need index >= {lowest}, have {index})"
            )
        estimate = model.forecast(index, horizon, float(threshold), with_path=True)
        anchors = [
            {"sessions_ahead": point.horizon, "median_close": round(point.median, 4)}
            for point in estimate.path
        ]
        return {
            "symbol": symbol,
            "anchored_at": days[-1].isoformat(),
            "as_of_close": round(estimate.as_of_close, 4),
            "threshold": float(threshold),
            "horizon_sessions": horizon,
            "probability": round(estimate.probability, 4),
            "sample_size": estimate.sample_size,
            "rule": estimate.rule,
            "median_path": anchors,
            "note": "state-analogue model: same state, outcome already printed; multiple of its own start",
        }

    def _tool_search_news(self, query: str, limit: int = 5) -> dict:
        documents = self._load_news()
        if not documents:
            return {
                "query": query,
                "documents": [],
                "note": (
                    "no news corpus is configured in this repo (data/news/*.json is absent), "
                    "so price data is the only evidence available for this question"
                ),
            }
        terms = [term for term in str(query).lower().split() if term]
        scored = []
        for document in documents:
            haystack = f"{document.get('title', '')} {document.get('text', '')}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score or not terms:
                scored.append((score, document))
        scored.sort(key=lambda pair: (pair[0], pair[1]["published_at"]), reverse=True)
        return {
            "query": query,
            "documents": [
                {
                    "id": document["id"],
                    "published_at": document["published_at"],
                    "title": document.get("title", ""),
                    "excerpt": document.get("text", "")[:600],
                }
                for _, document in scored[: max(1, min(int(limit), 10))]
            ],
        }

    def _tool_read_doc(self, doc_id: str) -> dict:
        for document in self._load_news():
            if document["id"] == str(doc_id):
                return {
                    "id": document["id"],
                    "published_at": document["published_at"],
                    "title": document.get("title", ""),
                    "text": document.get("text", "")[:MAX_DOC_CHARS],
                }
        raise ValueError(f"no document with id {doc_id!r}")

    def _tool_market_implied(self, question_id: str) -> dict:
        path = self.root / "data" / "market" / f"{question_id}.json"
        if not path.is_file():
            return {
                "question_id": question_id,
                "available": False,
                "note": "no prediction-market quote was recorded for this question",
            }
        quote = json.loads(path.read_text())
        published = datetime.fromisoformat(quote["quoted_at"])
        self.as_of.check_published(published, f"market quote {question_id}")
        return {
            "question_id": question_id,
            "available": True,
            "probability": quote["probability"],
            "venue": quote.get("venue", "unknown"),
            "quoted_at": quote["quoted_at"],
        }

    # ------------------------------------------------------------------- corpus

    def _load_news(self) -> list[dict]:
        default_dir = self.root / "data" / "news"
        if self.news_dir == default_dir and db_path(self.root).is_file():
            rows = load_all_news_items(self.root)
            if rows:
                for row in rows:
                    published = datetime.fromisoformat(row["published_at"])
                    if published.tzinfo is None:
                        raise ValueError(f"news {row.get('id', '?')}: published_at needs a timezone")
                    self.as_of.check_published(published, f"news {row.get('id', '?')}")
                return rows
        if self.news_dir is None or not self.news_dir.is_dir():
            return []
        documents = []
        for path in sorted(self.news_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            rows = payload.get("documents", []) if isinstance(payload, dict) else payload
            for row in rows:
                published = datetime.fromisoformat(row["published_at"])
                if published.tzinfo is None:
                    raise ValueError(f"{path.name}: published_at needs a timezone")
                # Gate before the document is even shown to the model.
                self.as_of.check_published(published, f"news {row.get('id', '?')}")
                documents.append(row)
        return documents


def _spec(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


_SYMBOL = {"type": "string", "description": "Ticker as stored, e.g. ^VXN or ^NDX"}

SPECS: tuple[tuple[str, str, dict], ...] = (
    (
        "price_history",
        "Daily closes for one symbol, oldest first, ending at the as-of session. "
        "Every row is a session you are allowed to see.",
        {
            "type": "object",
            "properties": {"symbol": _SYMBOL, "sessions": {"type": "integer", "minimum": 2, "maximum": 756}},
            "required": ["symbol"],
        },
    ),
    (
        "series_stats",
        "Summary statistics of one symbol over a trailing window: last close, mean, "
        "median, 10th and 90th percentile, z-score of the last close, changes over "
        "5/21/252 sessions, and 21-session realised volatility.",
        {
            "type": "object",
            "properties": {"symbol": _SYMBOL, "window": {"type": "integer", "minimum": 30, "maximum": 1000}},
            "required": ["symbol"],
        },
    ),
    (
        "base_rate",
        "Unconditional hit rate for 'close more than threshold exactly horizon sessions "
        "later', measured only on outcomes already printed. This is the number to beat.",
        {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "threshold": {"type": "number"},
                "horizon": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["symbol", "threshold", "horizon"],
        },
    ),
    (
        "analogue_forecast",
        "State-analogue model: find the 200 past sessions whose 1w/1m/1y changes and "
        "distance from the 1-year midpoint look most like today, then count how often "
        "their multiple of their own start cleared the threshold. Returns a probability "
        "and the median path. Treat it as one opinion, not the answer.",
        {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "threshold": {"type": "number"},
                "horizon": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["symbol", "threshold", "horizon"],
        },
    ),
    (
        "search_news",
        "Keyword search over the local, time-stamped document corpus (news, statements, "
        "releases). Returns only documents published at or before the decision instant. "
        "May be empty; if it is, say so instead of inventing coverage.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    ),
    (
        "read_doc",
        "Read one document by id from the local corpus. Same time gate as search_news.",
        {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    ),
    (
        "market_implied",
        "Prediction-market price for this question, if one was recorded. Returns "
        "available=false when there is no quote; that is not a price of zero.",
        {
            "type": "object",
            "properties": {"question_id": {"type": "string"}},
            "required": ["question_id"],
        },
    ),
)
