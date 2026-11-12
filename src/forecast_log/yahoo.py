from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from forecast_log.bars import Bar, Series

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_ATTEMPTS = 5
NY = ZoneInfo("America/New_York")
# `range=max` silently returns monthly or quarterly bars. Ask for explicit windows instead.
WINDOW_YEARS = 5
HISTORY_START = date(1985, 1, 1)


def fetch_daily(symbol: str, timeout: float = 60) -> Series:
    """Daily Close from the Yahoo chart API. This is the only price source."""
    quoted = urllib.parse.quote(symbol, safe="")
    by_date: dict[date, float] = {}
    tz_name = "America/New_York"
    start = HISTORY_START
    stop = datetime.now(NY).date() + timedelta(days=1)
    while start < stop:
        end = min(_add_years(start, WINDOW_YEARS), stop)
        query = (
            CHART.format(symbol=quoted)
            + f"?interval=1d&period1={_unix(start)}&period2={_unix(end)}"
        )
        try:
            payload = _load(query, symbol, timeout)
        except _NotYetStarted:
            # The series has not started yet. Later windows still count.
            start = end
            continue
        chunk_zone, rows = _read_chart(symbol, payload)
        tz_name = chunk_zone or tz_name
        by_date.update(rows)
        start = end
    bars = tuple(Bar(date=day, close=price) for day, price in sorted(by_date.items()))
    if not bars:
        raise RuntimeError(f"Yahoo returned no closes for {symbol}")
    _assert_daily(symbol, bars)
    fetched_at = datetime.now(NY).isoformat(timespec="seconds")
    return Series(symbol=symbol, timezone=tz_name, fetched_at=fetched_at, bars=bars)


class _NotYetStarted(Exception):
    pass


def _load(query: str, symbol: str, timeout: float) -> dict:
    """One window, with backoff. Yahoo answers 429 when asked too often."""
    delay = 2.0
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        request = urllib.request.Request(query, headers={"User-Agent": "forecast-log"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            if error.code == 400 and "Data doesn't exist" in body:
                raise _NotYetStarted from error
            if error.code in RETRY_STATUS and attempt < RETRY_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"Yahoo chart error for {symbol}: {error.code} {body}") from error
        except (urllib.error.URLError, TimeoutError):
            if attempt >= RETRY_ATTEMPTS:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Yahoo chart error for {symbol}: out of attempts")


def _read_chart(symbol: str, payload: dict) -> tuple[str | None, dict[date, float]]:
    chart = payload["chart"]
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {symbol}: {chart['error']}")
    result = chart["result"][0]
    meta = result["meta"]
    if meta.get("dataGranularity") not in (None, "1d"):
        raise RuntimeError(f"Yahoo returned {meta.get('dataGranularity')} bars for {symbol}")
    tz_name = meta.get("exchangeTimezoneName") or "America/New_York"
    zone = ZoneInfo(tz_name)
    stamps = result.get("timestamp") or []
    closes = result["indicators"]["quote"][0]["close"]
    rows = {}
    for stamp, close in zip(stamps, closes):
        if close is None:
            continue
        rows[datetime.fromtimestamp(int(stamp), zone).date()] = round(float(close), 4)
    return tz_name, rows


def _assert_daily(symbol: str, bars: tuple[Bar, ...]) -> None:
    gaps = sorted((b.date - a.date).days for a, b in zip(bars, bars[1:]))
    if not gaps or gaps[len(gaps) // 2] > 5:
        raise RuntimeError(f"{symbol} bars are not daily")


def _add_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)


def _unix(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
