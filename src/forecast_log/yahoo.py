from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from forecast_log.bars import Bar, Series, is_session_closed

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_ATTEMPTS = 5
NY = ZoneInfo("America/New_York")
SHANGHAI = ZoneInfo("Asia/Shanghai")
# `range=max` silently returns monthly or quarterly bars. Ask for explicit windows instead.
WINDOW_YEARS = 5
HISTORY_START = date(1985, 1, 1)


def fetch_daily(symbol: str, timeout: float = 60, start: date | None = None) -> Series:
    """Daily OHLCV from the Yahoo chart API. This is the only price source."""
    quoted = urllib.parse.quote(symbol, safe="")
    by_date: dict[date, tuple[float | None, float | None, float | None, float, int | None]] = {}
    tz_name = "Asia/Shanghai" if symbol.endswith((".SS", ".SZ")) else "America/New_York"
    cursor = start or HISTORY_START
    # Take the later of NY and Beijing dates so A-share morning sessions are never cut off
    # when New York is still on the previous calendar day.
    stop = max(datetime.now(NY).date(), datetime.now(SHANGHAI).date()) + timedelta(days=2)
    while cursor < stop:
        end = min(_add_years(cursor, WINDOW_YEARS), stop)
        query = (
            CHART.format(symbol=quoted)
            + f"?interval=1d&period1={_unix(cursor)}&period2={_unix(end)}"
        )
        try:
            payload = _load(query, symbol, timeout)
        except _NotYetStarted:
            # The series has not started yet. Later windows still count.
            cursor = end
            continue
        chunk_zone, rows = _read_chart(symbol, payload)
        tz_name = chunk_zone or tz_name
        by_date.update(rows)
        cursor = end
    fetched_at = datetime.now(NY).isoformat(timespec="seconds")
    bars = tuple(
        Bar(
            date=day,
            close=close,
            open=open_,
            high=high,
            low=low,
            volume=volume,
            is_closed=is_session_closed(day, tz_name, fetched_at),
        )
        for day, (open_, high, low, close, volume) in sorted(by_date.items())
    )
    if not bars:
        raise RuntimeError(f"Yahoo returned no closes for {symbol}")
    if start is None:
        _assert_daily(symbol, bars)
    return Series(symbol=symbol, timezone=tz_name, fetched_at=fetched_at, bars=bars)


def fetch_incremental(symbol: str, existing: Series | None, timeout: float = 60) -> Series:
    """只拉取数据库中最早未收盘日期（或最近日期）之后的 K 线并合并，确保未收盘数据被重新拉取并更新收盘状态。"""
    if existing is None or not existing.bars:
        return fetch_daily(symbol, timeout=timeout)
    last_day = existing.bars[-1].date
    unclosed_days = [bar.date for bar in existing.bars if not bar.is_closed]
    anchor_day = min(unclosed_days) if unclosed_days else last_day
    # 往前留 5 天余量，覆盖盘中未收盘重抓、跨日收盘确认或节假日修正
    start = min(anchor_day, last_day) - timedelta(days=5)
    delta = fetch_daily(symbol, timeout=timeout, start=start)
    tz_name = delta.timezone or existing.timezone
    merged: dict[date, Bar] = {bar.date: bar for bar in existing.bars}
    for bar in delta.bars:
        if bar.date >= start or bar.date not in merged or not merged[bar.date].is_closed:
            merged[bar.date] = bar
    bars = tuple(merged[d] for d in sorted(merged))
    _assert_daily(symbol, bars)
    return Series(symbol=symbol, timezone=tz_name, fetched_at=delta.fetched_at, bars=bars)


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


def _read_chart(
    symbol: str, payload: dict
) -> tuple[str | None, dict[date, tuple[float | None, float | None, float | None, float, int | None]]]:
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
    quote = result["indicators"]["quote"][0]
    closes = quote.get("close") or []
    opens = quote.get("open") or [None] * len(stamps)
    highs = quote.get("high") or [None] * len(stamps)
    lows = quote.get("low") or [None] * len(stamps)
    volumes = quote.get("volume") or [None] * len(stamps)
    rows = {}
    for stamp, close, open_, high, low, volume in zip(stamps, closes, opens, highs, lows, volumes):
        if close is None:
            continue
        rows[datetime.fromtimestamp(int(stamp), zone).date()] = (
            None if open_ is None else round(float(open_), 4),
            None if high is None else round(float(high), 4),
            None if low is None else round(float(low), 4),
            round(float(close), 4),
            None if volume is None else int(volume),
        )
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
