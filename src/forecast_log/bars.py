from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Bar:
    date: date
    close: float


@dataclass(frozen=True)
class Series:
    symbol: str
    timezone: str
    fetched_at: str
    bars: tuple[Bar, ...]


def validate_bars(bars: tuple[Bar, ...] | list[Bar]) -> None:
    if not bars:
        raise ValueError("no bars")
    dates = [bar.date for bar in bars]
    if dates != sorted(dates):
        raise ValueError("bars must be sorted by date")
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate session dates")
    if any(bar.close <= 0 or bar.close != bar.close for bar in bars):
        raise ValueError("closes must be finite and positive")


def save_series(path: Path, series: Series) -> None:
    validate_bars(series.bars)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": series.symbol,
        "timezone": series.timezone,
        "fetched_at": series.fetched_at,
        "bars": [
            {"date": bar.date.isoformat(), "close": bar.close} for bar in series.bars
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_series(path: Path) -> Series:
    payload = json.loads(path.read_text())
    bars = tuple(
        Bar(date=date.fromisoformat(row["date"]), close=float(row["close"]))
        for row in payload["bars"]
    )
    series = Series(
        symbol=payload["symbol"],
        timezone=payload["timezone"],
        fetched_at=payload["fetched_at"],
        bars=bars,
    )
    validate_bars(series.bars)
    return series


def bar_on(series: Series, session: date) -> Bar | None:
    for bar in series.bars:
        if bar.date == session:
            return bar
    return None


def last_completed_bar(series: Series, now: datetime) -> Bar:
    """Drop today's bar until 16:15 America/New_York, when both prints are out."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_ny = now.astimezone(NY)
    last = series.bars[-1]
    if last.date != now_ny.date():
        return last
    close_at = datetime(now_ny.year, now_ny.month, now_ny.day, 16, 15, tzinfo=NY)
    if now_ny >= close_at:
        return last
    if len(series.bars) < 2:
        raise ValueError("no completed session")
    return series.bars[-2]
