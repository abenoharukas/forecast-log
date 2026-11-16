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
    # 开盘价、最高价、最低价、成交量。旧缓存文件缺这些列时仍然能读。
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: int | None = None


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
    for field in ("open", "high", "low"):
        if any(
            getattr(bar, field) is not None
            and (getattr(bar, field) <= 0 or getattr(bar, field) != getattr(bar, field))
            for bar in bars
        ):
            raise ValueError(f"{field}s must be finite and positive when present")
    if any(
        bar.high is not None and bar.low is not None and bar.low > bar.high
        for bar in bars
    ):
        raise ValueError("low must not exceed high")
    if any(bar.volume is not None and bar.volume < 0 for bar in bars):
        raise ValueError("volumes must be non-negative when present")


def _root_from_series_path(path: Path) -> Path | None:
    if path.parent.name == "raw" and path.parent.parent.name == "data":
        return path.parent.parent.parent
    return None


def has_series(root: Path, symbol: str) -> bool:
    from forecast_log.db import has_series_in_db
    from forecast_log.paths import series_path

    if has_series_in_db(root, symbol):
        return True
    return series_path(root, symbol).is_file()


def save_series(path: Path, series: Series) -> None:
    validate_bars(series.bars)
    root = _root_from_series_path(path)
    if root is not None:
        from forecast_log.db import upsert_series

        upsert_series(root, series)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": series.symbol,
        "timezone": series.timezone,
        "fetched_at": series.fetched_at,
        "bars": [
            {
                "date": bar.date.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in series.bars
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_series(path: Path) -> Series:
    root = _root_from_series_path(path)
    if root is not None:
        from forecast_log.db import get_series_by_slug

        loaded = get_series_by_slug(root, path.stem)
        if loaded is not None:
            return loaded
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
