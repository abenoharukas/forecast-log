from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class Bar:
    date: date
    close: float
    # 开盘价、最高价、最低价、成交量。旧缓存文件缺这些列时仍然能读。
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: int | None = None
    # 该交易日在抓取时刻是否已正式收盘（未收盘数据在后续刷新时必须重新拉取并覆盖）。
    is_closed: bool = True


@dataclass(frozen=True)
class Series:
    symbol: str
    timezone: str
    fetched_at: str
    bars: tuple[Bar, ...]


def is_session_closed(
    session_date: date,
    tz_name: str = "America/New_York",
    at: datetime | str | None = None,
) -> bool:
    """判断某个交易日 session_date 在时间点 at 时是否已经收盘。

    - A 股（Asia/Shanghai）：交易日当天 15:05（北京时间）及之后视为已收盘；早于当天的历史日期恒为已收盘。
    - 美股及其他（America/New_York 等）：交易日当天 16:15（美东时间）及之后视为已收盘；早于当天的历史日期恒为已收盘。
    """
    if at is None:
        now_dt = datetime.now(NY)
    elif isinstance(at, str):
        try:
            now_dt = datetime.fromisoformat(at)
        except ValueError:
            return True
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=NY)
    else:
        now_dt = at if at.tzinfo is not None else at.replace(tzinfo=NY)

    if tz_name == "Asia/Shanghai":
        market_tz = SHANGHAI
        close_hm = (15, 5)
    else:
        market_tz = NY
        close_hm = (16, 15)

    local_now = now_dt.astimezone(market_tz)
    if session_date < local_now.date():
        return True
    if session_date > local_now.date():
        return False
    return (local_now.hour, local_now.minute) >= close_hm


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
                "is_closed": bar.is_closed,
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
    tz_name = payload.get("timezone") or "America/New_York"
    fetched_at = payload.get("fetched_at")
    bars = tuple(
        Bar(
            date=date.fromisoformat(row["date"]),
            close=float(row["close"]),
            open=float(row["open"]) if row.get("open") is not None else None,
            high=float(row["high"]) if row.get("high") is not None else None,
            low=float(row["low"]) if row.get("low") is not None else None,
            volume=int(row["volume"]) if row.get("volume") is not None else None,
            is_closed=(
                bool(row["is_closed"])
                if "is_closed" in row and row["is_closed"] is not None
                else is_session_closed(date.fromisoformat(row["date"]), tz_name, fetched_at)
            ),
        )
        for row in payload["bars"]
    )
    series = Series(
        symbol=payload["symbol"],
        timezone=tz_name,
        fetched_at=fetched_at,
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
