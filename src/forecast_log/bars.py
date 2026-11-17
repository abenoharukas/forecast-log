from __future__ import annotations

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


def has_series(root: Path, symbol: str) -> bool:
    from forecast_log.db import has_series_in_db

    return has_series_in_db(root, symbol)


def save_series(root: Path, series: Series) -> None:
    """唯一存储：SQLite 的 bars 表。"""
    validate_bars(series.bars)
    from forecast_log.db import upsert_series

    upsert_series(root, series)


def load_series(root: Path, symbol: str) -> Series:
    from forecast_log.db import get_series

    loaded = get_series(root, symbol)
    if loaded is None:
        raise FileNotFoundError(f"{symbol} 不在 data/forecast.db 的 bars 表里，先跑 make fetch")
    return loaded


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
