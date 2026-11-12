from __future__ import annotations

from datetime import date, timedelta

# The question file names this day as closed. It is not a general holiday calendar.
CLOSED_SESSIONS = {date(2026, 11, 26)}


def sessions_after(as_of: date, count: int, closed: set[date] | None = None) -> list[date]:
    """The next `count` sessions after `as_of`, skipping weekends and `closed`."""
    if count < 1:
        raise ValueError("count must be positive")
    skipped = CLOSED_SESSIONS if closed is None else closed
    found: list[date] = []
    day = as_of + timedelta(days=1)
    while len(found) < count:
        if day.weekday() < 5 and day not in skipped:
            found.append(day)
        day += timedelta(days=1)
    return found


def sessions_from_as_of_through_resolve(
    as_of: date,
    resolve: date,
    closed: set[date] | None = None,
) -> int:
    """Trading sessions in (as_of, resolve], skipping weekends and `closed`."""
    if resolve <= as_of:
        raise ValueError("resolve session must be after as_of")
    skipped = CLOSED_SESSIONS if closed is None else closed
    if resolve.weekday() >= 5 or resolve in skipped:
        raise ValueError(f"{resolve.isoformat()} is not a session")
    count = 0
    day = as_of + timedelta(days=1)
    while day <= resolve:
        if day.weekday() < 5 and day not in skipped:
            count += 1
        day += timedelta(days=1)
    if count < 1:
        raise ValueError("horizon is empty")
    return count
