"""The one rule that makes a forecast auditable: nothing newer than `as_of` reaches the model.

Every tool in `tools.py` calls `AsOf.check` before it returns a row. The check raises
instead of clipping, because a silent clip looks exactly like a correct answer and would
never show up in a trace. A lookahead bug that raises gets fixed in an afternoon; one that
clips survives to production and quietly inflates every backtest.

`as_of` carries two things on purpose:

- `session`, a date: the last session whose close the agent is allowed to read.
- `at`, a tz-aware instant: the wall clock of the decision. News and documents are
  timestamped with times, not sessions, so they are gated against this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
# Daily closes are out at 16:15 America/New_York for both the index and CBOE series.
CLOSE_AT = (16, 15)


class LookAheadError(RuntimeError):
    """Raised when a value dated after the decision instant would reach the model."""


@dataclass(frozen=True)
class AsOf:
    session: date
    at: datetime

    def __post_init__(self) -> None:
        if self.at.tzinfo is None:
            raise ValueError("as_of.at must be timezone-aware")

    @classmethod
    def at_session_close(cls, session: date) -> AsOf:
        """The decision instant for a backtest: just after that session's close."""
        hour, minute = CLOSE_AT
        return cls(
            session=session,
            at=datetime(session.year, session.month, session.day, hour, minute, tzinfo=NY),
        )

    @classmethod
    def today(cls, session: date, now: datetime | None = None) -> AsOf:
        """Live use: the decision happens now, so `at` is now, not the close."""
        now = datetime.now(NY) if now is None else now
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(NY)
        if now.date() < session:
            raise ValueError("as_of session is in the future")
        return cls(session=session, at=now)

    def check(self, day: date, source: str) -> date:
        """Reject a session-dated value from after `session`."""
        if day > self.session:
            raise LookAheadError(
                f"{source} returned {day.isoformat()}, "
                f"after the as-of session {self.session.isoformat()}"
            )
        return day

    def check_published(self, when: datetime, source: str) -> datetime:
        """Reject a timestamped value (news, filings) from after the decision instant."""
        if when.tzinfo is None:
            raise ValueError(f"{source}: published timestamp must be timezone-aware")
        if when.astimezone(NY) > self.at:
            raise LookAheadError(
                f"{source} was published {when.isoformat()}, "
                f"after the decision instant {self.at.isoformat()}"
            )
        return when

    def to_json(self) -> dict:
        return {
            "session": self.session.isoformat(),
            "at": self.at.astimezone(NY).isoformat(timespec="seconds"),
        }


def check_bars(rows: list[dict], as_of: AsOf, source: str) -> list[dict]:
    """Gate a list of {date, close} rows. Used by the tools and by the answer validator."""
    for row in rows:
        as_of.check(date.fromisoformat(row["date"]), source)
    return rows
