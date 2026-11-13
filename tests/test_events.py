"""LLM 拉来的事件日历：存、读、算距离。

不联网：只测纯函数，`fetch` 需要 API key，靠真实运行的产物验证（`data/events.json` 的
`problems` 字段就是校验结果）。
"""

from __future__ import annotations

import json
import math
from datetime import date

from forecast_log import events


def test_missing_file_degrades_to_none(tmp_path):
    assert events.load(tmp_path) is None


def test_load_round_trips_and_sorts(tmp_path):
    path = events.path_of(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "source": "llm",
                "calendar": {
                    "cpi": ["2026-02-11", "2026-01-13"],
                    "fomc": ["2026-01-28"],
                    "nfp": ["2026-01-02"],
                },
            }
        )
    )
    calendar = events.load(tmp_path)
    assert calendar is not None
    assert calendar["cpi"] == [date(2026, 1, 13), date(2026, 2, 11)]
    assert calendar["fomc"] == [date(2026, 1, 28)]


def test_days_to_next_counts_calendar_days_and_saturates():
    days = [date(2026, 1, 13), date(2026, 2, 11)]
    assert events.days_to_next(days, date(2026, 1, 10)) == 3
    assert events.days_to_next(days, date(2026, 1, 13)) == 0
    assert events.days_to_next(days, date(2026, 1, 14)) == 28
    assert math.isnan(events.days_to_next(days, date(2026, 3, 1)))
