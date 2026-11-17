from pathlib import Path

from forecast_log import china
from forecast_log.artifacts_db import load_cross_market
from forecast_log.bars import has_series, load_series

ROOT = Path(__file__).resolve().parents[1]


def _have_inputs() -> bool:
    return load_cross_market(ROOT, "ndx") is not None and has_series(ROOT, china.TARGET)


def test_every_signal_enters_on_the_first_a_share_session_after_the_us_date():
    if not _have_inputs():
        return
    series = load_series(ROOT, china.TARGET)
    days = [bar.date.isoformat() for bar in series.bars]
    rows = china.entries(ROOT)
    assert rows
    for row in rows:
        assert row.us_date < row.date
        index = days.index(row.date)
        assert index > 0
        # No A-share session may sit between the US signal date and the entry.
        assert days[index - 1] <= row.us_date


def test_the_entry_is_the_open_and_the_window_starts_there():
    if not _have_inputs():
        return
    series = load_series(ROOT, china.TARGET)
    days = [bar.date.isoformat() for bar in series.bars]
    bars = {bar.date.isoformat(): bar for bar in series.bars}
    position = {day: index for index, day in enumerate(days)}
    for row in china.entries(ROOT):
        index = position[row.date]
        # 入场价必须是那天的开盘价：收盘价那一刻信号才成立，买不到。
        assert row.entry == bars[row.date].open
        if row.outcome is None:
            assert row.peak is None and row.target is None
            continue
        # 窗口是买入日之后的 5 个收盘，买入当天不算。
        window = days[index + 1 : index + 1 + china.HORIZON]
        best = max(bars[day].close for day in window)
        assert row.target == days[index + china.HORIZON]
        assert row.peak == best
        # 判定基准是箭头那天的收盘价；比较的对象是第 3 天、第 5 天的收盘。
        day = days[index]
        assert row.base_close == bars[day].close
        assert row.close3 == bars[days[index + 3]].close
        assert row.close5 == bars[days[index + 5]].close
        assert row.hit3 == int(row.close3 > row.base_close)
        assert row.outcome == int(row.close5 > row.base_close)


def test_the_signal_itself_is_not_rebuilt_for_this_market():
    if not _have_inputs():
        return
    signals = load_cross_market(ROOT, "ndx")
    assert signals is not None
    rows = china.entries(ROOT)
    entered = {row.us_date for row in rows if row.outcome is not None}
    flagged = {day for day, flag in zip(signals["time"], signals["flag"]) if flag}
    # Every entry must trace back to a flag the NDX study already raised, one A-share session each.
    assert entered <= flagged
    assert len({row.date for row in rows}) == len(rows)
    # The few flags without an entry are the ones that pile onto an A-share session already taken
    # (a US signal on either side of a Chinese holiday lands on the same session), plus the tail.
    missing = flagged - entered
    assert len(missing) < 0.1 * len(flagged)
    taken = {row.date for row in rows}
    assert len(taken) == len(rows)


def test_document_reports_both_markets_and_the_bands():
    if not _have_inputs():
        return
    doc = china.document(ROOT)
    assert doc["instrument"] == "000001.SS"
    assert len(doc["time"]) == len(doc["close"]) == len(doc["flag"])
    assert sum(doc["flag"]) > 500
    assert doc["summary"]["all"]["lift"] is not None
    assert doc["reference"]["instrument"] == "^NDX"
    assert len(doc["bands"]) == 10
