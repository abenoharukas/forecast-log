from pathlib import Path

from forecast_log import china
from forecast_log.bars import load_series
from forecast_log.paths import series_path

ROOT = Path(__file__).resolve().parents[1]


def _have_inputs() -> bool:
    return (ROOT / "artifacts" / "cross.json").is_file() and series_path(ROOT, china.TARGET).exists()


def test_every_signal_enters_on_the_first_a_share_session_after_the_us_date():
    if not _have_inputs():
        return
    series = load_series(series_path(ROOT, china.TARGET))
    days = [bar.date.isoformat() for bar in series.bars]
    rows = china.entries(ROOT)
    assert rows
    for row in rows:
        assert row.us_date < row.date
        index = days.index(row.date)
        assert index > 0
        # No A-share session may sit between the US signal date and the entry.
        assert days[index - 1] <= row.us_date


def test_the_window_is_five_a_share_sessions_and_the_hit_is_any_close_above_entry():
    if not _have_inputs():
        return
    series = load_series(series_path(ROOT, china.TARGET))
    days = [bar.date.isoformat() for bar in series.bars]
    closes = {bar.date.isoformat(): bar.close for bar in series.bars}
    position = {day: index for index, day in enumerate(days)}
    for row in china.entries(ROOT):
        if row.outcome is None:
            assert row.peak is None and row.target is None
            continue
        index = position[row.date]
        window = days[index + 1 : index + 1 + china.HORIZON]
        best = max(closes[day] for day in window)
        assert row.target == days[index + china.HORIZON]
        assert row.peak == best
        assert row.outcome == int(best > row.close)


def test_the_signal_itself_is_not_rebuilt_for_this_market():
    if not _have_inputs():
        return
    import json

    signals = json.loads((ROOT / "artifacts" / "cross.json").read_text())
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
