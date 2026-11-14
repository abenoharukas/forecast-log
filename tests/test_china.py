from pathlib import Path

import numpy as np

from forecast_log import china, cross
from forecast_log.bars import load_series
from forecast_log.paths import series_path

ROOT = Path(__file__).resolve().parents[1]


def _have_inputs() -> bool:
    return all(series_path(ROOT, symbol).exists() for symbol in china.CHINA_SYMBOLS)


def test_composite_uses_the_legs_it_is_given():
    series = load_series(series_path(ROOT, "000001.SS"))
    default = cross.composite(ROOT, series)
    single = cross.composite(ROOT, series, china.LEGS[:1])
    assert np.isfinite(default[-1])
    assert np.isfinite(single[-1])
    # A one-leg composite cannot equal the four-leg one on every session.
    assert not np.allclose(default[300:], single[300:], equal_nan=True)


def test_the_ninety_band_and_the_five_session_window():
    if not _have_inputs():
        return
    series = load_series(series_path(ROOT, "000001.SS"))
    rows = china.calls(ROOT, series)
    closes = {bar.date.isoformat(): bar.close for bar in series.bars}
    days = [bar.date.isoformat() for bar in series.bars]
    position = {day: index for index, day in enumerate(days)}
    checked = 0
    for row in rows:
        index = position[row.date]
        assert row.flag == int(row.score >= row.threshold)
        if row.outcome is None:
            assert row.target is None and row.peak is None
            continue
        window = days[index + 1 : index + 1 + china.HORIZON]
        best = max(closes[day] for day in window)
        assert row.target == days[index + china.HORIZON]
        assert row.peak == best
        assert row.outcome == int(best > row.close)
        checked += 1
    assert checked > 500


def test_document_carries_the_diagnostics_that_say_whether_it_works():
    if not _have_inputs():
        return
    doc = china.document(ROOT)
    summary = doc["summary"]
    assert doc["instrument"] == "000001.SS"
    assert len(doc["time"]) == len(doc["close"]) == len(doc["flag"])
    assert 0 < summary["flagged"] < summary["n"]
    assert summary["holdout_from"] == china.HOLDOUT
    # The page prints these, so they have to be there even when the answer is "no edge".
    assert len(doc["legs"]) == len(china.SINGLE_LEGS)
    assert len(doc["variants"]) == len(china.VARIANTS)
    assert len(doc["deciles"]) == 10
    assert doc["deciles"][0]["from"] <= doc["deciles"][-1]["to"]
    # No US calendar and no news layer: those inputs do not exist for this market.
    assert "calendar" not in doc and "news" not in doc


def test_imported_check_puts_the_us_signal_on_the_next_sssec_session():
    if not _have_inputs() or not (ROOT / "artifacts" / "cross.json").is_file():
        return
    doc = china.document(ROOT)
    imported = doc["imported"]
    assert imported is not None
    # Same market, same hit rule: from 2021 both studies must score the same sessions.
    assert abs(imported["holdout"]["base_rate"] - doc["summary"]["holdout_base_rate"]) < 1e-9
    assert imported["all"]["signals"] >= doc["summary"]["flagged"]
