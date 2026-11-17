from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from forecast_log.bars import Bar, Series
from forecast_log.estimate import LONG_LOOKBACK, NEIGHBOURS, AnalogModel, min_index
from forecast_log.horizon import sessions_after

NY = ZoneInfo("America/New_York")
CLIM_MIN = 30
# The rolling path: every session states a 7-session fan, and each step is scored when
# that session prints. This is the only place in the repo that compares a forecast made
# H sessions ago with what actually printed on the session it was about.
PATH_HORIZON = 7
# A "shock" is a move this big or larger by the resolve session, in either direction.
SHOCK = 0.12


def brier(probability, outcome) -> float:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if p.size == 0:
        raise ValueError("no forecasts")
    return float(np.mean((p - y) ** 2))


def _rows(series: Series, horizon: int, threshold_at, extra=None) -> list[dict]:
    """One forecast per session. `threshold_at(i)` gives that day's threshold."""
    bars = series.bars
    closes = np.array([bar.close for bar in bars], dtype=float)
    model = AnalogModel(closes, extra)
    indices = [i for i in range(model.first_index(horizon), len(bars) - horizon) if model.complete[i]]
    thresholds = [float(threshold_at(i, closes)) for i in indices]
    estimates = model.forecast_batch(indices, horizon, thresholds, with_path=False)
    rows = []
    for i, threshold, est in zip(indices, thresholds, estimates):
        actual = float(closes[i + horizon])
        rows.append(_row(bars[i], bars[i + horizon].date, est, actual, actual > threshold))
    return rows


def _rows_vxn(series: Series, horizon: int, threshold: float, extra=None) -> list[dict]:
    return _rows(series, horizon, lambda i, closes: threshold, extra)


def _rows_ndx(series: Series, horizon: int) -> list[dict]:
    # The live NDX question is measured against the baseline day's close.
    return _rows(series, horizon, lambda i, closes: closes[i])


def _row(as_of: Bar, resolve: date, estimate, actual: float, yes: bool) -> dict:
    return {
        "as_of": as_of.date.isoformat(),
        "resolve": resolve.isoformat(),
        "as_of_close": as_of.close,
        "probability": estimate.probability,
        "median_close": estimate.median_close,
        "actual_close": actual,
        "outcome": 1 if yes else 0,
        "sample_size": estimate.sample_size,
        "rule": estimate.rule,
    }


def _climatology(rows: list[dict]) -> list[float | None]:
    """Expanding frequency of outcomes whose resolve session is already known."""
    resolved_sum = 0.0
    resolved_n = 0
    pointer = 0
    climatology: list[float | None] = []
    for row in rows:
        as_of = date.fromisoformat(row["as_of"])
        while pointer < len(rows) and date.fromisoformat(rows[pointer]["resolve"]) <= as_of:
            resolved_sum += rows[pointer]["outcome"]
            resolved_n += 1
            pointer += 1
        if resolved_n >= CLIM_MIN:
            climatology.append(resolved_sum / resolved_n)
        else:
            climatology.append(None)
    return climatology


def warning_skill(rows: list[dict], top: int = 20) -> dict:
    """Can a forecast made on the day of a move warn before the move? Measured, not assumed.

    `lead` is how many sessions earlier the forecast's turning points happen than the
    actual ones. `called_up` is, among the `top` largest jumps, how many the forecast
    already pointed up on the session the jump started.
    """
    predicted = np.array([row["median_close"] for row in rows], dtype=float)
    actual = np.array([row["actual_close"] for row in rows], dtype=float)
    anchor = np.array([row["as_of_close"] for row in rows], dtype=float)

    def extrema(values: np.ndarray, width: int = 4) -> tuple[list[int], list[int]]:
        peaks, troughs = [], []
        for i in range(width, len(values) - width):
            window = values[i - width : i + width + 1]
            if values[i] == window.max():
                peaks.append(i)
            if values[i] == window.min():
                troughs.append(i)
        return peaks, troughs

    lead = []
    for forecast_side, actual_side in zip(extrema(predicted), extrema(actual)):
        for here in actual_side:
            near = [p for p in forecast_side if abs(p - here) <= 6]
            if near:
                lead.append(min(near, key=lambda p: abs(p - here)) - here)

    jump = actual / anchor - 1.0
    biggest = np.argsort(jump)[-top:]
    return {
        "turn_lead_median": None if not lead else float(np.median(lead)),
        "turn_lead_share_early": None if not lead else round(float(np.mean(np.array(lead) < 0)), 4),
        "turn_points": len(lead),
        "jumps_checked": int(len(biggest)),
        "jumps_called_up": int(np.sum(predicted[biggest] > anchor[biggest])),
        "jumps_mean_expected": round(float(np.mean(predicted[biggest] / anchor[biggest] - 1.0)), 4),
        "jumps_mean_realized": round(float(np.mean(jump[biggest])), 4),
    }


def _bins(rows: list[dict]) -> list[dict]:
    probability = np.array([row["probability"] for row in rows])
    outcome = np.array([row["outcome"] for row in rows])
    bins = []
    for step in range(10):
        lo = step / 10
        hi = lo + 0.1
        if step < 9:
            mask = (probability >= lo) & (probability < hi)
        else:
            mask = (probability >= lo) & (probability <= hi + 1e-12)
        count = int(mask.sum())
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": count,
                "mean_probability": None if count == 0 else float(probability[mask].mean()),
                "actual_frequency": None if count == 0 else float(outcome[mask].mean()),
            }
        )
    return bins


def _march_examples(rows: list[dict]) -> list[dict]:
    chosen = []
    seen: set[int] = set()
    for row in rows:
        as_of = date.fromisoformat(row["as_of"])
        if as_of.month == 3 and as_of.year not in seen:
            seen.add(as_of.year)
            chosen.append(_public_row(row))
    return chosen


def _public_row(row: dict) -> dict:
    return {
        "as_of": row["as_of"],
        "resolve": row["resolve"],
        "as_of_close": row["as_of_close"],
        "probability": round(row["probability"], 6),
        "median_close": round(row["median_close"], 4),
        "actual_close": row["actual_close"],
        "outcome": row["outcome"],
        "sample_size": row["sample_size"],
        "rule": row["rule"],
    }


def summarize(template_id: str, horizon: int, rows: list[dict], detail: dict) -> dict:
    probability = [row["probability"] for row in rows]
    outcome = [row["outcome"] for row in rows]
    climatology = _climatology(rows)
    clim_p = []
    clim_y = []
    for row, clim in zip(rows, climatology):
        if clim is None:
            continue
        clim_p.append(clim)
        clim_y.append(row["outcome"])
    step = rows[::horizon]
    error = [abs(row["median_close"] - row["actual_close"]) for row in rows]
    pct_error = [
        abs(row["median_close"] - row["actual_close"]) / row["actual_close"] for row in rows
    ]
    return {
        "id": template_id,
        "horizon_sessions": horizon,
        "n_forecasts": len(rows),
        "first_as_of": rows[0]["as_of"],
        "last_as_of": rows[-1]["as_of"],
        "last_resolve": rows[-1]["resolve"],
        "mean_probability": float(np.mean(probability)),
        "actual_frequency": float(np.mean(outcome)),
        "brier": brier(probability, outcome),
        "brier_half": brier([0.5] * len(rows), outcome),
        "brier_climatology": None if not clim_p else brier(clim_p, clim_y),
        "n_climatology": len(clim_p),
        "brier_nonoverlapping": brier(
            [row["probability"] for row in step],
            [row["outcome"] for row in step],
        ),
        "n_nonoverlapping": len(step),
        "median_close_mae": float(np.mean(error)),
        "median_abs_pct_error": float(np.median(pct_error)),
        "rules": dict(Counter(row["rule"] for row in rows)),
        "warning": warning_skill(rows),
        "bins": _bins(rows),
        "march_examples": _march_examples(rows),
        "latest": [_public_row(row) for row in rows[-8:]],
        **detail,
    }


def forecast_rows(vxn: Series, ndx: Series, extra: np.ndarray | None = None) -> dict[str, list[dict]]:
    return {
        "vxn-h7-gt-30": _rows_vxn(vxn, horizon=7, threshold=30.0, extra=extra),
        "vxn-h12-gt-30": _rows_vxn(vxn, horizon=12, threshold=30.0, extra=extra),
        "ndx-h12-gt-asof": _rows_ndx(ndx, horizon=12),
    }


def _column(rows: list[dict], key: str, digits: int | None = None) -> list:
    values = [row[key] for row in rows]
    if digits is None:
        return values
    return [round(float(value), digits) for value in values]


def forward_levels(
    series: Series, horizon: int, threshold: float | None = None, extra=None
) -> list[dict]:
    """Point forecast on sessions that have not printed yet. Nothing is written on past dates."""
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    model = AnalogModel(closes, extra)
    if len(closes) - 1 < model.first_index(horizon):
        return []
    estimate = model.forecast(
        len(closes) - 1,
        horizon,
        float(closes[-1]) if threshold is None else float(threshold),
        with_path=True,
    )
    days = sessions_after(series.bars[-1].date, horizon)
    return [
        {
            "time": day.isoformat(),
            "horizon": point.horizon,
            "median": round(point.median, 2),
            "low": round(point.low, 2),
            "high": round(point.high, 2),
        }
        for point, day in zip(estimate.path, days, strict=True)
    ]


def skill(sample: list[dict]) -> dict:
    """Does the forecast follow the change it is supposed to predict, or just restate the level?"""
    if len(sample) < 2:
        return {"change_corr": None, "direction_hit": None, "n": 0}
    anchor = np.array([row["as_of_close"] for row in sample], dtype=float)
    predicted = np.log(np.array([row["median_close"] for row in sample]) / anchor)
    realized = np.log(np.array([row["actual_close"] for row in sample]) / anchor)
    return {
        "change_corr": round(float(np.corrcoef(predicted, realized)[0, 1]), 4),
        "direction_hit": round(float(np.mean(np.sign(predicted) == np.sign(realized))), 4),
        "n": len(sample),
    }


def daily_forecasts(series: Series, extra=None) -> dict:
    """每根 K 线都是「用前一根及以前的数据预测这一根」，画在这一根上。

    这是「当天」的预测：t 日收盘前你不可能知道 t 日的收盘，所以只能用 t-1。
    """
    bars = series.bars
    closes = np.array([bar.close for bar in bars], dtype=float)
    model = AnalogModel(closes, extra)
    start = model.first_index(1) + 1
    ts = [t for t in range(start, len(bars)) if model.complete[t - 1]]
    ests = model.forecast_batch([t - 1 for t in ts], 1, [1.0] * len(ts), with_path=True)
    rows = []
    for t, est in zip(ts, ests):
        point = est.path[0]
        rows.append(
            {
                "time": bars[t].date.isoformat(),
                "previous": closes[t - 1],
                "median": round(point.median, 2),
                "low": round(point.low, 2),
                "high": round(point.high, 2),
                "actual": closes[t],
            }
        )
    # 风险等级 = 今天这个区间宽度，在过去一年里排第几（0-100）。
    widths = np.array(
        [(row["high"] - row["low"]) / row["median"] for row in rows], dtype=float
    )
    for k, row in enumerate(rows):
        window = widths[max(0, k - 252) : k + 1]
        row["risk"] = int(round(float((window < widths[k]).mean()) * 100))

    tomorrow = None
    if len(bars) >= model.first_index(1) + 1:
        point = model.forecast(len(bars) - 1, 1, 1.0, with_path=True).path[0]
        when = sessions_after(bars[-1].date, 1)[0]
        width = (point.high - point.low) / point.median
        tomorrow = {
            "time": when.isoformat(),
            "median": round(point.median, 2),
            "low": round(point.low, 2),
            "high": round(point.high, 2),
            "risk": int(round(float((widths[-252:] < width).mean()) * 100)),
        }

    error = np.array([abs(row["median"] - row["actual"]) for row in rows], dtype=float)
    pct = np.array(
        [abs(row["median"] - row["actual"]) / row["actual"] for row in rows], dtype=float
    )
    predicted = np.array([row["median"] for row in rows], dtype=float)
    actual = np.array([row["actual"] for row in rows], dtype=float)
    previous = np.array([row["previous"] for row in rows], dtype=float)
    hi = np.array([row["high"] for row in rows], dtype=float)
    lo = np.array([row["low"] for row in rows], dtype=float)
    return {
        "rows": rows,
        "tomorrow": tomorrow,
        "skill": {
            "n": len(rows),
            "mae": round(float(error.mean()), 4) if len(rows) else None,
            "median_abs_pct_error": round(float(np.median(pct)), 4) if len(rows) else None,
            "mean_abs_pct_error": round(float(np.mean(pct)), 4) if len(rows) else None,
            "level_corr": round(float(np.corrcoef(predicted, actual)[0, 1]), 4) if len(rows) > 2 else None,
            "baseline_mae": round(float(np.abs(previous - actual).mean()), 4) if len(rows) else None,
            "median_abs_move": round(float(np.median(np.abs(actual / previous - 1.0))), 4)
            if len(rows)
            else None,
            "mean_abs_move": round(float(np.mean(np.abs(actual / previous - 1.0))), 4)
            if len(rows)
            else None,
            "mean_width": round(float(np.mean((hi - lo) / predicted)), 4) if len(rows) else None,
            "band_coverage": round(
                float(
                    np.mean(
                        (actual >= np.array([r["low"] for r in rows]))
                        & (actual <= np.array([r["high"] for r in rows]))
                    )
                ),
                4,
            )
            if len(rows)
            else None,
            "direction_hit": round(
                float(np.mean(np.sign(predicted - previous) == np.sign(actual - previous))), 4
            )
            if len(rows)
            else None,
        },
    }


def rolling_paths(series: Series, horizon: int = PATH_HORIZON, extra=None) -> list[dict]:
    """One forecast per session: a full fan, plus the closes it will be scored against.

    `forecast(i, ...)` only reads closes[0..i] and analogues that finished at or before
    `i`, so every row here is what the model would have printed that evening. `actual`
    holds the later closes; nothing in the forecast saw them.
    """
    bars = series.bars
    closes = np.array([bar.close for bar in bars], dtype=float)
    model = AnalogModel(closes, extra)
    start = model.first_index(horizon) + 1
    indices = [i for i in range(start, len(bars)) if model.complete[i]]
    estimates = model.forecast_batch(indices, horizon, [closes[i] for i in indices], with_path=True)
    rows = []
    # Runs to the last printed session: a forecast made today still has a path, it just has
    # no `actual` yet for the steps that are still in the future (None, never dropped early).
    for i, estimate in zip(indices, estimates):
        recent = np.diff(np.log(closes[i - horizon : i + 1]))
        rows.append(
            {
                "as_of": bars[i].date.isoformat(),
                # The session the whole `horizon`-long path lands on, if it has printed.
                "resolve": bars[i + horizon].date.isoformat() if i + horizon < len(bars) else None,
                "anchor": float(closes[i]),
                "recent_vol": float(np.std(recent)),
                # What a "revert to the one-year middle" rule would say, for the direction check.
                "center": float(np.median(closes[max(0, i - LONG_LOOKBACK) : i + 1])),
                "path": [
                    {
                        "step": point.horizon,
                        "date": bars[i + point.horizon].date.isoformat()
                        if i + point.horizon < len(bars)
                        else None,
                        "median": round(point.median, 4),
                        "low": round(point.low, 4),
                        "high": round(point.high, 4),
                    }
                    for point in estimate.path
                ],
                "actual": [
                    float(closes[i + step]) if i + step < len(bars) else None
                    for step in range(1, horizon + 1)
                ],
            }
        )
    return rows


def path_summary(rows: list[dict], horizon: int = PATH_HORIZON) -> dict:
    """How far off a forecast made H sessions earlier is, by step and at the terminal step."""
    # Only steps whose session has printed can be scored; the newest rows still run past the
    # last close, and those cells stay empty rather than getting a made-up value.
    scored = [row for row in rows if row["actual"][-1] is not None]
    if not scored:
        return {"horizon": horizon, "n": 0, "steps": [], "latest": []}
    anchor = np.array([row["anchor"] for row in scored], dtype=float)
    steps = []
    for step in range(1, horizon + 1):
        sample = [row for row in rows if row["actual"][step - 1] is not None]
        step_anchor = np.array([row["anchor"] for row in sample], dtype=float)
        actual = np.array([row["actual"][step - 1] for row in sample], dtype=float)
        median = np.array([row["path"][step - 1]["median"] for row in sample], dtype=float)
        low = np.array([row["path"][step - 1]["low"] for row in sample], dtype=float)
        high = np.array([row["path"][step - 1]["high"] for row in sample], dtype=float)
        error = np.abs(median - actual)
        # The baseline a longer horizon has to beat: nothing moves from where it is today.
        naive = np.abs(step_anchor - actual)
        coverage = float(np.mean((actual >= low) & (actual <= high)))
        relative = np.abs(actual / step_anchor - 1.0)
        steps.append(
            {
                "step": step,
                "n": len(sample),
                "mae": round(float(error.mean()), 4),
                "naive_mae": round(float(naive.mean()), 4),
                "beats_naive": round(float(1.0 - error.mean() / naive.mean()), 4),
                "median_abs_pct": round(float(np.median(error / actual)), 4),
                "mean_abs_pct": round(float(np.mean(error / actual)), 4),
                "naive_median_abs_pct": round(float(np.median(naive / actual)), 4),
                "coverage": round(coverage, 4),
                "mean_width": round(float(np.mean((high - low) / median)), 4),
                # 这一档实际走了多远。带宽和它的比值才是「包住」的原因。
                "median_abs_move": round(
                    float(np.median(np.abs(actual / step_anchor - 1.0))), 4
                ),
                "mean_abs_move": round(float(np.mean(np.abs(actual / step_anchor - 1.0))), 4),
                # The constant "today's close +/- x%" band that would cover just as often.
                "naive_width": round(2.0 * float(np.quantile(relative, coverage)), 4),
                "direction_hit": round(
                    float(np.mean(np.sign(median - step_anchor) == np.sign(actual - step_anchor))), 4
                ),
            }
        )

    terminal = steps[-1]
    actual = np.array([row["actual"][-1] for row in scored], dtype=float)
    median = np.array([row["path"][-1]["median"] for row in scored], dtype=float)
    low = np.array([row["path"][-1]["low"] for row in scored], dtype=float)
    high = np.array([row["path"][-1]["high"] for row in scored], dtype=float)
    width = (high - low) / median
    shock = np.abs(actual / anchor - 1.0) > SHOCK
    recent = np.array([row["recent_vol"] for row in scored], dtype=float)
    center = np.array([row["center"] for row in scored], dtype=float)

    def terciles(signal: np.ndarray) -> list[float | None]:
        if np.all(np.isnan(signal)):
            return [None, None, None]
        cuts = np.nanquantile(signal, [1 / 3, 2 / 3])
        bands = [signal <= cuts[0], (signal > cuts[0]) & (signal <= cuts[1]), signal > cuts[1]]
        return [None if not band.any() else round(float(shock[band].mean()), 4) for band in bands]

    latest = [
        {
            "as_of": row["as_of"],
            "resolve": row["resolve"],
            "anchor": round(row["anchor"], 2),
            "median": round(row["path"][-1]["median"], 2),
            "low": round(row["path"][-1]["low"], 2),
            "high": round(row["path"][-1]["high"], 2),
            "actual": round(row["actual"][-1], 2),
            "error_pct": round((row["path"][-1]["median"] / row["actual"][-1] - 1) * 100, 2),
            "inside": bool(
                row["path"][-1]["low"] <= row["actual"][-1] <= row["path"][-1]["high"]
            ),
        }
        for row in scored[-8:]
    ]
    return {
        "horizon": horizon,
        "n": len(scored),
        "first_as_of": rows[0]["as_of"],
        "last_as_of": rows[-1]["as_of"],
        "last_resolve": scored[-1]["resolve"],
        "steps": steps,
        "terminal": {
            "step": terminal["step"],
            "mae": terminal["mae"],
            "naive_mae": terminal["naive_mae"],
            "beats_naive": terminal["beats_naive"],
            "median_abs_pct": terminal["median_abs_pct"],
            "naive_median_abs_pct": terminal["naive_median_abs_pct"],
            "coverage": terminal["coverage"],
            "direction_hit": terminal["direction_hit"],
            "direction_hit_reversion": round(
                float(np.mean(np.sign(center - anchor) == np.sign(actual - anchor))), 4
            ),
            "always_down_hit": round(float(np.mean(actual < anchor)), 4),
            "change_corr": round(
                float(np.corrcoef(np.log(median / anchor), np.log(actual / anchor))[0, 1]), 4
            ),
            "shock_threshold": SHOCK,
            "shock_base_rate": round(float(shock.mean()), 4),
            "shock_by_width": terciles(width),
            "shock_by_recent_vol": terciles(recent),
        },
        "latest": latest,
    }


def ladder_document(rows: list[dict], dates: list[str], pending: list[dict] | None = None) -> dict:
    """One column per lead time N, aligned to the session axis (null where nothing was said).

    N = 1 是提前一个交易日说的，N = 7 是提前七个。列上每一格都是「目标日那天收盘前 N 天
    说的中位和区间」，与实际收盘同一根时间轴对齐，图和表都能直接比。
    """
    horizon = len(rows[0]["path"]) if rows else PATH_HORIZON
    index = {day: position for position, day in enumerate(dates)}
    blank = [None] * len(dates)
    columns = {
        str(step): {"median": list(blank), "low": list(blank), "high": list(blank)}
        for step in range(1, horizon + 1)
    }
    for row in rows:
        for point in row["path"]:
            position = index.get(point["date"])
            if position is None:
                continue
            column = columns[str(point["step"])]
            column["median"][position] = round(point["median"], 2)
            column["low"][position] = round(point["low"], 2)
            column["high"][position] = round(point["high"], 2)
    return {
        "horizons": list(range(1, horizon + 1)),
        "columns": columns,
        # 今天出发、还没结算的那一次：目标日在时间轴之外。
        "pending": pending or [],
        "summary": path_summary(rows),
    }


def path_document(rows: list[dict], horizon: int = PATH_HORIZON) -> dict:
    """Columnar terminal step, keyed by the resolve session, for the comparison chart."""
    resolved = [row for row in rows if row["resolve"] is not None]
    return {
        "horizon": horizon,
        "time": [row["resolve"] for row in resolved],
        "as_of": [row["as_of"] for row in resolved],
        "anchor": [round(row["anchor"], 2) for row in resolved],
        "median": [round(row["path"][-1]["median"], 2) for row in resolved],
        "low": [round(row["path"][-1]["low"], 2) for row in resolved],
        "high": [round(row["path"][-1]["high"], 2) for row in resolved],
        "actual": [round(row["actual"][-1], 2) for row in resolved],
        "summary": path_summary(rows, horizon),
    }


def timeline_document(
    vxn: Series,
    ndx: Series,
    rows: dict[str, list[dict]],
    extra=None,
    paths: dict[str, list[dict]] | None = None,
) -> dict:
    """Columnar series for the zoomable chart. One forecast row per session."""

    def closes(series: Series) -> dict:
        return {
            "time": [bar.date.isoformat() for bar in series.bars],
            "close": [bar.close for bar in series.bars],
        }

    def forecast(sample: list[dict]) -> dict:
        # The level is drawn on the session the forecast was made, not on the session it is
        # about. Drawn on the resolve session, a level forecast is the actual series shifted
        # right by the horizon, which reads as a lag even when the forecast has skill.
        return {
            "time": _column(sample, "as_of"),
            "resolve": _column(sample, "resolve"),
            "probability": _column(sample, "probability", 4),
            "outcome": _column(sample, "outcome"),
            "median": _column(sample, "median_close", 2),
            "actual": _column(sample, "actual_close", 4),
            "anchor": _column(sample, "as_of_close", 2),
            "skill": skill(sample),
        }

    daily = {"vxn": daily_forecasts(vxn, extra), "ndx": daily_forecasts(ndx)}
    if paths is None:
        paths = {
            "vxn": rolling_paths(vxn, PATH_HORIZON, extra),
            "ndx": rolling_paths(ndx, PATH_HORIZON),
        }

    # 下一次对比：今天出发的 7 日路径，最后一点就是还没结算的那一天。
    forward = {
        "vxn": forward_levels(vxn, PATH_HORIZON, None, extra),
        "ndx": forward_levels(ndx, PATH_HORIZON),
    }

    return {
        "rolling": {
            symbol: path_document(sample) for symbol, sample in paths.items() if sample
        },
        # 提前 N 天说的 vs 那天真的到的，N 从 1 到 7，对齐到目标日那根 K 线。
        "ladder": {
            "vxn": ladder_document(
                paths["vxn"], [bar.date.isoformat() for bar in vxn.bars], forward["vxn"]
            )
        },
        "forward": forward,
        "vxn": closes(vxn),
        "ndx": closes(ndx),
        "daily": {
            key: {
                "time": _column(value["rows"], "time"),
                "median": _column(value["rows"], "median", 2),
                "low": _column(value["rows"], "low", 2),
                "high": _column(value["rows"], "high", 2),
                "actual": _column(value["rows"], "actual", 2),
                "previous": _column(value["rows"], "previous", 2),
                "risk": _column(value["rows"], "risk"),
                "tomorrow": value["tomorrow"],
                "skill": value["skill"],
            }
            for key, value in daily.items()
        },
        "questions": {
            "vxn-h7-gt-30": {
                "title": "VXN，7 个交易日后高于 30",
                "symbol": "vxn",
                "horizon": 7,
                "threshold": 30,
                "forecast": forecast(rows["vxn-h7-gt-30"]),
                "forward": forward_levels(vxn, 7, 30.0, extra),
            },
            "vxn-h12-gt-30": {
                "title": "VXN，12 个交易日后高于 30",
                "symbol": "vxn",
                "horizon": 12,
                "threshold": 30,
                "forecast": forecast(rows["vxn-h12-gt-30"]),
                "forward": forward_levels(vxn, 12, 30.0, extra),
            },
            "ndx-h12-gt-asof": {
                "title": "纳斯达克 100，12 个交易日后高于当天",
                "symbol": "ndx",
                "horizon": 12,
                "threshold": None,
                "forecast": forecast(rows["ndx-h12-gt-asof"]),
                "forward": forward_levels(ndx, 12),
            },
        },
    }


def run_backtest(
    vxn: Series,
    ndx: Series,
    rows: dict[str, list[dict]] | None = None,
    extra=None,
    paths: dict[str, list[dict]] | None = None,
) -> dict:
    if rows is None:
        rows = forecast_rows(vxn, ndx, extra)
    if paths is None:
        paths = {
            "vxn": rolling_paths(vxn, PATH_HORIZON, extra),
            "ndx": rolling_paths(ndx, PATH_HORIZON),
        }
    vxn_7 = rows["vxn-h7-gt-30"]
    vxn_12 = rows["vxn-h12-gt-30"]
    ndx_12 = rows["ndx-h12-gt-asof"]
    templates = [
        summarize(
            "vxn-h7-gt-30",
            7,
            vxn_7,
            {"symbol": "^VXN", "threshold": 30.0, "neighbours": NEIGHBOURS},
        ),
        summarize(
            "vxn-h12-gt-30",
            12,
            vxn_12,
            {"symbol": "^VXN", "threshold": 30.0, "neighbours": NEIGHBOURS},
        ),
        summarize(
            "ndx-h12-gt-asof",
            12,
            ndx_12,
            {"symbol": "^NDX", "neighbours": NEIGHBOURS},
        ),
    ]
    return {
        "generated_at": datetime.now(NY).isoformat(timespec="seconds"),
        "vxn_fetched_at": vxn.fetched_at,
        "ndx_fetched_at": ndx.fetched_at,
        "vxn_span": [vxn.bars[0].date.isoformat(), vxn.bars[-1].date.isoformat()],
        "vxn_inputs": "state + CBOE vol complex" if extra is not None else "state only",
        "ndx_span": [ndx.bars[0].date.isoformat(), ndx.bars[-1].date.isoformat()],
        "templates": templates,
        "path": {
            symbol: path_summary(sample)
            for symbol, sample in paths.items()
            if sample
        },
    }


def write_report(path, report: dict, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=indent, separators=(",", ":")) + "\n")


def render_report(report: dict) -> str:
    lines = [
        f"数据 VXN {report['vxn_span'][0]} → {report['vxn_span'][1]}，取数 {report['vxn_fetched_at']}",
        f"数据 NDX {report['ndx_span'][0]} → {report['ndx_span'][1]}，取数 {report['ndx_fetched_at']}",
        "",
        "概率对照的是发生频率，不是某一次的收盘价。说 20% 的那一组，后来大约两成越过门槛，才算对上。",
        "中位点位从当天收盘出发，由类比日之后的实际涨跌幅推出来，不是把历史价位搬过来。",
        "类比只看 1 周、1 月、1 年的涨跌和相对 1 年中位数的偏离，从 200 个最像的已完成类比里取。",
        "",
    ]
    for template in report["templates"]:
        lines.append(_render_template(template))
    for symbol, summary in report.get("path", {}).items():
        lines.append(_render_path(symbol, summary))
    return "\n".join(lines).rstrip() + "\n"


def _render_path(symbol: str, summary: dict) -> str:
    if not summary.get("n"):
        return ""
    horizon = summary["horizon"]
    terminal = summary["terminal"]
    lines = [
        f"## {symbol.upper()} 滚动 {horizon} 日区间（每一格都对过实际）",
        (
            f"样本 {summary['n']} 次，{summary['first_as_of']} → {summary['last_as_of']}，"
            f"最后一次结算是 {summary['last_resolve']}"
        ),
        "每个交易日都发出一个区间；到这个区间对应的那一天收盘后，用实际收盘打分。",
        "步长  平均绝对误差  猜今天不动  只好  中位百分比误差  猜今天  区间覆盖  平均带宽  同覆盖的固定带宽  方向命中",
    ]
    for step in summary["steps"]:
        lines.append(
            f"{step['step']:4d}  {step['mae']:12.2f}  {step['naive_mae']:10.2f}  "
            f"{step['beats_naive']:+5.1%}  {step['median_abs_pct']:14.1%}  "
            f"{step['naive_median_abs_pct']:6.1%}  {step['coverage']:7.1%}  "
            f"{step['mean_width']:7.1%}  {step['naive_width']:14.1%}  {step['direction_hit']:7.1%}"
        )
    width = terminal["shock_by_width"]
    recent = terminal["shock_by_recent_vol"]
    lines.append(
        f"第 {horizon} 日大动（|涨跌| > {terminal['shock_threshold']:.0%}）的基准概率 "
        f"{terminal['shock_base_rate']:.1%}；按区间宽度分窄/中/宽："
        + " / ".join("—" if value is None else f"{value:.1%}" for value in width)
        + "；按过去一周已实现波动分："
        + " / ".join("—" if value is None else f"{value:.1%}" for value in recent)
    )
    lines.append(
        f"方向命中 {terminal['direction_hit']:.1%}，而「总是猜跌」是 {terminal['always_down_hit']:.1%}，"
        f"「朝一年中位数回归」是 {terminal['direction_hit_reversion']:.1%}；"
        f"变动相关性 {terminal['change_corr']:+.2f}"
    )
    lines.append("最近 8 次：发出日 → 结算日，当时的收盘，那时说的中位与区间，实际收盘，偏差")
    for row in summary["latest"]:
        mark = "在" if row["inside"] else "出"
        lines.append(
            f"{row['as_of']} → {row['resolve']}  {row['anchor']:8.2f}  "
            f"中位 {row['median']:8.2f} 区间 {row['low']:8.2f}–{row['high']:8.2f}  "
            f"实际 {row['actual']:8.2f}  {row['error_pct']:+6.1f}%  {mark}"
        )
    lines.append("")
    return "\n".join(lines)


def _render_warning(warning: dict) -> str:
    lead = warning["turn_lead_median"]
    lead_text = "样本不足" if lead is None else (
        f"拐点提前中位 {lead:+.1f} 根，提前的占 {warning['turn_lead_share_early']:.0%}"
        f"（{warning['turn_points']} 个配对）"
    )
    return (
        f"预警能力：{lead_text}；{warning['jumps_checked']} 次最大暴涨里，"
        f"起点当天预测上涨的 {warning['jumps_called_up']} 次"
        f"（预测平均 {warning['jumps_mean_expected']:+.1%}，实际 {warning['jumps_mean_realized']:+.1%}）"
    )


def _render_template(template: dict) -> str:
    gap = template["mean_probability"] - template["actual_frequency"]
    clim = template["brier_climatology"]
    clim_text = "尚无" if clim is None else f"{clim:.4f}"
    lines = [
        f"## {template['id']}",
        (
            f"样本 {template['n_forecasts']} 次，{template['first_as_of']} → {template['last_as_of']}，"
            f"最后结算 {template['last_resolve']}"
        ),
        (
            f"平均概率 {template['mean_probability']:.4f}，实际频率 {template['actual_frequency']:.4f}，"
            f"差 {gap:+.4f}"
        ),
        (
            f"Brier {template['brier']:.4f}，始终答 0.5 为 {template['brier_half']:.4f}，"
            f"当时已知的历史基率 {clim_text}"
        ),
        (
            f"每 {template['horizon_sessions']} 个交易日取一题的 Brier {template['brier_nonoverlapping']:.4f}"
            f"（{template['n_nonoverlapping']} 次，窗口不重叠）"
        ),
        (
            f"中位数收盘的平均绝对误差 {template['median_close_mae']:.2f}，"
            f"百分比误差的中位数 {template['median_abs_pct_error']:.1%}"
        ),
        f"估计规则次数 {template['rules']}",
        _render_warning(template["warning"]),
        "概率区间  题数  平均概率  实际频率",
    ]
    for bucket in template["bins"]:
        if bucket["n"] == 0:
            continue
        lines.append(
            f"{bucket['lo']:.1f}–{bucket['hi']:.1f}  {bucket['n']:5d}  "
            f"{bucket['mean_probability']:.3f}    {bucket['actual_frequency']:.3f}"
        )
    lines.append("每年 3 月第一个交易日的推演（最近 12 年）：")
    lines.append("截止日        当时收盘  概率    中位数   实际收盘  越过门槛")
    for row in template["march_examples"][-12:]:
        lines.append(
            f"{row['as_of']}  {row['as_of_close']:8.2f}  {row['probability']:.3f}  "
            f"{row['median_close']:8.2f}  {row['actual_close']:8.2f}  {row['outcome']}"
        )
    lines.append("")
    return "\n".join(lines)
