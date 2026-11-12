from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from forecast_log.bars import Bar, Series
from forecast_log.estimate import NEIGHBOURS, AnalogModel, min_index
from forecast_log.horizon import sessions_after

NY = ZoneInfo("America/New_York")
CLIM_MIN = 30


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
    rows = []
    for i in range(model.first_index(horizon), len(bars) - horizon):
        # The model only reads closes[0..i]; analogues finish at or before i.
        threshold = float(threshold_at(i, closes))
        estimate = model.forecast(i, horizon, threshold)
        actual = float(closes[i + horizon])
        rows.append(_row(bars[i], bars[i + horizon].date, estimate, actual, actual > threshold))
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
    rows = []
    for t in range(start, len(bars)):
        point = model.forecast(t - 1, 1, 1.0, with_path=True).path[0]
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
    return {
        "rows": rows,
        "tomorrow": tomorrow,
        "skill": {
            "n": len(rows),
            "mae": round(float(error.mean()), 4) if len(rows) else None,
            "median_abs_pct_error": round(float(np.median(pct)), 4) if len(rows) else None,
            "level_corr": round(float(np.corrcoef(predicted, actual)[0, 1]), 4) if len(rows) > 2 else None,
            "baseline_mae": round(float(np.abs(previous - actual).mean()), 4) if len(rows) else None,
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


def timeline_document(vxn: Series, ndx: Series, rows: dict[str, list[dict]], extra=None) -> dict:
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

    return {
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
    vxn: Series, ndx: Series, rows: dict[str, list[dict]] | None = None, extra=None
) -> dict:
    if rows is None:
        rows = forecast_rows(vxn, ndx, extra)
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
    return "\n".join(lines).rstrip() + "\n"


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
