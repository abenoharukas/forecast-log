from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from forecast_log.bars import Bar, Series
from forecast_log.estimate import (
    NDX_MIN_N,
    NDX_VOL_BAND,
    NDX_VOL_WINDOW,
    VIX_BAND,
    VIX_MIN_N,
    estimate_ndx_from_closes,
    estimate_vix_from_closes,
    ndx_path,
    vix_path,
)
from forecast_log.horizon import sessions_after

NY = ZoneInfo("America/New_York")
MIN_HISTORY = 252
CLIM_MIN = 30


def brier(probability, outcome) -> float:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if p.size == 0:
        raise ValueError("no forecasts")
    return float(np.mean((p - y) ** 2))


def _rows_vix(series: Series, horizon: int, threshold: float) -> list[dict]:
    bars = series.bars
    closes = np.array([bar.close for bar in bars], dtype=float)
    rows = []
    last = len(bars) - horizon
    for i in range(MIN_HISTORY, last):
        # Prefix ends at i, so the H-session outcome is not in the array the estimator sees.
        estimate = estimate_vix_from_closes(closes[: i + 1], horizon, threshold)
        actual = float(closes[i + horizon])
        rows.append(_row(bars[i], bars[i + horizon].date, estimate, actual, actual > threshold))
    return rows


def _rows_ndx(series: Series, horizon: int) -> list[dict]:
    bars = series.bars
    closes = np.array([bar.close for bar in bars], dtype=float)
    rows = []
    last = len(bars) - horizon
    for i in range(MIN_HISTORY, last):
        # Locked on the baseline day, the live NDX question is "above today's close".
        estimate = estimate_ndx_from_closes(closes[: i + 1], horizon, baseline=float(closes[i]))
        actual = float(closes[i + horizon])
        rows.append(_row(bars[i], bars[i + horizon].date, estimate, actual, actual > closes[i]))
    return rows


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
        "bins": _bins(rows),
        "march_examples": _march_examples(rows),
        "latest": [_public_row(row) for row in rows[-8:]],
        **detail,
    }


def forecast_rows(vix: Series, ndx: Series) -> dict[str, list[dict]]:
    return {
        "vix-h7-gt-20": _rows_vix(vix, horizon=7, threshold=20.0),
        "vix-h12-gt-20": _rows_vix(vix, horizon=12, threshold=20.0),
        "ndx-h12-gt-asof": _rows_ndx(ndx, horizon=12),
    }


def _column(rows: list[dict], key: str, digits: int | None = None) -> list:
    values = [row[key] for row in rows]
    if digits is None:
        return values
    return [round(float(value), digits) for value in values]


def forward_levels(series: Series, horizon: int, kind: str, threshold: float | None = None) -> list[dict]:
    """Point forecast on sessions that have not printed yet. Nothing is written on past dates."""
    closes = np.array([bar.close for bar in series.bars], dtype=float)
    if kind == "vix":
        if len(closes) <= horizon:
            return []
        path = vix_path(closes, horizon, 20.0 if threshold is None else threshold)
    else:
        if len(closes) <= horizon + NDX_VOL_WINDOW:
            return []
        path = ndx_path(closes, horizon)
    days = sessions_after(series.bars[-1].date, horizon)
    return [
        {
            "time": day.isoformat(),
            "horizon": point["horizon"],
            "median": round(point["median"], 2),
            "low": round(point["low"], 2),
            "high": round(point["high"], 2),
        }
        for point, day in zip(path, days, strict=True)
    ]


def timeline_document(vix: Series, ndx: Series, rows: dict[str, list[dict]]) -> dict:
    """Columnar series for the zoomable chart. One forecast row per session."""

    def closes(series: Series) -> dict:
        return {
            "time": [bar.date.isoformat() for bar in series.bars],
            "close": [bar.close for bar in series.bars],
        }

    def forecast(sample: list[dict]) -> dict:
        return {
            "time": _column(sample, "as_of"),
            "resolve": _column(sample, "resolve"),
            "probability": _column(sample, "probability", 4),
            "outcome": _column(sample, "outcome"),
            "median": _column(sample, "median_close", 2),
            "actual": _column(sample, "actual_close", 4),
        }

    return {
        "vix": closes(vix),
        "ndx": closes(ndx),
        "questions": {
            "vix-h7-gt-20": {
                "title": "VIX，7 个交易日后高于 20",
                "symbol": "vix",
                "threshold": 20,
                "forecast": forecast(rows["vix-h7-gt-20"]),
                "forward": forward_levels(vix, 7, "vix", 20.0),
            },
            "vix-h12-gt-20": {
                "title": "VIX，12 个交易日后高于 20",
                "symbol": "vix",
                "threshold": 20,
                "forecast": forecast(rows["vix-h12-gt-20"]),
                "forward": forward_levels(vix, 12, "vix", 20.0),
            },
            "ndx-h12-gt-asof": {
                "title": "纳斯达克 100，12 个交易日后高于当天",
                "symbol": "ndx",
                "threshold": None,
                "forecast": forecast(rows["ndx-h12-gt-asof"]),
                "forward": forward_levels(ndx, 12, "ndx"),
            },
        },
    }


def run_backtest(vix: Series, ndx: Series, rows: dict[str, list[dict]] | None = None) -> dict:
    if rows is None:
        rows = forecast_rows(vix, ndx)
    vix_7 = rows["vix-h7-gt-20"]
    vix_12 = rows["vix-h12-gt-20"]
    ndx_12 = rows["ndx-h12-gt-asof"]
    templates = [
        summarize(
            "vix-h7-gt-20",
            7,
            vix_7,
            {"symbol": "^VIX", "threshold": 20.0, "band": VIX_BAND, "min_n": VIX_MIN_N},
        ),
        summarize(
            "vix-h12-gt-20",
            12,
            vix_12,
            {"symbol": "^VIX", "threshold": 20.0, "band": VIX_BAND, "min_n": VIX_MIN_N},
        ),
        summarize(
            "ndx-h12-gt-asof",
            12,
            ndx_12,
            {
                "symbol": "^NDX",
                "vol_window": NDX_VOL_WINDOW,
                "vol_band": NDX_VOL_BAND,
                "min_n": NDX_MIN_N,
            },
        ),
    ]
    return {
        "generated_at": datetime.now(NY).isoformat(timespec="seconds"),
        "vix_fetched_at": vix.fetched_at,
        "ndx_fetched_at": ndx.fetched_at,
        "vix_span": [vix.bars[0].date.isoformat(), vix.bars[-1].date.isoformat()],
        "ndx_span": [ndx.bars[0].date.isoformat(), ndx.bars[-1].date.isoformat()],
        "templates": templates,
    }


def write_report(path, report: dict, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=indent, separators=(",", ":")) + "\n")


def render_report(report: dict) -> str:
    lines = [
        f"数据 VIX {report['vix_span'][0]} → {report['vix_span'][1]}，取数 {report['vix_fetched_at']}",
        f"数据 NDX {report['ndx_span'][0]} → {report['ndx_span'][1]}，取数 {report['ndx_fetched_at']}",
        "",
        "概率对照的是发生频率，不是某一次的收盘价。说 20% 的那一组，后来大约两成越过门槛，才算对上。",
        "中位数收盘是同一批历史样本的中心，用来看价位差多少，不拿来当提交的预测。",
        "",
    ]
    for template in report["templates"]:
        lines.append(_render_template(template))
    return "\n".join(lines).rstrip() + "\n"


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
