from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from forecast_log.backtest import (
    forecast_rows,
    render_report,
    run_backtest,
    timeline_document,
    write_report,
)
from forecast_log.bars import bar_on, last_completed_bar, load_series, save_series
from forecast_log.estimate import (
    NDX_MIN_N,
    NDX_VOL_BAND,
    NDX_VOL_WINDOW,
    VIX_BAND,
    VIX_MIN_N,
    estimate_ndx,
    estimate_vix,
)
from forecast_log.horizon import sessions_from_as_of_through_resolve
from forecast_log.paths import series_path
from forecast_log.questions import load_questions, reference_closes
from forecast_log.yahoo import fetch_daily

NY = ZoneInfo("America/New_York")
SYMBOLS = ("^VIX", "^NDX")


def fetch_command(root: Path) -> dict:
    loaded = {}
    for symbol in SYMBOLS:
        series = fetch_daily(symbol)
        save_series(series_path(root, symbol), series)
        last = series.bars[-1]
        print(
            f"{symbol} {len(series.bars)} 根，{series.bars[0].date.isoformat()} → "
            f"{last.date.isoformat()} 收盘 {last.close}，写入 {series_path(root, symbol)}"
        )
        loaded[symbol] = series
    report = reference_report(root, loaded["^VIX"], loaded["^NDX"])
    print(reference_text(report))
    if not report["matches"]:
        raise SystemExit(1)
    return report


def backtest_command(root: Path) -> dict:
    vix, ndx = _load_both(root)
    ref = reference_report(root, vix, ndx)
    print(reference_text(ref))
    rows = forecast_rows(vix, ndx)
    report = run_backtest(vix, ndx, rows)
    report["reference"] = ref
    path = root / "artifacts" / "backtest_report.json"
    write_report(path, report)
    timeline_path = root / "artifacts" / "timeline.json"
    write_report(timeline_path, timeline_document(vix, ndx, rows), indent=None)
    print(render_report(report))
    print(f"完整结果写在 {path}")
    print(f"时间轴写在 {timeline_path}")
    if not ref["matches"]:
        raise SystemExit(1)
    return report


def lock_command(root: Path, now: datetime | None = None) -> list[Path]:
    now = datetime.now(NY) if now is None else now
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    questions = load_questions(root / "questions.json")
    forecast_dir = root / "forecasts"
    existing = [forecast_dir / f"{question.id}.json" for question in questions]
    found = [path for path in existing if path.exists()]
    if found:
        names = ", ".join(path.name for path in found)
        raise SystemExit(f"已经锁定，不再改写：{names}")
    series = {symbol: load_series(series_path(root, symbol)) for symbol in SYMBOLS}
    as_of_bar = {
        symbol: last_completed_bar(loaded, now) for symbol, loaded in series.items()
    }
    written = []
    forecast_dir.mkdir(parents=True, exist_ok=True)
    for question in questions:
        loaded = series[question.symbol]
        as_of = as_of_bar[question.symbol]
        horizon = sessions_from_as_of_through_resolve(as_of.date, question.resolve_session)
        if question.symbol == "^VIX":
            estimate = estimate_vix(
                loaded.bars, as_of.date, horizon, threshold=question.threshold
            )
            method = "empirical_band_v1"
            parameters = {"band": VIX_BAND, "min_n": VIX_MIN_N, "threshold": question.threshold}
        else:
            estimate = estimate_ndx(
                loaded.bars,
                as_of.date,
                horizon,
                baseline=question.threshold,
            )
            method = "empirical_vol_band_v1"
            parameters = {
                "vol_window": NDX_VOL_WINDOW,
                "vol_band": NDX_VOL_BAND,
                "min_n": NDX_MIN_N,
                "baseline": question.threshold,
            }
        payload = {
            "question_id": question.id,
            "locked_at": now.astimezone(NY).isoformat(timespec="seconds"),
            "as_of_session": as_of.date.isoformat(),
            "probability": round(estimate.probability, 6),
            "method": {
                "name": method,
                "horizon_sessions": horizon,
                "sample_size": estimate.sample_size,
                "rule": estimate.rule,
                "median_close": round(estimate.median_close, 4),
                "parameters": parameters,
            },
            "evidence": [
                {
                    "date": bar.date.isoformat(),
                    "symbol": question.symbol,
                    "close": bar.close,
                    "fetched_at": loaded.fetched_at,
                }
                for bar in loaded.bars
                if bar.date <= as_of.date
            ],
        }
        path = forecast_dir / f"{question.id}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        written.append(path)
        print(
            f"{question.id} 概率 {payload['probability']:.4f}，"
            f"截止 {payload['as_of_session']}，{horizon} 个交易日，写入 {path.name}"
        )
    return written


def resolve_command(root: Path, now: datetime | None = None) -> None:
    now = datetime.now(NY) if now is None else now
    questions = load_questions(root / "questions.json")
    path = root / "resolutions.json"
    document = {"resolutions": [], "failures": []}
    if path.exists():
        document = json.loads(path.read_text())
        document.setdefault("resolutions", [])
        document.setdefault("failures", [])
    settled = {row["question_id"] for row in document["resolutions"]}
    fresh = {}
    for question in questions:
        if question.id in settled:
            print(f"{question.id} 已经结算，保持原记录")
            continue
        if question.symbol not in fresh:
            fresh[question.symbol] = fetch_daily(question.symbol)
        series = fresh[question.symbol]
        if not _session_complete(question.resolve_session, now):
            print(f"{question.id} 尚未收盘")
            continue
        bar = bar_on(series, question.resolve_session)
        if bar is None:
            if series.bars[-1].date <= question.resolve_session:
                print(f"{question.id} 收盘后行情里还没有这一天，先不记失败")
                continue
            document["failures"] = [
                row for row in document["failures"] if row["question_id"] != question.id
            ]
            document["failures"].append(
                {
                    "question_id": question.id,
                    "resolve_session": question.resolve_session.isoformat(),
                    "reason": "no daily close for the resolve session",
                    "fetched_at": series.fetched_at,
                }
            )
            print(f"{question.id} 这一天没有日线，记失败，不改用别的日期")
            continue
        document["failures"] = [
            row for row in document["failures"] if row["question_id"] != question.id
        ]
        outcome = 1 if bar.close > question.threshold else 0
        document["resolutions"].append(
            {
                "question_id": question.id,
                "resolve_session": question.resolve_session.isoformat(),
                "symbol": question.symbol,
                "close": bar.close,
                "threshold": question.threshold,
                "outcome": outcome,
                "fetched_at": series.fetched_at,
            }
        )
        print(f"{question.id} 收盘 {bar.close}，结果 {outcome}")
    path.write_text(json.dumps(document, indent=2) + "\n")


def score_command(root: Path) -> float | None:
    questions = load_questions(root / "questions.json")
    path = root / "resolutions.json"
    resolutions = {}
    if path.exists():
        for row in json.loads(path.read_text()).get("resolutions", []):
            resolutions[row["question_id"]] = row
    scores = []
    for question in questions:
        forecast_path = root / "forecasts" / f"{question.id}.json"
        if not forecast_path.exists():
            print(f"{question.id} 还没有锁定")
            continue
        if question.id not in resolutions:
            print(f"{question.id} 已锁定，尚未结算")
            continue
        probability = json.loads(forecast_path.read_text())["probability"]
        outcome = resolutions[question.id]["outcome"]
        squared = (probability - outcome) ** 2
        scores.append(squared)
        print(f"{question.id} 概率 {probability:.4f}，结果 {outcome}，Brier {squared:.4f}")
    if not scores:
        print("还没有可以打分的题目。三题都答 0.5 时，平均 Brier 是 0.25。")
        return None
    mean = sum(scores) / len(scores)
    print(f"已结算 {len(scores)} 题，平均 Brier {mean:.4f}。始终答 0.5 的对照是 0.25。")
    return mean


def reference_report(root: Path, vix, ndx) -> dict:
    ref = reference_closes(root / "questions.json")
    vix_bar = bar_on(vix, ref["session_date"])
    ndx_bar = bar_on(ndx, ref["session_date"])
    if vix_bar is None or ndx_bar is None:
        raise SystemExit(f"缓存里没有 {ref['session_date'].isoformat()} 的收盘")
    vix_matches = round(vix_bar.close, 2) == round(ref["vix"], 2)
    ndx_matches = round(ndx_bar.close, 2) == round(ref["ndx"], 2)
    return {
        "session_date": ref["session_date"].isoformat(),
        "vix_close": vix_bar.close,
        "ndx_close": ndx_bar.close,
        "vix_reference": ref["vix"],
        "ndx_reference": ref["ndx"],
        "matches": vix_matches and ndx_matches,
    }


def reference_text(report: dict) -> str:
    state = "与 questions.json 一致" if report["matches"] else "与 questions.json 不一致"
    return (
        f"{report['session_date']} 收盘 VIX {report['vix_close']}（合同 {report['vix_reference']}），"
        f"NDX {report['ndx_close']}（合同 {report['ndx_reference']}）。{state}。"
    )


def _load_both(root: Path):
    missing = [symbol for symbol in SYMBOLS if not series_path(root, symbol).exists()]
    if missing:
        raise SystemExit("还没有日线缓存。先运行 forecast-log fetch")
    return load_series(series_path(root, "^VIX")), load_series(series_path(root, "^NDX"))


def _session_complete(session, now: datetime) -> bool:
    now_ny = now.astimezone(NY)
    if session < now_ny.date():
        return True
    if session > now_ny.date():
        return False
    close_at = datetime(now_ny.year, now_ny.month, now_ny.day, 16, 15, tzinfo=NY)
    return now_ny >= close_at
