from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from forecast_log.backtest import (
    PATH_HORIZON,
    forecast_rows,
    render_report,
    rolling_paths,
    run_backtest,
    timeline_document,
    write_report,
)
from forecast_log.bars import bar_on, last_completed_bar, load_series, save_series
from forecast_log.estimate import STATE_LOOKBACKS, estimate
from forecast_log.cross import CROSS_SYMBOLS, comparison, document as cross_document
from forecast_log.features import FEATURE_SYMBOLS, describe as describe_features, load_matrix
from forecast_log.horizon import sessions_from_as_of_through_resolve
from forecast_log.paths import series_path
from forecast_log.questions import load_questions, reference_closes
from forecast_log.yahoo import fetch_daily

NY = ZoneInfo("America/New_York")
# The settlement symbols. questions.json is the contract; these are what it points at.
SYMBOLS = ("^VXN", "^NDX")
# 同一套信号，换一只能真下单的基金：用来检查结论是不是绑在指数点位上。
COMPARE_SYMBOLS = ("QQQ",)


def _features_for(root: Path, series) -> object:
    """Forward-looking inputs aligned to this series' sessions, or None if not fetched."""
    matrix = load_matrix(root, [bar.date for bar in series.bars])
    if matrix is None:
        print(f"提示：{', '.join(FEATURE_SYMBOLS)} 还没取，只用价格自身的历史")
    return matrix


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
    for symbol in FEATURE_SYMBOLS:
        try:
            series = fetch_daily(symbol)
        except (RuntimeError, OSError) as error:
            print(f"{symbol} 取数失败，跳过：{error}")
            continue
        save_series(series_path(root, symbol), series)
        last = series.bars[-1]
        print(
            f"{symbol} {len(series.bars)} 根，{series.bars[0].date.isoformat()} → "
            f"{last.date.isoformat()} 收盘 {last.close}，写入 {series_path(root, symbol)}"
        )
    for symbol in (*CROSS_SYMBOLS, *COMPARE_SYMBOLS):
        try:
            series = fetch_daily(symbol)
        except (RuntimeError, OSError, ValueError) as error:
            print(f"{symbol} 取数失败，跳过：{error}")
            continue
        save_series(series_path(root, symbol), series)
        print(
            f"{symbol} {len(series.bars)} 根，{series.bars[0].date.isoformat()} → "
            f"{series.bars[-1].date.isoformat()} 收盘 {series.bars[-1].close}，写入 {series_path(root, symbol)}"
        )
    report = reference_report(root, loaded["^VXN"], loaded["^NDX"])
    print(reference_text(report))
    if not report["matches"]:
        raise SystemExit(1)
    return report


def backtest_command(root: Path) -> dict:
    vxn, ndx = _load_both(root)
    ref = reference_report(root, vxn, ndx)
    print(reference_text(ref))
    extra = _features_for(root, vxn)
    if extra is not None:
        print(f"输入含前瞻数据：{describe_features()}")
    rows = forecast_rows(vxn, ndx, extra)
    # Concave once: the report and the chart read the same rolling paths.
    paths = {
        "vxn": rolling_paths(vxn, PATH_HORIZON, extra),
        "ndx": rolling_paths(ndx, PATH_HORIZON),
    }
    report = run_backtest(vxn, ndx, rows, extra, paths)
    report["reference"] = ref
    path = root / "artifacts" / "backtest_report.json"
    write_report(path, report)
    timeline_path = root / "artifacts" / "timeline.json"
    write_report(timeline_path, timeline_document(vxn, ndx, rows, extra, paths), indent=None)
    print(render_report(report))
    print(f"完整结果写在 {path}")
    print(f"时间轴写在 {timeline_path}")
    if not ref["matches"]:
        raise SystemExit(1)
    return report


def cross_command(root: Path) -> dict:
    """跨资产风险偏好偏多时，之后 5 个交易日内偏涨的点（artifacts/cross.json）。"""
    ndx = load_series(series_path(root, "^NDX"))
    document = cross_document(root, ndx)
    path = root / "artifacts" / "cross.json"
    write_report(path, document, indent=None)
    stats = document["summary"]
    print(
        f"{document['question']['text']}\n"
        f"全部 {stats['n']} 天：打点 {stats['flagged']} 个，命中 {stats['hit']:.1%}，基率 {stats['base_rate']:.1%}\n"
        f"留出期（{stats['holdout_from']} 起）{stats['holdout_n']} 天：打点 {stats['holdout_flagged']} 个，"
        f"命中 {stats['holdout_hit']:.1%}，基率 {stats['holdout_base_rate']:.1%}\n"
        f"最新 {document['latest']['date']}：分数 {document['latest']['score']:.3f}，"
        f"阈值 {document['latest']['threshold']:.3f}，{'打点' if document['latest']['flag'] else '不打点'}\n"
        f"写在 {path}"
    )
    for symbol in COMPARE_SYMBOLS:
        compare_path = series_path(root, symbol)
        if not compare_path.is_file():
            print(f"{symbol} 还没取，跳过对照（forecast-log fetch）")
            continue
        # 信号还是那一条：只把入场/出场价换成这只标的的收盘。
        other = cross_document(root, ndx, instrument=symbol, target=load_series(compare_path))
        other["comparison"] = comparison(document, other)
        other_path = root / "artifacts" / f"cross_{symbol.lower()}.json"
        write_report(other_path, other, indent=None)
        rows = other["comparison"]["rows"]
        print(
            f"\n同一套信号、换到 {symbol}（{other['first_session']} → {other['last_session']}，"
            f"日历对齐 {'是' if other['comparison']['same_flags'] else '否'}）："
        )
        for row in rows:
            print(
                f"  {row['instrument']:6s} 全样本 {row['signals']} 个信号 "
                f"命中 {row['precision']:.1%} 对基率 {row['base_rate']:.1%}（{row['lift']:+.1%}）"
                if row["lift"] is not None
                else f"  {row['instrument']:6s} 无信号"
            )
            print(
                f"         样本外 {row['holdout_signals']} 个 "
                f"命中 {row['holdout_precision']:.1%} 对基率 {row['holdout_base_rate']:.1%}"
                f"（{row['holdout_lift']:+.1%}）"
            )
        print(
            f"  判定不一致 {len(other['comparison']['disagreements'])} 天，写在 {other_path}"
        )
    return document


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
    extra = {symbol: (_features_for(root, loaded) if symbol == "^VXN" else None)
             for symbol, loaded in series.items()}
    as_of_bar = {
        symbol: last_completed_bar(loaded, now) for symbol, loaded in series.items()
    }
    written = []
    forecast_dir.mkdir(parents=True, exist_ok=True)
    for question in questions:
        loaded = series[question.symbol]
        as_of = as_of_bar[question.symbol]
        horizon = sessions_from_as_of_through_resolve(as_of.date, question.resolve_session)
        forecast = estimate(
            loaded.bars, as_of.date, horizon, question.threshold, extra=extra[question.symbol]
        )
        parameters = {
            "threshold": question.threshold,
            "neighbours": forecast.sample_size,
            "state_lookbacks": list(STATE_LOOKBACKS),
            "matched_as": "multiple of the analogue's own start",
            "inputs": "state + CBOE vol complex" if extra[question.symbol] is not None else "state only",
        }
        payload = {
            "question_id": question.id,
            "locked_at": now.astimezone(NY).isoformat(timespec="seconds"),
            "as_of_session": as_of.date.isoformat(),
            "probability": round(forecast.probability, 6),
            "method": {
                "name": "analog_state_v2",
                "horizon_sessions": horizon,
                "sample_size": forecast.sample_size,
                "rule": forecast.rule,
                "median_close": round(forecast.median_close, 4),
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


def reference_report(root: Path, vxn, ndx) -> dict:
    ref = reference_closes(root / "questions.json")
    vxn_bar = bar_on(vxn, ref["session_date"])
    ndx_bar = bar_on(ndx, ref["session_date"])
    if vxn_bar is None or ndx_bar is None:
        raise SystemExit(f"缓存里没有 {ref['session_date'].isoformat()} 的收盘")
    vxn_matches = round(vxn_bar.close, 2) == round(ref["vxn"], 2)
    ndx_matches = round(ndx_bar.close, 2) == round(ref["ndx"], 2)
    return {
        "session_date": ref["session_date"].isoformat(),
        "vxn_close": vxn_bar.close,
        "ndx_close": ndx_bar.close,
        "vxn_reference": ref["vxn"],
        "ndx_reference": ref["ndx"],
        "matches": vxn_matches and ndx_matches,
    }


def reference_text(report: dict) -> str:
    state = "与 questions.json 一致" if report["matches"] else "与 questions.json 不一致"
    return (
        f"{report['session_date']} 收盘 VXN {report['vxn_close']}（合同 {report['vxn_reference']}），"
        f"NDX {report['ndx_close']}（合同 {report['ndx_reference']}）。{state}。"
    )


def _load_both(root: Path):
    missing = [symbol for symbol in SYMBOLS if not series_path(root, symbol).exists()]
    if missing:
        raise SystemExit("还没有日线缓存。先运行 forecast-log fetch")
    return load_series(series_path(root, "^VXN")), load_series(series_path(root, "^NDX"))


def _session_complete(session, now: datetime) -> bool:
    now_ny = now.astimezone(NY)
    if session < now_ny.date():
        return True
    if session > now_ny.date():
        return False
    close_at = datetime(now_ny.year, now_ny.month, now_ny.day, 16, 15, tzinfo=NY)
    return now_ny >= close_at
