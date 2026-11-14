"""The eval bench: build historical questions, run predictors on them, score honestly.

The questions are generated from one series' own history, so every one of them is already
resolved and nothing has to be waited on. For each as-of session the threshold is a
trailing quantile of what was visible then, which keeps the base rate near 40% instead of
letting everything resolve yes.

Three things this bench does that a demo does not:

- It compares against the number that is hard to beat: the unconditional base rate.
  A predictor that cannot beat it has not earned a line in a report.
- It reports calibration (ECE and a reliability table), not just Brier. Brier rewards
  hedging; calibration shows whether "0.8" meant 0.8.
- It reports cost and latency next to quality, because a forecast that costs $0.40 and
  40 seconds is a different product from one that costs $0.03 and 6 seconds.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np

from forecast_log.agent.gate import AsOf
from forecast_log.agent.llm import LLM
from forecast_log.agent.loop import Agent, QuestionView, Trace
from forecast_log.agent.tools import Toolbox
from forecast_log.bars import Series, load_series
from forecast_log.horizon import sessions_from_as_of_through_resolve
from forecast_log.paths import series_path
from forecast_log.questions import load_questions


@dataclass(frozen=True)
class LabeledQuestion:
    question: QuestionView
    outcome: int
    as_of: date
    as_of_close: float
    family: str = "level"

    @property
    def id(self) -> str:
        return self.question.id


@dataclass
class Prediction:
    probability: float
    cost_usd: float = 0.0
    latency_s: float = 0.0
    tool_calls: int = 0
    gate_hits: int = 0
    note: str = ""


class Predictor(Protocol):
    name: str

    def predict(self, row: LabeledQuestion) -> Prediction: ...


# --------------------------------------------------------------------- questions


THRESHOLD_MODES = ("level", "quantile", "band_up", "band_down")


def _threshold(mode: str, trailing: np.ndarray, as_of_close: float, quantile: float) -> float:
    """Four question families, so no single predictor gets to look good on one shape.

    level:     "higher in h sessions?" - the pure directional question, base rate ~55%.
    quantile:  "above its own 60th percentile?" - a state question.
    band_up:   "+5% in h sessions?" - a rare event, where calibration is hard.
    band_down: "-5% in h sessions?" - the other rare tail.
    """
    if mode == "level":
        return round(as_of_close, 2)
    if mode == "quantile":
        return round(float(np.quantile(trailing, quantile)), 2)
    if mode == "band_up":
        return round(as_of_close * 1.05, 2)
    if mode == "band_down":
        return round(as_of_close * 0.95, 2)
    raise ValueError(f"unknown threshold mode {mode!r}")


def build_questions(
    series: Series,
    *,
    stride: int = 10,
    horizons: tuple[int, ...] = (5, 10),
    modes: tuple[str, ...] = ("level", "quantile"),
    quantile: float = 0.6,
    window: int = 504,
    start: date | None = None,
    limit: int | None = None,
) -> list[LabeledQuestion]:
    """Binary questions whose answer is already printed in the series.

    `window` sessions of history are required before the first question, so the trailing
    distribution is never estimated from a short or partially future stretch.
    """
    bars = [bar for bar in series.bars if start is None or bar.date >= start]
    closes = np.array([bar.close for bar in bars], dtype=float)
    if len(bars) < window + max(horizons) + 1:
        raise ValueError("series is too short for these parameters")
    for mode in modes:
        if mode not in THRESHOLD_MODES:
            raise ValueError(f"unknown threshold mode {mode!r}")

    rows: list[LabeledQuestion] = []
    for index in range(window - 1, len(bars) - max(horizons)):
        if (index - (window - 1)) % stride:
            continue
        as_of = bars[index].date
        as_of_close = float(closes[index])
        trailing = closes[index - window + 1 : index + 1]
        for horizon in horizons:
            target = index + horizon
            resolve = bars[target].date
            for mode in modes:
                threshold = _threshold(mode, trailing, as_of_close, quantile)
                outcome = 1 if float(closes[target]) > threshold else 0
                tag = "" if mode == "level" else f"-{mode}"
                rows.append(
                    LabeledQuestion(
                        question=QuestionView(
                            id=f"{series.symbol}-{as_of.isoformat()}-h{horizon}{tag}".replace("^", ""),
                            text=(
                                f"Will {series.symbol} close strictly above {threshold} on "
                                f"{resolve.isoformat()}, the session {horizon} sessions after "
                                f"{as_of.isoformat()}?"
                            ),
                            symbol=series.symbol,
                            threshold=threshold,
                            resolve_session=resolve,
                            horizon_sessions=horizon,
                            kind="synthetic",
                        ),
                        outcome=outcome,
                        as_of=as_of,
                        as_of_close=as_of_close,
                        family=mode,
                    )
                )
    if limit is not None and len(rows) > limit:
        # One slice per question family, so a short run is balanced instead of
        # accidental: a 12-question run still covers every shape of question.
        groups: dict[tuple[int, str], list[LabeledQuestion]] = {}
        for row in rows:
            groups.setdefault((row.question.horizon_sessions, row.family), []).append(row)
        per_group = max(1, limit // len(groups))
        picked: list[LabeledQuestion] = []
        for items in groups.values():
            step = len(items) / per_group
            picked.extend(items[int(position * step)] for position in range(per_group))
        picked.sort(key=lambda row: (row.as_of, row.question.horizon_sessions, row.question.id))
        rows = picked[:limit]
    return rows


def settlement_questions(root: Path) -> list[QuestionView]:
    """The three real questions, as the agent sees them."""
    views = []
    for question in load_questions(root / "questions.json"):
        as_of = load_series(series_path(root, question.symbol)).bars[-1].date
        horizon = sessions_from_as_of_through_resolve(as_of, question.resolve_session)
        views.append(
            QuestionView(
                id=question.id,
                text=question.yes_when,
                symbol=question.symbol,
                threshold=question.threshold,
                resolve_session=question.resolve_session,
                horizon_sessions=horizon,
                kind="settlement",
            )
        )
    return views



# -------------------------------------------------------------------- predictors


@dataclass
class ConstantPredictor:
    name: str = "constant 0.5"
    value: float = 0.5

    def predict(self, row: LabeledQuestion) -> Prediction:
        return Prediction(probability=self.value)


@dataclass
class BaseRatePredictor:
    """Answer every question with the point-in-time unconditional hit rate."""

    root: Path
    name: str = "base rate"

    def predict(self, row: LabeledQuestion) -> Prediction:
        question = row.question
        toolbox = Toolbox(self.root, AsOf.at_session_close(_as_of_of(row)))
        started = _now()
        outcome = toolbox.call(
            "base_rate",
            {
                "symbol": question.symbol,
                "threshold": question.threshold,
                "horizon": question.horizon_sessions,
            },
        )
        if outcome.error:
            raise RuntimeError(outcome.error)
        return Prediction(
            probability=float(outcome.payload["base_rate"]),
            latency_s=_now() - started,
            tool_calls=1,
            gate_hits=int(outcome.gate_hit),
        )


@dataclass
class AnaloguePredictor:
    """The repo's own model, used as a baseline the agent has to beat or defer to."""

    root: Path
    name: str = "analogue model"

    def predict(self, row: LabeledQuestion) -> Prediction:
        question = row.question
        toolbox = Toolbox(self.root, AsOf.at_session_close(_as_of_of(row)))
        started = _now()
        outcome = toolbox.call(
            "analogue_forecast",
            {
                "symbol": question.symbol,
                "threshold": question.threshold,
                "horizon": question.horizon_sessions,
            },
        )
        if outcome.error:
            raise RuntimeError(outcome.error)
        return Prediction(
            probability=float(outcome.payload["probability"]),
            latency_s=_now() - started,
            tool_calls=1,
            gate_hits=int(outcome.gate_hit),
        )


@dataclass
class AgentPredictor:
    """One or more independent LLM runs, pooled in log-odds."""

    root: Path
    llm: LLM
    runs: int = 1
    blind: bool = False
    max_steps: int = 8
    temperature: float = 0.7
    traces: list[Trace] = field(default_factory=list)
    name: str = "agent"

    def __post_init__(self) -> None:
        if not self.name or self.name == "agent":
            suffix = " blind" if self.blind else ""
            label = getattr(self.llm, "model", None) or getattr(self.llm, "name", "?")
            self.name = f"agent {label}{suffix}"

    def predict(self, row: LabeledQuestion) -> Prediction:
        from forecast_log.agent.ensemble import pool, spread

        question = row.question
        as_of = AsOf.at_session_close(_as_of_of(row))
        box = Toolbox(self.root, as_of)
        agent = Agent(
            self.llm,
            box,
            max_steps=self.max_steps,
            temperature=self.temperature,
            blind=self.blind,
        )
        probabilities, traces, elapsed = [], [], 0.0
        for _ in range(self.runs):
            started = _now()
            trace = agent.run(question)
            elapsed += _now() - started
            self.traces.append(trace)
            traces.append(trace)
            if trace.answer is not None:
                probabilities.append(trace.answer.probability)
        if not probabilities:
            raise RuntimeError(f"{question.id}: agent produced no answer")
        return Prediction(
            probability=pool(probabilities),
            cost_usd=sum(trace.usage.cost_usd for trace in traces),
            latency_s=elapsed,
            tool_calls=sum(trace.tool_calls for trace in traces),
            gate_hits=sum(trace.gate_hits for trace in traces),
            note=f"spread {spread(probabilities):.2f}" if len(probabilities) > 1 else "",
        )


# ----------------------------------------------------------------------- metrics


def brier(pairs: list[tuple[float, int]]) -> float:
    return sum((probability - outcome) ** 2 for probability, outcome in pairs) / len(pairs)


def log_loss(pairs: list[tuple[float, int]]) -> float:
    total = 0.0
    for probability, outcome in pairs:
        clipped = min(max(probability, 1e-6), 1 - 1e-6)
        total += -(outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped))
    return total / len(pairs)


def reliability(pairs: list[tuple[float, int]], bins: int = 5) -> list[dict]:
    table = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [
            (probability, outcome)
            for probability, outcome in pairs
            if (low <= probability < high) or (index == bins - 1 and probability == 1.0)
        ]
        if not bucket:
            continue
        table.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "n": len(bucket),
                "mean_predicted": round(statistics.fmean(p for p, _ in bucket), 4),
                "observed": round(statistics.fmean(o for _, o in bucket), 4),
            }
        )
    return table


def ece(pairs: list[tuple[float, int]], bins: int = 5) -> float:
    table = reliability(pairs, bins)
    if not table:
        return float("nan")
    total = sum(row["n"] for row in table)
    return sum(
        row["n"] / total * abs(row["mean_predicted"] - row["observed"]) for row in table
    )


def evaluate(
    predictors: list[Predictor],
    questions: list[LabeledQuestion],
    *,
    verbose: bool = True,
    workers: int = 1,
) -> dict:
    """Score every predictor on the questions where all of them produced a number.

    `workers` runs questions concurrently. A tool-using run is 6-8 blocking HTTP calls, so
    at one worker a 96-question bench is 20 minutes of mostly waiting; the forecast itself
    has no shared state, which is what makes this safe. Order is preserved, so the report
    does not depend on scheduling.
    """
    results: dict[str, dict[str, float]] = {}
    for predictor in predictors:
        scores: dict[str, float] = {}
        stats = {"cost_usd": 0.0, "latency": [], "tool_calls": [], "gate_hits": 0, "failures": 0}

        done = 0
        lock = threading.Lock()

        def attempt(row: LabeledQuestion) -> Prediction | Exception:
            nonlocal done
            try:
                outcome: Prediction | Exception = predictor.predict(row)
            except Exception as error:  # noqa: BLE001 - a failed forecast is data too
                outcome = error
            with lock:
                done += 1
                if verbose and workers > 1 and done % 10 == 0:
                    print(f"   {predictor.name}: {done}/{len(questions)}", flush=True)
            return outcome

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outcomes = list(pool.map(attempt, questions))
        else:
            outcomes = [attempt(row) for row in questions]

        for row, outcome in zip(questions, outcomes):
            if isinstance(outcome, Exception):
                stats["failures"] += 1
                if verbose:
                    print(f"   {predictor.name}: {row.id} failed: {outcome}", flush=True)
                continue
            scores[row.id] = outcome.probability
            stats["cost_usd"] += outcome.cost_usd
            stats["latency"].append(outcome.latency_s)
            stats["tool_calls"].append(outcome.tool_calls)
            stats["gate_hits"] += outcome.gate_hits
        if verbose:
            print(f"   {predictor.name}: {len(scores)}/{len(questions)} 道题给出了答案", flush=True)
        results[predictor.name] = {
            "_scores": scores,  # type: ignore[dict-item]
            "_stats": stats,  # type: ignore[dict-item]
        }

    # A predictor that answered nothing is excluded rather than allowed to void the
    # comparison for everyone else; it stays in the report with n=0 and its failures.
    alive = [entry for entry in results.values() if entry["_scores"]]
    shared = set.intersection(*(set(entry["_scores"].keys()) for entry in alive)) if alive else set()
    by_id = {row.id: row for row in questions}
    ordered = [by_id[key] for key in sorted(shared, key=lambda key: (by_id[key].question.resolve_session, key))]

    report = {
        "n": len(ordered),
        "questions": [row.id for row in ordered],
        "outcomes": {row.id: row.outcome for row in ordered},
        "predictors": {},
    }
    for name, entry in results.items():
        if not entry["_scores"]:
            # 一道都没答出来的预测者：照样留一行（n=0 + 失败原因），但不参与 Brier 汇总，
            # 也不许把整次评测带崩——它没有有序列表里的任何一个 key。
            stats = entry["_stats"]
            report["predictors"][name] = {
                "brier": None, "log_loss": None, "ece": None, "hit_direction": None,
                "mean_probability": None, "observed_rate": float(np.mean([row.outcome for row in ordered])) if ordered else None,
                "cost_usd_total": round(stats["cost_usd"], 6),
                "cost_usd_per_forecast": None,
                "latency_p50_s": round(statistics.median(stats["latency"]), 3) if stats["latency"] else None,
                "latency_p95_s": None,
                "tool_calls_mean": round(statistics.fmean(stats["tool_calls"]), 2) if stats["tool_calls"] else 0.0,
                "gate_hits": stats["gate_hits"], "failures": stats["failures"],
                "reliability": [], "n": 0,
            }
            continue
        pairs = [(entry["_scores"][row.id], row.outcome) for row in ordered]
        if not pairs:
            continue
        stats = entry["_stats"]
        latency = sorted(stats["latency"])
        report["predictors"][name] = {
            "brier": round(brier(pairs), 4),
            "log_loss": round(log_loss(pairs), 4),
            "ece": round(ece(pairs), 4),
            "hit_direction": round(
                statistics.fmean(
                    1.0 if (probability >= 0.5) == (outcome == 1) else 0.0 for probability, outcome in pairs
                ),
                4,
            ),
            "mean_probability": round(statistics.fmean(p for p, _ in pairs), 4),
            "observed_rate": round(statistics.fmean(o for _, o in pairs), 4),
            "cost_usd_total": round(stats["cost_usd"], 4),
            "cost_usd_per_forecast": round(stats["cost_usd"] / len(pairs), 5),
            "latency_p50_s": round(_percentile(latency, 0.50), 2),
            "latency_p95_s": round(_percentile(latency, 0.95), 2),
            "tool_calls_mean": round(statistics.fmean(stats["tool_calls"]), 2),
            "gate_hits": stats["gate_hits"],
            "failures": stats["failures"],
            "reliability": reliability(pairs),
            "pairs": {row.id: round(entry["_scores"][row.id], 6) for row in ordered},
            "by_family": {
                family: {
                    "n": len(family_pairs),
                    "brier": round(brier(family_pairs), 4),
                    "observed": round(statistics.fmean(outcome for _, outcome in family_pairs), 3),
                }
                for family in sorted({row.family for row in ordered})
                for family_pairs in [[(entry["_scores"][row.id], row.outcome) for row in ordered if row.family == family]]
                if family_pairs
            },
        }
    if ordered:
        outcomes = [(0.5, row.outcome) for row in ordered]
        report["references"] = {
            "constant_0.5": {"brier": round(brier(outcomes), 4), "log_loss": round(log_loss(outcomes), 4)},
            "observed_rate": round(statistics.fmean(row.outcome for row in ordered), 4),
        }
    return report


def render(report: dict) -> str:
    if not report.get("predictors"):
        return "没有可比的预测者。"
    header = f"{'预测者':<34}{'Brier':>8}{'LogLoss':>9}{'ECE':>8}{'方向':>7}{'$/条':>8}{'p50 s':>8}{'工具':>6}"
    lines = [f"{report['n']} 道已结算的历史题（同一批题，所有预测者都必须作答）", "", header, "-" * len(header)]
    for name, row in sorted(report["predictors"].items(), key=lambda pair: pair[1]["brier"]):
        lines.append(
            f"{name[:33]:<34}{row['brier']:>8.4f}{row['log_loss']:>9.4f}{row['ece']:>8.3f}"
            f"{row['hit_direction']:>7.1%}{row['cost_usd_per_forecast']:>8.4f}"
            f"{row['latency_p50_s']:>8.2f}{row['tool_calls_mean']:>6.2f}"
        )
    reference = report.get("references") or {}
    if "constant_0.5" in reference:
        lines.append("-" * len(header))
        lines.append(
            f"{'参照：始终 0.5':<34}{reference['constant_0.5']['brier']:>8.4f}"
            f"{reference['constant_0.5']['log_loss']:>9.4f}"
        )
        lines.append(f"这批题的实际发生率 {reference['observed_rate']:.1%}")
    families = sorted(
        {family for row in report["predictors"].values() for family in row.get("by_family", {})}
    )
    if families:
        names = list(report["predictors"])
        lines.append("")
        lines.append("按题型拆开（同一列上下可比，题型说明见 eval.py）：")
        lines.append(f"{'题型':<12}{'n':>5}" + "".join(f"{name[:16]:>18}" for name in names))
        label = {"level": "level 高于今天", "quantile": "quantile 60 分位", "band_up": "band_up +5%", "band_down": "band_down -5%"}
        for family in families:
            cells = ""
            counts = []
            for name in names:
                entry = report["predictors"][name].get("by_family", {}).get(family)
                if entry:
                    counts.append(entry["n"])
                    cells += f"{entry['brier']:>18.4f}"
                else:
                    cells += f"{'-':>18}"
            lines.append(f"{label.get(family, family):<12}{max(counts, default=0):>5}" + cells)
    lines.append("")
    for name, row in report["predictors"].items():
        if row.get("failures"):
            lines.append(f"{name}：{row['failures']} 次没有给出答案")
        if row.get("gate_hits"):
            lines.append(f"{name}：时间闸门拦住 {row['gate_hits']} 次")
    return "\n".join(lines)


def write_report(path: Path, report: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


def trace_documents(traces: list[Trace]) -> list[dict]:
    return [trace.to_json() for trace in traces]


# ----------------------------------------------------------------------- helpers


def _now() -> float:
    import time

    return time.perf_counter()


def _as_of_of(row: LabeledQuestion) -> date:
    return row.as_of


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    position = min(max(int(round(fraction * (len(values) - 1))), 0), len(values) - 1)
    return values[position]
