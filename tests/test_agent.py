"""The agent layer, tested without a network and without an API key.

The tests that matter here are the ones about leakage and rejection. Anything can produce
a probability; the claim being made is that the probability could not have seen the answer,
and that a bad submission is caught rather than published.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from forecast_log.agent.ensemble import logit, pool, sigmoid, spread
from forecast_log.agent.eval import (
    LabeledQuestion,
    build_questions,
    brier,
    ece,
    evaluate,
    reliability,
    ConstantPredictor,
)
from forecast_log.agent.gate import AsOf, LookAheadError
from forecast_log.agent.llm import Reply, ScriptedLLM, ToolCall
from forecast_log.agent.loop import Agent, AnswerRejected, QuestionView
from forecast_log.agent.loop import SUBMIT_SPEC  # noqa: F401  (imported for the blind-mode check)
from forecast_log.agent.tools import Toolbox

NY = ZoneInfo("America/New_York")
from forecast_log.bars import Bar, Series, save_series
from forecast_log.paths import series_path


# ----------------------------------------------------------------------- fixtures


def write_series(root: Path, closes: list[float], symbol: str = "^VXN", start: date | None = None) -> Series:
    start = start or date(2015, 1, 5)
    bars = tuple(
        Bar(date=start + timedelta(days=index), close=float(close))
        for index, close in enumerate(closes)
    )
    series = Series(symbol=symbol, timezone="America/New_York", fetched_at="2026-11-12T06:48:23-05:00", bars=bars)
    save_series(series_path(root, symbol), series)
    return series


def as_of_at(series: Series, index: int) -> AsOf:
    return AsOf.at_session_close(series.bars[index].date)


def tool_reply(name: str, arguments: dict, call_id: str = "call-1") -> Reply:
    return Reply(
        content=None,
        tool_calls=(
            ToolCall(id=call_id, name=name, arguments=arguments, raw_arguments=json.dumps(arguments)),
        ),
    )


def submission(probability: float = 0.4, evidence_date: str | None = None) -> Reply:
    return tool_reply(
        "submit_forecast",
        {
            "probability": probability,
            "rationale": "Base rate is the anchor and nothing in the printed history moves me far from it.",
            "anchored_on": "base_rate",
            "evidence": [{"date": evidence_date or "2015-01-09", "source": "price_history"}],
        },
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


# --------------------------------------------------------------------------- gate


def test_gate_rejects_a_session_after_the_as_of_date() -> None:
    gate = AsOf.at_session_close(date(2026, 11, 11))
    assert gate.check(date(2026, 11, 11), "close") == date(2026, 11, 11)
    with pytest.raises(LookAheadError):
        gate.check(date(2026, 11, 12), "close")


def test_gate_rejects_a_document_published_after_the_decision_instant() -> None:
    gate = AsOf.at_session_close(date(2026, 11, 11))
    late = datetime(2026, 11, 11, 20, 0, tzinfo=NY)
    with pytest.raises(LookAheadError):
        gate.check_published(late, "news")
    with pytest.raises(ValueError):
        gate.check_published(datetime(2026, 11, 11, 9, 0), "news")  # naive timestamp


# -------------------------------------------------------------------------- tools


def test_price_history_never_returns_a_bar_after_the_as_of_session(root: Path) -> None:
    series = write_series(root, [10 + index for index in range(20)])
    gate = as_of_at(series, 9)
    outcome = Toolbox(root, gate).call("price_history", {"symbol": "^VXN", "sessions": 500})
    assert outcome.error is None
    assert outcome.payload["rows"][-1]["date"] == series.bars[9].date.isoformat()
    assert all(row["date"] <= series.bars[9].date.isoformat() for row in outcome.payload["rows"])
    assert outcome.payload["sessions"] == 10


def test_base_rate_only_counts_outcomes_already_printed(root: Path) -> None:
    # Ten quiet sessions, then a jump that would wreck the base rate if it were counted.
    closes = [20.0] * 40 + [20.0, 20.0, 20.0, 20.0, 99.0]
    series = write_series(root, closes)
    gate = as_of_at(series, 39)  # as-of is before the jump
    outcome = Toolbox(root, gate).call("base_rate", {"symbol": "^VXN", "threshold": 50, "horizon": 1})
    assert outcome.error is None
    assert outcome.payload["base_rate"] == 0.0
    assert outcome.payload["sessions_measured"] == 40 - 1


def test_a_future_document_in_the_corpus_is_an_error_not_a_skip(root: Path) -> None:
    write_series(root, [20.0] * 30)
    news = root / "data" / "news"
    news.mkdir(parents=True)
    (news / "2026-11-12.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "late-1",
                        "published_at": "2026-11-12T09:00:00-05:00",
                        "title": "This came out after the close",
                        "text": "not allowed",
                    }
                ]
            }
        )
    )
    gate = AsOf.at_session_close(date(2026, 11, 11))
    outcome = Toolbox(root, gate).call("search_news", {"query": "vix"})
    assert outcome.gate_hit is True
    assert outcome.error and "after the decision instant" in outcome.error


def test_missing_market_quote_is_not_a_price(root: Path) -> None:
    series = write_series(root, [20.0] * 30)
    outcome = Toolbox(root, as_of_at(series, 20)).call("market_implied", {"question_id": "vxn-x"})
    assert outcome.payload["available"] is False


def test_unknown_symbol_lists_what_is_available(root: Path) -> None:
    write_series(root, [20.0] * 30)
    outcome = Toolbox(root, AsOf.at_session_close(date(2015, 1, 9))).call(
        "price_history", {"symbol": "^NOPE"}
    )
    assert outcome.error and "^VXN" in outcome.error


# --------------------------------------------------------------------------- loop


def question(resolve: date, as_of: date, threshold: float = 30.0) -> QuestionView:
    return QuestionView(
        id="q-1",
        text=f"Will ^VXN close strictly above {threshold} on {resolve.isoformat()}?",
        symbol="^VXN",
        threshold=threshold,
        resolve_session=resolve,
        horizon_sessions=5,
    )


def test_loop_accepts_a_valid_submission_after_using_a_tool(root: Path) -> None:
    series = write_series(root, [20.0] * 40)
    gate = as_of_at(series, 39)
    llm = ScriptedLLM([tool_reply("price_history", {"symbol": "^VXN", "sessions": 5}), submission()])
    trace = Agent(llm, Toolbox(root, gate)).run(question(date(2015, 3, 1), series.bars[39].date))
    assert trace.answer is not None
    assert trace.answer.probability == pytest.approx(0.4)
    assert trace.tool_calls == 2
    assert trace.gate_hits == 0
    payload = trace.to_json()
    assert payload["answer"]["evidence"][0]["source"] == "price_history"
    assert len(payload["turns"]) == 2


def test_evidence_from_the_future_is_rejected_and_the_retry_is_kept(root: Path) -> None:
    series = write_series(root, [20.0] * 40)
    gate = as_of_at(series, 39)
    future = (series.bars[39].date + timedelta(days=1)).isoformat()
    llm = ScriptedLLM([submission(evidence_date=future), submission()])
    trace = Agent(llm, Toolbox(root, gate)).run(question(date(2015, 3, 1), series.bars[39].date))
    assert trace.answer is not None
    assert len(trace.rejections) == 1
    assert "after the as-of session" in trace.rejections[0]


def test_probability_outside_the_unit_interval_is_rejected(root: Path) -> None:
    series = write_series(root, [20.0] * 40)
    gate = as_of_at(series, 39)
    llm = ScriptedLLM([submission(probability=1.4), submission(probability=0.3)])
    trace = Agent(llm, Toolbox(root, gate)).run(question(date(2015, 3, 1), series.bars[39].date))
    assert trace.rejections and "outside [0, 1]" in trace.rejections[0]
    assert trace.answer is not None and trace.answer.probability == pytest.approx(0.3)


def test_missing_evidence_is_rejected(root: Path) -> None:
    series = write_series(root, [20.0] * 40)
    gate = as_of_at(series, 39)
    empty = tool_reply(
        "submit_forecast",
        {"probability": 0.5, "rationale": "x" * 60, "evidence": []},
    )
    llm = ScriptedLLM([empty, submission()])
    trace = Agent(llm, Toolbox(root, gate)).run(question(date(2015, 3, 1), series.bars[39].date))
    assert trace.rejections and "non-empty" in trace.rejections[0]


def test_blind_mode_offers_only_the_submit_tool(root: Path) -> None:
    series = write_series(root, [20.0] * 40)
    llm = ScriptedLLM([submission()])
    Agent(llm, Toolbox(root, as_of_at(series, 39)), blind=True).run(
        question(date(2015, 3, 1), series.bars[39].date)
    )
    assert llm.tool_names_seen[0] == ["submit_forecast"]


def test_running_out_of_steps_is_an_error_not_a_guess(root: Path) -> None:
    series = write_series(root, [20.0] * 40)
    llm = ScriptedLLM([tool_reply("series_stats", {"symbol": "^VXN"}) for _ in range(2)])
    trace = Agent(llm, Toolbox(root, as_of_at(series, 39)), max_steps=2).run(
        question(date(2015, 3, 1), series.bars[39].date)
    )
    assert trace.answer is None
    assert trace.error and "no valid submission" in trace.error


def test_a_tool_error_is_shown_to_the_model_instead_of_crashing(root: Path) -> None:
    series = write_series(root, [20.0] * 40)
    gate = as_of_at(series, 39)
    llm = ScriptedLLM([tool_reply("analogue_forecast", {"symbol": "^VXN", "threshold": 30, "horizon": 40}), submission()])
    trace = Agent(llm, Toolbox(root, gate)).run(question(date(2015, 3, 1), series.bars[39].date))
    assert trace.answer is not None
    assert trace.turns[0].outcomes[0].error is not None


# ----------------------------------------------------------------------- ensemble


def test_pooling_two_opposite_runs_lands_in_the_middle() -> None:
    assert pool([0.05, 0.95]) == pytest.approx(0.5, abs=1e-6)
    assert pool([0.9]) == pytest.approx(0.9, abs=1e-9)
    assert pool([0.5, 0.5, 0.5]) == pytest.approx(0.5, abs=1e-9)


def test_logit_and_sigmoid_round_trip_and_spread_is_zero_when_runs_agree() -> None:
    assert sigmoid(logit(0.2)) == pytest.approx(0.2, abs=1e-9)
    assert spread([0.4, 0.4]) == 0.0
    assert spread([0.1, 0.9]) > 1.0


# ------------------------------------------------------------------ the bench itself


def test_generated_questions_cannot_see_their_own_answer(root: Path) -> None:
    rng = np.random.default_rng(7)
    closes = list(20 * np.exp(rng.normal(0, 0.02, 900).cumsum()))
    write_series(root, closes)
    rows = build_questions(
        __import__("forecast_log.bars", fromlist=["load_series"]).load_series(series_path(root, "^VXN")),
        stride=50,
        modes=("level", "quantile", "band_up"),
        limit=30,
    )
    assert rows
    assert len({row.id for row in rows}) == len(rows)
    actual = {bar.date: bar.close for bar in __import__("forecast_log.bars", fromlist=["load_series"]).load_series(series_path(root, "^VXN")).bars}
    for row in rows:
        assert row.question.resolve_session > row.as_of
        expected = 1 if actual[row.question.resolve_session] > row.question.threshold else 0
        assert row.outcome == expected
        if row.family == "level":
            assert row.question.threshold == pytest.approx(round(row.as_of_close, 2))


def test_metrics_match_hand_arithmetic() -> None:
    pairs = [(0.9, 1), (0.8, 1), (0.4, 1), (0.1, 0)]
    assert brier(pairs) == pytest.approx((0.01 + 0.04 + 0.36 + 0.01) / 4)
    assert ece([(0.2, 0), (0.2, 0), (0.2, 0), (0.2, 0)]) == pytest.approx(0.2)
    table = reliability([(0.9, 1)] * 3 + [(0.1, 0)] * 1)
    assert sum(row["n"] for row in table) == 4


def test_evaluate_scores_only_the_questions_every_predictor_answered(root: Path) -> None:
    as_of = date(2026, 1, 5)
    rows = [
        LabeledQuestion(
            question=QuestionView(
                id=f"q{index}",
                text="text",
                symbol="^VXN",
                threshold=20.0,
                resolve_session=date(2026, 1, 12),
                horizon_sessions=5,
            ),
            outcome=outcome,
            as_of=as_of,
            as_of_close=20.0,
        )
        for index, outcome in enumerate([1, 0, 1, 0])
    ]
    report = evaluate(
        [ConstantPredictor(name="always 0.5"), ConstantPredictor(name="always 0.9", value=0.9)],
        rows,
        verbose=False,
    )
    assert report["n"] == 4
    assert report["predictors"]["always 0.5"]["brier"] == pytest.approx(0.25)
    assert report["predictors"]["always 0.9"]["brier"] == pytest.approx(0.41)
