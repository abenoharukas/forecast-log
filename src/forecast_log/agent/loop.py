"""The loop: question in, locked probability out, with a trace that can be re-read later.

There is no framework here on purpose. The loop is ~150 lines and every one of them is
something this job needs to control: the step budget, the token budget per tool result,
what happens when the model submits an invalid answer, and whether a rejected answer was
rejected because it tried to read the future.

Two behaviours are worth pointing at:

- A submission is validated *before* it is accepted, and the rejection is fed back into
  the transcript. A model that cites a date after the as-of session gets told so and has
  to try again; the trace keeps the attempt.
- `blind=True` removes every tool. Running the same eval with and without tools is the
  ablation that shows whether the LLM knows anything about prices, or whether the tools
  and the loop are doing the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from forecast_log.agent.gate import AsOf, LookAheadError
from forecast_log.agent.llm import LLM, Reply, Usage
from forecast_log.agent.tools import ToolOutcome, Toolbox

MAX_TOOL_CHARS = 2400
MIN_RATIONALE_CHARS = 40


class AgentError(RuntimeError):
    pass


class AnswerRejected(ValueError):
    pass


@dataclass(frozen=True)
class QuestionView:
    """A question as the agent sees it. Real ones come from questions.json, the eval
    builds synthetic ones from history, and both go through the same loop."""

    id: str
    text: str
    symbol: str
    threshold: float
    resolve_session: date
    horizon_sessions: int
    kind: str = "settlement"


@dataclass(frozen=True)
class Evidence:
    date: date
    source: str
    note: str = ""

    def to_json(self) -> dict:
        return {"date": self.date.isoformat(), "source": self.source, "note": self.note}


@dataclass(frozen=True)
class ForecastAnswer:
    probability: float
    rationale: str
    evidence: tuple[Evidence, ...]
    anchored_on: str = "unspecified"

    def to_json(self) -> dict:
        return {
            "probability": round(self.probability, 6),
            "rationale": self.rationale,
            "anchored_on": self.anchored_on,
            "evidence": [row.to_json() for row in self.evidence],
        }


@dataclass
class Turn:
    index: int
    reply: Reply
    outcomes: list[ToolOutcome] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "content": self.reply.content,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments, "parse_error": call.parse_error}
                for call in self.reply.tool_calls
            ],
            "tool_results": [
                {"name": outcome.name, "error": outcome.error, "gate_hit": outcome.gate_hit}
                for outcome in self.outcomes
            ],
            "usage": self.reply.usage.to_json(),
        }


@dataclass
class Trace:
    question: QuestionView
    as_of: AsOf
    model: str
    turns: list[Turn] = field(default_factory=list)
    answer: ForecastAnswer | None = None
    rejections: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def usage(self) -> Usage:
        total = Usage()
        for turn in self.turns:
            total = total + turn.reply.usage
        return total

    @property
    def tool_calls(self) -> int:
        return sum(len(turn.reply.tool_calls) for turn in self.turns)

    @property
    def gate_hits(self) -> int:
        return sum(1 for turn in self.turns for outcome in turn.outcomes if outcome.gate_hit)

    def to_json(self) -> dict:
        return {
            "question_id": self.question.id,
            "question": self.question.text,
            "as_of": self.as_of.to_json(),
            "model": self.model,
            "answer": None if self.answer is None else self.answer.to_json(),
            "rejections": self.rejections,
            "error": self.error,
            "usage": self.usage.to_json(),
            "tool_calls": self.tool_calls,
            "gate_hits": self.gate_hits,
            "turns": [turn.to_json() for turn in self.turns],
        }


SUBMIT_SPEC = {
    "type": "function",
    "function": {
        "name": "submit_forecast",
        "description": (
            "Submit the final forecast. Call this exactly once, when you are done gathering "
            "evidence. The probability is what gets locked and scored, so it must be your "
            "honest degree of belief, not a hedge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "probability": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "P(the question resolves yes)",
                },
                "rationale": {
                    "type": "string",
                    "description": "Two to five sentences: the base rate, what moved you off it, and why not further",
                },
                "anchored_on": {
                    "type": "string",
                    "enum": ["base_rate", "analogue", "news", "market", "prior"],
                    "description": "The single thing the number is anchored on",
                },
                "evidence": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "Session or publication date, YYYY-MM-DD"},
                            "source": {"type": "string", "description": "Tool name or document id"},
                            "note": {"type": "string"},
                        },
                        "required": ["date", "source"],
                    },
                },
            },
            "required": ["probability", "rationale", "evidence"],
        },
    },
}

SYSTEM_PROMPT = """You are the forecasting agent in forecast-log. You get one closed question \
about a financial time series and must return a probability that will be locked, published, \
and scored with a Brier score after reality resolves.

Hard rules:
1. You may only use data dated on or before the as-of session. The tools enforce this and \
will return a LookAheadError if you reach past it. When that happens, do not retry the same \
request with a later date; use what is available.
2. Submit exactly once, with submit_forecast, before you run out of steps.
3. Every evidence row needs a real date at or before the as-of session and the tool or \
document it came from. Never cite something you did not read through a tool.
4. If a tool returns nothing (empty corpus, no market quote, an error), say so in the \
rationale. An empty result is information; a fabricated one is a scoring bug.

How to pick the number:
- Start from the unconditional base rate for this threshold and horizon.
- Move away from it only as far as the evidence supports, and say what moved you.
- A probability of 0.8 means that across many questions where you said 0.8, about eight in \
ten should resolve yes. Extreme numbers need strong evidence, and a 7-session question is \
rarely extreme.
- If two sources disagree (analogue model versus base rate, say), do not average them \
blindly. Say which one you trust here and why.

Be concise. Your rationale is read by a human who will also read the trace."""


class Agent:
    def __init__(
        self,
        llm: LLM,
        toolbox: Toolbox,
        max_steps: int = 8,
        temperature: float = 0.7,
        blind: bool = False,
        max_tool_chars: int = MAX_TOOL_CHARS,
    ):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.temperature = temperature
        self.blind = blind
        self.max_tool_chars = max_tool_chars

    def run(self, question: QuestionView) -> Trace:
        trace = Trace(question=question, as_of=self.toolbox.as_of, model=getattr(self.llm, "name", "?"))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": self._brief(question)}]
        specs = [SUBMIT_SPEC] if self.blind else self.toolbox.specs() + [SUBMIT_SPEC]

        for step in range(self.max_steps):
            # In blind mode the only legal move is to submit, so ask for exactly that.
            # Otherwise a model that prefers prose burns the whole step budget and the
            # ablation row measures timeouts instead of forecasting skill.
            force = {"type": "function", "function": {"name": "submit_forecast"}} if self.blind else None
            reply = self.llm.chat(
                messages, tools=specs, temperature=self.temperature, tool_choice=force
            )
            turn = Turn(index=step, reply=reply)
            trace.turns.append(turn)

            if reply.content and not reply.tool_calls:
                messages.append({"role": "assistant", "content": reply.content})
                messages.append({"role": "user", "content": "You have not submitted yet. Call submit_forecast now."})
                continue
            messages.append(_assistant_message(reply))
            if not reply.tool_calls:
                continue

            for call in reply.tool_calls:
                if call.name == "submit_forecast":
                    try:
                        trace.answer = self._validate(call.arguments)
                    except (AnswerRejected, LookAheadError) as error:
                        trace.rejections.append(str(error))
                        messages.append(_tool_message(call.id, f"rejected: {error}. Fix it and submit again."))
                        continue
                    return trace
                if self.blind:
                    messages.append(
                        _tool_message(call.id, "error: no tools are available in this run")
                    )
                    continue
                if call.parse_error:
                    messages.append(_tool_message(call.id, f"error: {call.parse_error}"))
                    continue
                outcome = self.toolbox.call(call.name, call.arguments)
                turn.outcomes.append(outcome)
                messages.append(_tool_message(call.id, outcome.as_message(self.max_tool_chars)))

        trace.error = f"no valid submission after {self.max_steps} steps"
        return trace

    # ------------------------------------------------------------------ internals

    def _brief(self, question: QuestionView) -> str:
        return (
            f"Question id: {question.id}\n"
            f"Question: {question.text}\n"
            f"Series: {question.symbol}\n"
            f"Threshold: {question.threshold}\n"
            f"Settlement session: {question.resolve_session.isoformat()} "
            f"({question.horizon_sessions} sessions ahead)\n"
            f"As-of session: {self.toolbox.as_of.session.isoformat()}\n"
            f"Decision instant: {self.toolbox.as_of.at.isoformat(timespec='seconds')}\n"
            + (
                "\nNo tools are available in this run: answer from what you already know, "
                "and note in the rationale that you had no data."
                if self.blind
                else "\nYou may call at most one tool per turn."
            )
        )

    def _validate(self, arguments: dict) -> ForecastAnswer:
        if "probability" not in arguments:
            raise AnswerRejected("missing probability")
        try:
            probability = float(arguments["probability"])
        except (TypeError, ValueError) as error:
            raise AnswerRejected(f"probability is not a number: {error}") from error
        if probability != probability or not 0.0 <= probability <= 1.0:
            raise AnswerRejected(f"probability {probability} is outside [0, 1]")

        rationale = str(arguments.get("rationale") or "").strip()
        if len(rationale) < MIN_RATIONALE_CHARS:
            raise AnswerRejected(f"rationale is shorter than {MIN_RATIONALE_CHARS} characters")

        rows = arguments.get("evidence")
        if not isinstance(rows, list) or not rows:
            raise AnswerRejected("evidence must be a non-empty list")
        evidence = []
        for row in rows:
            if not isinstance(row, dict) or "date" not in row or "source" not in row:
                raise AnswerRejected("each evidence row needs date and source")
            try:
                day = date.fromisoformat(str(row["date"])[:10])
            except ValueError as error:
                raise AnswerRejected(f"evidence date {row['date']!r} is not YYYY-MM-DD") from error
            self.toolbox.as_of.check(day, f"evidence {row.get('source')}")
            evidence.append(Evidence(date=day, source=str(row["source"]), note=str(row.get("note") or "")))

        anchored = str(arguments.get("anchored_on") or "unspecified")
        return ForecastAnswer(
            probability=probability,
            rationale=rationale,
            evidence=tuple(evidence),
            anchored_on=anchored,
        )


def _assistant_message(reply: Reply) -> dict:
    message: dict = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments or "{}",
                },
            }
            for call in reply.tool_calls
        ]
    return message


def _tool_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}

