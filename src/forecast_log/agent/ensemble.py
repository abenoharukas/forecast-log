"""Independent runs, pooled in log-odds.

Averaging probabilities is the wrong average: it drags a confident 0.05 towards 0.5 much
harder than it drags 0.95, so the pooled number is biased towards the middle. Averaging
logits is symmetric: two runs that disagree 0.05 / 0.95 pool to 0.5, which is the honest
reading of "these two disagree".

This is a multi-agent system in the smallest useful form: N identical actors that do not
see each other, one aggregation rule, and a disagreement number that says whether the
aggregate is worth anything.
"""

from __future__ import annotations

from statistics import pstdev

import numpy as np

from forecast_log.agent.loop import Agent, QuestionView, Trace

EPS = 1e-3


def logit(probability: float) -> float:
    clipped = min(max(float(probability), EPS), 1 - EPS)
    return float(np.log(clipped / (1 - clipped)))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def pool(probabilities: list[float]) -> float:
    if not probabilities:
        raise ValueError("nothing to pool")
    return sigmoid(float(np.mean([logit(p) for p in probabilities])))


def spread(probabilities: list[float]) -> float:
    """Standard deviation of the logits. 0 means every run said the same thing."""
    if len(probabilities) < 2:
        return 0.0
    return float(pstdev([logit(p) for p in probabilities]))


def run_ensemble(
    agent: Agent,
    question: QuestionView,
    runs: int = 3,
    temperature: float | None = None,
) -> tuple[float, list[Trace]]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if temperature is not None:
        agent.temperature = temperature
    traces = [agent.run(question) for _ in range(runs)]
    answered = [trace.answer.probability for trace in traces if trace.answer is not None]
    if not answered:
        raise RuntimeError(f"{question.id}: no run produced an answer")
    return pool(answered), traces
