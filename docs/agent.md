# The agent

A forecasting agent that is graded by `forecast-log`'s own contract: read only what was
published before the decision, lock a probability, keep the trace, get a Brier score after
reality resolves.

Six modules, about 1,700 lines, plus 324 lines of offline tests:

| file | lines | what it is |
|---|---|---|
| `agent/gate.py` | 90 | `AsOf` + `LookAheadError`. The rule that nothing from after the decision reaches the model. |
| `agent/llm.py` | 271 | `/chat/completions` over `urllib`, tool calls, retries, token and dollar accounting. No SDK. |
| `agent/tools.py` | 404 | Seven gated tools over local price history, the analogue model, a document corpus and prediction-market quotes. |
| `agent/loop.py` | 355 | The tool-calling loop, the submission validator, the trace. No framework. |
| `agent/ensemble.py` | 60 | Log-odds pooling across independent runs. |
| `agent/eval.py` | 539 | Question generator, baselines, metrics, calibration table, cost, latency, concurrency. |

```
questions.json ──► QuestionView ──► Agent.run ──► ForecastAnswer ──► forecasts/ (locked)
                                        │                              │
                                        └── Trace (every turn) ────────┴──► Brier
```

## The gate is the product

Anyone can put a probability on a chart. The claim here is narrower and checkable: the
number could not have been computed from data that came out after the decision.

```python
def check(self, day: date, source: str) -> date:
    if day > self.session:
        raise LookAheadError(...)
    return day
```

Three places call it, and each one is a place a leak would otherwise hide:

1. **Tool return values.** `price_history`, `base_rate`, `analogue_forecast` and
   `series_stats` all filter to `date <= as_of.session` *before* the gate, and then assert
   it anyway. Belt and braces on purpose.
2. **Document timestamps.** `search_news` / `read_doc` gate `published_at` against the
   decision *instant*, not the session date, because a wire story at 14:00 on the as-of day
   is not something a 16:15 decision could have read and a 09:00 one is.
3. **The submission itself.** Every `evidence[].date` is gated before the answer is
   accepted. A model that cites the future gets its submission rejected, is told why, and
   has to try again. The rejection stays in the trace.

Raising beats clipping. A clip looks exactly like a correct answer, and survives every
test you would think to write. An exception shows up in the first trace you read.

## Tools

| tool | returns | why it exists |
|---|---|---|
| `price_history` | last N daily closes | the raw evidence |
| `series_stats` | level, z-score, percentiles, 5/21/252 changes, realised vol | the state, compressed to ~15 numbers instead of 400 |
| `base_rate` | unconditional hit rate for this threshold and horizon | the number that is embarrassingly hard to beat |
| `analogue_forecast` | the repo's own model: probability + median path | the old answer, demoted to one input |
| `search_news` | time-stamped documents matching a query | the retrieval path |
| `read_doc` | one document by id | same gate |
| `market_implied` | prediction-market price, or `available: false` | the price to beat when one exists |

`analogue_forecast` is the interesting one. The model in `estimate.py` used to *be* the
forecast. Here it is a tool with a known weakness (it re-states its neighbour pool and
cannot see regime change), and the agent has to decide how much to defer to it. That
decision, written down in the rationale, is the thing a human reviewer actually wants.

### News search is behind a null implementation

`data/news/*.json` does not exist in this repo, so `search_news` returns an empty corpus
and says so. Wiring a real provider is one class, and the tool already gates on
`published_at`. It is missing because a licensed feed and a search key were not available
while this was written, and a fake corpus would make the eval look better than it is — an
empty result that the model must disclose is more useful than a fabricated one.

The consequence is stated in every trace: **the eval measures the price-only path.**

## The loop

```
system prompt (the contract)
  ↓
model ──tool call──► gate ──► JSON result (truncated to 2.4 KB)
  ↑                                  │
  └──────────── transcript ◄─────────┘
  ↓
submit_forecast ──► validate ──► accept
                       │
                       └── reject ──► back into the transcript, try again
```

Details that were choices, not defaults:

- **8 steps.** Measured: a normal run takes 5–6 tool calls. Runs that hit 8 are almost
  always model confusion, and a confused forecast is worse than no forecast.
- **2.4 KB per tool result.** A 400-row price history is ~12 KB of tokens for information
  the model will not use. Truncation is visible in the payload (`… [truncated, n chars]`)
  so the model knows it has the head of a longer answer.
- **Structured submission via a tool, not JSON in prose.** A tool call is validated by the
  provider's own schema before you ever see it, and a failed submission is cheap to retry
  in-loop rather than re-prompting from scratch.
- **Rejection is a first-class state.** `probability` outside [0,1], a missing evidence
  row, a rationale under 40 characters, or a future-dated citation all come back as
  `rejected: <reason>` and the loop continues. `Trace.rejections` is what you read when a
  forecast looks off.
- **No provider SDK.** `urllib` + `/chat/completions` in about 120 lines. The same class
  drives DeepSeek, OpenAI, OpenRouter, vLLM and Ollama; swapping is a base URL.

## Ensemble

`agent/ensemble.py` runs the agent N times and pools **log-odds**, not probabilities.
Averaging probabilities is asymmetric — it drags 0.05 towards 0.5 much harder than 0.95,
so the pooled number is biased towards the middle. Pooling logits is symmetric, and the
standard deviation of the logits is returned as `spread`: two runs that disagree 0.05 /
0.95 pool to 0.5, which is the honest reading of "these two disagree".

## The eval

`agent-eval` generates questions from the series' own history, so every one is already
resolved. Thresholds come in four families, because a single shape lets a single predictor
look good:

| family | threshold | base rate |
|---|---|---|
| `level` | today's close | ~55% |
| `quantile` | 60th percentile of the trailing 2 years | ~40% |
| `band_up` | +5% | ~20% |
| `band_down` | −5% | ~20% |

Every predictor answers the same questions, and scoring happens on the intersection, so
no one gets to skip the hard ones. Baselines: constant 0.5, the point-in-time base rate,
the analogue model, the agent, and (with `--also-blind`) the same model with no tools.

Metrics: Brier, log loss, ECE, direction, a 5-bin reliability table, Brier per family,
dollars per forecast, p50/p95 latency, mean tool calls, gate hits, failures.

96 questions on `^VXN` (2003-01-28 → 2025-10-07), 48 `level` and 48 `quantile`,
one DeepSeek `deepseek-chat` run each, `--workers 6`:

| predictor | Brier | LogLoss | ECE | direction | $/forecast | p50 |
|---|---|---|---|---|---|---|
| agent + tools | **0.1631** | 0.4815 | **0.036** | **72.9%** | 0.0011 | 7.6 s |
| analogue model | 0.1636 | 0.4719 | 0.053 | 71.9% | 0 | 0.04 s |
| agent, no tools | 0.2293 | 0.6486 | 0.115 | 55.2% | 0.0002 | 2.3 s |
| constant 0.5 | 0.2500 | 0.6931 | 0.073 | 42.7% | 0 | 0 |
| point-in-time base rate | 0.2647 | 0.7953 | 0.162 | 55.2% | 0 | 0 |

By family (Brier): `level` 0.2353 / analogue 0.2410 / blind 0.2458; `quantile` 0.0908 /
analogue 0.0863 / blind 0.2128.

What the numbers actually say:

- **The tools are the forecast.** Remove them and 0.1631 becomes 0.2293 with direction back
  at 55%. The model has no prior about `^VXN`; the loop and the seven tools do the work.
- **It ties the analogue model and is better calibrated** (ECE 0.036 vs 0.053): it inherits
  the model's judgement and then un-hedges the overconfidence around round levels. That is
  an argument for using the agent as the *explainer*, not the estimator, on numeric questions
  — which is exactly what the `$` and latency columns are there to make arguable.
- **The base rate loses to a coin flip** (0.2647 > 0.25). A hit rate on an absolute price
  level is carried away by drift. The agent caught this on the real NDX question — 1.15%
  base rate, threshold equal to today's close — and said so in its rationale.
- **The gate never fired** in 96 runs. That is the right amount of drama: it is cheap enough
  to keep for the case where it does, and `tests/test_agent.py` has three cases proving it
  would.

One run is one run: 96 questions gives a standard error near 0.02 on Brier differences, so
"ties the analogue model" is the honest reading of 0.1631 vs 0.1636, and "beats the blind
agent" is not in doubt at all.

## What this does not do

Said plainly, because it is the difference between a demo and a delivery:

- **No licensed news feed.** One interface, one null implementation. The retrieval path is
  built and gated but has no corpus behind it here.
- **No policy simulation.** Multi-agent policy runs need an economic model to act inside;
  `ensemble.py` is the multi-agent machinery without the domain model.
- **No queue, no database, no service.** Everything is files and a CLI. `forecasts/` is
  append-only by convention (`lock` refuses to overwrite), not by a database constraint.
  That is the next thing to build, not something to claim.
- **The Brier numbers are small-sample.** The `n` in every table is the number of
  questions actually run; a 96-question bench has wide intervals, and the report says so
  rather than rounding to a headline.

## Run it

```bash
export DEEPSEEK_API_KEY=...            # or FORECAST_LOG_API_KEY / OPENAI_API_KEY
forecast-log agent-forecast --runs 3   # shadow forecasts; never touches forecasts/
forecast-log agent-eval --limit 96 --stride 60 --also-blind
pytest tests/test_agent.py             # 19 tests, no network, no key
```

`FORECAST_LOG_MODEL` and `FORECAST_LOG_BASE_URL` point the same code at any
OpenAI-compatible endpoint, including a local vLLM or Ollama server.
